import asyncio
import json
from logging import getLogger

logger = getLogger(__name__)

from pydantic import BaseModel, Field
from openai import OpenAI
from agents import (
    Agent,
    Runner,
    RunConfig,
    OpenAIChatCompletionsModel,
    ModelSettings,
    function_tool,
)
from agents.memory.sqlite_session import SQLiteSession
from agents.run import ModelInputData, CallModelData
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document

from rogueone.utils import embedding_cfg, llm_cfg, knowledge_agent_cfg
from rogueone.utils.console import ConsoleManager
from rogueone.utils.config import ExperimentConfig
from rogueone.utils import retrieval_audit
from rogueone.llm.clients import create_openai_client_for_vllm


class SearchArgs(BaseModel):
    """Arguments for vector similarity search.

    Attributes:
        query (str): The search query string.
        k (int): Number of most similar documents to return.
    """

    query: str = Field(..., description="The search query.")
    k: int = Field(5, description="The number of similar documents to return.")


async def _run_explanation_agent(query: str, db: Chroma) -> str:
    """Execute a RAG agent to answer queries using vector database retrieval.

    Uses similarity search to retrieve relevant documents from the vector database,
    then passes them to an LLM agent to synthesize an answer.

    Args:
        query (str): The user's query to answer.
        db (Chroma): The vector database to search.

    Returns:
        str: The agent's synthesized answer based on retrieved documents.
    """
    client = create_openai_client_for_vllm(
        base_url=llm_cfg.endpoint,
        api_key=llm_cfg.api_key,
        agent_name="KnowledgeAgent",
    )

    @function_tool
    async def _similarity_search(args: SearchArgs):
        """Search for semantically similar documents in the vector database.

        Args:
            args (SearchArgs): Query string and number of results to return.

        Returns:
            str: Concatenated content from the k most similar documents.
        """
        # !!! THIS TOOL HAS NO IMPLEMENTATION AND RETURNS None ON EVERY CALL. !!!
        #
        # `db` is never queried -- it is used only to name the SQLiteSession
        # below. Every reported run therefore had the vector-DB retrieval
        # channel inert: the Extractor's `search_in_literature_tool` and the
        # Tester's `methodology_lookup_tool` both route here. See Finding 40 in
        # paper/reviewer_feedback/CAMERA_READY_EXPERIMENT_PLAN.md.
        #
        # Left INERT deliberately. Implementing it would change what the system
        # does relative to every number already measured, which is a
        # paper-claim-altering decision reserved for Henrik (cf. the Finding 39
        # "freeze and disclose" precedent). The call below only records the
        # fact; it does not repair it.
        retrieval_audit.record(
            channel="vector_db",
            event="query",
            caller="KnowledgeAgent._similarity_search",
            query=getattr(args, "query", None),
            result=None,
            ok=False,
            error="tool_not_implemented: body is a docstring only, returns None",
            k=getattr(args, "k", None),
            collection=getattr(db, "_collection_name", None),
        )

    agent = Agent(
        model_settings=ModelSettings(
            temperature=knowledge_agent_cfg.temperature,
            reasoning=knowledge_agent_cfg.reasoning,
        ),
        name="Knowledge lookup assistant",
        instructions=f"""
                    
            # The Setup:
            You are a knowledge lookup assistant that helps researchers find relevant information from a database.
            You are connected to a vector database that contains relevant literature and documents.
            By using the 'similarity_search' tool, you can retrieve information that semantically similar to the user's query.
                
            # Your Role and Tasks:
            Your goal is to answer the user's query in the best possible way using the information available in the vector database.
            You are operating as a RAG (Retrieval-Augmented Generation) agent.

            # Your Workflow:
            1. Break down the user's query into key concepts.
            2. For each key concept, use the 'similarity_search' tool to find relevant information from the database.
            3. Summarize the information retrieved from the database to answer the user's query.
            4. Provide a final response containing a relevant summary to the user based on the user's query.
                
            # Important Guidelines:
            - Always provide a final answer based on your search results, even if limited information is found.
            - You must only base your summaries on the information found using the 'similarity_search' tool.
            - Respond with "Unable to provide an explanation at this time." if no relevant information is found.
            """,
        model=OpenAIChatCompletionsModel(
            model=llm_cfg.model,
            openai_client=client,
        ),
        tools=[_similarity_search],
    )

    def _strip_reasoning(data: CallModelData) -> ModelInputData:
        """Strip reasoning history items for vLLM chat-completions compatibility."""
        filtered = [
            item
            for item in data.model_data.input
            if not (isinstance(item, dict) and item.get("type") == "reasoning")
        ]
        return ModelInputData(input=filtered, instructions=data.model_data.instructions)

    session = SQLiteSession(f"knowledge_agent_session_db_{db._collection_name}")

    try:
        res = await Runner.run(
            agent,
            f"user's query: {query}",
            max_turns=knowledge_agent_cfg.max_iterations,
            session=session,
            run_config=RunConfig(call_model_input_filter=_strip_reasoning),
        )

        if not res.final_output or res.final_output == "":
            return "Unable to provide an explanation at this time."

        return res.final_output

    except Exception as e:
        print(f"Error occurred: {e}")
        return "Internal error."


class KnowledgeAgent:
    """Agent that retrieves and synthesizes information from a vector knowledge database.

    Uses embeddings-based similarity search to find relevant documents, then passes them
    to an LLM to generate answers. Suitable for RAG (Retrieval-Augmented Generation)
    workflows.

    Attributes:
        _cfg (ExperimentConfig): Experiment configuration.
        _vector_db (Chroma): The vector database containing knowledge documents.
    """

    def __init__(self, cfg: ExperimentConfig, collection_name: str | None = None):
        """Initialize the KnowledgeAgent with vector database connection.

        Args:
            cfg (ExperimentConfig): Experiment configuration.
            collection_name (str, optional): Name of the vector database collection.
                Defaults to value from cfg.knowledge_db_collection_name.
        """
        self._cfg = cfg

        self._vector_db = Chroma(
            collection_name=collection_name or self._cfg.knowledge_db_collection_name,
            persist_directory=str(self._cfg.knowledge_db_path),
            embedding_function=OpenAIEmbeddings(
                base_url=embedding_cfg.endpoint,
                api_key=embedding_cfg.api_key,
                model=embedding_cfg.model,
                tiktoken_enabled=True,
                chunk_size=embedding_cfg.chunk_size,
            ),
        )

    async def explain_query(self, query: str) -> str:
        """Retrieve and synthesize information to answer a query.

        Searches the vector database for relevant documents using semantic similarity,
        then uses an RAG agent to synthesize an answer based on the retrieved documents.

        Args:
            query (str): The query to answer.

        Returns:
            str: The synthesized answer from the knowledge base.
        """
        try:
            doc_count = self._vector_db._collection.count()  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - a diagnostic must never break a run
            doc_count = None

        with retrieval_audit.timed_retrieval(
            channel="vector_db",
            event="answer",
            caller="KnowledgeAgent.explain_query",
            query=query,
            collection=getattr(self._vector_db, "_collection_name", None),
            doc_count=doc_count,
        ) as audit:
            ans = await _run_explanation_agent(query, db=self._vector_db)
            audit.result = ans
        return ans

    def upload_pdf(self, pdf_path: str) -> int:
        """
        Loads a PDF, splits it into chunks, and stores it in the vector DB.
        Returns the number of chunks added.
        """
        print(f"Uploading PDF: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        for i in range(0, len(documents), 5000):
            self._vector_db.add_documents(documents[i : min(i + 5000, len(documents))])

        return len(documents)

    def upload_json(self, json_path: str) -> int:
        """
        Loads a JSON file, splits it into chunks, and stores it in the vector DB.
        Returns the number of chunks added.
        """
        print(f"Uploading JSON: {json_path}")

        with open(json_path, "r") as f:
            data = json.load(f)

        title = data.get("title", "Unknown Title")
        print(f"Title: {title}")
        sections = data.get("content", [])
        print(f"Number of sections: {len(sections)}")

        # Assuming the JSON structure is a list of text entries
        documents = [
            Document(
                page_content=dedent(
                    f"""
                Title: {title}
                Section: {entry.get("section_title", "Unknown Section")}
                {entry.get("content", "")}
                """
                ).strip(),
            )
            for entry in sections
        ]

        for i in range(0, len(documents), 5000):
            self._vector_db.add_documents(documents[i : min(i + 5000, len(documents))])

        return len(documents)

    def num_documents(self) -> int:
        """Return the number of documents in the vector store."""
        return self._vector_db._collection.count()  # type: ignore


if __name__ == "__main__":

    from pathlib import Path
    import asyncio

    base_path = Path("./tasks/vehicle/literatur/json")

    cfg = ExperimentConfig.from_yaml("./tasks/vehicle/config.yml")

    agent = KnowledgeAgent(cfg, collection_name=cfg.knowledge_db_collection_name)

    query = "What are common practices for feature extraction?"
    print(f"Top 3 results for query '{query}':")

    print(f"Number of documents in the vector DB: {agent.num_documents()}")

    async def main():
        res = await agent.explain_query(query)
        print(f"Query: {query}")
        print(f"Summary: {res}")

    asyncio.run(main())
