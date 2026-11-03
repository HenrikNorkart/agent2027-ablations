import asyncio
import re
import time
from logging import getLogger
import os

logger = getLogger(__name__)

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
from langchain_community.utilities import GoogleSerperAPIWrapper

from rogueone.utils import knowledge_agent_cfg, llm_cfg
from rogueone.utils.console import ConsoleManager
from rogueone.utils.prompt_dump import dump_prompt
from rogueone.utils.config import ExperimentConfig
from rogueone.utils import retrieval_audit
from rogueone.utils import instrumentation
from rogueone.utils import search_cache
from rogueone.llm.agents.core.knowledge import backend as knowledge_backend
from rogueone.llm.clients import create_openai_client_for_vllm

# Maximum query length accepted by the Serper/Google Search API.
_MAX_QUERY_LENGTH = 256

# Web-search provider (HQ #266: Serper ran out of credits). ``serper`` (default,
# legacy) or ``searxng``: a self-hosted SearXNG metasearch instance queried over
# its JSON API at ROGUE_ONE_SEARXNG_URL. The tool name, docstring and summariser
# prompt are unchanged, so rendered prompts are byte-identical across providers.
_SEARXNG_TOP_K = 10
_SEARXNG_TIMEOUT_S = 30.0


class SearxngSearch:
    """Minimal SearXNG client with the ``run(query) -> str`` shape of
    ``GoogleSerperAPIWrapper``: direct answers and infobox text first, then the
    snippets of the top ``k`` results joined by spaces. Zero results returns
    ``""`` (the caller then tries the shorter query variants); HTTP and network
    errors raise, so the caller's audit path records them."""

    def __init__(self, base_url: str, k: int = _SEARXNG_TOP_K, timeout_s: float = _SEARXNG_TIMEOUT_S):
        self.base_url = base_url.rstrip("/")
        self.k = k
        self.timeout_s = timeout_s

    def results(self, query: str) -> dict:
        import requests

        r = requests.get(
            f"{self.base_url}/search",
            params={"q": query, "format": "json", "safesearch": 0},
            timeout=self.timeout_s,
        )
        r.raise_for_status()
        return r.json()

    def run_with_meta(self, query: str) -> tuple[str, list[str], list[str]]:
        """``(text, urls, engines)``: ``text`` is exactly :meth:`run`'s return;
        ``urls`` are the sources whose text made it into ``text`` (E4 URL
        logging) and ``engines`` the SearXNG engines behind them."""
        data = self.results(query)
        parts: list[str] = []
        urls: list[str] = []
        engines: set[str] = set()

        def _note(item: dict) -> None:
            url = item.get("url") or item.get("id")
            if isinstance(url, str) and url.startswith("http") and url not in urls:
                urls.append(url)
            for eng in item.get("engines") or ([item["engine"]] if item.get("engine") else []):
                engines.add(str(eng))

        for a in data.get("answers") or []:
            text = a.get("answer") if isinstance(a, dict) else a
            if text:
                parts.append(str(text))
                if isinstance(a, dict):
                    _note(a)
        for box in data.get("infoboxes") or []:
            if box.get("content"):
                parts.append(str(box["content"]))
                _note(box)
        for res in (data.get("results") or [])[: self.k]:
            snippet = (res.get("content") or "").strip()
            if snippet:
                parts.append(snippet)
                _note(res)
        return " ".join(parts), urls, sorted(engines)

    def run(self, query: str) -> str:
        return self.run_with_meta(query)[0]


def web_search_provider() -> str:
    return os.getenv("ROGUE_ONE_WEB_SEARCH_PROVIDER", "serper").strip().lower() or "serper"


def make_search_client():
    provider = web_search_provider()
    if provider == "searxng":
        return SearxngSearch(os.getenv("ROGUE_ONE_SEARXNG_URL", "http://127.0.0.1:8888"))
    if provider != "serper":
        raise ValueError(f"ROGUE_ONE_WEB_SEARCH_PROVIDER must be serper or searxng, got {provider!r}")
    serper_key = os.getenv("SERPER_API_KEY", "")
    if serper_key:
        os.environ["SERPER_API_KEY"] = serper_key
    return GoogleSerperAPIWrapper()


def _serper_urls(results: dict) -> list[str]:
    """Source URLs in a Serper ``results()`` payload, in display order."""
    urls: list[str] = []
    for block in ("answerBox", "knowledgeGraph"):
        item = results.get(block) or {}
        for k in ("link", "descriptionLink", "website"):
            if isinstance(item.get(k), str) and item[k] not in urls:
                urls.append(item[k])
    for item in results.get("organic") or []:
        link = item.get("link")
        if isinstance(link, str) and link not in urls:
            urls.append(link)
    return urls


def _search_with_meta(search, query: str) -> tuple[str, list[str], list[str]]:
    """``(text, urls, engines)`` for one upstream query (E4).

    ``text`` is what ``search.run(query)`` returns. For Serper the wrapper's
    ``run`` is ``_parse_results`` over the same API payload that ``results``
    returns, so the text is unchanged while the URLs become visible; a wrapper
    without those methods falls back to ``run`` with no URLs.
    """
    if hasattr(search, "run_with_meta"):
        return search.run_with_meta(query)
    if hasattr(search, "results") and hasattr(search, "_parse_results"):
        payload = search.results(query)
        return search._parse_results(payload), _serper_urls(payload or {}), ["serper"]
    return search.run(query), [], []


def _sanitize_query(query: str) -> str:
    """Sanitize a search query for the Google Serper API.

    The Serper API proxies Google Search and inherits its restrictions:
    - The ``q`` parameter is sent as a URL query-string value.
    - Mathematical operators (``=``, ``+`` used as literal, ``<``, ``>``) and
      special Unicode characters (Greek letters, superscripts, …) may trigger
      a 400 Bad Request when URL-encoded.
    - Excessively long queries (>~256 chars) are also rejected.

    This function strips problematic characters and truncates to a safe length
    while preserving meaningful search terms.
    """
    # Replace mathematical / formula characters with spaces
    query = re.sub(r"[=<>≥≤≠≈∝∞±×÷√∑∏∫∂∇∆θφπσμλαβγδεζηικνξρτυψω]+", " ", query)
    # Remove stray special characters that are not useful in a search
    query = re.sub(r"[{}()\[\]|\\^~`@#$%&*_]", " ", query)
    # Collapse multiple spaces / newlines
    query = re.sub(r"\s+", " ", query).strip()
    # Truncate to safe length at a word boundary
    if len(query) > _MAX_QUERY_LENGTH:
        query = query[:_MAX_QUERY_LENGTH].rsplit(" ", 1)[0]
    return query


def _query_variants(sanitized: str) -> list[str]:
    """The sanitised query, then shorter/simpler fallbacks (first 8, 5, 3 words)."""
    # Build a list of query variants: original sanitized, then shorter/simpler fallbacks
    query_variants = [sanitized]

    # Fallback 1: keep only the first ~8 words
    words = sanitized.split()
    if len(words) > 8:
        query_variants.append(" ".join(words[:8]))

    # Fallback 2: keep only the first ~5 words
    if len(words) > 5:
        query_variants.append(" ".join(words[:5]))

    # Fallback 3: keep only the first ~3 words
    if len(words) > 3:
        query_variants.append(" ".join(words[:3]))
    return query_variants


def _web_search_impl(search, provider: str, cache, query: str) -> str:
    """Body of the ``_web_search`` tool (module level so it is unit-testable).

    Sanitises ``query``, then tries it and progressively shorter variants
    against the E4 cache and the upstream provider. Every attempt is one
    retrieval-transcript row carrying the URLs, the engines, the caller role
    and (when the cache is on) the cache state. Return strings are unchanged
    from the pre-E4 tool.
    """
    sanitized = _sanitize_query(query)
    caller_role = instrumentation.current_caller()
    if not sanitized:
        retrieval_audit.record(
            channel="web_search",
            event="query",
            caller="WebSearchAgent._web_search",
            provider=provider,
            query=query,
            result=None,
            ok=False,
            error="empty_after_sanitization",
            query_sanitized="",
            caller_role=caller_role,
        )
        return "Search query was empty after sanitization. Please provide a simpler, natural-language query."

    query_variants = _query_variants(sanitized)

    def _row(variant: str, attempt: int, t0: float, cache_state: str | None, **fields) -> None:
        if cache.enabled and cache_state is not None:
            fields.update(cache=cache.mode, cache_state=cache_state)
        retrieval_audit.record(
            channel="web_search",
            event="query",
            caller="WebSearchAgent._web_search",
            provider=provider,
            query=query,
            latency_s=time.monotonic() - t0,
            query_sanitized=variant,
            variant_index=attempt,
            caller_role=caller_role,
            **fields,
        )

    last_error = None
    for attempt, variant in enumerate(query_variants):
        _t0 = time.monotonic()
        cached = cache.get(provider, variant)
        if cached is not None:
            instrumentation.count("search_cache_hits")
            _row(variant, attempt, _t0, f"hit_{cached.source}", result=cached.text, ok=True,
                 urls=cached.urls, engines=cached.engines)
            return cached.text
        if cache.enabled:
            instrumentation.count("search_cache_misses")
        if not cache.reads_upstream:
            # replay: a miss never reaches the provider; it reads as an empty answer.
            _row(variant, attempt, _t0, "miss", result=None, ok=False, error="cache_miss_replay")
            continue
        instrumentation.count("serper_calls")
        try:
            search_results, urls, engines = _search_with_meta(search, variant)
            if search_results and str(search_results).strip():
                cache.put(provider, variant, search_results, urls, engines)
                _row(variant, attempt, _t0, "miss", result=search_results, ok=True,
                     urls=urls, engines=engines)
                return search_results
            instrumentation.count("serper_ok_false")
            _row(variant, attempt, _t0, "miss", result=search_results, ok=False, error="empty_result")
        except Exception as e:
            last_error = e
            instrumentation.count("serper_ok_false")
            _row(variant, attempt, _t0, "miss", result=None, ok=False, error=f"{type(e).__name__}: {e}")
            # Extract the JSON body from Serper HTTP errors for clearer diagnostics
            error_detail = str(e)
            if hasattr(e, "response") and e.response is not None:
                try:
                    body = e.response.json()
                    error_detail = body.get("message", error_detail)
                except Exception:
                    pass
            ConsoleManager.console_error_print(
                f"Web search failed for query '{variant}': {error_detail}"
            )
            # Non-retryable errors: credits exhausted, auth failures
            if "credit" in error_detail.lower() or "unauthorized" in error_detail.lower():
                return f"Search API unavailable: {error_detail}. Please check your SERPER_API_KEY credits."
            continue

    # All variants exhausted
    msg = f"All search attempts failed. Last error: {last_error}" if last_error else "No results found."
    return msg


_LOCAL_CORPUS_CACHE: dict = {}


def _get_local_corpus():
    """The E8b corpus for ``ROGUE_ONE_KNOWLEDGE_BACKEND=local``, loaded once per path."""
    key = str(knowledge_backend.local_corpus_dir())
    if key not in _LOCAL_CORPUS_CACHE:
        _LOCAL_CORPUS_CACHE[key] = knowledge_backend.load_local_corpus()
    return _LOCAL_CORPUS_CACHE[key]


def _local_search_impl(corpus, query: str) -> str:
    """Body of ``_web_search`` under ``local`` (E8b wiring point, A14 §4).

    Same sanitising and query-variant loop as the web tool; each variant asks
    ``corpus.search(variant)`` for ``(doc_id, score, text)`` hits. The first
    variant with hits returns them joined as the tool's text; all-zero scores move
    to the next variant, then "No results found.". Never touches the web provider,
    the E4 cache or the ``serper_calls`` counter. One transcript row per attempt,
    ``channel="local_corpus"``, with ``doc_ids`` / ``doc_scores``.
    """
    sanitized = _sanitize_query(query)
    caller_role = instrumentation.current_caller()
    if not sanitized:
        retrieval_audit.record(
            channel="local_corpus", event="query", caller="WebSearchAgent._web_search",
            provider="local_corpus", query=query, result=None, ok=False,
            error="empty_after_sanitization", query_sanitized="", caller_role=caller_role,
        )
        return "Search query was empty after sanitization. Please provide a simpler, natural-language query."
    for attempt, variant in enumerate(_query_variants(sanitized)):
        _t0 = time.monotonic()
        hits = [h for h in (corpus.search(variant) or []) if h[1] > 0]
        text = "\n\n".join(h[2] for h in hits)
        retrieval_audit.record(
            channel="local_corpus", event="query", caller="WebSearchAgent._web_search",
            provider="local_corpus", query=query, latency_s=time.monotonic() - _t0,
            query_sanitized=variant, variant_index=attempt, caller_role=caller_role,
            result=text or None, ok=bool(hits),
            doc_ids=[h[0] for h in hits], doc_scores=[float(h[1]) for h in hits],
            **({} if hits else {"error": "empty_result"}),
        )
        if hits:
            return text
    return "No results found."


async def _run_search_agent(query: str) -> str:
    """Execute a web search agent to answer a query using Google Serper API.

    Performs web searches and synthesizes results using an LLM agent to provide
    a comprehensive answer to the user's query.

    Args:
        query (str): The query string to search for on the web.

    Returns:
        str: The synthesized answer based on web search results.
    """
    # E9e: ``web`` (default) builds the provider client and cache exactly as before;
    # ``local`` and ``parametric`` never touch the provider or the E4 cache.
    kb = knowledge_backend.backend()
    if kb == "off":
        raise knowledge_backend.KnowledgeBackendError(
            "ROGUE_ONE_KNOWLEDGE_BACKEND=off: the search summariser must not be called"
        )
    if kb == "local":
        corpus = _get_local_corpus()
    else:
        search = make_search_client() if kb == "web" else None
        provider = web_search_provider()
        cache = search_cache.get_cache() if kb == "web" else None

    @function_tool
    async def _web_search(query: str):
        """Perform a web search for the provided query using Google Serper API.

        The query must be plain natural-language text — like what a human would
        type into Google.  Do NOT include mathematical notation, formulas,
        special symbols (=, <, >, θ, π, σ …), or programming syntax.

        Good: "variance formula statistics explained"
        Bad:  "var = sum((x - mean)^2) / n"

        If the first attempt fails, the tool will automatically retry with
        progressively simplified versions of the query.

        Args:
            query (str): A plain-text search query (no formulas or special chars).

        Returns:
            str: Search results as formatted text, or error message if search fails.
        """
        if kb == "local":
            return _local_search_impl(corpus, query)
        return _web_search_impl(search, provider, cache, query)

    # E7: ``global`` (default) = llm_cfg.endpoint / llm_cfg.model, as before.
    _endpoint, _model = knowledge_backend.summariser_route(
        llm_cfg, instrumentation.current_caller()
    )
    client = create_openai_client_for_vllm(
        base_url=_endpoint,
        api_key=llm_cfg.api_key,
        agent_name="WebSearchAgent",
    )

    agent = Agent(
        model_settings=ModelSettings(
            temperature=knowledge_agent_cfg.temperature,
            reasoning=knowledge_agent_cfg.reasoning,
        ),
        name="Knowledge lookup assistant",
        instructions=knowledge_backend.summariser_instructions(f"""
                    
            # The Setup:
            You are a knowledge lookup assistant that helps researchers find relevant information from the web.
            You are connected to the web via the web search tool.
            The user will provide you with queries related to their research.
                
            # Your Role and Tasks:
            Your goal is to answer the user's query in the best possible way using the information available from the web-search.
            You are operating as a RAG (Retrieval-Augmented Generation) agent.

            # Your Workflow:
            1. Break down the user's query into key concepts.
            2. For each key concept, use the web search tool to find relevant information from the web.
            3. If a search returns no useful results, rephrase and try again with different keywords. Do NOT give up after one failed search.
            4. Summarize the information retrieved from the web to answer the user's query.
            5. Provide a final response containing a relevant summary to the user based on the user's query.
                
            # CRITICAL — Search Query Rules:
            The web search tool sends queries to Google via the Serper API. You MUST follow these rules when crafting queries:
            - Use plain natural-language keywords ONLY — like what a human would type into Google.
            - NEVER include mathematical notation, formulas, or equations (e.g. "= sum(x) / n", "y = mx + b").
            - NEVER include special symbols: =, <, >, ≥, ≤, θ, π, σ, α, β, ∑, ∫, √, ±, ×, ÷, ^, or similar.
            - NEVER include programming syntax, code, or brackets.
            - Keep queries short: ideally 3-8 words. Longer queries are more likely to fail.
            - Use descriptive words instead of symbols: say "variance formula" not "s = sum(x - m) / n".
            - If a search fails, simplify the query further and retry with fewer, more common words.

            Good query examples: "variance formula statistics explained", "feature engineering classification", "ratio interaction features"
            Bad query examples: "mean = sum(x) / n", "F1 > 0.8 classification threshold", "σ = sqrt(variance)"

            # Important Guidelines:
            - Always provide a final answer based on your search results, even if limited information is found.
            - You must only base your summaries on the information found using the web search tool.
            - If a search fails, rephrase with simpler keywords and try again. Repeat up to 3-4 times before giving up.
            - Only respond with "Unable to provide an explanation at this time." as a LAST RESORT when ALL retry attempts have produced no results.
            """),
        model=OpenAIChatCompletionsModel(
            model=_model,
            openai_client=client,
        ),
        tools=[] if kb == "parametric" else [_web_search],
    )

    def _strip_reasoning(data: CallModelData) -> ModelInputData:
        """Strip reasoning history items for vLLM chat-completions compatibility."""
        filtered = [
            item
            for item in data.model_data.input
            if not (isinstance(item, dict) and item.get("type") == "reasoning")
        ]
        return ModelInputData(input=filtered, instructions=data.model_data.instructions)

    session = SQLiteSession(f"web_search_agent")

    dump_prompt(
        "websearch", "search", 0, agent.instructions, "user's query: <query>",
        tools=agent.tools, once=True,
    )
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
        ConsoleManager.console_error_print(f"Error occurred: {e}")
        return "Internal error."


class WebSearchAgent:
    """Agent that searches the web and synthesizes information to answer queries.

    Uses Google Serper API to perform web searches and an LLM to synthesize results
    into comprehensive answers. Suitable for current events and general knowledge queries.

    Attributes:
        cfg (ExperimentConfig): Experiment configuration.
    """

    def __init__(self, cfg: ExperimentConfig):
        """Initialize the WebSearchAgent.

        Args:
            cfg (ExperimentConfig): Configuration for the experiment.

        Raises:
            KnowledgeBackendError / NotImplementedError: the E9e/E7 switches hold an
                unknown value, or ``local`` cannot load its corpus (run refuses to start).
        """
        knowledge_backend.validate()
        self.cfg = cfg

    async def explain_query(self, query: str, caller_role: str | None = None) -> str:
        """Search the web and synthesize an answer to the query.

        Args:
            query (str): The query to answer via web search.
            caller_role (str | None): Role that asked (``scientist``,
                ``extractor`` or ``tester``). Its tokens and upstream search
                calls are charged to that role (E4 caller attribution); None
                keeps whatever caller is already in context.

        Returns:
            str: The synthesized answer from web search results.
        """
        role = caller_role or instrumentation.current_caller()
        with instrumentation.caller_scope(role):
            with retrieval_audit.timed_retrieval(
                channel=knowledge_backend.answer_channel(),
                event="answer",
                caller="WebSearchAgent.explain_query",
                query=query,
                caller_role=role,
            ) as audit:
                ans = await _run_search_agent(query)
                audit.result = ans
        return ans
