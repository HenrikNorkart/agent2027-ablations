import io
import contextlib
import asyncio
import re
from typing_extensions import Any
from logging import getLogger
from collections import defaultdict

from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
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
from langchain_chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings

from rogueone.utils import (
    llm_cfg,
    rouge_one_cfg,
    scientist_agent_cfg,
)
from rogueone.utils import scientist_signal as _sci_signal
from rogueone.utils.loop_switches import FEEDBACK_L0 as _FEEDBACK_L0
from rogueone.utils.config import ExperimentConfig
from rogueone.llm.agents.knowledge import KnowledgeAgent
from rogueone.llm.agents.web_search import WebSearchAgent
from rogueone.utils.console import ConsoleManager, tool_feedback
from rogueone.utils.prompt_dump import dump_prompt
from rogueone.llm.agents.core.knowledge import backend as knowledge_backend
from rogueone.dataclasses import AttributeExplanation
from rogueone.utils.wandb_utils import safe_wandb_log
from rogueone.llm.prompts.agent_roles import AgentRoles
from rogueone.llm.prompts import set_phased as _set_phased
from rogueone.llm.agents.core.tool_error_reporting import ToolErrorReporter, report_bypass
from rogueone.utils import instrumentation
from rogueone.llm.clients import create_openai_client_for_vllm
from rogueone.utils.sandbox import sandbox_scope
from rogueone.utils.pandas_exec import exec_with_last_expr

logger = getLogger(__name__)


def build_pandas_tool_scope(df_test_results, df_raw_data) -> dict:
    """Exec scope of ``generic_pandas_tool`` (single dict: globals == locals).

    ``ROGUE_ONE_SANDBOX=on`` restricts builtins and hides module file I/O (E13);
    ``off`` returns exactly the pre-E13 dict.
    """
    scope = {
        "pd": pd,
        "np": np,
        "df_test_results": df_test_results,
        "df_raw_data": df_raw_data,
    }
    return sandbox_scope(scope)


def apply_scientist_signal_to_pandas_tool(tool, signal: str):
    """E14: under ``inner_cv_l1`` rewrite the tool description's performance columns.

    ``train_fit`` returns the very same tool object (prompt unchanged).
    """
    if signal != _sci_signal.INNER_CV_L1:
        return tool
    tool.description = _sci_signal.l1_pandas_tool_description(tool.description)
    return tool


# --------------------------------------------------------------------------- E9b (L0)
# ROGUE_ONE_FEEDBACK_LEVEL=l0 (AGENT 2027 A11, open-loop planner): the Scientist loses
# the verifier-to-planner channel. Everything here is a no-op under ``full`` (default).
# The edits apply to the Scientist's rendered prompt only; ``AgentRoles`` is never
# edited, because ``AgentRoles.TESTER`` is also rendered into the Extractor prompt.

#: Attribute-table columns the L0 Scientist must not see (``attribute_lookup_tool``).
L0_HIDDEN_ATTRIBUTE_COLUMNS: tuple[str, ...] = ("Status", "Pruned Iteration", "Prune Rationale")

#: ``AgentRoles.TESTER`` without the two sentences about feedback to the Scientist (edit 3).
_TESTER_L0_TEXT = "\n".join(
    line
    for line in AgentRoles.TESTER.value.split("\n")
    if not line.strip().startswith(
        (
            "The Tester Agent provides detailed feedback to the Scientist Agent",
            "This feedback is crucial for guiding the Scientist Agent",
        )
    )
)

#: Edits 1-6 to the Scientist's rendered instructions (A11-T6); each ``old`` must occur
#: exactly once.
L0_PROMPT_EDITS: tuple[tuple[str, str], ...] = (
    (  # 1
        "based on the test results provided by the tester agent and the explanations of the attributes extracted by the extractor agent.",
        "based on the explanations of the attributes extracted by the extractor agent.",
    ),
    (  # 2
        "Your task is to analyze the test results, the attribute explanations, the focus history,",
        "Your task is to analyze the attribute explanations, the focus history,",
    ),
    (AgentRoles.TESTER.value, _TESTER_L0_TEXT),  # 3
    (  # 4
        "2. Identify strongest and weakest tested attributes.",
        "2. Review the attributes extracted so far.",
    ),
    (  # 5
        "Focus on exploratory strategies if current performance is poor or an early round, and exploitative refinements if performance is strong.",
        "Choose between exploratory and exploitative focuses using the focus history and the attributes extracted so far.",
    ),
    (  # 6
        ", or restoration of features the Tester has previously pruned with concrete arguments for why they should come back",
        "",
    ),
)

#: Edit 7: the ``generic_pandas_tool`` description (exact-once substring edits, applied
#: after the E14 signal rewrite, so they hold under ``train_fit`` and ``inner_cv_l1``).
_L0_TOOL_INTRO = (
    "Execute a generic pandas command on the scientist feedback dataframe 'df_test_results'",
    "Execute a generic pandas command on the trial-index dataframe 'df_test_results'",
)
_L0_TOOL_COLUMNS_RE = re.compile(
    r"The dataframe 'df_test_results' is organized as follows:.*?(?:"
    + re.escape(_sci_signal.TRAIN_FIT_COLUMNS_LINE)
    + "|"
    + re.escape(_sci_signal.L1_COLUMNS_LINE)
    + ")",
    re.S,
)
_L0_TOOL_COLUMNS_NEW = (
    "The dataframe 'df_test_results' is a trial-index dataframe with one row per earlier "
    "trial and only the column `id` (trial identifier with the form 'Trial <n>'; newer "
    "trials have higher numbers)."
)
_L0_TOOL_USE = (
    "Use this tool to explore the dataframe and generate insights into the test results.",
    "Use this tool to explore the data.",
)
_L0_TOOL_EXAMPLE_INTRO = (
    "For example, if you want to get the mean accuracy, you should use:",
    "For example, if you want to summarise the raw data, you should use:",
)
_L0_TOOL_EXAMPLE_RE = re.compile(
    re.escape(_sci_signal.TRAIN_FIT_EXAMPLE) + "|" + re.escape(_sci_signal.L1_EXAMPLE)
)
_L0_TOOL_EXAMPLE_NEW = "result = df_raw_data.describe()"


def _replace_once(text: str, old: str, new: str, where: str) -> str:
    n = text.count(old)
    if n != 1:
        raise RuntimeError(
            f"E9b L0 edit expects exactly one occurrence in the {where}, found {n}: {old[:80]!r}"
        )
    return text.replace(old, new)


def _sub_once(text: str, pattern: "re.Pattern[str]", new: str, where: str) -> str:
    out, n = pattern.subn(lambda _m: new, text)
    if n != 1:
        raise RuntimeError(
            f"E9b L0 edit expects exactly one match in the {where}, found {n}: {pattern.pattern[:80]!r}"
        )
    return out


def _scientist_view_test_results(df: pd.DataFrame | None, level: str) -> pd.DataFrame:
    """The Scientist's copy of the feedback pool (``df_test_results``).

    ``full``: exactly the pre-E9b copy. ``l0``: only the trial ``id`` column.
    """
    if level != _FEEDBACK_L0:
        return df.copy() if df is not None else pd.DataFrame()
    if df is None or "id" not in df.columns:
        return pd.DataFrame(columns=["id"])
    return df[["id"]].copy()


def _scientist_view_attributes(df: pd.DataFrame, level: str) -> pd.DataFrame:
    """``attribute_lookup_tool`` output; ``l0`` drops Status / Pruned Iteration / Prune Rationale."""
    if level != _FEEDBACK_L0:
        return df
    return df.drop(columns=list(L0_HIDDEN_ATTRIBUTE_COLUMNS), errors="ignore")


def _apply_l0_prompt_edits(instructions: str) -> str:
    """Edits 1-6 on the Scientist's rendered instructions (each applied exactly once)."""
    for old, new in L0_PROMPT_EDITS:
        instructions = _replace_once(instructions, old, new, "Scientist instructions")
    return instructions


def l0_pandas_tool_description(description: str) -> str:
    """Edit 7: the ``generic_pandas_tool`` description for ``l0``."""
    where = "generic_pandas_tool description"
    description = _replace_once(description, *_L0_TOOL_INTRO, where)
    description = _sub_once(description, _L0_TOOL_COLUMNS_RE, _L0_TOOL_COLUMNS_NEW, where)
    description = _replace_once(description, *_L0_TOOL_USE, where)
    description = _replace_once(description, *_L0_TOOL_EXAMPLE_INTRO, where)
    description = _sub_once(description, _L0_TOOL_EXAMPLE_RE, _L0_TOOL_EXAMPLE_NEW, where)
    return description


def apply_feedback_level_to_pandas_tool(tool, level: str):
    """E9b: under ``l0`` rewrite the tool description (edit 7); ``full`` returns the same object.

    The tool name stays ``generic_pandas_tool`` (E3 counters key on it).
    """
    if level != _FEEDBACK_L0:
        return tool
    tool.description = l0_pandas_tool_description(tool.description)
    return tool


def apply_feedback_level_to_instructions(instructions: str, level: str) -> str:
    """E9b: the Scientist's rendered instructions; unchanged under ``full``."""
    if level != _FEEDBACK_L0:
        return instructions
    return _apply_l0_prompt_edits(instructions)


def _build_scientist_model_settings(cfg) -> ModelSettings:
    """ModelSettings for the Scientist; ``max_tokens`` only when configured (E2)."""
    kwargs = {"temperature": cfg.temperature, "reasoning": cfg.reasoning}
    if getattr(cfg, "max_tokens", None) is not None:
        kwargs["max_tokens"] = cfg.max_tokens
    return ModelSettings(**kwargs)


class ScientistAgent:
    """Agent that determines feature evaluation focus areas based on test results.

    Analyzes model performance metrics and generates strategy recommendations for the next
    feature extraction iteration. Maintains a notebook of observations across iterations.
    Can search literature for domain knowledge to inform strategy decisions.

    Attributes:
        cfg_experiment (ExperimentConfig): Experiment configuration.
        knowledge_agent (KnowledgeAgent): Agent for literature searches.
        web_search_agent (WebSearchAgent): Agent for web searches.
        _note_book (str): Accumulated notes from previous iterations.
    """

    def __init__(self, cfg: ExperimentConfig):
        """Initialize the ScientistAgent.

        Args:
            cfg (ExperimentConfig): Configuration for the experiment.
        """
        self.cfg_experiment = cfg
        self.knowledge_agent = KnowledgeAgent(
            cfg=self.cfg_experiment, collection_name="cardiac_arrest"
        )
        self.web_search_agent = WebSearchAgent(cfg=self.cfg_experiment)

        self._note_book = ""
        self._tool_call_counter = defaultdict(int)
        self._tool_call_counter_success = defaultdict(int)
        self._tool_call_counter_failed = defaultdict(int)
        self._tool_error_reporter: ToolErrorReporter | None = None

    @staticmethod
    def _round_label(
        index: int, max_index: int, iteration_in_fold: int | None = None
    ) -> str:
        if iteration_in_fold is None:
            return f"Round {index}/{max_index}"
        return f"Round {iteration_in_fold + 1}/{max_index}"

    async def determine_focus(
        self,
        df_raw_data: pd.DataFrame,
        df_test_results: pd.DataFrame | None,
        df_attribute_explanations: pd.DataFrame,
        focus_history: list[str],
        index: int,
        max_index: int,
        iteration_in_fold: int | None = None,
        shared_session=None,
    ) -> str:
        """Determine the focus area for the next feature extraction iteration.

        Analyzes previous test results and generates a strategy for what features to
        extract next. Can search literature and web for domain insights to inform strategy.

        Args:
            df_raw_data (pd.DataFrame): The original raw dataset.
            df_test_results (pd.DataFrame | None): Test metrics from previous iteration.
            df_attribute_explanations (pd.DataFrame): Explanations of previously extracted features.
            focus_history (list[str]): List of focus areas from previous iterations.
            index (int): Current iteration number (0-indexed).
            max_index (int): Total number of iterations planned (per fold).
            iteration_in_fold (int, optional): 0-indexed iteration within the current
                fold. When given, the prompt renders "Round {iteration_in_fold+1}/{max_index}";
                otherwise the legacy global ``index`` is shown.
            shared_session (topology.SharedIterationSession, optional): E9f
                ``set_phased`` only. Replaces this call's session and system prompt and
                sends the PLAN phase message; ``None`` (default) is Full-v2.

        Returns:
            str: Description of the focus area for feature extraction.
        """

        _feedback_level = rouge_one_cfg.feedback_level
        _df_test_results = _scientist_view_test_results(df_test_results, _feedback_level)
        _df_base_data = df_raw_data.copy()

        self._tool_call_counter = defaultdict(int)
        self._tool_call_counter_success = defaultdict(int)
        self._tool_call_counter_failed = defaultdict(int)
        self._tool_error_reporter = ToolErrorReporter(
            agent_name=self.__class__.__name__,
            output_dir=self.cfg_experiment.output_dir,
            session_name=f"scientist_agent_step_{index or 0}",
        )
        if shared_session is not None:  # E9f: error records carry the SET phase
            self._tool_error_reporter.phase = "S"
            shared_session.enter("S")

        def _report_tool_error(
            *,
            tool_name: str,
            message: str,
            exception: Exception,
            code: str | None = None,
        ) -> str:
            self._tool_call_counter_failed[tool_name] += 1
            if self._tool_error_reporter is not None:
                self._tool_error_reporter.report(
                    tool_name=tool_name,
                    message=message,
                    exception=exception,
                    step=index,
                    context={"code": code} if code is not None else None,
                )
            return tool_feedback(ConsoleManager.console_error_print(message, code=code))

        self.web_search_agent

        @function_tool
        async def search_tool(query: str) -> str:
            """
            Search the web for relevant information to expand your knowledge.
            Use this tool to find supporting evidence or context.
            Use this tool to get a better understanding of relevant concepts.

            Args:
                query (str): The search query.
            Returns:
                str: A summary of the explanation.
            """
            try:
                self._tool_call_counter["search_tool"] += 1
                explanation = await self.web_search_agent.explain_query(query, caller_role="scientist")

                # Normalize and check for empty responses
                explanation = explanation if explanation is not None else ""
                if str(explanation).strip() == "":
                    report_bypass(self._tool_error_reporter, site="h2", tool_name="search_tool", message=f"No explanation found for query: {query}", step=index, kind="empty_search", agent_name="ScientistAgent")
                    return tool_feedback(ConsoleManager.console_error_print(
                        f"No explanation found for query: {query}"
                    ))

                max_len = min(1000, len(explanation))

                ConsoleManager.console_agent_logging(
                    "ScientistAgent",
                    "search_tool",
                    query,
                    post_message=explanation[:max_len]
                    + ("..." if len(explanation) > max_len else ""),
                )
                self._tool_call_counter_success["search_tool"] += 1
                return explanation
            except Exception as e:
                return _report_tool_error(
                    tool_name="search_tool",
                    message=f"Error during literature search: {e}",
                    exception=e,
                )

        # @function_tool
        # async def search_in_literature_tool(query: str) -> str:

        @function_tool
        async def generic_pandas_tool(command: str) -> Any:
            """
            Execute a generic pandas command on the scientist feedback dataframe 'df_test_results' or the base data dataframe 'df_raw_data'.
            The dataframe 'df_test_results' is organized as follows:
            - Each row is the result from one round of investigations (one trial).
            - Columns include:
                - `id`: trial identifier with the form 'Trial <n>' (capital T, single space). Newer trials have higher numbers.
                - `status`: 'completed', 'aborted', or 'skipped'.
                - `dataset_id`, `fold_seed`, `aborted_fold_id`, `skip_reason`: per-trial metadata.
                - `feature_assessment`: the markdown assessment report produced by the Tester for that trial.
                - Performance columns are TRAIN-set metrics only: `train_accuracy`, `train_precision`, `train_recall`, `train_f1`, `train_AUROC`, `train_MSE`, `train_MAE`, `train_R2`. Held-out metrics are deliberately not visible to the Scientist (Constitution II — leakage prevention).

            The dataframe 'df_raw_data' contains the original attributes used for aggregating.

            Use this tool to explore the dataframe and generate insights into the test results.
            This tool can be used to filter, aggregate, or view the data.

            The command should always:
            1. Use 'result' as the variable name for the output of the command.
               For example, if you want to get the mean accuracy, you should use:
               result = df_test_results['train_accuracy'].mean()
            2. Ensure that the command is a valid pandas command that can be executed in the context of the dataframe 'df_test_results' or 'df_raw_data'.
               You can use any pandas function or method that is applicable to dataframes.

            The tool will execute the following code:
                ```python
                locals_dict = {}

                with contextlib.redirect_stdout(io.StringIO()) as f:
                    exec(
                        f"{command}",
                        {
                            "pd": pd,
                            "np": np,
                            "df_test_results": _df_test_results,
                            "df_raw_data": _df_base_data,
                        },
                        locals_dict,
                    )

                result = locals_dict.get("result")
                return result
                ```

            Args:
                command (str): The pandas command to execute.

            Returns:
                Any: The result of the pandas command.
            """
            try:
                self._tool_call_counter["generic_pandas_tool"] += 1
                instrumentation.note_scientist_pandas(command)

                normalized_command = command
                if isinstance(command, str):
                    from textwrap import dedent as _dedent

                    normalized_command = _dedent(command).strip()
                    normalized_command = re.sub(
                        r";[ \t]*\n[ \t]+", "\n", normalized_command
                    )

                # Single namespace dict so nested lambdas/defs see
                # multi-statement assignments. See _exec_user_command in the
                # Extractor module for the rationale.
                scope = build_pandas_tool_scope(_df_test_results, _df_base_data)
                with contextlib.redirect_stdout(io.StringIO()) as f:
                    exec_with_last_expr(f"{normalized_command}", scope)

                result = scope.get("result", None)
                if result is None and rouge_one_cfg.exec_compat and f.getvalue().strip():
                    result = f.getvalue()
                if result is None:
                    report_bypass(self._tool_error_reporter, site="h3", tool_name="generic_pandas_tool", message="Command did not set 'result' variable or 'result' was set to None!", step=index, code=command, kind="no_result", agent_name="ScientistAgent")
                    return tool_feedback(ConsoleManager.console_error_print(
                        f"Error: Command did not set 'result' variable or 'result' was set to None! Command: {command}"
                    ))

                ConsoleManager.console_agent_logging(
                    "ScientistAgent", "generic_pandas_tool", code=command
                )
                self._tool_call_counter_success["generic_pandas_tool"] += 1
                return result
            except Exception as e:
                return _report_tool_error(
                    tool_name="generic_pandas_tool",
                    message=f"Error executing command: {e}",
                    exception=e,
                    code=command,
                )

        generic_pandas_tool = apply_scientist_signal_to_pandas_tool(
            generic_pandas_tool, rouge_one_cfg.scientist_signal
        )
        generic_pandas_tool = apply_feedback_level_to_pandas_tool(
            generic_pandas_tool, _feedback_level
        )

        @function_tool
        async def list_all_attributes() -> str:
            """
            List all attribute names available in the dataframe 'df'.
            These names can be used as input for the 'attribute_lookup_tool'.

            Returns:
                str: A comma-separated list of attribute names.
            """

            try:
                self._tool_call_counter["list_all_attributes"] += 1
                if len(df_attribute_explanations) == 0:
                    res = "No attributes available. This indicates that this is the first round of investigation."

                else:
                    res = df_attribute_explanations["Attribute"].to_list()
                ConsoleManager.console_agent_logging(
                    "ScientistAgent",
                    "list_all_attributes",
                    f"List all attributes ({len(df_attribute_explanations)} total)",
                )

                self._tool_call_counter_success["list_all_attributes"] += 1
                return res
            except Exception as e:
                return _report_tool_error(
                    tool_name="list_all_attributes",
                    message=f"Error listing attributes: {e}",
                    exception=e,
                )

        @function_tool
        async def attribute_lookup_tool(attribute_names_list: list[str]) -> str:
            """
            Look up the explanation for a given list of attribute names from the attribute_explanations dictionary.

            Args:
                attribute_names_list (list[str]): The names of the attributes to look up.

            Returns:
                str: The explanation for the attribute, or a message indicating it was not found.
            """

            try:
                self._tool_call_counter["attribute_lookup_tool"] += 1
                res = _scientist_view_attributes(
                    df_attribute_explanations[
                        df_attribute_explanations["Attribute"].isin(attribute_names_list)
                    ],
                    _feedback_level,
                )
                ConsoleManager.console_agent_logging(
                    "ScientistAgent",
                    "attribute_lookup_tool",
                    f"Lookup attributes: {attribute_names_list}",
                )

                self._tool_call_counter_success["attribute_lookup_tool"] += 1
                return res
            except Exception as e:
                return _report_tool_error(
                    tool_name="attribute_lookup_tool",
                    message=f"Error looking up attributes: {e}",
                    exception=e,
                )

        @function_tool
        async def take_a_note(note: str) -> str:
            """
            Take a note and add it to the notebook.
            This can be used to keep track of important information during the investigation and to pass information on to next iteration.

            Args:
                note (str): The note to be added.

            Returns:
                str: A confirmation message.
            """

            try:
                self._tool_call_counter["take_a_note"] += 1
                ConsoleManager.console_agent_logging(
                    "ScientistAgent",
                    "take_a_note",
                    f"Taking note: {note}",
                )

                self._note_book += f"Trail {index}:\n{note}\n"

                self._tool_call_counter_success["take_a_note"] += 1
                return "Note added successfully."
            except Exception as e:
                return _report_tool_error(
                    tool_name="take_a_note",
                    message=f"Error taking note: {e}",
                    exception=e,
                )

        @function_tool
        async def read_notes() -> str:
            """
            Read all notes from the notebook.

            Returns:
                str: The contents of the notebook.
            """

            try:
                self._tool_call_counter["read_notes"] += 1
                ConsoleManager.console_agent_logging(
                    "ScientistAgent",
                    "read_notes",
                    "Reading all notes",
                )

                self._tool_call_counter_success["read_notes"] += 1
                return self._note_book
            except Exception as e:
                return _report_tool_error(
                    tool_name="read_notes",
                    message=f"Error reading notes: {e}",
                    exception=e,
                )

        async def main():

            client = create_openai_client_for_vllm(
                base_url=llm_cfg.endpoint_for("scientist"),
                api_key=llm_cfg.api_key,
                agent_name="ScientistAgent",
            )

            agent = Agent(
                model_settings=_build_scientist_model_settings(scientist_agent_cfg),
                name="Researcher Agent",
                instructions=f"""
                
                    The setting:
                    You are the Scientist Agent leading focus selection for iterative feature engineering.
                    Your primary responsibility is to determine the focus of the investigation based on the test results provided by the tester agent and the explanations of the attributes extracted by the extractor agent.
                    Your task is to analyze the test results, the attribute explanations, the focus history, and compare with existing information from the web to determine the focus area of the investigation. Find the focus area that will lead to the most useful hypotheses.
                    Use the tools available to you to explore the data, gather information, and generate insights.
                    
                    Global goal:
                    {AgentRoles.GLOBAL_GOAL.value}
                    

                    # Team context:
                    - Extractor Agent: {AgentRoles.EXTRACTOR.value}
                    - Tester Agent: {AgentRoles.TESTER.value}

                    # Objective:
                    - Choose the next focus area that most likely improves model performance.

                    # Required workflow:
                    1. Review prior notes and focus history.
                    2. Identify strongest and weakest tested attributes.
                    3. Use tools to gather missing context (search, lookup, quick dataframe checks). Always perform at least one web search.
                    4. Select one focus for the next extractor round and classify it as exploratory or exploitative.
                    5. Save key reasoning in notes for future rounds.

                    # Hard rules:
                    - Keep focus within what Extractor can implement.
                    - Prefer first-round baseline on raw attributes unless evidence suggests otherwise.
                    - Use search_tool actively when uncertain.
                    - Avoid repeating previous focus unless new evidence justifies it.
                    - Focus on exploratory strategies if current performance is poor or an early round, and exploitative refinements if performance is strong.
                    - **Every focus MUST be additive: it must request at least one concrete new feature, encoding, interaction, or transform the Extractor can build via `append_new_attribute`.** Pure-pruning focuses ("remove low-importance features", "keep only top-K", "drop redundant columns") are forbidden — pruning is the Tester's job, the Extractor has no pruning tool, and a pruning-only focus produces zero new features and wastes the iteration. If you believe pruning is needed, pair it with at least one additive request (e.g. "build target-encoded versions of the categorical columns; the Tester may then prune redundant one-hots").
                    - **Replacement framing must still be additive.** Phrases like "replace X with Y" leave the Extractor unsure whether to add or delete. Phrase the focus as "introduce Y (the Tester will retire X if it is dominated)" instead, so the Extractor's job is unambiguously to call `append_new_attribute` for the new feature.
                    - **Avoid focus repetition once the obvious feature space is covered.** Small datasets (e.g. a dataset with only a handful of raw categorical columns) saturate after 1–2 iterations of "ordinal encodings + one-hot + pairwise interactions". When the focus history shows the same encoding/interaction theme twice, pivot to **less obvious** additive ideas: log/sqrt/box-cox transforms, three-way interactions, ratio features, target-conditional aggregates (only if you can compute them without leaking the label), or restoration of features the Tester has previously pruned with concrete arguments for why they should come back. Do not loop on the same theme.

                    # Extractor implementation constraints:
                    - Tabular features: pandas and numpy only.
                    - Extractor can ONLY add features via `append_new_attribute`. It cannot delete, rename, or replace existing features. Plan focuses around what the Extractor can do.

                    # Task context and predictive goal:
                    {self.cfg_experiment.context_and_goal}

                    # Output requirements:
                    - Return a concise focus statement plus a short justification.
                    - Keep the final answer to a few sentences.
                    - Do not provide long chain-of-thought narratives.
                    """,
                model=OpenAIChatCompletionsModel(
                    model=llm_cfg.scientist_model,
                    openai_client=client,
                ),
                tools=[
                    generic_pandas_tool,
                    # search_in_literature_tool,
                    search_tool,
                    attribute_lookup_tool,
                    list_all_attributes,
                    take_a_note,
                    read_notes,
                ],
            )
            # E9b: under l0 the rendered instructions get edits 1-6 (no-op under full).
            agent.instructions = apply_feedback_level_to_instructions(
                agent.instructions, _feedback_level
            )
            # E9e: drops search only under ROGUE_ONE_KNOWLEDGE_BACKEND=off (A13 edits 1-4).
            agent.instructions = knowledge_backend.role_instructions("scientist", agent.instructions)
            agent.tools = knowledge_backend.role_tools(agent.tools)

            # E9f set_phased: neutral system prompt; the role block moves into the PLAN
            # phase message (sent on the first attempt only). None = Full-v2, unchanged.
            _set_first_message = None
            if shared_session is not None:
                _set_full_instructions = agent.instructions
                agent.instructions = shared_session.system_prompt

            if len(focus_history) > 0:
                prompt = f"{self._round_label(index, max_index, iteration_in_fold)}: Determine the focus for the next round of attribute extraction based the focus history: {focus_history}."
            else:
                prompt = f"Round 1: Determine the focus for the next round of attribute extraction. This is the first round of investigation, thus there is no focus history or extracted attributes."

            def _strip_reasoning(data: CallModelData) -> ModelInputData:
                """Strip reasoning history items for vLLM chat-completions compatibility."""
                filtered = []
                for item in data.model_data.input:
                    if isinstance(item, dict) and item.get("type") == "reasoning":
                        continue
                    if isinstance(item, dict) and isinstance(item.get("content"), str):
                        cleaned_item = dict(item)
                        cleaned_item["content"] = re.sub(
                            r"<\|[^|]+\|>",
                            " ",
                            cleaned_item["content"],
                        )
                        filtered.append(cleaned_item)
                    else:
                        filtered.append(item)
                return ModelInputData(
                    input=filtered, instructions=data.model_data.instructions
                )

            _run_config = RunConfig(call_model_input_filter=_strip_reasoning)

            if shared_session is None:
                session = SQLiteSession(f"scientist_agent_session_step_{index or 0}")
                instrumentation.set_session(f"scientist_agent_session_step_{index or 0}")
            else:  # E9f: the iteration's shared session
                session = shared_session.session
                instrumentation.set_session(shared_session.session_name)
                _set_first_message = _set_phased.phase_message(
                    "S", _set_full_instructions, prompt
                )

            def _heuristic_fallback_focus() -> str:
                if len(focus_history) == 0:
                    return (
                        "Use a raw-attribute baseline first, then add simple domain "
                        "interaction features to test immediate predictive signal."
                    )

                if self.cfg_experiment.task_type == "classification":
                    return (
                        "Prioritize exploitative refinement of the strongest active "
                        "attributes and add one complementary interaction feature."
                    )

                return (
                    "Prioritize features that stabilize regression error and reduce "
                    "scale-sensitive variance across folds."
                )

            dump_prompt(
                "scientist", "focus", index, agent.instructions,
                prompt if shared_session is None else _set_first_message,
                tools=agent.tools, output_dir=self.cfg_experiment.output_dir,
            )
            for _ in range(100):
                instrumentation.count("runner_runs_scientist")
                try:
                    focus = await Runner().run(
                        agent,
                        prompt
                        if shared_session is None
                        else shared_session.input_for("S", _set_first_message, prompt),
                        session=session,
                        max_turns=scientist_agent_cfg.max_iterations,
                        run_config=_run_config,
                    )
                    if len(focus.final_output) > 10:
                        return focus.final_output
                except Exception as e:
                    self._tool_call_counter_failed["runner"] += 1
                    if shared_session is not None:  # E9f: off-phase tool call counter
                        shared_session.note_runner_exception("S", e)
                    if self._tool_error_reporter is not None:
                        self._tool_error_reporter.report(
                            tool_name="runner",
                            message=f"Error during Scientist Agent run: {e}",
                            exception=e,
                            step=index,
                            context=(
                                {"session_name": f"scientist_agent_session_step_{index or 0}"}
                                if shared_session is None
                                else {"session_name": shared_session.session_name, "phase": "S"}
                            ),
                        )
                    ConsoleManager.console_error_print(
                        f"Error during Scientist Agent run: {e}"
                    )
                    continue

            # Fallback path for model profiles that intermittently fail while planning tool calls.
            # Use a no-tool prompt to still produce a compact, actionable focus statement.
            try:
                fallback_agent = Agent(
                    model_settings=ModelSettings(
                        temperature=min(float(scientist_agent_cfg.temperature), 0.3),
                        reasoning=None,
                    ),
                    name="Researcher Agent Fallback",
                    instructions=(
                        "Return one short focus statement for the next extraction round. "
                        "No tools. No chain-of-thought."
                    ),
                    model=OpenAIChatCompletionsModel(
                        model=llm_cfg.scientist_model,
                        openai_client=client,
                    ),
                    tools=[],
                )
                dump_prompt(
                    "scientist", "fallback", index, fallback_agent.instructions, prompt,
                    tools=fallback_agent.tools, output_dir=self.cfg_experiment.output_dir,
                )
                instrumentation.set_session(f"scientist_agent_fallback_session_step_{index or 0}")
                instrumentation.count("runner_runs_scientist")
                fallback = await Runner().run(
                    fallback_agent,
                    prompt,
                    session=SQLiteSession(
                        f"scientist_agent_fallback_session_step_{index or 0}"
                    ),
                    max_turns=4,
                    run_config=_run_config,
                )
                final_output = str(getattr(fallback, "final_output", "")).strip()
                if len(final_output) > 10:
                    instrumentation.count("fallback_scientist_agent")
                    instrumentation.note_focus_source("fallback_agent")
                    if shared_session is not None:  # E9f: fallbacks are counted per phase
                        shared_session.note_fallback("S", "fallback_agent")
                    return final_output
            except Exception as fallback_exc:
                if shared_session is not None:
                    shared_session.note_runner_exception("S", fallback_exc)
                if self._tool_error_reporter is not None:
                    self._tool_error_reporter.report(
                        tool_name="runner_fallback",
                        message=f"Error during Scientist Agent fallback run: {fallback_exc}",
                        exception=fallback_exc,
                        step=index,
                        context={
                            "session_name": (
                                f"scientist_agent_fallback_session_step_{index or 0}"
                            ),
                            **({"phase": "S"} if shared_session is not None else {}),
                        },
                    )

            instrumentation.note_focus_source("heuristic")
            if shared_session is not None:
                shared_session.note_fallback("S", "heuristic")
            return _heuristic_fallback_focus()

        focus = await main()

        # safe_wandb_log demotes step to auto-increment whenever the requested
        # step would regress (which happens here on every iteration after the
        # first because the LLM client's per-call token logs auto-advance the
        # run step, see safe_wandb_log docstring).
        safe_wandb_log({"Scientist/Research_focus": focus}, prefer_step=index)
        safe_wandb_log({"Research_focus": focus}, prefer_step=index)

        safe_wandb_log(
            {
                f"Scientist/tool_usage/{tool_name}": count
                for tool_name, count in self._tool_call_counter.items()
            },
            prefer_step=index,
        )
        safe_wandb_log(
            {
                f"Scientist/tool_usage_success/{tool_name}": count
                for tool_name, count in self._tool_call_counter_success.items()
            },
            prefer_step=index,
        )
        safe_wandb_log(
            {
                f"Scientist/tool_usage_failed/{tool_name}": count
                for tool_name, count in self._tool_call_counter_failed.items()
            },
            prefer_step=index,
        )

        if self._tool_error_reporter is not None:
            safe_wandb_log(
                {
                    "Scientist/tool_errors/total": self._tool_error_reporter.total_errors,
                    "Scientist/tool_errors/log_path": str(
                        self._tool_error_reporter.file_path
                    ),
                },
                prefer_step=index,
            )

        return focus


if __name__ == "__main__":

    import json
    from pathlib import Path
    from rogueone.utils.config import ExperimentConfig

    cfg = ExperimentConfig.from_yaml(
        "./tasks/suppression_rato/config.yml"
    )

    patient_attributes = pd.read_csv("./tmp/df_patients.csv")
    attribute_explanations = json.loads(
        Path("./tmp/attribute_descriptions.json").read_text()
    )

    agent = ScientistAgent(cfg=cfg)
    focus = agent.determine_focus(
        df_raw_data=patient_attributes,
        df_test_results=pd.DataFrame(),
        df_attribute_explanations=attribute_explanations,
        focus_history=[],
        index=0,
        max_index=5,
    )

    print("-" * 80)

    print(focus)
