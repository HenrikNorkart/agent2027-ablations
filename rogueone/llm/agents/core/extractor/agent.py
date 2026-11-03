import io
import contextlib
import asyncio
import re
from logging import getLogger
from collections import defaultdict
from textwrap import dedent

import numpy as np
from pydantic import BaseModel, Field
import pandas as pd
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

from rogueone.dataclasses.attributes import AttributeExplanation
from rogueone.utils.console import (
    ConsoleManager,
    tool_feedback,
    truncate_tool_output,
)
from rogueone.utils.prompt_dump import dump_prompt
from rogueone.llm.agents.core.knowledge import backend as knowledge_backend
from rogueone.utils import llm_cfg, extractor_agent_cfg, rouge_one_cfg
from rogueone.utils.config import ExperimentConfig
from rogueone.utils.wandb_utils import safe_wandb_log
from rogueone.utils.pandas_exec import (
    exec_user_command as _exec_user_command,
    extract_missing_column as _extract_missing_column,
    normalize_user_command as _normalize_user_command,
)
from rogueone.llm.prompts.agent_roles import AgentRoles
from rogueone.llm.prompts import set_phased as _set_phased
from rogueone.llm.agents.knowledge import KnowledgeAgent
from rogueone.llm.agents.web_search import WebSearchAgent
from rogueone.llm.agents.core.tool_error_reporting import ToolErrorReporter, report_bypass
from rogueone.utils import instrumentation
from rogueone.llm.clients import create_openai_client_for_vllm

logger = getLogger(__name__)


def _ordinal_encode_categorical(
    df: pd.DataFrame,
    *,
    column: str,
    df_features: pd.DataFrame,
) -> pd.Series | None:
    """Return a deterministic ordinal encoding of ``df[column]`` aligned to
    ``df_features['id']``, or None when the column cannot be encoded
    (missing values, no unique categories, alignment failure).

    Categories are sorted alphabetically *before* encoding so the mapping is
    independent of row order and reproducible across runs. Used by the
    Extractor's categorical fallback path: CAR (and other all-categorical
    datasets) used to abort when the LLM Extractor produced 0 features
    because the prior numeric-only fallback had nothing to add.
    """
    if column not in df.columns:
        return None
    if "id" not in df.columns or "id" not in df_features.columns:
        return None

    try:
        grouped = df.groupby("id")[column].first()
    except Exception:
        return None

    unique_vals = grouped.dropna().astype(str).unique().tolist()
    if not unique_vals:
        return None

    categories = sorted(unique_vals)
    encoder = {c: i for i, c in enumerate(categories)}
    encoded = grouped.astype(str).map(encoder)
    if encoded.isna().any():
        return None

    result = df_features["id"].map(encoded)
    if result.isna().any():
        return None

    return result.astype(float)


def _build_extractor_model_settings(cfg) -> "ModelSettings":
    """Build a :class:`ModelSettings` from an :class:`AgentConfig`-like object.

    Includes ``max_tokens`` only when configured so we don't override SDK defaults.
    """
    kwargs = {
        "temperature": cfg.temperature,
        "reasoning": cfg.reasoning,
    }
    if getattr(cfg, "max_tokens", None) is not None:
        kwargs["max_tokens"] = cfg.max_tokens
    return ModelSettings(**kwargs)


class PandasCommandArgs(BaseModel):
    """Arguments for executing pandas analysis commands.

    Attributes:
        command (str): The pandas command to execute. Should operate on df or df_features.
        explanation (str): User's explanation of what the command does and why it's useful.
    """

    command: str = Field(
        ...,
        description="The pandas command to used to analyze the data. This command should return a pandas Series or DataFrame with the same length as df_features when executed. The command should be a valid pandas command starting with 'df' or 'df_features' as the dataframe variable.",
    )
    explanation: str = Field(
        ...,
        description="""
            A brief explanation of what the command does and a justification for why it is useful for extracting patient attributes.
            Answer how this command helps in identifying relevant attributes based on the focus area provided by the Scientist Agent.
        """,
    )


class AttributeExplainingArgs(BaseModel):
    """Arguments for adding and explaining a new extracted attribute.

    Attributes:
        name (str): Name of the new attribute to add.
        command (str): Pandas command that calculates the attribute value. Must include groupby('id').
        argumentation (str): Clinical or domain-specific reasoning for why this attribute is relevant.
    """

    name: str = Field(..., description="The name of the attribute to explain.")
    command: str = Field(
        ...,
        description=(
            "Python code that calculates the new attribute. May be a single statement or multiple "
            "statements separated by newlines or semicolons. Define any intermediate variables "
            "(e.g. ordinal mappings, helper masks) **within this same call** — they do not persist "
            "between calls. The final value must be assigned to a variable named `result`. The "
            "command must include a groupby('id') operation so the result aligns with df_features. "
            "Use bitwise `&`, `|`, `~` for boolean ops on pandas Series, never Python `and`/`or`."
        ),
    )
    argumentation: str = Field(
        ...,
        description="The reasoning behind why the attribute should be added. Use this to explain the clinical relevance of the attribute.",
    )


class ExtractorAgent:
    """Agent that extracts domain-relevant features from raw data using LLM guidance.

    Uses a knowledge agent for literature research and iteratively generates patient attributes
    through pandas-based feature engineering commands. Features are created based on a focus
    area provided by the ScientistAgent.

    Attributes:
        cfg_experiment (ExperimentConfig): Experiment configuration.
        knowledge_agent (WebSearchAgent): Agent for literature searches to inform feature extraction.
    """

    def __init__(self, cfg: ExperimentConfig):
        """Initialize the ExtractorAgent.

        Args:
            cfg (ExperimentConfig): Configuration for the experiment.
        """
        self.cfg_experiment = cfg
        self.knowledge_agent = WebSearchAgent(cfg=self.cfg_experiment)
        self._tool_call_counter = defaultdict(int)
        self._tool_call_counter_success = defaultdict(int)
        self._tool_call_counter_failed = defaultdict(int)
        self._tool_error_reporter: ToolErrorReporter | None = None

    async def generate_patient_attributes(
        self,
        _df: pd.DataFrame,
        _df_patients: pd.DataFrame,
        _df_attribute_explanations: pd.DataFrame,
        focus: str,
        *,
        step: int | None = None,
        self_focus=None,
        shared_session=None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Generate new patient attributes based on the provided focus area.

        Uses an LLM agent to iteratively design and execute pandas commands that create
        new attributes. Features are designed to be relevant to the specified focus area
        and include explanations of their clinical/domain significance.

        Args:
            _df (pd.DataFrame): Raw data records (may include timestamps, measurements, etc).
            _df_patients (pd.DataFrame): Patient feature table with patient IDs.
            _df_attribute_explanations (pd.DataFrame): Existing attribute metadata including status.
            focus (str): Domain focus area for guiding feature extraction (e.g., "cardiac risk").
            step (int, optional): Step identifier for logging and session management.
            self_focus (fx_switches.SelfFocus, optional): E9d P− only. Supplies the
                round-1 digest message, receives the round-1 output (its ``FOCUS:``
                line) and supplies "System now wants to focus on: <own focus>." for
                rounds 2..R. ``None`` (default) leaves every message unchanged.
            shared_session (topology.SharedIterationSession, optional): E9f
                ``set_phased`` only. Replaces the session and system prompt; round 1
                sends the BUILD phase message, rounds 2..R the focus line alone.
                ``None`` (default) is Full-v2.

        Returns:
            tuple: (DataFrame with newly generated attributes, DataFrame with explanations)
        """

        self._tool_call_counter = defaultdict(int)
        self._tool_call_counter_success = defaultdict(int)
        self._tool_call_counter_failed = defaultdict(int)
        self._tool_error_reporter = ToolErrorReporter(
            agent_name=self.__class__.__name__,
            output_dir=self.cfg_experiment.output_dir,
            session_name=f"extractor_agent_step_{step or 0}",
        )
        if shared_session is not None:  # E9f: error records carry the SET phase
            self._tool_error_reporter.phase = "E"
            shared_session.enter("E")

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
                    step=step,
                    context={"code": code} if code is not None else None,
                )
            return tool_feedback(ConsoleManager.console_error_print(message, code=code))

        df = _df.copy()
        df_features = _df_patients.copy()
        df_attribute_explanations = _df_attribute_explanations.copy()

        attribute_descriptions: list[AttributeExplanation] = []

        know_attributes = set(df_features.columns)
        if {
            "Attribute",
            "Status",
        }.issubset(df_attribute_explanations.columns):
            pruned_attributes = set(
                df_attribute_explanations.loc[
                    df_attribute_explanations["Status"].astype(str).str.lower()
                    == "pruned",
                    "Attribute",
                ].astype(str)
            )
        else:
            pruned_attributes = set()

        @function_tool
        async def search_in_literature_tool(query: str) -> str:
            """
            Search up and get a summary from relevant literature to help expand your knowledge.
            Use this tool to learn more about feature extraction techniques so that you can create better features.
            The tool does an online search and provides a summary of the most relevant information found.
            It can also be used to look up documentation for external libraries or frameworks like pandas.

            Args:
                query (str): The search query.
            Returns:
                str: A summary of the explanation.
            """
            try:
                self._tool_call_counter["search_in_literature_tool"] += 1
                explanation = await self.knowledge_agent.explain_query(query, caller_role="extractor")

                # Support both synchronous and asynchronous implementations of explain_query.
                explanation = explanation if explanation is not None else ""
                if str(explanation).strip() == "":
                    report_bypass(self._tool_error_reporter, site="h5", tool_name="search_in_literature_tool", message=f"No explanation found for query: {query}", step=step, kind="empty_search", agent_name="ExtractorAgent")
                    return tool_feedback(ConsoleManager.console_error_print(
                        f"No explanation found for query: {query}"
                    ))

                ConsoleManager.console_agent_logging(
                    "ExtractorAgent", "search_in_literature_tool", query
                )
                self._tool_call_counter_success["search_in_literature_tool"] += 1
                return explanation
            except Exception as e:
                return _report_tool_error(
                    tool_name="search_in_literature_tool",
                    message=f"Error during literature search: {e}",
                    exception=e,
                )

        @function_tool
        async def generic_pandas_tool(args: PandasCommandArgs) -> str:
            # The command should be a valid pandas command using 'df' as the dataframe variable for standard operations.
            """
            Execute a generic pandas command and return the result.
            The command should be a valid pandas command using 'df' or 'df_features' as the dataframe variable.
            The pandas package is already imported as 'pd' and numpy as 'np'.
            All commands must:
                - contain 'df' or 'df_features'
                - store the output in a variable named 'result'

            # IMPORTANT:
                EDITS TO DATAFRAMES ARE NOT PERMANENT AND WILL NOT BE SAVED.

            # Tool Functionality:
            The tool works as follows:
                ```
                exec(
                        f"{args.command}",
                        {
                            "pd": pd,
                            "df": df,
                            "np": np,
                            "df_features": df_features,
                        },
                        locals_dict,
                    )
                return locals_dict.get("result", None)
                ```

            # Example of valid commands:
                - result = df.head()
                - result = df.describe()
                - result = df['column_name'].value_counts()
                - result = df_features['id'].nunique()
                - tmp = df.groupby('id')['lab_value'].mean(); result = tmp.fillna(0)

            # Example of invalid commands:
                - df = df.dropna()  # Reassigning df is not allowed
                - df[df['column'] > 0]  # Not storing result in 'result' variable is not allowed
                - df_features = df_features.merge(other_df)  # Reassigning df_features is not allowed

            # Args:
                command (PandasCommandArgs): The pandas command to execute. It should be a valid pandas command using 'df' or 'df_features' as the dataframe variable.
            # Returns:
                str: The result of the pandas command as a string.

            """
            try:
                self._tool_call_counter["generic_pandas_tool"] += 1

                if (
                    args.command.strip().startswith("df =")
                    or args.command.strip().startswith("df_features =")
                    or not "df" in args.command.strip()
                ):
                    report_bypass(self._tool_error_reporter, site="h6", tool_name="generic_pandas_tool", message=f"ILLEGAL COMMAND: {args.command[:200]}", step=step, kind="illegal", agent_name="ExtractorAgent")
                    return tool_feedback(ConsoleManager.console_print(
                        f"[red]ILLEGAL COMMAND: Command must a valid pandas command containing 'df' or 'df_features'.[/red]\nCommand: {args.command}"
                    ))

                ConsoleManager.console_agent_logging(
                    self.__class__.__name__,
                    "generic_pandas_tool",
                    message=args.explanation,
                    code=args.command,
                )

                # Use eval to execute the command in the context of the dataframe

                with contextlib.redirect_stdout(io.StringIO()) as g:
                    result, _captured = _exec_user_command(
                        args.command, df, df_features
                    )
                    # _exec_user_command captures stdout internally; replay so the
                    # outer redirect sees the same output for the message below.
                    if _captured:
                        print(_captured, end="")

                if result is None and g.getvalue().strip() == "":
                    report_bypass(self._tool_error_reporter, site="h7", tool_name="generic_pandas_tool", message="Command did not set 'result' variable or 'result' was set to None!", step=step, code=args.command, kind="no_result", agent_name="ExtractorAgent")
                    return tool_feedback(ConsoleManager.console_error_print(
                        f"Error: Command did not set 'result' variable or 'result' was set to None! Command: {args.command}"
                    ))

                # If the result is a DataFrame or Series, convert it to string for better readability
                if isinstance(result, (pd.DataFrame, pd.Series)):
                    result = result.to_string()

                out = dedent(f"""
                    Result:
                    {result}    

                    Terminal Output:
                    {g.getvalue() if g.getvalue().strip() else "No output generated."}                    
                    """).strip()

                self._tool_call_counter_success["generic_pandas_tool"] += 1
                return truncate_tool_output(out)

            except Exception as e:
                return _report_tool_error(
                    tool_name="generic_pandas_tool",
                    message=f"Error executing command: {e}",
                    exception=e,
                    code=args.command,
                )

        @function_tool
        async def append_new_attribute(args: AttributeExplainingArgs) -> str:
            """
            Explain a patient attribute and its calculation method.
            This tool should be used to document new attributes added that should be added to the 'df_features' dataframe.
            The tool adds attributes to the 'df_features' dataframe by running the following code from the tool arguments:
                ```
                locals_dict = {}
                exec(
                    f"{args.command}",
                    {
                        "pd": pd,
                        "df": df,
                        "np": np,
                        "df_features": df_features,
                    },
                    locals_dict,
                )
                df_features[args.name] = locals_dict.get("result")
                ```

            In other words: args.commands should return a pandas dataframe or series with the same length as df_features when executed.
            This is done by including a groupby operation on 'id' in the command.
            The command should:
            - Be a valid pandas command.
            - Be using 'df' or 'df_features' as the dataframe variable.
            - Not reassign 'df' or 'df_features' (e.g., df = ...).
            - Store the output in a variable named 'result'. The variable 'result' is then added to 'df_features' as a new column with the name specified in 'args.name'.
            - The output stored in 'result' must be numeric (integer or float) and must not contain any NaN or Inf values and must be the same length as df_features.

            # IMPORTANT:
            - Each call runs in a fresh exec scope. Variables defined in earlier calls are NOT available — you must define helper variables (mappings, intermediate columns, lookup tables, helper functions) in the SAME call that uses them, OR first add the building-block as a separate feature via append_new_attribute and then reference it via df_features['name'] in a later call.
            - You CAN write multi-statement code: separate statements with newlines or semicolons. Do not indent continuation lines. Use REAL newline characters in your tool argument; do NOT emit literal `\\n` escape sequences.
            - The TARGET / LABEL column is NOT in `df`. Accessing it (e.g. `df['class']`, `df['target']`, `df.groupby(...)['class']`) raises KeyError. The system prompt lists the only columns available in `df`.
            - `result` MUST be a NUMERIC Series, ndarray, or scalar of length len(df_features). Never assign a column-name string to it.
            - Boolean ops on pandas Series MUST use `&`, `|`, `~` with parentheses; Python `and`/`or` raises ValueError.
            - Do not create new dataframes, only read.
            - The command must include a groupby operation on 'id'.
            - The pandas package is imported as `pd`, numpy as `np`.

            # CORRECT vs WRONG (boolean ops):
              # WRONG: result = (df['x'] == 1) and (df['y'] == 2)         <- ValueError
              # RIGHT: result = ((df['x'] == 1) & (df['y'] == 2)).astype(int).values

            # CORRECT vs WRONG (target column):
              # WRONG: result = df['class'].map({'a': 0, 'b': 1})  <- KeyError, target is held back
              # RIGHT: build features only from the columns the system prompt lists as available in `df`.

            # Examples (single-statement):
              result = df.groupby('id')['lab_value'].mean()
              result = df_features['id'].map(df.groupby('id')['age'].first())

            # Examples (multi-statement; required for categorical encoding, lookup tables, helper masks):
              size_map = {'low': 1, 'med': 2, 'high': 3}
              size_ord = df.groupby('id')['size'].first().map(size_map)
              result = df_features['id'].map(size_ord).fillna(0).values

              groups = [['col_a','col_b','col_c'],
                        ['col_d','col_e','col_f']]
              hits = pd.concat([(df.groupby('id')[cols].first() == 'yes').all(axis=1) for cols in groups], axis=1).any(axis=1)
              result = df_features['id'].map(hits.astype(int)).fillna(0).values

            # Example of invalid commands:
                - df = df.dropna()  # Reassigning df is not allowed
                - df_features = df_features.merge(other_df)  # Reassigning df_features is not allowed
                - result = 'col_a'  # WRONG: result must be numeric, not a column name string
                - def helper(b): ...  in one call, then  result = df.apply(helper, ...)  in a different call  # WRONG: helper does not survive across calls

            # Args:
                args (AttributeExplainingArgs): The arguments containing attribute information.

            # Returns:
                str: A confirmation message.
            """
            try:
                self._tool_call_counter["append_new_attribute"] += 1
                instrumentation.count("append_attempts")

                if args.name in know_attributes:
                    if args.name in pruned_attributes:
                        # Auto-unprune: treat a suggestion for an existing pruned feature
                        # as an intent to restore it rather than an error.
                        attribute_descriptions.append(
                            AttributeExplanation(
                                name=args.name,
                                argumentation=args.argumentation,
                                command="",
                                status="Active",
                                added=f"Trial {step}" if step is not None else "N/A",
                            )
                        )
                        pruned_attributes.discard(args.name)
                        instrumentation.count("append_restored")
                        ConsoleManager.console_agent_logging(
                            self.__class__.__name__,
                            "append_new_attribute",
                            f"Auto-unprune: [green]{args.name}[/green] restored to Active.\nRationale: {args.argumentation}",
                        )
                        self._tool_call_counter_success["append_new_attribute"] += 1
                        return f"Attribute {args.name} was pruned — it has been restored to Active status."
                    ConsoleManager.console_print(
                        f"[red]Error: Attribute {args.name} already exists and is active.[/red]"
                    )
                    report_bypass(self._tool_error_reporter, site="v1", tool_name="append_new_attribute", message=f"Attribute {args.name} already exists and is active.", step=step, kind="duplicate", agent_name="ExtractorAgent")
                    return f"Attribute {args.name} already exists and is active."

                locals_dict = {}

                with contextlib.redirect_stdout(io.StringIO()) as f:
                    result, _captured = _exec_user_command(
                        args.command, df, df_features
                    )
                    if _captured:
                        print(_captured, end="")

                if result is None:
                    report_bypass(self._tool_error_reporter, site="h8", tool_name="append_new_attribute", message=f"Command did not set 'result' variable for attribute '{args.name}'.", step=step, code=args.command, kind="no_result", agent_name="ExtractorAgent")
                    return tool_feedback(ConsoleManager.console_error_print(
                        f"Error: Command did not set 'result' variable for attribute '{args.name}'.",
                        code=args.command,
                    ))

                # Ensure alignment by position, not index
                if isinstance(result, (pd.Series, pd.DataFrame)):
                    result = result.reset_index(drop=True).values

                if isinstance(result, np.ndarray):
                    if np.isnan(result).any() or np.isinf(result).any():
                        report_bypass(self._tool_error_reporter, site="h8", tool_name="append_new_attribute", message=f"Result for attribute '{args.name}' contains NaN or Inf values.", step=step, kind="nan", agent_name="ExtractorAgent")
                        return tool_feedback(ConsoleManager.console_error_print(
                            f"Error: Result for attribute '{args.name}' contains NaN or Inf values."
                        ))
                else:
                    if pd.isnull(result) or result in [float("inf"), float("-inf")]:
                        report_bypass(self._tool_error_reporter, site="h8", tool_name="append_new_attribute", message=f"Result for attribute '{args.name}' contains NaN or Inf values.", step=step, kind="nan", agent_name="ExtractorAgent")
                        return tool_feedback(ConsoleManager.console_error_print(
                            f"Error: Result for attribute '{args.name}' contains NaN or Inf values."
                        ))

                # Check that all values are numeric
                if isinstance(result, np.ndarray):
                    if not np.issubdtype(result.dtype, np.number):
                        report_bypass(self._tool_error_reporter, site="h8", tool_name="append_new_attribute", message=f"Result for attribute '{args.name}' contains non-numeric values.", step=step, kind="non_numeric", agent_name="ExtractorAgent")
                        return tool_feedback(ConsoleManager.console_error_print(
                            f"Error: Result for attribute '{args.name}' contains non-numeric values."
                        ))
                else:
                    # For scalar or Series/DataFrame converted to values
                    try:
                        _ = pd.Series(
                            result.flatten() if hasattr(result, "flatten") else result
                        ).astype(float)
                    except Exception:
                        report_bypass(self._tool_error_reporter, site="h8", tool_name="append_new_attribute", message=f"Result for attribute '{args.name}' contains non-numeric values.", step=step, kind="non_numeric", agent_name="ExtractorAgent")
                        return tool_feedback(ConsoleManager.console_error_print(
                            f"Error: Result for attribute '{args.name}' contains non-numeric values."
                        ))

                df_features[args.name] = result

                attribute_descriptions.append(
                    AttributeExplanation(
                        name=args.name,
                        argumentation=args.argumentation,
                        command=args.command,
                        status="Active",
                        added=f"Trial {step}" if step is not None else "N/A",
                    )
                )
                know_attributes.add(args.name)

                ConsoleManager.console_agent_logging(
                    self.__class__.__name__,
                    "append_new_attribute",
                    f"Name: [green]{args.name}[/green] \nArgumentation: {args.argumentation}",
                    code=args.command,
                    language="python",
                )

                self._tool_call_counter_success["append_new_attribute"] += 1
                instrumentation.count("append_accepted")
                return f"Attribute {args.name} explained, documented, and added to df_features."
            except Exception as e:
                return _report_tool_error(
                    tool_name="append_new_attribute",
                    message=f"Error updating attributes: {e}",
                    exception=e,
                    code=args.command,
                )

        @function_tool
        async def list_known_attributes_tool() -> str:
            """
            List all currently known attributes in the df_features dataframe, each with its status.
            This tool can be used to check which attributes have already been explained and added to the df_features dataframe,
            and which of them are currently Pruned (a Pruned attribute can be restored by calling append_new_attribute with its exact name).

            Args:
                args (Any): No arguments are needed for this tool.

            Returns:
                str: A comma-separated string of known attribute names, each followed by its status, e.g. "age (Active), bmi (Pruned)".
            """
            try:
                self._tool_call_counter["list_known_attributes_tool"] += 1

                # D7 (Henrik, 2026-10-09): expose Status so the restore-pruned instruction is actionable.
                known_attrs = ", ".join(
                    f"{name} ({'Pruned' if name in pruned_attributes else 'Active'})"
                    for name in sorted(set(df_features.columns) - {"id"})
                )
                ConsoleManager.console_agent_logging(
                    self.__class__.__name__,
                    "list_known_attributes_tool",
                    f"Known attributes ({len(set(df_features.columns) - {'id'})}) : {known_attrs}",
                )
                self._tool_call_counter_success["list_known_attributes_tool"] += 1
                return known_attrs
            except Exception as e:
                return _report_tool_error(
                    tool_name="list_known_attributes_tool",
                    message=f"Error listing known attributes: {e}",
                    exception=e,
                )

        def _apply_baseline_fallback_attributes() -> int:
            """Add a minimal, deterministic baseline feature set when tool-calls fail.

            This keeps the run alive for fold evaluation on model profiles that fail during
            tool planning/execution.
            """

            excluded_cols = {"id", "time", "datetime", "target"}
            candidate_cols = [
                col
                for col in df.columns
                if col not in excluded_cols and pd.api.types.is_numeric_dtype(df[col])
            ]
            categorical_cols = [
                col
                for col in df.columns
                if col not in excluded_cols
                and not pd.api.types.is_numeric_dtype(df[col])
            ]
            # CAR (and similar all-categorical datasets) used to fall through with
            # nothing because the fallback only handled numeric columns. Without an
            # ordinal-encoding path for categoricals, an empty Extractor turn on
            # CAR aborted the entire dataset.
            if not candidate_cols and not categorical_cols and not pruned_attributes:
                return 0

            added_count = 0

            # Step 0 — Restore previously pruned features. This is the cheapest
            # win mid-run: every Tester pruning step usually leaves several
            # rows in df_attributes_explanations marked Pruned. Late
            # iterations (where the model has already exhausted simple
            # encodings/interactions) often emit zero append calls and used
            # to abort the dataset. A trivial restoration of any pruned
            # feature is always productive and keeps the run alive. It also
            # gives the Scientist a chance to continue exploring rather than
            # losing the entire fold mid-flight.
            for pruned_name in list(pruned_attributes):
                if added_count >= 4:  # cap to leave room for fresh features
                    break
                if pruned_name in know_attributes and pruned_name in df_features.columns:
                    attribute_descriptions.append(
                        AttributeExplanation(
                            name=pruned_name,
                            argumentation=(
                                "Fallback restoration of previously pruned feature. "
                                "Triggered because the LLM Extractor produced 0 "
                                "successful append calls this iteration. The Tester "
                                "may re-prune it; restoring it here keeps the run "
                                "alive without inventing arbitrary new features."
                            ),
                            command="",
                            status="Active",
                            added=f"Trial {step}" if step is not None else "N/A",
                        )
                    )
                    pruned_attributes.discard(pruned_name)
                    added_count += 1

            # Prefer stable per-entity aggregates first.
            for col in candidate_cols:
                if added_count >= 8:
                    break

                attr_name = str(col)
                if attr_name in know_attributes:
                    continue

                try:
                    grouped = df.groupby("id")[col].mean()
                    result = df_features["id"].map(grouped)
                    if result.isna().any():
                        continue
                    if np.isinf(result.to_numpy(dtype=float)).any():
                        continue

                    df_features[attr_name] = result.astype(float)
                    attribute_descriptions.append(
                        AttributeExplanation(
                            name=attr_name,
                            argumentation=(
                                f"Fallback baseline feature from raw column '{col}' "
                                "using per-id mean aggregation."
                            ),
                            command=f"result = df.groupby('id')['{col}'].mean()",
                            status="Active",
                            added=f"Trial {step}" if step is not None else "N/A",
                        )
                    )
                    know_attributes.add(attr_name)
                    added_count += 1
                except Exception:
                    continue

            # If no grouped features were possible, attempt position-aligned raw columns.
            if added_count == 0 and len(df) == len(df_features):
                for col in candidate_cols:
                    if added_count >= 8:
                        break

                    attr_name = str(col)
                    if attr_name in know_attributes:
                        continue

                    series = pd.to_numeric(df[col], errors="coerce")
                    if (
                        series.isna().any()
                        or np.isinf(series.to_numpy(dtype=float)).any()
                    ):
                        continue

                    df_features[attr_name] = series.reset_index(drop=True).astype(float)
                    attribute_descriptions.append(
                        AttributeExplanation(
                            name=attr_name,
                            argumentation=(
                                f"Fallback baseline feature from raw column '{col}' "
                                "using row-aligned values."
                            ),
                            command=f"result = pd.to_numeric(df['{col}'], errors='coerce')",
                            status="Active",
                            added=f"Trial {step}" if step is not None else "N/A",
                        )
                    )
                    know_attributes.add(attr_name)
                    added_count += 1

            # If raw baselines are already present, create deterministic interactions.
            numeric_feature_cols = [
                col
                for col in df_features.columns
                if col != "id" and pd.api.types.is_numeric_dtype(df_features[col])
            ]

            for i, left_col in enumerate(numeric_feature_cols):
                if added_count >= 8:
                    break
                for right_col in numeric_feature_cols[i + 1 :]:
                    if added_count >= 8:
                        break

                    left = pd.to_numeric(df_features[left_col], errors="coerce")
                    right = pd.to_numeric(df_features[right_col], errors="coerce")
                    if left.isna().any() or right.isna().any():
                        continue

                    interaction_specs = [
                        (
                            f"{left_col}_mul_{right_col}",
                            left * right,
                            f"result = df_features['{left_col}'] * df_features['{right_col}']",
                            "product interaction",
                        ),
                        (
                            f"{left_col}_minus_{right_col}",
                            left - right,
                            f"result = df_features['{left_col}'] - df_features['{right_col}']",
                            "difference interaction",
                        ),
                        (
                            f"{left_col}_ratio_{right_col}",
                            left / (right + 1e-6),
                            (
                                f"result = df_features['{left_col}'] / "
                                f"(df_features['{right_col}'] + 1e-6)"
                            ),
                            "ratio interaction",
                        ),
                    ]

                    for (
                        attr_name,
                        series,
                        command,
                        interaction_kind,
                    ) in interaction_specs:
                        if added_count >= 8:
                            break
                        if attr_name in know_attributes:
                            continue
                        if (
                            series.isna().any()
                            or np.isinf(series.to_numpy(dtype=float)).any()
                        ):
                            continue

                        df_features[attr_name] = series.astype(float)
                        attribute_descriptions.append(
                            AttributeExplanation(
                                name=attr_name,
                                argumentation=(
                                    "Fallback engineered feature using deterministic "
                                    f"{interaction_kind} between '{left_col}' and '{right_col}'."
                                ),
                                command=command,
                                status="Active",
                                added=f"Trial {step}" if step is not None else "N/A",
                            )
                        )
                        know_attributes.add(attr_name)
                        added_count += 1

            # Categorical fallback: ordinal-encode raw categorical columns. Used
            # when the dataset has no numeric raw columns (e.g. CAR with
            # buying/maint/doors/persons/lug_boot/safety) AND the LLM Extractor
            # produced 0 features this iteration. Without this branch, an empty
            # Extractor turn aborted CAR even on the very first iteration.
            for col in categorical_cols:
                if added_count >= 8:
                    break

                attr_name = f"{col}_ord"
                if attr_name in know_attributes or col in know_attributes:
                    continue

                encoded_series = _ordinal_encode_categorical(
                    df, column=col, df_features=df_features
                )
                if encoded_series is None:
                    continue

                df_features[attr_name] = encoded_series
                attribute_descriptions.append(
                    AttributeExplanation(
                        name=attr_name,
                        argumentation=(
                            f"Fallback ordinal encoding of categorical column "
                            f"'{col}' (categories sorted alphabetically for "
                            "reproducibility)."
                        ),
                        command=(
                            f"cats = sorted(map(str, df.groupby('id')['{col}']."
                            f"first().dropna().unique()))\n"
                            "enc = {c: i for i, c in enumerate(cats)}\n"
                            f"result = df_features['id'].map("
                            f"df.groupby('id')['{col}'].first().astype(str)"
                            ".map(enc))"
                        ),
                        status="Active",
                        added=f"Trial {step}" if step is not None else "N/A",
                    )
                )
                know_attributes.add(attr_name)
                added_count += 1

            return added_count

        async def main():

            client = create_openai_client_for_vllm(
                base_url=llm_cfg.endpoint_for("extractor"),
                api_key=llm_cfg.api_key,
                agent_name="ExtractorAgent",
            )

            # Authoritative list of columns the model is allowed to reference in `df`.
            # The target / label column is held back from `df` upstream, so without
            # this list the model invents references like `df['class']` and gets
            # KeyError. Surfaced by car run 09vnwi4s (agent_toolcall_errors_car:v23,
            # 2x KeyError: 'class').
            _raw_columns_str = ", ".join(
                f"`{c}`" for c in sorted(map(str, df.columns.tolist()))
            )

            agent = Agent(
                model_settings=_build_extractor_model_settings(extractor_agent_cfg),
                name="Extractor Agent",
                instructions=f"""
                
                    Global goal:
                    {AgentRoles.GLOBAL_GOAL.value}
                
                    # Team context:
                    - The Scientist Agent: {AgentRoles.SCIENTIST.value}
                    - The Tester Agent: {AgentRoles.TESTER.value}

                    You all work in a loop where the Scientist Agent defines a focus area for the investigation, you extract relevant attributes based on that focus, and the Tester Agent evaluates the quality of those attributes.


                    # Objective:
                    - Create useful new features in 'df_features' with high predictive power aligned with the provided focus area from the Scientist Agent.

                    # Dataframes:
                    - df: raw data
                    - df_features: cumulative feature table

                    # Required workflow:
                    1. Explore with generic_pandas_tool.
                    2. Before proposing features, check existing columns with list_known_attributes_tool.
                    3. Add each candidate via append_new_attribute with clear rationale and valid command.
                    4. Repeat until you have added a sufficient number of relevant features.

                    # Hard rules:
                    - Follow the provided focus strictly.
                    - Prefer concise tool calls over long narration.
                    - **You MUST call `append_new_attribute` at least once before ending your turn — even if the focus is unclear, partially implementable, or already covered by existing features.** Returning zero successful append calls on the FIRST iteration of a fold aborts the dataset entirely. Subsequent zero-output iterations are merely wasted compute. When in doubt, fall back to one of these always-applicable last-resort additions (in order of preference):
                        1. **Restore a pruned feature**: call `list_known_attributes_tool`, look at `df_attributes_explanations` for any feature with `Status="Pruned"`, then call `append_new_attribute` with that exact name — it auto-unprunes and counts as productive work.
                        2. **A polynomial / log / sqrt transform** of any existing numeric feature (e.g. `result = np.log1p(df_features['some_feature'].abs() + 1).values`).
                        3. **A pairwise interaction** between any two existing numeric features (`result = (df_features['a'] * df_features['b']).values`).
                        4. **A simple ordinal encoding** of any raw categorical column you have not yet encoded.
                       It is **always better to add one trivial feature than to return zero**. The Tester will prune it if it is uninformative.
                    - **Do NOT give up after a single failure.** After ANY tool error, try at least two more variations of the idea (different formulation, different columns) or pivot to a different feature before stopping.
                    - **Each append_new_attribute call has a fresh Python scope.** Helper functions (`def helper(...)`, `score`, `solve`, ...), lookup tables (`groups = [...]`), and constants defined in EARLIER calls are gone. To use a helper, either:
                        (a) define it inline in the SAME call that uses it, or
                        (b) materialize the building-block as its own feature first (e.g. add `count_a`, `count_b` separately), then in a later call read `df_features['count_a']` and combine.
                    - **Boolean ops on pandas Series MUST use `&`, `|`, `~`** with parentheses around each comparison: `result = ((df['x'] == 1) & (df['y'] == 2)).astype(int)`. Python `and`/`or` raises `ValueError: The truth value of a Series is ambiguous`.
                    - **`result` MUST be numeric** (int or float Series/array of length df_features). Never assign a column-name string (e.g. `result = 'col_a'` is wrong; use `result = (df['col_a'] == 'yes').astype(int).values`).
                    - **Split genuinely multi-step features into successive append_new_attribute calls building intermediate features** (e.g. compute a lookup/aggregation as its own feature first, then combine it in a later call) — not because of a length limit, but because each call has its own fresh Python scope (see below), so building step-by-step is more reliable than one large expression.
                    - You are not limited to products, ratios, z-scoring, trig encodings, or weighted sums — logs, roots, polynomials, quantile binning, and lag/window or rolling-aggregate operators (where the data has a natural order, e.g. by id or time) are all valid and encouraged when they fit the focus area.
                    - Categorical features added to df_features should be one-hot flags.
                    - To restore a pruned feature, call append_new_attribute with its name — it will be automatically restored to Active.
                    - Validate ideas with generic_pandas_tool before append_new_attribute.
                    - **`df` is read-only and contains ONLY the raw columns listed below. ANY column you create with append_new_attribute lands in `df_features`, NEVER in `df`.** Reading a derived column back via `df['size_ord']` or `df.groupby('id')['size_ord']` raises `KeyError` — use `df_features['size_ord']` (or call `list_known_attributes_tool` to confirm what is available) and group by `df_features.groupby('id')[...]` if you need a grouped view of derived columns.
                    - **The target / label column is HELD BACK from `df`** — it is NOT accessible. Building features that reference it (e.g. `df['class']`, `df['target']`, `df.groupby(...)['class']`) raises `KeyError`. Build features only from the raw columns listed below.

                    # Raw columns available in `df` (authoritative — these are the ONLY columns you may read from `df`):
                    {_raw_columns_str}

                    # Available tools:
                    - generic_pandas_tool
                    - append_new_attribute
                    - list_known_attributes_tool
                    - search_in_literature_tool

                    # Task context and predictive goal:
                    {self.cfg_experiment.context_and_goal}

                    """,
                # Response style:
                # - Be concise.
                # - Do not provide long chain-of-thought narratives.
                # Provide a final summary of the new attributes you have added to 'df_features' as a markdown table.
                model=OpenAIChatCompletionsModel(
                    model=llm_cfg.extractor_model,
                    openai_client=client,
                ),
                tools=[
                    generic_pandas_tool,
                    append_new_attribute,
                    list_known_attributes_tool,
                    search_in_literature_tool,
                ],
            )
            # E9e: drops search only under ROGUE_ONE_KNOWLEDGE_BACKEND=off (A13 edit 5).
            agent.instructions = knowledge_backend.role_instructions("extractor", agent.instructions)
            agent.tools = knowledge_backend.role_tools(agent.tools)
            # E9f set_phased: neutral system prompt; role block -> BUILD phase message.
            _set_first_message = None
            if shared_session is not None:
                _set_full_instructions = agent.instructions
                agent.instructions = shared_session.system_prompt

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
                session = SQLiteSession(f"extractor_agent_session_step_{step or 0}")
                instrumentation.set_session(f"extractor_agent_session_step_{step or 0}")
            else:  # E9f: the iteration's shared session
                session = shared_session.session
                instrumentation.set_session(shared_session.session_name)

            _user_message = f"System now wants to focus on: {focus}."
            if self_focus is not None:  # E9d P−: round-1 digest message
                _user_message = self_focus.message_for_round(0)
            if shared_session is not None:
                _set_first_message = _set_phased.phase_message(
                    "E", _set_full_instructions, _user_message
                )
            dump_prompt(
                "extractor", "append", step, agent.instructions,
                _user_message if shared_session is None else _set_first_message,
                tools=agent.tools, output_dir=self.cfg_experiment.output_dir,
            )
            for _round in range(rouge_one_cfg.extractor_rounds):
                instrumentation.count("runner_runs_extractor")
                if self_focus is not None:  # E9d P−: digest once, then own focus
                    _user_message = self_focus.message_for_round(_round)
                try:
                    res = await Runner().run(
                        agent,
                        _user_message
                        if shared_session is None
                        else shared_session.input_for("E", _set_first_message, _user_message),
                        max_turns=extractor_agent_cfg.max_iterations,
                        session=session,
                        run_config=_run_config,
                    )
                    if self_focus is not None:
                        self_focus.observe(_round, res)
                    # ConsoleManager.console_print(f"Agent Result: {res.final_output}")
                except Exception as e:
                    self._tool_call_counter_failed["runner"] += 1
                    if shared_session is not None:  # E9f: off-phase tool call counter
                        shared_session.note_runner_exception("E", e)
                    if self._tool_error_reporter is not None:
                        self._tool_error_reporter.report(
                            tool_name="runner",
                            message=f"Error during Extractor Agent run: {e}",
                            exception=e,
                            step=step,
                            context=(
                                {"session_name": f"extractor_agent_session_step_{step or 0}"}
                                if shared_session is None
                                else {"session_name": shared_session.session_name, "phase": "E"}
                            ),
                        )
                    ConsoleManager.console_error_print(
                        f"Error during Extractor Agent run: {e}"
                    )
                    continue

        await main()

        if len(attribute_descriptions) == 0:
            fallback_added = _apply_baseline_fallback_attributes()
            _fb_new = [a.name for a in attribute_descriptions if a.command]
            instrumentation.count("fallback_extractor")
            if shared_session is not None:  # E9f: fallbacks are counted per phase
                shared_session.note_fallback("E", "deterministic")
            instrumentation.count("fallback_extractor_features", len(_fb_new))
            instrumentation.count("fallback_extractor_restorations", len(attribute_descriptions) - len(_fb_new))
            instrumentation.note_fallback_features(_fb_new)
            if fallback_added > 0:
                ConsoleManager.console_print(
                    "[yellow]Extractor fallback activated: "
                    f"added {fallback_added} deterministic baseline attributes.[/yellow]"
                )

        # safe_wandb_log handles step regression: passing the small per-step
        # iteration index directly to wandb.log gets dropped silently because
        # the LLM client's per-call token logging auto-advances run.step.
        safe_wandb_log(
            {
                f"Extractor_tool_usage/{tool_name}": count
                for tool_name, count in self._tool_call_counter.items()
            },
            prefer_step=step,
        )
        safe_wandb_log(
            {
                f"Extractor_tool_usage_success/{tool_name}": count
                for tool_name, count in self._tool_call_counter_success.items()
            },
            prefer_step=step,
        )
        safe_wandb_log(
            {
                f"Extractor_tool_usage_failed/{tool_name}": count
                for tool_name, count in self._tool_call_counter_failed.items()
            },
            prefer_step=step,
        )

        if self._tool_error_reporter is not None:
            safe_wandb_log(
                {
                    "Extractor_tool_errors/total": self._tool_error_reporter.total_errors,
                    "Extractor_tool_errors/log_path": str(
                        self._tool_error_reporter.file_path
                    ),
                },
                prefer_step=step,
            )

        df_attribute_descriptions = pd.DataFrame(
            [attr.as_dict for attr in attribute_descriptions]
        )

        return df_features, df_attribute_descriptions


if __name__ == "__main__":

    import json
    from pathlib import Path

    ConsoleManager.console_rule("Testing Extractor Agent", color="red")

    df = pd.read_csv("./tmp/df_patients.csv")
    focus = "Medical history"

    cfg_placeholder = ExperimentConfig.from_yaml(
        "./tasks/suppression_rato/config.yml"
    )
    agent = ExtractorAgent(cfg_placeholder)
    df_features, attribute_descriptions = asyncio.run(
        agent.generate_patient_attributes(
            df,
            pd.DataFrame({"id": sorted(df["id"].unique())}),
            pd.DataFrame(
                columns=[
                    "Attribute",
                    "Description",
                    "Pandas Command",
                    "Status",
                    "Added",
                ]
            ),
            focus,
        )
    )

    _tool_counts_str = "\n".join(
        f"{k}: {v}" for k, v in dict(agent._tool_call_counter).items()
    )
    ConsoleManager.console_print(f"Tool call counts: {_tool_counts_str}")

    ConsoleManager.console_print(
        f"Extracted {len(df_features.columns) - 1} patient attributes."
    )

    ConsoleManager.console_print(df_features)
    ConsoleManager.console_print(json.dumps(attribute_descriptions, indent=4))
