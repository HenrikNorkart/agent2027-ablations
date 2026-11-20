import os
from pathlib import Path
from typing import Literal
from dataclasses import dataclass

from dotenv import load_dotenv
from openai.types.shared.reasoning import Reasoning

from rogueone.utils.singelton import Singleton
from rogueone.utils.config_keys import (
    ENV_KEY_ALIASES,
    OPTIONAL_ENV_DEFAULTS,
    REQUIRED_ENV_KEYS,
)
from rogueone.utils.env_validation import (
    apply_env_aliases,
    apply_optional_defaults,
    env_as_bool,
    validate_required_env_keys,
)

p = Path(__file__).resolve().parent.parent.parent
# .env, then .dev.env with override=True, then .secret.env -- unchanged order; the
# helper also records which files existed and which keys .dev.env assigned, so the
# E1 flag registry can put the evidence into W&B (a baked-in .dev.env beats docker -e).
from rogueone.utils.flag_registry import load_dotenv_chain

load_dotenv_chain(p)

print(
    f"Loaded environment variables from: {p / '.env'}, {p / '.dev.env'}, {p / '.secret.env'}"
)

apply_env_aliases(ENV_KEY_ALIASES)
apply_optional_defaults(OPTIONAL_ENV_DEFAULTS)
validate_required_env_keys(REQUIRED_ENV_KEYS)


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration object for a single experiment run.

    Encapsulates all configuration parameters for an experiment including:
    - Experiment metadata (name, description)
    - Data file paths (features, labels)
    - Knowledge database configuration
    - Task configuration (classification/regression, tabular)
    - Output directory

    Attributes:
        experiment_name (str): Name identifier for the experiment.
        experiment_description (str): Human-readable description of the experiment.
        knowledge_db_path (Path): Path to the knowledge database directory.
        knowledge_db_collection_name (str): Name of the collection in the knowledge DB.
        output_dir (Path): Directory where experiment results will be saved.
        modality (Literal["tabular"]): Type of data modality.
        task_type (Literal["classification", "regression"]): Type of ML task.
        feature_file_path (Path): Path to the CSV/data file containing features.
        label_file_path (Path): Path to the CSV/data file containing labels.
        context_and_goal (str): Context description and goal for the experiment.
    """

    experiment_name: str
    experiment_description: str

    knowledge_db_path: Path
    knowledge_db_collection_name: str

    output_dir: Path

    modality: Literal["tabular"]
    task_type: Literal["classification", "regression"]

    feature_file_path: Path
    label_file_path: Path

    context_and_goal: str

    def __post_init__(self):
        """Validate and normalize configuration paths after initialization.

        Converts string paths to Path objects and validates that required files exist.
        Creates output directory if it does not exist.
        """
        if isinstance(self.knowledge_db_path, str):
            object.__setattr__(self, "knowledge_db_path", Path(self.knowledge_db_path))
        if isinstance(self.feature_file_path, str):
            object.__setattr__(self, "feature_file_path", Path(self.feature_file_path))
        if isinstance(self.label_file_path, str):
            object.__setattr__(self, "label_file_path", Path(self.label_file_path))
        if isinstance(self.output_dir, str):
            object.__setattr__(self, "output_dir", Path(self.output_dir))

        assert (
            self.feature_file_path.exists()
        ), f"Feature file path {self.feature_file_path} does not exist."
        assert (
            self.label_file_path.exists()
        ), f"Label file path {self.label_file_path} does not exist."

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        # baGY Q1: every run gets a query-level retrieval audit trail alongside
        # its other artifacts. Purely additive -- the recorder never raises and
        # never feeds the evaluator. Imported here rather than at module scope
        # because `rogueone.utils.__init__` imports this module, and wrapped
        # because no diagnostic may ever prevent a config from being built.
        # See rogueone.utils.retrieval_audit.
        try:
            from rogueone.utils import retrieval_audit

            retrieval_audit.configure_from_output_dir(self.output_dir)
        except Exception:  # noqa: BLE001 - a diagnostic must never break a run
            pass

    @staticmethod
    def _resolve_path(
        path_value: str | Path, *, config_dir: Path | None = None
    ) -> Path:
        """Resolve a config path across common workspace layouts.

        Supports:
        - Existing absolute paths
        - Relative paths (resolved from YAML location when available)
        - `/workspace/...` paths remapped to current repository root
        """
        path = Path(path_value)
        if path.exists():
            return path

        if not path.is_absolute() and config_dir is not None:
            candidate = (config_dir / path).resolve()
            if candidate.exists():
                return candidate

        raw = str(path)
        if raw.startswith("/workspace/"):
            repo_root = Path(__file__).resolve().parents[2]
            mapped = repo_root / raw.removeprefix("/workspace/")
            if mapped.exists():
                return mapped

        return path

    @classmethod
    def from_dict(
        cls, config_dict: dict, *, config_dir: Path | None = None
    ) -> "ExperimentConfig":
        """Create an ExperimentConfig instance from a dictionary.

        Args:
            config_dict (dict): Dictionary containing configuration key-value pairs.

        Returns:
            ExperimentConfig: Instantiated configuration object.
        """
        return cls(
            experiment_name=config_dict.get("experiment_name", "default_experiment"),
            experiment_description=config_dict.get(
                "experiment_description", "No description provided."
            ),
            knowledge_db_path=cls._resolve_path(
                config_dict["knowledge_db_path"], config_dir=config_dir
            ),
            knowledge_db_collection_name=config_dict["knowledge_db_collection_name"],
            feature_file_path=cls._resolve_path(
                config_dict["feature_file_path"], config_dir=config_dir
            ),
            label_file_path=cls._resolve_path(
                config_dict["label_file_path"], config_dir=config_dir
            ),
            context_and_goal=config_dict["context_and_goal"],
            output_dir=cls._resolve_path(
                config_dict.get("output_dir", "./output"), config_dir=config_dir
            ),
            modality=config_dict.get("modality", "tabular"),
            task_type=config_dict.get("task_type", "classification"),
        )

    @classmethod
    def from_yaml(cls, yaml_path: Path | str) -> "ExperimentConfig":
        """Create an ExperimentConfig instance from a YAML file.

        Args:
            yaml_path (Path | str): Path to the YAML configuration file.

        Returns:
            ExperimentConfig: Instantiated configuration object.

        Raises:
            AssertionError: If YAML file does not exist.
        """
        import yaml

        if isinstance(yaml_path, str):
            yaml_path = Path(yaml_path)

        assert yaml_path.exists(), f"YAML file {yaml_path} does not exist."

        with open(yaml_path, "r") as file:
            config_dict = yaml.safe_load(file)
        return cls.from_dict(config_dict, config_dir=yaml_path.parent)

    def to_dict(self) -> dict:
        return {
            "experiment_name": self.experiment_name,
            "experiment_description": self.experiment_description,
            "knowledge_db_path": str(self.knowledge_db_path),
            "knowledge_db_collection_name": self.knowledge_db_collection_name,
            "feature_file_path": str(self.feature_file_path),
            "label_file_path": str(self.label_file_path),
            "context_and_goal": self.context_and_goal,
            "output_dir": str(self.output_dir),
            "modality": self.modality,
            "task_type": self.task_type,
        }


#: Roles that can be pointed at their own model/endpoint. These are the three core
#: agents; the retrieval helpers (KnowledgeAgent, WebSearchAgent) deliberately stay
#: on ``LLM_MODEL`` so that overriding one core agent does not silently move its
#: search helper too.
AGENT_ROLES = ("scientist", "extractor", "tester")


def _role_env(role: str, suffix: str) -> str:
    return f"ROGUE_ONE_{role.upper()}_{suffix}"


class LLMConfig(metaclass=Singleton):
    """Endpoint/model settings for the LLM backend.

    Besides the global ``LLM_MODEL`` / ``LLM_PORT``, each of :data:`AGENT_ROLES`
    may be pointed at its own model and its own vLLM endpoint via
    ``ROGUE_ONE_<ROLE>_MODEL`` and ``ROGUE_ONE_<ROLE>_PORT``. Both fall back to the
    global setting when unset or empty, so runs that set neither behave exactly as
    before. ``REBUTTAL_FINAL.md`` (reply to baGY, Q3) states that per-agent model
    scale was not configurable; this is what makes it configurable.

    The port override exists because a model override alone is not usable in
    practice: one vLLM server serves one model, so a mixed 120B/20B configuration
    needs a second server and therefore a second port. Overriding only the model
    while leaving the port shared points the role at a model the endpoint does not
    serve, and the request fails at the server.
    """

    def __init__(self):
        self.model: Literal["gpt-oss-120b", "gpt-oss-20b"] = os.getenv("LLM_MODEL")
        self.temperature: float = float(os.getenv("LLM_TEMPERATURE", 0.7))
        self.max_retries: int = int(os.getenv("LLM_MAX_RETRIES", 3))
        self.api_key: str = os.getenv("LLM_API_KEY", "")
        self.port: str = os.getenv("LLM_PORT", "")
        self.endpoint: str = f"http://localhost:{self.port}/v1"

        self._role_models: dict[str, str] = {
            role: (os.getenv(_role_env(role, "MODEL"), "").strip() or self.model)
            for role in AGENT_ROLES
        }
        self._role_ports: dict[str, str] = {
            role: (os.getenv(_role_env(role, "PORT"), "").strip() or self.port)
            for role in AGENT_ROLES
        }

    def model_for(self, role: str) -> str:
        """Model id for ``role``, falling back to ``LLM_MODEL``.

        Raises:
            KeyError: If ``role`` is not one of :data:`AGENT_ROLES`. Failing loudly
                beats silently handing back the global model for a typo'd role,
                which would look like a working override in an ablation.
        """
        return self._role_models[role.lower()]

    def endpoint_for(self, role: str) -> str:
        """vLLM endpoint for ``role``, falling back to ``LLM_PORT``."""
        return f"http://localhost:{self._role_ports[role.lower()]}/v1"

    @property
    def scientist_model(self) -> str:
        return self.model_for("scientist")

    @property
    def extractor_model(self) -> str:
        return self.model_for("extractor")

    @property
    def tester_model(self) -> str:
        return self.model_for("tester")


class EmbeddingConfig(metaclass=Singleton):
    def __init__(self):
        self.model: Literal["Qwen/Qwen3-Embedding-4B"] = os.getenv("EMBEDD_MODEL")
        self.chunk_size: int = int(os.getenv("EMBEDD_CHUNK_SIZE", 1))
        self.api_key: str = os.getenv("EMBEDD_API_KEY", "")
        self.port: str = os.getenv("EMBEDD_PORT", "")
        self.endpoint: str = f"http://localhost:{self.port}/v1"


class ConsoleConfig(metaclass=Singleton):
    def __init__(self):
        self.disable: bool = env_as_bool("CONSOLE_DISABLE", False)


class rogueoneConfig(metaclass=Singleton):
    def __init__(self):
        self.num_iterations: int = int(os.getenv("ROGUE_ONE_NUM_ITERATIONS", 3))
        self.train_size: float = float(os.getenv("ROGUE_ONE_TRAIN_SIZE", 0.8))
        self.test_size: float = 1 - self.train_size
        self.do_logging: bool = env_as_bool("ROGUE_ONE_DO_LOGGING", True)
        self.k_folds: int = int(os.getenv("ROGUE_ONE_K_FOLDS", 5))
        self.fold_seed: int = int(os.getenv("ROGUE_ONE_FOLD_SEED", 42))
        self.fold_retry_limit: int = int(os.getenv("ROGUE_ONE_FOLD_RETRY_LIMIT", 1))
        self.do_balancing = env_as_bool("ROGUE_ONE_DO_BALANCING", True)
        self.enable_rollback: bool = env_as_bool("ROGUE_ONE_ENABLE_ROLLBACK", False)
        # E9a / E9b (AGENT 2027 A10, A11): default-off loop switches
        # (rogueone.utils.loop_switches). Anything other than the listed values raises.
        from rogueone.utils.loop_switches import parse_feedback_level, parse_restart_mode

        self.restart_mode: str = parse_restart_mode(os.getenv("ROGUE_ONE_RESTART_MODE"))
        self.feedback_level: str = parse_feedback_level(
            os.getenv("ROGUE_ONE_FEEDBACK_LEVEL")
        )
        # E9f (AGENT 2027 A16): "three_agent" (default, Full-v2) | "set_phased"
        # (rogueone.utils.topology). set_phased refuses every other non-Full switch.
        from rogueone.utils.topology import check_exclusive, parse_topology

        self.topology: str = parse_topology(os.getenv("ROGUE_ONE_TOPOLOGY"))
        check_exclusive(self.topology)
        # "inner_cv" selects the reported iteration by a cross-validation score computed
        # inside the training folds (nested CV). "test" reproduces the legacy protocol
        # where the held-out metric both selects and is reported.
        self.selection_signal: str = os.getenv(
            "ROGUE_ONE_SELECTION_SIGNAL", "inner_cv"
        ).strip().lower()
        # Stratify the outer CV folds for classification. Off by default so existing
        # results stay reproducible; enabled to match a stratifying baseline's fold
        # partition and remove the fold definition as a confound.
        self.stratified_folds: bool = env_as_bool("ROGUE_ONE_STRATIFIED_FOLDS", False)
        # Seed forwarded to the LLM's sampler on every chat completion. Unset (None)
        # leaves generation unseeded, which is how the submitted runs were produced;
        # fold_seed alone only varies the data partition, not the model's sampling, so
        # a seed sweep without this measures fold variance only. vLLM honours the
        # OpenAI `seed` field, but verify determinism against the live endpoint before
        # relying on it -- support is per-backend and not guaranteed.
        _llm_seed = os.getenv("ROGUE_ONE_LLM_SEED", "").strip()
        self.llm_seed: int | None = int(_llm_seed) if _llm_seed else None
        # E2 (AGENT 2027). "message" returns tool-error text to the agent;
        # "none" returns None from the error sites (what t17 did by accident).
        _tef = os.getenv("ROGUE_ONE_TOOL_ERROR_FEEDBACK", "message").strip().lower()
        if _tef not in ("message", "none"):
            raise ValueError(
                f"ROGUE_ONE_TOOL_ERROR_FEEDBACK must be 'message' or 'none', got {_tef!r}"
            )
        self.tool_error_feedback: str = _tef
        # Max characters of a tool result returned to an agent; 0 = unbounded.
        self.tool_output_max_chars: int = int(
            os.getenv("ROGUE_ONE_TOOL_OUTPUT_MAX_CHARS", 0) or 0
        )
        # 1 = forward reasoning_effort to the endpoint instead of popping it.
        self.reasoning_effort_passthrough: bool = env_as_bool(
            "ROGUE_ONE_REASONING_EFFORT_PASSTHROUGH", False
        )
        # 1 = inline "$ref"/"$defs" in tool schemas before sending them, and accept a JSON
        # object string for the Tester's code_input. vLLM's qwen3_coder tool parser does not
        # follow "$ref", so object arguments arrive as strings (A17/A19 Qwen runs). Off by
        # default so O120 requests stay byte-identical.
        self.inline_tool_schema_refs: bool = env_as_bool(
            "ROGUE_ONE_INLINE_TOOL_SCHEMA_REFS", False
        )
        # Lenient exec for LLM-authored code (A17 Q4 smoke): turn literal "\n" into
        # newlines only when the code does not compile as sent, let a trailing bare
        # expression set `result`, and let the Scientist's pandas tool return printed
        # output when `result` is None. Off by default so O120 behaviour is unchanged.
        self.exec_compat: bool = env_as_bool("ROGUE_ONE_EXEC_COMPAT", False)
        # Loop guard (A17 Q4 smoke 2: the Scientist repeated one identical tool call
        # up to 587 times). When the last N tool calls of a conversation are identical,
        # the next request is sent without tools and with a note to answer. 0 = off,
        # the default, so O120 requests stay byte-identical.
        self.loop_guard_repeats: int = int(os.getenv("ROGUE_ONE_LOOP_GUARD", "0") or 0)
        # Inf guard (A17 relaunch: Q4/Q122 rows ended on XGBoostError "Input data
        # contains inf"). Clips +-inf in the Tester's fold matrices to +-1e6, as the
        # scoring fits already do. Off by default so O120 behaviour is unchanged.
        self.inf_guard: bool = env_as_bool("ROGUE_ONE_INF_GUARD", False)
        # Rounds of the Extractor's per-iteration loop (code default 10).
        self.extractor_rounds: int = int(os.getenv("ROGUE_ONE_EXTRACTOR_ROUNDS", 10))
        # E6 (AGENT 2027). "replace" rebuilds the pool from the id column every fold
        # (pre-E6 behaviour); "augment" also seeds every raw column as a replayable,
        # prune-protected row (rogueone.utils.fold_safe_features.seed_raw_feature_rows).
        _fpm = os.getenv("ROGUE_ONE_FEATURE_POOL_MODE", "replace").strip().lower()
        if _fpm not in ("replace", "augment"):
            raise ValueError(
                f"ROGUE_ONE_FEATURE_POOL_MODE must be 'replace' or 'augment', got {_fpm!r}"
            )
        self.feature_pool_mode: str = _fpm
        # E14 (AGENT 2027). "train_fit" (default) = the Scientist sees the in-sample
        # train_* metrics; "inner_cv_l1" = one inner-CV scalar on a split disjoint from
        # the selection split (rogueone.utils.scientist_signal). Anything else raises.
        from rogueone.utils.scientist_signal import parse_scientist_signal

        self.scientist_signal: str = parse_scientist_signal(
            os.getenv("ROGUE_ONE_SCIENTIST_SIGNAL")
        )
        # E6: 1 = run the end-of-run replay-validity check (Finding 39 authoring frame).
        self.replay_validity_check: bool = env_as_bool(
            "ROGUE_ONE_REPLAY_VALIDITY_CHECK", False
        )
        # E1b: run only the first n outer folds (smoke / Stage-0 only); None = all.
        from rogueone.utils.task_registry import parse_smoke_folds

        self.smoke_folds: int | None = parse_smoke_folds(
            os.getenv("ROGUE_ONE_SMOKE_FOLDS")
        )
        # Comma-separated dataset names; empty means run the full sweep.
        self.dataset_filter: list[str] = [
            name.strip().lower()
            for name in os.getenv("ROGUE_ONE_DATASETS", "").split(",")
            if name.strip()
        ]
        # E9c (AGENT 2027). "llm" (default) = the Scientist writes the focus (Full-v2);
        # "oracle" / "misdirected" = fixed focus per (fold, iteration) from
        # ROGUE_ONE_PLANNER_FOCUS_FILE; "none" is reserved for E9d. Anything else raises.
        from rogueone.utils.planner_mode import parse_planner_mode

        self.planner_mode: str = parse_planner_mode(os.getenv("ROGUE_ONE_PLANNER_MODE"))
        self.planner_focus_file: str | None = (
            os.getenv("ROGUE_ONE_PLANNER_FOCUS_FILE", "").strip() or None
        )


class WandBConfig(metaclass=Singleton):
    def __init__(self):
        self.project_name: str = os.getenv("WANDB_PROJECT_NAME", "rogueone_project")
        self.entity: str = os.getenv("WANDB_ENTITY", "rogueone_entity")
        self.api_key: str = os.getenv("WANDB_API_KEY", "")


class AgentConfig:
    def __init__(
        self,
        temperature: float = 0.7,
        reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = "medium",
        max_iterations: int = 100,
        max_tokens: int | None = None,
    ):
        self.temperature = temperature
        self.reasoning = Reasoning(effort=reasoning_effort)
        self.max_iterations = max_iterations
        self.max_tokens = max_tokens


def _opt_int_env(key: str) -> int | None:
    val = os.getenv(key)
    if val is None or str(val).strip() == "":
        return None
    return int(val)


class ExtractorAgentConfig(AgentConfig, metaclass=Singleton):
    def __init__(self):
        super().__init__(
            temperature=os.getenv("EXTRACTOR_AGENT_TEMPERATURE", 0.7),
            reasoning_effort=os.getenv("EXTRACTOR_AGENT_REASONING_EFFORT", None),
            max_iterations=int(os.getenv("EXTRACTOR_AGENT_MAX_ITERATIONS", 100)),
            # 16384 leaves headroom for gpt-oss-120b reasoning tokens, which count
            # against the response budget. Repeated truncation errors at 8192
            # ('(' was never closed') in tic-tac-toe runs si9h32kk and lciydoty
            # showed the model running out mid-list / mid-lambda.
            max_tokens=int(os.getenv("EXTRACTOR_AGENT_MAX_TOKENS", 32768)),
        )


class MedicalKnowledgeAgentConfig(AgentConfig, metaclass=Singleton):
    def __init__(self):
        super().__init__(
            temperature=os.getenv("MEDICAL_KNOWLEDGE_AGENT_TEMPERATURE", 0.7),
            reasoning_effort=os.getenv(
                "MEDICAL_KNOWLEDGE_AGENT_REASONING_EFFORT", None
            ),
            max_iterations=int(
                os.getenv("MEDICAL_KNOWLEDGE_AGENT_MAX_ITERATIONS", 100)
            ),
            max_tokens=_opt_int_env("MEDICAL_KNOWLEDGE_AGENT_MAX_TOKENS"),
        )


class TestAgentConfig(AgentConfig, metaclass=Singleton):
    def __init__(self):
        super().__init__(
            temperature=os.getenv("TESTER_AGENT_TEMPERATURE", 0.7),
            reasoning_effort=os.getenv("TESTER_AGENT_REASONING_EFFORT", None),
            max_iterations=int(os.getenv("TESTER_AGENT_MAX_ITERATIONS", 100)),
            max_tokens=int(os.getenv("TESTER_AGENT_MAX_TOKENS", 8192)),
        )


class ScientistAgentConfig(AgentConfig, metaclass=Singleton):
    def __init__(self):
        super().__init__(
            temperature=os.getenv("SCIENTIST_AGENT_TEMPERATURE", 0.7),
            reasoning_effort=os.getenv("SCIENTIST_AGENT_REASONING_EFFORT", None),
            max_iterations=int(os.getenv("SCIENTIST_AGENT_MAX_ITERATIONS", 200)),
            max_tokens=_opt_int_env("SCIENTIST_AGENT_MAX_TOKENS"),
        )
