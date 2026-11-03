import os
from pathlib import Path

# Keep W&B useful but quiet in container logs unless explicitly overridden.
os.environ["WANDB_SILENT"] = "true"
os.environ["WANDB_QUIET"] = "true"
os.environ["WANDB_CONSOLE"] = "off"

import wandb

from rogueone.utils import wandb_cfg, rouge_one_cfg
from rogueone.utils import flag_registry
from rogueone.utils.config import ExperimentConfig
from rogueone.utils.console import ConsoleManager


def wandb_logging_wrapper(func):
    """Decorator that conditionally enables W&B logging based on configuration.

    If ROGUE_ONE_DO_LOGGING is disabled, returns None instead of executing the wrapped function.

    Args:
        func: The function to wrap.

    Returns:
        wrapper: A wrapper function that checks logging configuration before executing.
    """

    def wrapper(*args, **kwargs):
        if not rouge_one_cfg.do_logging:
            return None
        # Prevent implicit wandb auto-init when a run is intentionally disabled.
        if wandb.run is None:
            return None
        return func(*args, **kwargs)

    return wrapper


def safe_wandb_log(payload: dict, *, prefer_step: int | None = None) -> None:
    """Log to W&B using ``prefer_step`` only when it would not regress the run step.

    W&B requires ``step`` to be monotonically non-decreasing across the run; it
    silently drops calls whose ``step`` is below ``wandb.run.step``. The LLM
    client logs token usage on every model call without an explicit step (so
    each call auto-increments the run step), which means small per-iteration
    step values such as ``step=index`` (0..K) almost always end up below the
    current step and disappear. CAR runs ``71g183ic`` / ``o5zyks6e`` /
    ``m4i2r4v9`` exhibited this — all ``extractor_iteration/*`` and
    ``extractor_abort/*`` keys were missing from history despite the code
    paths firing, because each ``wandb.log(..., step=index)`` was dropped.

    This helper falls back to wandb's auto-incremented step whenever
    ``prefer_step`` would go backwards. Call sites that pass an iteration index
    in the payload itself (e.g. ``extractor_iteration/iteration_global``) keep
    full plottability — readers chart against that field instead of the global
    ``_step``.
    """
    if not rouge_one_cfg.do_logging or wandb.run is None:
        return

    current_step = getattr(wandb.run, "step", 0) or 0
    if prefer_step is not None and prefer_step >= current_step:
        wandb.log(payload, step=prefer_step)
    else:
        wandb.log(payload)


def sign_in_wandb():
    """Authenticate with Weights & Biases using API key from environment configuration.

    Uses WANDB_API_KEY from environment to sign in to W&B. Logs error message if authentication fails.
    """
    try:
        wandb.login(key=wandb_cfg.api_key, verify=True)
        ConsoleManager.console_print(
            "Logged in to Weights & Biases using WANDB_API_KEY from environment."
        )
        return
    except Exception as e:
        ConsoleManager.console_error_print(f"Failed to login with WANDB_API_KEY: {e}")


def _dataset_id_for(cfg: ExperimentConfig) -> str:
    """Task folder name (= R10 ``dataset_id``) of a config, falling back to the
    experiment name. The folder is the parent of ``data.csv``; it is what
    ``ROGUE_ONE_DATASETS`` / ``select_paths`` match on."""
    try:
        name = Path(cfg.feature_file_path).parent.name
    except Exception:  # noqa: BLE001
        name = ""
    return name or str(cfg.experiment_name)


def _summariser_model_label(llm_cfg) -> str:
    """``model_W``: ``LLM_MODEL`` under global routing; under ``caller`` the common
    role model when S, E and T share one, else ``caller``."""
    from rogueone.llm.agents.core.knowledge import backend as knowledge_backend

    if knowledge_backend.routing() != "caller":
        return llm_cfg.model
    role_models = {llm_cfg.model_for(r) for r in knowledge_backend.CALLER_ROLES}
    return role_models.pop() if len(role_models) == 1 else "caller"


def kt3_corpus_config() -> dict:
    """E8b / A14-T7: ``{"kt3_corpus_sha256": <A14-T1 hash>}`` under
    ``ROGUE_ONE_KNOWLEDGE_BACKEND=local``, else ``{}`` (Full-v2's config is unchanged).

    The hash is of the mounted corpus folder the run actually serves; A14-T10 checks it
    against the pre-registered hash. A missing folder is logged, not raised here: the
    run already refuses to start in ``WebSearchAgent.__init__``.
    """
    from rogueone.llm.agents.core.knowledge import backend as knowledge_backend

    try:
        if knowledge_backend.backend() != "local":
            return {}
        from rogueone.llm.agents.core.knowledge import local_corpus

        return {"kt3_corpus_sha256": local_corpus.corpus_sha256(knowledge_backend.local_corpus_dir())}
    except Exception as e:  # noqa: BLE001 - metadata must never block a run
        return {"kt3_corpus_sha256": f"ERROR: {type(e).__name__}: {e}"}


def registry_wandb_kwargs(cfg: ExperimentConfig) -> dict:
    """Flag-registry extras for ``wandb.init`` (see :mod:`rogueone.utils.flag_registry`)."""
    from rogueone.utils import llm_cfg

    try:
        models = {
            "S": llm_cfg.model_for("scientist"),
            "E": llm_cfg.model_for("extractor"),
            "T": llm_cfg.model_for("tester"),
            # E7: the summariser is on LLM_MODEL (global) or follows its caller.
            "W": _summariser_model_label(llm_cfg),
        }
    except Exception:  # noqa: BLE001 - metadata must never block a run
        models = {}
    out = flag_registry.wandb_run_kwargs(
        dataset_id=_dataset_id_for(cfg),
        task_type=getattr(cfg, "task_type", ""),
        fold_seed=rouge_one_cfg.fold_seed,
        llm_seed=rouge_one_cfg.llm_seed,
        models=models,
    )
    out["config"].update(kt3_corpus_config())  # E8b: only under the local backend
    return out


def init_wandb_run(cfg: ExperimentConfig, **kwargs):
    """Initialize a Weights & Biases run for the current experiment.

    Args:
        cfg (ExperimentConfig): The experiment configuration containing experiment metadata.
        **kwargs: Additional keyword arguments to pass to wandb.init().

    Returns:
        wandb.Run or None: The initialized W&B run object, or None if logging is disabled.
    """

    if not rouge_one_cfg.do_logging:
        ConsoleManager.console_print(
            "ROGUE_ONE_DO_LOGGING is disabled. Skipping Weights & Biases run initialization.",
            style="yellow",
        )
        return None

    sign_in_wandb()

    ConsoleManager.console_print(
        f"Initializing Weights & Biases run for {cfg.experiment_name}", style="cyan"
    )

    conf = cfg.to_dict()
    # E1a: every R3/R4 flag as flags/<KEY> with its effective (post-dotenv) value, the
    # run-identity keys, and -- when the manifest launcher set the identity -- the R12
    # run name, group, job_type, tags and the PlanBench entity. Legacy runs (no
    # identity keys) keep name=experiment_name and their configured entity.
    extras = registry_wandb_kwargs(cfg)
    conf.update(extras.pop("config"))
    conf.update(kwargs)

    run = wandb.init(
        project=wandb_cfg.project_name,
        entity=extras.pop("entity", None) or wandb_cfg.entity,
        name=extras.pop("name", None) or cfg.experiment_name,
        config=conf,
        settings=wandb.Settings(silent=True, console="off"),
        **extras,
    )
    ConsoleManager.console_print(
        f"Weights & Biases run initialized: {run.name} ({run.id})", style="green"
    )

    return run


@wandb_logging_wrapper
def log_fold_payload(payload: dict, *, step: int | None = None) -> None:
    """Log fold-level metrics and payloads to Weights & Biases when logging is enabled.

    Args:
        payload (dict): Dictionary containing metrics/data to log.
        step (int, optional): Step number for the log entry. If None, uses auto-incrementing step.
    """
    wandb.log(payload, step=step)


@wandb_logging_wrapper
def log_llm_token_usage(
    *,
    agent_name: str,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
    cumulative_input_tokens: int,
    cumulative_output_tokens: int,
    cumulative_total_tokens: int,
    step: int | None = None,
) -> None:
    """Log per-agent LLM token usage to Weights & Biases.

    Args:
        agent_name (str): Logical agent identifier (e.g., "TesterAgent").
        input_tokens (int): Prompt/input tokens used by the latest model call.
        output_tokens (int): Completion/output tokens used by the latest model call.
        total_tokens (int): Total tokens used by the latest model call.
        cumulative_input_tokens (int): Running input-token total for this agent.
        cumulative_output_tokens (int): Running output-token total for this agent.
        cumulative_total_tokens (int): Running total-token total for this agent.
        step (int, optional): Optional W&B step.
    """
    payload = {
        f"LLM_tokens/{agent_name}/input": input_tokens,
        f"LLM_tokens/{agent_name}/output": output_tokens,
        f"LLM_tokens/{agent_name}/total": total_tokens,
        f"LLM_tokens_cumulative/{agent_name}/input": cumulative_input_tokens,
        f"LLM_tokens_cumulative/{agent_name}/output": cumulative_output_tokens,
        f"LLM_tokens_cumulative/{agent_name}/total": cumulative_total_tokens,
    }
    wandb.log(payload, step=step)


@wandb_logging_wrapper
def log_file_artifact(
    file_path: str | Path,
    *,
    artifact_name: str,
    artifact_type: str,
    metadata: dict | None = None,
) -> bool:
    """Upload a file to Weights & Biases as an artifact if it exists.

    Args:
        file_path (str | Path): File to upload.
        artifact_name (str): Name of the created artifact.
        artifact_type (str): Artifact type for W&B organization.
        metadata (dict, optional): Optional metadata attached to the artifact.

    Returns:
        bool: True when an artifact was uploaded, otherwise False.
    """
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        ConsoleManager.console_print(
            f"[yellow]Skipping W&B artifact upload; file not found: {path}[/yellow]"
        )
        return False

    artifact = wandb.Artifact(
        name=artifact_name,
        type=artifact_type,
        metadata=metadata or {},
    )
    artifact.add_file(str(path))
    wandb.log_artifact(artifact)
    return True


if __name__ == "__main__":
    from math import cos

    from pathlib import Path
    import io

    wandb_run = init_wandb_run(
        ExperimentConfig.from_yaml(
            Path("./tasks/suppression_rato/config.yml")
        )
    )

    for i in range(100):
        wandb.log({"test_metric": cos(i / 10)}, step=i)
