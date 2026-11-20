"""Flag registry (AGENT 2027 E1a): every R3/R4 flag and run-identity key, with the
values the process actually runs with.

Why this exists. ``config.py`` loads ``.env``, then ``.dev.env`` with
``override=True``, then ``.secret.env``. A ``.dev.env`` baked into an image therefore
silently beats the ``docker run -e`` flags the launcher sets, and nothing in W&B
showed it. This module reads the **effective** values (``os.environ`` after the
dotenv chain has run) and hands them to ``init_wandb_run`` as ``flags/<KEY>``, next
to the run-identity keys the manifest launcher sets (01 §7 R4, R12).

The module is deliberately standard-library only at import time (``python-dotenv``
is imported lazily inside :func:`load_dotenv_chain`) and imports nothing from
``rogueone``, so ``scripts/agent2027/make_manifest.py`` can load it straight from
its file path without running the package's config side effects.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Mapping, Optional

# --------------------------------------------------------------------------- dotenv chain
#: The order ``config.py`` loads its dotenv files in, with each file's override mode.
DOTENV_CHAIN: tuple[tuple[str, bool], ...] = (
    (".env", False),
    (".dev.env", True),
    (".secret.env", False),
)

#: What the last :func:`load_dotenv_chain` call did (logged into W&B config).
DOTENV_STATE: dict = {"root": None, "files_loaded": [], "dev_env_keys": []}


def _parse_dotenv_keys(path: Path) -> list[str]:
    """Keys a dotenv file assigns (comments and blank lines skipped)."""
    keys: list[str] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return keys
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key = line.split("=", 1)[0].strip()
        if key:
            keys.append(key)
    return keys


def load_dotenv_chain(root: Path | str) -> dict:
    """Load ``.env`` / ``.dev.env`` (override) / ``.secret.env`` from ``root``.

    This is exactly the sequence ``config.py`` has always run; it is factored out
    here so the registry can report which files were present and which keys the
    overriding ``.dev.env`` assigned. Returns (and stores in
    :data:`DOTENV_STATE`) ``{"root", "files_loaded", "dev_env_keys"}``.
    """
    from dotenv import load_dotenv

    root = Path(root)
    loaded: list[str] = []
    dev_keys: list[str] = []
    for name, override in DOTENV_CHAIN:
        path = root / name
        load_dotenv(path, override=override)
        if path.is_file():
            loaded.append(name)
            if name == ".dev.env":
                dev_keys = _parse_dotenv_keys(path)
    DOTENV_STATE.update(root=str(root), files_loaded=loaded, dev_env_keys=sorted(set(dev_keys)))
    return dict(DOTENV_STATE)


# --------------------------------------------------------------------------- key lists
# Each entry: key -> code default as the reading site applies it when the variable is
# unset (None = the code has no default / the flag is not read yet / unset is a state).

#: 01 §7 R3, existing flags (verified reading sites in the comments of R3).
R3_FLAGS: dict[str, Optional[str]] = {
    "ROGUE_ONE_NUM_ITERATIONS": "3",
    "ROGUE_ONE_K_FOLDS": "5",
    "ROGUE_ONE_FOLD_SEED": "42",
    "ROGUE_ONE_LLM_SEED": None,
    "ROGUE_ONE_SELECTION_SIGNAL": "inner_cv",
    "ROGUE_ONE_ENABLE_ROLLBACK": "False",
    "ROGUE_ONE_STRATIFIED_FOLDS": "False",
    "ROGUE_ONE_DO_BALANCING": "True",
    "ROGUE_ONE_FOLD_RETRY_LIMIT": "1",
    "ROGUE_ONE_DATASETS": "",
    "ROGUE_ONE_DO_LOGGING": "True",
    "ROGUE_ONE_START_VLLM": "True",
    "ROGUE_ONE_DISABLE_WANDB": "0",
    "ROGUE_ONE_VERBOSE_AGENT_ERRORS": "0",
    "ROGUE_ONE_RETRIEVAL_TRANSCRIPT": None,
    "CONSOLE_DISABLE": "False",
    "LLM_MODEL": None,
    "LLM_PORT": "",
    "LLM_API_KEY": "",
    "LLM_TEMPERATURE": "0.7",
    "ROGUE_ONE_SCIENTIST_MODEL": "",
    "ROGUE_ONE_SCIENTIST_PORT": "",
    "ROGUE_ONE_EXTRACTOR_MODEL": "",
    "ROGUE_ONE_EXTRACTOR_PORT": "",
    "ROGUE_ONE_TESTER_MODEL": "",
    "ROGUE_ONE_TESTER_PORT": "",
    "SCIENTIST_AGENT_TEMPERATURE": "0.7",
    "EXTRACTOR_AGENT_TEMPERATURE": "0.7",
    "TESTER_AGENT_TEMPERATURE": "0.7",
    "MEDICAL_KNOWLEDGE_AGENT_TEMPERATURE": "0.7",
    "EXTRACTOR_AGENT_MAX_TOKENS": "32768",
    "TESTER_AGENT_MAX_TOKENS": "8192",
    "SCIENTIST_AGENT_MAX_TOKENS": None,
    "EXTRACTOR_AGENT_MAX_ITERATIONS": "100",
    "TESTER_AGENT_MAX_ITERATIONS": "100",
    "SCIENTIST_AGENT_MAX_ITERATIONS": "200",
    "EXTRACTOR_AGENT_REASONING_EFFORT": None,
    "TESTER_AGENT_REASONING_EFFORT": None,
    "SCIENTIST_AGENT_REASONING_EFFORT": None,
    "MEDICAL_KNOWLEDGE_AGENT_REASONING_EFFORT": None,
    "WANDB_ENTITY": None,
    "WANDB_PROJECT_NAME": None,
    "WANDB_API_KEY": None,
    "SERPER_API_KEY": None,
}

#: 01 §7 R4, new flags. Defaults are the code defaults (not the Full-v2 values; those
#: live in ``scripts/agent2027/v2_defaults.env``). None = not read by the code yet.
R4_FLAGS: dict[str, Optional[str]] = {
    "ROGUE_ONE_TOOL_ERROR_FEEDBACK": "message",
    "ROGUE_ONE_EXTRACTOR_ROUNDS": "10",
    "ROGUE_ONE_WEB_SEARCH_PROVIDER": "serper",
    "ROGUE_ONE_SEARXNG_URL": "http://127.0.0.1:8888",
    "ROGUE_ONE_TOOL_OUTPUT_MAX_CHARS": "0",
    "ROGUE_ONE_REASONING_EFFORT_PASSTHROUGH": "0",
    "ROGUE_ONE_LLM_TRACE": "0",
    "ROGUE_ONE_PROMPT_DUMP": "0",
    "ROGUE_ONE_TASK_REGISTRY": None,
    "ROGUE_ONE_SMOKE_FOLDS": None,
    "ROGUE_ONE_FEATURE_POOL_MODE": "replace",
    "ROGUE_ONE_REPLAY_VALIDITY_CHECK": "0",
    "ROGUE_ONE_SEARCH_CACHE": "off",
    "ROGUE_ONE_SEARCH_CACHE_PATH": "/workspace/cache/serper_cache.sqlite",
    "ROGUE_ONE_WEBSEARCH_ROUTING": "global",
    "ROGUE_ONE_KNOWLEDGE_BACKEND": "web",
    "ROGUE_ONE_LOCAL_CORPUS": None,
    "ROGUE_ONE_TOPOLOGY": "three_agent",  # E9f (rogueone/utils/topology.py)
    "ROGUE_ONE_PLANNER_MODE": "llm",
    "ROGUE_ONE_PLANNER_FOCUS_FILE": None,
    "ROGUE_ONE_MEMORY_MODE": "full",  # E9d (rogueone/utils/fx_switches.py)
    "ROGUE_ONE_ASSESSMENT_MODE": "llm",  # E9d
    "ROGUE_ONE_PRUNING_MODE": "llm",  # E9d
    "ROGUE_ONE_PRUNING_SHADOW": "0",  # E9d
    "ROGUE_ONE_RESTART_MODE": "off",
    "ROGUE_ONE_FEEDBACK_LEVEL": "full",
    "ROGUE_ONE_SANDBOX": "off",
    "ROGUE_ONE_FALLBACKS": None,
    "ROGUE_ONE_SCIENTIST_SIGNAL": "train_fit",
}

FLAG_KEYS: tuple[str, ...] = tuple(R3_FLAGS) + tuple(R4_FLAGS)
FLAG_DEFAULTS: dict[str, Optional[str]] = {**R3_FLAGS, **R4_FLAGS}

#: Run-identity keys (R4 "Run-identity keys") -> W&B config key (R12).
IDENTITY_KEYS: dict[str, str] = {
    "ROGUE_ONE_ABLATION_ID": "ablation_id",
    "ROGUE_ONE_ARM_ID": "arm_id",
    "ROGUE_ONE_REPLICATE": "replicate",
    "ROGUE_ONE_SESSION_ID": "session_id",
    "ROGUE_ONE_MANIFEST_ROW": "manifest_row",
    "ROGUE_ONE_IS_SENTINEL": "is_sentinel",
    "ROGUE_ONE_CODE_PHASE": "code_phase",
    "ROGUE_ONE_IMAGE_TAG": "image_tag",
    "ROGUE_ONE_GIT_SHA": "git_sha",
    "ROGUE_ONE_SERVER_TOPOLOGY": "server_topology",
    "ROGUE_ONE_RUNG_S": "rung_S",
    "ROGUE_ONE_RUNG_E": "rung_E",
    "ROGUE_ONE_RUNG_T": "rung_T",
}

#: Extra launcher-provided metadata for the R12 config keys that R4 gives no env name
#: (``image_id``, ``vllm_image_id``, ``prereg_tag``, ``parallel``). Set by
#: ``ro27_session_launch.sh`` / ``ro27_clients_manifest.sh``.
LAUNCHER_META_KEYS: dict[str, str] = {
    "ROGUE_ONE_IMAGE_ID": "image_id",
    "ROGUE_ONE_VLLM_IMAGE_ID": "vllm_image_id",
    "ROGUE_ONE_PREREG_TAG": "prereg_tag",
    "ROGUE_ONE_PARALLEL": "parallel",
}

#: Values never written to W&B in clear text.
SECRET_KEYS = frozenset({"WANDB_API_KEY", "SERPER_API_KEY", "LLM_API_KEY"})

#: Identity values that must be present for a run to take the R12 name.
NAME_KEYS = ("ROGUE_ONE_ABLATION_ID", "ROGUE_ONE_ARM_ID", "ROGUE_ONE_REPLICATE")

#: Entity for every AGENT 2027 run (R12). A01 uses the legacy launcher, which sets
#: WANDB_ENTITY=brainflux itself and never sets the identity keys.
AGENT2027_ENTITY = "PlanBench"

UNSET = "<unset>"


def _mask(key: str, value: Optional[str]) -> Optional[str]:
    if key in SECRET_KEYS and value:
        return "<set>"
    return value


def effective_flags(environ: Optional[Mapping[str, str]] = None) -> dict[str, Optional[str]]:
    """``{KEY: effective value}`` for every R3/R4 flag.

    The value is the environment value when the variable is set (after dotenv), and
    otherwise the reading site's code default from :data:`FLAG_DEFAULTS` (``None``
    when the code has none). Secrets are masked to ``"<set>"``.
    """
    env = os.environ if environ is None else environ
    out: dict[str, Optional[str]] = {}
    for key in FLAG_KEYS:
        if key in env:
            out[key] = _mask(key, env[key])
        else:
            out[key] = FLAG_DEFAULTS[key]
    return out


def unset_flags(environ: Optional[Mapping[str, str]] = None) -> list[str]:
    env = os.environ if environ is None else environ
    return [k for k in FLAG_KEYS if k not in env]


def identity_from_env(environ: Optional[Mapping[str, str]] = None) -> dict[str, str]:
    """W&B config keys for the identity and launcher-metadata env vars that are set."""
    env = os.environ if environ is None else environ
    out: dict[str, str] = {}
    for env_key, cfg_key in {**IDENTITY_KEYS, **LAUNCHER_META_KEYS}.items():
        val = env.get(env_key)
        if val is not None and str(val).strip() != "":
            out[cfg_key] = str(val).strip()
    return out


def has_run_identity(environ: Optional[Mapping[str, str]] = None) -> bool:
    env = os.environ if environ is None else environ
    return all(str(env.get(k, "")).strip() for k in NAME_KEYS)


def run_name(ablation_id: str, arm_id: str, dataset_id: str, fold_seed, replicate) -> str:
    """R12 run name ``<AID>__<arm_id>__<dataset_id>__f<fold_seed>__r<replicate>``."""
    return f"{ablation_id}__{arm_id}__{dataset_id}__f{fold_seed}__r{replicate}"


def wandb_run_kwargs(
    *,
    dataset_id: str,
    task_type: str,
    fold_seed,
    llm_seed,
    models: Optional[Mapping[str, str]] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> dict:
    """Extra ``wandb.init`` arguments for a run.

    Always returns ``{"config": {...}}`` holding ``flags/<KEY>`` for every R3/R4 flag,
    the dotenv report and whatever identity keys are set. When the run identity is
    present (``ROGUE_ONE_ABLATION_ID``/``_ARM_ID``/``_REPLICATE``) it also returns the
    R12 ``name``, ``group`` (= AID), ``job_type`` (= arm), ``tags``
    ([session, code_phase, image_tag]) and the ``entity`` default ``PlanBench``.
    Without identity keys only ``config`` is returned, so legacy runs keep their
    legacy name, group and entity.
    """
    env = os.environ if environ is None else environ
    config: dict = {f"flags/{k}": v for k, v in effective_flags(env).items()}
    config["flags_unset"] = unset_flags(env)
    config["dotenv_files_loaded"] = list(DOTENV_STATE.get("files_loaded") or [])
    config["dotenv_dev_env_keys"] = list(DOTENV_STATE.get("dev_env_keys") or [])
    config.update(identity_from_env(env))
    config["dataset_id"] = dataset_id
    config["task_type"] = task_type
    config["fold_seed"] = fold_seed
    config["llm_seed"] = llm_seed
    for role, model in (models or {}).items():
        config[f"model_{role}"] = model

    out: dict = {"config": config}
    if not has_run_identity(env):
        return out

    aid = str(env["ROGUE_ONE_ABLATION_ID"]).strip()
    arm = str(env["ROGUE_ONE_ARM_ID"]).strip()
    rep = str(env["ROGUE_ONE_REPLICATE"]).strip()
    out["name"] = run_name(aid, arm, dataset_id, fold_seed, rep)
    out["group"] = aid
    out["job_type"] = arm
    tags = [
        str(env.get(k, "")).strip()
        for k in ("ROGUE_ONE_SESSION_ID", "ROGUE_ONE_CODE_PHASE", "ROGUE_ONE_IMAGE_TAG")
    ]
    out["tags"] = [t for t in tags if t]
    entity = str(env.get("WANDB_ENTITY", "")).strip()
    out["entity"] = entity or AGENT2027_ENTITY
    return out


def all_registered_keys() -> Iterable[str]:
    """Every env key the registry knows about (flags, identity, launcher metadata)."""
    return (*FLAG_KEYS, *IDENTITY_KEYS, *LAUNCHER_META_KEYS)
