"""X7 / A17-T7: generate one A17 F0 pool (``f0pool_<rung>``): 5 outer folds x 128 single-shot drafts.

A17 §4 (F0 framework level), A17-T2 (frozen prompt; pre-registration Amendment 6, 2026-10-21:
``docs/agent2027/X7_F0_prompt_DRAFT.md`` as drafted), A17-T7 (this generator), A17-T13 (runs).

What it does
------------
* Reads one F0 row (``A17-101`` ... ``A17-177`` minus the dropped O20 pool ``A17-143`` ... ``149``)
  from ``manifests/agent2027/A17_F0.csv``. F0 rows live in their own file on purpose: they are
  not Rogue One runs and must never be planned into a session that ``ro27_clients_manifest.sh``
  would run as Full-v2.
* Builds the outer folds exactly as ``rogue_one_main.py`` does (entity ids = ``data.csv``'s
  unique ``id`` values in file order; optional stratification labels for classification when the
  row's ``env_overrides`` set ``ROGUE_ONE_STRATIFIED_FOLDS=True``; duplicate groups from the raw
  table; ``deterministic_kfold_indices`` with ``k=5`` and the row's ``fold_seed``).
* Per fold, renders the frozen F0 prompt (task ``context_and_goal``; raw columns and dtypes;
  the first 5 outer-fold *training* rows by ``id``, without the target, which lives in
  ``labels.csv`` and is never read except for stratification) and sends 128 independent
  requests through Rogue One's own client wrapper ``VLLMCompatibleAsyncOpenAI``
  (``rogueone/llm/clients.py``), so thinking-off and parameter stripping match F2. No tools,
  ``temperature=0.7``, ``max_tokens=32768``, request ``seed = llm_seed*1000 + 128*f + j``
  (the F0 seed rule; passing it explicitly overrides ``ROGUE_ONE_LLM_SEED``).
* Logs per draw: raw text, parsed JSON, ``fenced``, parse error, prompt and completion tokens,
  ``finish_reason``, latency, error kind. Draws are appended to ``draws.jsonl`` as they finish,
  so a killed job resumes where it stopped (``--resume``, the default). Infra errors (server
  unreachable, 5xx) are re-sent once at the end with the same seed.
* Writes ``folds.json`` (test ids per fold, for the A17-T8 scorer and the fold guard),
  ``prompts.jsonl`` (the exact messages per fold) and ``summary.json``; with ``--wandb`` it
  logs run ``A17__f0pool_<rung>__<dataset_id>__f<seed>__r0`` (group A17) with the R12
  identity keys, ``model_S/E/T/W`` = the server's model, ``flags/ROGUE_ONE_LLM_SEED`` =
  ``llm_seed``, summary ``tok_compl_total``, ``wall_s``, ``f0_parse_rate`` and uploads the
  output folder as an artifact of type ``f0-pool``.

``f0_parse_rate`` here is the share of draws whose answer parses as a JSON list of objects with
string ``name`` and ``pandas_command`` (fence stripped). Whether a snippet *executes* is decided
by the scorer (A17-T8, ``f0_score.py``), which runs the code in a network-less container; this
generator never executes model output.

Runs inside a ``rogue_one`` image (``--entrypoint python3 --network host``); see
``scripts/agent2027/f0_clients.sh`` and ``docs/agent2027/sessions/LAUNCH.md`` §7.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------------- frozen design
N_FOLDS = 5
N_DRAWS = 128
TEMPERATURE = 0.7
MAX_TOKENS = 32768
N_EXAMPLE_ROWS = 5
MAX_ITEMS = 10
DEFAULT_ENTITY = "PlanBench"
DEFAULT_PROJECT = "RogueOne-AGENT2027"
F0_ROW_RANGE = (101, 177)
F0_DROPPED_ROWS = frozenset(range(143, 150))  # O20 pool, SR4 fallback

# Frozen text (A17-T2 / X7, Amendment 6). tests check it against X7_F0_prompt_DRAFT.md.
SYSTEM_PROMPT = (
    "You are a feature-engineering assistant for tabular machine learning. You propose new\n"
    "features as pandas code. You cannot run code, call tools or search the web: answer in one\n"
    "message."
)
USER_TEMPLATE = (
    "# Task\n"
    "{context_and_goal}\n"
    "\n"
    "# Raw table `df`\n"
    "Columns (name and dtype):\n"
    "{columns}\n"
    "\n"
    "Five example rows (training data; the prediction target is not shown):\n"
    "{rows}\n"
    "\n"
    "# How each feature is computed\n"
    "Each feature is one Python snippet that runs in a fresh scope with these names available:\n"
    "- `df`: the raw table above (several rows may share one `id`),\n"
    "- `df_features`: a table with one row per `id` (column `id`) to which features are added,\n"
    "- `pd` (pandas) and `np` (numpy).\n"
    "Rules:\n"
    "- Assign the final value to a variable named `result`.\n"
    "- `result` must be numeric, without NaN or inf, and have one value per row of\n"
    "  `df_features`, aligned with `df_features['id']`. Use a `groupby('id')` and map it onto\n"
    "  `df_features['id']`, for example:\n"
    "  result = df_features['id'].map(df.groupby('id')['age'].first())\n"
    "- Define any helper (mapping, mask, function) inside the same snippet; nothing persists\n"
    "  between snippets.\n"
    "- Use `&`, `|` and `~` for boolean operations on pandas Series, never `and` / `or` / `not`.\n"
    "- Do not reassign `df` or `df_features`, and do not read files.\n"
    "\n"
    "# Answer format\n"
    "Return ONLY a JSON list of at most 10 objects, with no text before or after it:\n"
    '[{{"name": "<feature name>", "pandas_command": "<snippet>", "argumentation": "<why this '
    'feature should help predict the target>"}}, ...]'
)

R7_HEADER = (
    "row_id,ablation_id,arm_id,dataset_id,task_type,fold_seed,llm_seed,replicate,is_sentinel,"
    "session_id,order_key,image_tag,rung_S,rung_E,rung_T,port_S,port_E,port_T,env_overrides,"
    "cap_hours,expected_u,notes"
).split(",")


# --------------------------------------------------------------------------- pure helpers
def seed_for(llm_seed: int, fold: int, draw: int) -> int:
    """A17 §4 F0 seed rule: request seed of proposal j of fold f."""
    if not (0 <= fold < N_FOLDS and 0 <= draw < N_DRAWS):
        raise ValueError(f"fold {fold} / draw {draw} out of range")
    return int(llm_seed) * 1000 + N_DRAWS * int(fold) + int(draw)


def is_f0_row_id(row_id: str) -> bool:
    m = re.fullmatch(r"A17-(\d{3})", row_id or "")
    if not m:
        return False
    n = int(m.group(1))
    return F0_ROW_RANGE[0] <= n <= F0_ROW_RANGE[1] and n not in F0_DROPPED_ROWS


def load_row(manifest: Path, row_id: str) -> dict[str, str]:
    """The F0 row ``row_id`` as a dict; refuses non-F0 rows and covtype."""
    with open(manifest, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != R7_HEADER:
            raise ValueError(f"{manifest}: header is not the R7 header")
        for row in reader:
            if row["row_id"] == row_id:
                break
        else:
            raise KeyError(f"row {row_id} not in {manifest}")
    if "covtype" in json.dumps(row).lower():
        raise ValueError("covtype is never run (Henrik, 2026-10-16)")
    if not is_f0_row_id(row_id) or row["ablation_id"] != "A17":
        raise ValueError(f"{row_id} is not an A17 F0 pool row")
    rung = row["rung_S"]
    if not (row["rung_E"] == row["rung_T"] == rung) or row["arm_id"] != f"f0pool_{rung}":
        raise ValueError(f"{row_id}: arm_id {row['arm_id']} does not match rung {rung}")
    return row


def env_overrides(row: dict[str, str]) -> dict[str, str]:
    raw = (row.get("env_overrides") or "").strip()
    return json.loads(raw) if raw else {}


def parse_answer(text: Optional[str]) -> dict[str, Any]:
    """Parse one draw (X7 Q4: a ```json fence is stripped, then the text must be a JSON list).

    Returns ``{"ok", "items", "n_items", "fenced", "over_limit", "error"}``. ``ok`` needs a
    JSON list whose entries are all objects with string ``name`` and ``pandas_command``.
    Lists longer than 10 are kept whole and flagged (``over_limit``); the scorer decides.
    """
    out: dict[str, Any] = {"ok": False, "items": None, "n_items": 0, "fenced": False,
                           "over_limit": False, "error": None}
    if text is None or not str(text).strip():
        out["error"] = "empty"
        return out
    s = str(text).strip()
    m = re.fullmatch(r"```[ \t]*(?:json|JSON)?[ \t]*\r?\n(.*?)\r?\n?```", s, flags=re.S)
    if m:
        s = m.group(1).strip()
        out["fenced"] = True
    try:
        data = json.loads(s)
    except json.JSONDecodeError as exc:
        out["error"] = f"json: {exc.msg}"
        return out
    if not isinstance(data, list):
        out["error"] = f"not a list ({type(data).__name__})"
        return out
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            out["error"] = f"item {i} is not an object"
            return out
        for key in ("name", "pandas_command"):
            if not isinstance(item.get(key), str) or not item[key].strip():
                out["error"] = f"item {i} lacks a string {key!r}"
                return out
    out.update(ok=True, items=data, n_items=len(data), over_limit=len(data) > MAX_ITEMS)
    return out


def render_columns(df) -> str:
    return "\n".join(f"- {c} ({df[c].dtype})" for c in df.columns if c != "id")


def render_rows(df, train_ids) -> str:
    """First 5 outer-fold training rows of ``data.csv`` by ``id`` (stable), as CSV."""
    train = df[df["id"].isin(set(train_ids))]
    train = train.sort_values("id", kind="mergesort").head(N_EXAMPLE_ROWS)
    return train.to_csv(index=False).rstrip("\n")


def build_messages(context_and_goal: str, columns: str, rows: str) -> list[dict[str, str]]:
    user = USER_TEMPLATE.format(context_and_goal=str(context_and_goal).rstrip("\n"),
                                columns=columns, rows=rows)
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


# --------------------------------------------------------------------------- task + folds
def load_task(tasks_root: Path, task_type: str, dataset_id: str):
    import pandas as pd
    import yaml

    d = Path(tasks_root) / task_type / dataset_id
    cfg = yaml.safe_load((d / "config.yml").read_text(encoding="utf-8"))
    df = pd.read_csv(d / "data.csv")
    labels = pd.read_csv(d / "labels.csv") if (d / "labels.csv").exists() else None
    return cfg, df, labels


def outer_folds(df, labels, task_type: str, fold_seed: int, stratified: bool,
                k: int = N_FOLDS) -> list[dict[str, list]]:
    """Outer folds as ``rogue_one_main.py`` builds them (``__init__``, around ``:285-345``)."""
    import pandas as pd

    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from rogueone.utils.train_test_splitter import (  # noqa: WPS433
        compute_entity_duplicate_groups,
        deterministic_kfold_indices,
    )

    try:
        ent = pd.DataFrame(df["id"].unique(), columns=["id"])
    except Exception:  # same fallback as main
        ent = pd.DataFrame({"id": list(df.index)})
    stratify_labels = None
    if stratified and task_type == "classification" and labels is not None:
        try:
            lab = labels.copy()
            if "id" not in lab.columns and "patient_id" in lab.columns:
                lab = lab.rename(columns={"patient_id": "id"})
            ordered = ent[["id"]].merge(lab[["id", "target"]], on="id", how="left")
            if ordered["target"].isna().any():
                raise ValueError("some feature-frame ids have no matching label")
            stratify_labels = ordered["target"].to_numpy()
        except Exception as exc:  # main prints and falls back to KFold
            print(f"[f0] stratified folds requested but labels could not be aligned ({exc}); "
                  "using unstratified KFold", flush=True)
            stratify_labels = None
    groups = compute_entity_duplicate_groups(df, ent["id"].tolist(), exclude_cols=("id",))
    idx = deterministic_kfold_indices(n_rows=len(ent), n_splits=k, random_state=int(fold_seed),
                                      stratify_labels=stratify_labels, dedup_groups=groups)
    ids = ent["id"].tolist()
    return [{"train_ids": [ids[i] for i in tr], "test_ids": [ids[i] for i in te]} for tr, te in idx]


# --------------------------------------------------------------------------- generation
def _usage(resp: Any) -> tuple[Optional[int], Optional[int]]:
    u = getattr(resp, "usage", None)
    if u is None:
        return None, None
    return getattr(u, "prompt_tokens", None), getattr(u, "completion_tokens", None)


def classify_error(exc: BaseException) -> str:
    name = type(exc).__name__.lower()
    status = getattr(exc, "status_code", None)
    msg = str(exc).lower()
    if "timeout" in name or "timed out" in msg:
        return "timeout"
    if "context" in msg and ("length" in msg or "maximum" in msg):
        return "context"
    if "connection" in name or "connect" in msg or (isinstance(status, int) and status >= 500):
        return "infra"
    return "other"


async def draw_one(client: Any, *, model: str, messages: list, seed: int) -> dict[str, Any]:
    t0 = time.monotonic()
    rec: dict[str, Any] = {"seed": seed, "error_kind": None, "error_message": None}
    try:
        resp = await client.chat.completions.create(
            model=model, messages=messages, temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
            seed=seed,
        )
        choice = resp.choices[0]
        text = getattr(choice.message, "content", None)
        pt, ct = _usage(resp)
        rec.update(raw_text=text, finish_reason=getattr(choice, "finish_reason", None),
                   prompt_tokens=pt, completion_tokens=ct)
    except Exception as exc:
        rec.update(raw_text=None, finish_reason=None, prompt_tokens=None, completion_tokens=None,
                   error_kind=classify_error(exc), error_message=f"{type(exc).__name__}: {str(exc)[:500]}")
    rec["latency_s"] = round(time.monotonic() - t0, 3)
    parsed = parse_answer(rec.get("raw_text"))
    rec.update(parse_ok=parsed["ok"], parsed=parsed["items"], n_items=parsed["n_items"],
               fenced=parsed["fenced"], over_limit=parsed["over_limit"], parse_error=parsed["error"])
    return rec


def read_done(path: Path) -> dict[tuple[int, int], dict]:
    done: dict[tuple[int, int], dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[(int(r["fold"]), int(r["draw"]))] = r
    return done


async def generate(client: Any, *, model: str, llm_seed: int, fold_messages: list[list],
                   out: Path, concurrency: int = 16, n_draws: int = N_DRAWS,
                   resume: bool = True, requeue_infra: bool = True) -> list[dict]:
    """All draws of all folds; appends to ``out/draws.jsonl``; returns the final records."""
    out.mkdir(parents=True, exist_ok=True)
    path = out / "draws.jsonl"
    done = read_done(path) if resume else {}
    if not resume and path.exists():
        path.unlink()
    todo = [(f, j) for f in range(len(fold_messages)) for j in range(n_draws) if (f, j) not in done]
    sem = asyncio.Semaphore(max(1, int(concurrency)))
    lock = asyncio.Lock()

    async def one(f: int, j: int, attempt: int) -> dict:
        async with sem:
            rec = await draw_one(client, model=model, messages=fold_messages[f],
                                 seed=seed_for(llm_seed, f, j))
        rec.update(fold=f, draw=j, attempt=attempt)
        async with lock:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")
        return rec

    results = await asyncio.gather(*(one(f, j, 1) for f, j in todo))
    for r in results:
        done[(r["fold"], r["draw"])] = r
    if requeue_infra:
        again = [k for k, r in done.items() if r.get("error_kind") == "infra" and r.get("attempt", 1) == 1]
        for r in await asyncio.gather(*(one(f, j, 2) for f, j in again)):
            done[(r["fold"], r["draw"])] = r
    return [done[k] for k in sorted(done)]


def summarise(records: list[dict], *, n_expected: int, wall_s: float) -> dict[str, Any]:
    n = len(records)
    ok = sum(1 for r in records if r.get("parse_ok"))
    err = sum(1 for r in records if r.get("error_kind"))
    ct = sum(int(r.get("completion_tokens") or 0) for r in records)
    pt = sum(int(r.get("prompt_tokens") or 0) for r in records)
    per_fold: dict[str, Any] = {}
    for f in sorted({int(r["fold"]) for r in records}):
        rr = [r for r in records if int(r["fold"]) == f]
        per_fold[f"fold{f}_parse_rate"] = round(sum(1 for r in rr if r.get("parse_ok")) / len(rr), 4)
        per_fold[f"fold{f}_tok_compl"] = sum(int(r.get("completion_tokens") or 0) for r in rr)
    return {
        "n_draws": n, "n_expected": n_expected, "complete": n == n_expected,
        "f0_parse_rate": round(ok / n, 4) if n else 0.0,
        "n_parse_ok": ok, "n_errors": err,
        "n_error_infra": sum(1 for r in records if r.get("error_kind") == "infra"),
        "n_error_context": sum(1 for r in records if r.get("error_kind") == "context"),
        "n_error_timeout": sum(1 for r in records if r.get("error_kind") == "timeout"),
        "fenced_share": round(sum(1 for r in records if r.get("fenced")) / n, 4) if n else 0.0,
        "over_limit_share": round(sum(1 for r in records if r.get("over_limit")) / n, 4) if n else 0.0,
        "finish_length_share": round(sum(1 for r in records if r.get("finish_reason") == "length") / n, 4) if n else 0.0,
        "tok_compl_total": ct, "tok_prompt_total": pt,
        "tok_compl_mean_per_draw": round(ct / n, 1) if n else 0.0,
        "wall_s": round(wall_s, 1), **per_fold,
    }


# --------------------------------------------------------------------------- CLI
def _import_clients(model: str, port: str, api_key: str):
    os.environ.setdefault("LLM_MODEL", model)
    os.environ.setdefault("LLM_PORT", str(port))
    os.environ.setdefault("LLM_API_KEY", api_key)
    for k in ("EMBEDD_MODEL", "EMBEDD_PORT", "EMBEDD_API_KEY"):
        os.environ.setdefault(k, "unused")
    os.environ["ROGUE_ONE_LLM_TRACE"] = "0"
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from rogueone.llm import clients  # noqa: WPS433

    return clients


def run_name_for(row: dict[str, str]) -> str:
    return f"A17__{row['arm_id']}__{row['dataset_id']}__f{row['fold_seed']}__r{row['replicate']}"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", required=True, help="A17_F0.csv (R7 columns)")
    ap.add_argument("--row", required=True, help="one F0 row id, e.g. A17-164")
    ap.add_argument("--tasks", required=True, help="tasks root (classification/, regression/)")
    ap.add_argument("--model", default=os.getenv("LLM_MODEL"))
    ap.add_argument("--port", default=os.getenv("LLM_PORT"))
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--n-draws", type=int, default=N_DRAWS, help="testing only; frozen at 128")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--entity", default=os.getenv("WANDB_ENTITY") or DEFAULT_ENTITY)
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--session-id", default=os.getenv("SESSION_ID"))
    ap.add_argument("--image-tag", default=os.getenv("ROGUE_ONE_IMAGE_TAG"))
    ap.add_argument("--git-sha", default=os.getenv("ROGUE_ONE_GIT_SHA"))
    ap.add_argument("--code-version", default=os.getenv("F0_CODE_VERSION"))
    ap.add_argument("--dry-run", action="store_true", help="build folds and prompts, send nothing")
    args = ap.parse_args(argv)
    if not args.dry_run and (not args.model or not args.port):
        ap.error("--model and --port are required (or LLM_MODEL / LLM_PORT)")
    return args


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    row = load_row(Path(args.manifest), args.row)
    if args.n_draws != N_DRAWS:
        print(f"[f0] WARNING: --n-draws {args.n_draws} != frozen {N_DRAWS} (test pool)", flush=True)
    llm_seed = int(row["llm_seed"])
    fold_seed = int(row["fold_seed"])
    stratified = str(env_overrides(row).get("ROGUE_ONE_STRATIFIED_FOLDS", "")).lower() == "true"
    cfg, df, labels = load_task(Path(args.tasks), row["task_type"], row["dataset_id"])
    folds = outer_folds(df, labels, row["task_type"], fold_seed, stratified)
    columns = render_columns(df)
    fold_messages = [build_messages(cfg["context_and_goal"], columns, render_rows(df, f["train_ids"]))
                     for f in folds]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "folds.json").write_text(json.dumps(
        {"row_id": args.row, "dataset_id": row["dataset_id"], "fold_seed": fold_seed,
         "stratified": stratified, "test_ids": [f["test_ids"] for f in folds]}, default=str),
        encoding="utf-8")
    with open(out / "prompts.jsonl", "w", encoding="utf-8") as fh:
        for i, msgs in enumerate(fold_messages):
            fh.write(json.dumps({"fold": i, "messages": msgs}) + "\n")
    run_name = run_name_for(row)
    print(f"[f0] {run_name}: row={args.row} rung={row['rung_S']} model={args.model} port={args.port} "
          f"folds={len(folds)} draws/fold={args.n_draws} stratified={stratified} "
          f"seeds {seed_for(llm_seed, 0, 0)}..{seed_for(llm_seed, len(folds) - 1, args.n_draws - 1)}",
          flush=True)
    if args.dry_run:
        print(f"[f0] DRY_RUN: folds and prompts written to {out}; no request sent", flush=True)
        return 0

    api_key = os.getenv("LLM_API_KEY") or "EMPTY"
    clients = _import_clients(args.model, args.port, api_key)
    client = clients.create_openai_client_for_vllm(
        base_url=f"http://{args.host}:{args.port}/v1", api_key=api_key, agent_name="F0Generator",
        timeout=args.timeout, max_retries=0,
    )
    t0 = time.monotonic()
    records = asyncio.run(generate(client, model=args.model, llm_seed=llm_seed,
                                   fold_messages=fold_messages, out=out,
                                   concurrency=args.concurrency, n_draws=args.n_draws,
                                   resume=not args.no_resume))
    wall = time.monotonic() - t0
    summary = summarise(records, n_expected=len(folds) * args.n_draws, wall_s=wall)
    summary.update(row_id=args.row, run_name=run_name, model=args.model)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str), flush=True)

    if args.wandb:
        import wandb  # noqa: WPS433

        rung = row["rung_S"]
        run = wandb.init(entity=args.entity, project=args.project, group="A17",
                         job_type=row["arm_id"], name=run_name,
                         tags=[t for t in (args.session_id, args.image_tag, "v2", "f0pool", args.row) if t])
        run.config.update({
            "ablation_id": "A17", "arm_id": row["arm_id"], "dataset_id": row["dataset_id"],
            "task_type": row["task_type"], "fold_seed": fold_seed, "llm_seed": llm_seed,
            "replicate": int(row["replicate"]), "is_sentinel": 0, "manifest_row": args.row,
            "session_id": args.session_id, "code_phase": "v2", "image_tag": args.image_tag,
            "git_sha": args.git_sha, "f0_code_version": args.code_version,
            "rung_S": rung, "rung_E": rung, "rung_T": rung,
            "model": args.model, "model_S": args.model, "model_E": args.model, "model_T": args.model,
            "model_W": args.model, "port": args.port, "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS, "n_folds": len(folds), "n_draws_per_fold": args.n_draws,
            "concurrency": args.concurrency, "stratified_folds": stratified,
            "seed_rule": "llm_seed*1000 + 128*f + j", "flags/ROGUE_ONE_LLM_SEED": str(llm_seed),
            "f0_prompt": "docs/agent2027/X7_F0_prompt_DRAFT.md (Amendment 6)",
            "generator": "scripts/agent2027/f0_generate.py",
        }, allow_val_change=True)
        run.summary.update({k: v for k, v in summary.items() if not isinstance(v, (list, dict))})
        art = wandb.Artifact(f"f0-pool-{rung}-{row['dataset_id']}-f{fold_seed}".replace(".", "p"),
                             type="f0-pool",
                             metadata={k: summary[k] for k in ("n_draws", "f0_parse_rate",
                                                               "tok_compl_total", "complete")})
        for name in ("draws.jsonl", "folds.json", "prompts.jsonl", "summary.json"):
            art.add_file(str(out / name))
        run.log_artifact(art, aliases=["latest", args.row])
        run.finish()
    return 0 if summary["complete"] else 2


if __name__ == "__main__":
    sys.exit(main())
