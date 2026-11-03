"""A04-T5: freeze the Stage-0 prompt set (``stage0-prompts``) from E3 traces.

Samples ``--n`` (50) Extractor turns, RNG seed ``--seed`` (42), in which O120 emitted
a structured tool call, first from the primary sessions (the G0a smoke traces) and,
only if fewer than ``--n`` eligible turns exist there, tops up from the secondary
sessions (the A02 traces). Each output record holds the request (messages, tool
schemas, temperature, max_tokens, extra_body), the recorded O120 response, its
provenance and ``max_prompt_tokens``; the ``--kv-probe`` (6) longest are flagged
``kv_probe=true``.

Eligible turn: ``agent_name == "ExtractorAgent"``, no ``error``, the final message
carries >= 1 structured ``tool_calls``, no text rescue (``rescue`` is null), and the
response came from ``openai/gpt-oss-120b``. Only the final attempt directory of a
row is read (``<row>.attempt*`` folders of failed attempts are skipped).

Candidates are ordered by (session, row, trace seq) before sampling, so the draw is a
pure function of the trace files and the seed.

``stage0-prompts:v0`` (2026-10-14) was made with ``--primary S1012a S1012b``: the
first G0a smoke session ``S1012a`` (image pre.1) wrote no ``llm_traces.jsonl.gz``, so
all 50 turns come from its rerun ``S1012b`` (same rows A02-901/902, image pre.2; 230
eligible turns), and the A02 top-up (``S1013c``/``S1013d``) was not needed.

Run on CAIR (CPU-only container; the W&B key comes from the env file)::

    docker run --rm --cpus 4 --env-file /raid/home/henrbr16/code/PlanBench/.env \\
      -v <dir with this script>:/x2:ro \\
      -v $HOME/rogueone-runs-agent2027:/runs:ro -v $HOME/ro27_a04_prompts:/out \\
      --entrypoint python3 rogue_one:agent2027-pre.3 /x2/stage0_prompts.py \\
        --runs-root /runs --primary S1012a S1012b --secondary S1013c S1013d \\
        --out /out/stage0-prompts-v0.jsonl --wandb
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Optional

O120 = "openai/gpt-oss-120b"
ROW_DIR_RE = re.compile(r"^[A-Z][A-Z0-9]*-\d+$")  # A02-001, never A02-006.attempt1.<epoch>
REQUEST_FIELDS = ("messages", "tools", "tool_choice", "parallel_tool_calls", "temperature",
                  "max_tokens", "top_p", "extra_body", "model", "seed")


def _is_placeholder(v: Any) -> bool:
    return isinstance(v, str) and v.startswith("<openai.")


def trace_files(runs_root: Path, session: str) -> list[tuple[str, Path]]:
    sdir = runs_root / session
    out = []
    for row in sorted(p for p in sdir.iterdir() if p.is_dir() and ROW_DIR_RE.match(p.name)):
        for f in sorted(row.glob("tasks/*/*/output/llm_traces.jsonl.gz")):
            out.append((row.name, f))
    return out


def max_prompt_tokens(rec: dict) -> int:
    usage = rec.get("usage") or {}
    return max([int(usage.get("prompt_tokens") or 0)]
               + [int(a.get("prompt_tokens") or 0) for a in rec.get("attempts") or []])


def is_eligible(rec: dict, model: str = O120) -> bool:
    if rec.get("agent_name") != "ExtractorAgent" or rec.get("error"):
        return False
    msg = rec.get("message") or {}
    if not msg.get("tool_calls"):
        return False
    if rec.get("rescue"):
        return False
    return (rec.get("response_model") or rec.get("model")) == model


def candidates(runs_root: Path, sessions: list[str], group: str, model: str = O120) -> tuple[list[dict], dict]:
    cands: list[dict] = []
    stats: dict[str, Any] = {}
    for sid in sessions:
        for row, f in trace_files(runs_root, sid):
            n_ext = n_ok = 0
            with gzip.open(f, "rt", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    if rec.get("agent_name") == "ExtractorAgent":
                        n_ext += 1
                    if not is_eligible(rec, model):
                        continue
                    n_ok += 1
                    cands.append(make_record(rec, sid, row, f, runs_root, group))
            stats[f"{sid}/{row}"] = {"extractor_calls": n_ext, "eligible": n_ok,
                                     "dataset": f.parts[-3], "trace": str(f.relative_to(runs_root))}
    cands.sort(key=lambda r: (r["provenance"]["session_id"], r["provenance"]["row_id"],
                              r["provenance"]["seq"]))
    return cands, stats


def make_record(rec: dict, sid: str, row: str, f: Path, runs_root: Path, group: str) -> dict:
    req = {k: rec.get(k) for k in REQUEST_FIELDS if rec.get(k) is not None and not _is_placeholder(rec.get(k))}
    return {
        "prompt_id": None,
        "role": rec.get("role"),
        "agent_name": rec.get("agent_name"),
        "stage": rec.get("stage"),
        "request": req,
        "recorded_response": {
            "message": rec.get("message"),
            "finish_reason": rec.get("finish_reason"),
            "usage": rec.get("usage"),
            "attempts": rec.get("attempts"),
            "rerolls": rec.get("rerolls"),
            "rescue": rec.get("rescue"),
            "response_model": rec.get("response_model"),
        },
        "max_prompt_tokens": max_prompt_tokens(rec),
        "kv_probe": False,
        "provenance": {
            "source_group": group,
            "session_id": sid,
            "row_id": row,
            "dataset_id": f.parts[-3],
            "trace_path": str(f.relative_to(runs_root)),
            "seq": rec.get("seq"),
            "call_seq": rec.get("call_seq"),
            "session_name": rec.get("session_name"),
            "index": rec.get("index"),
            "fold_run_id": rec.get("fold_run_id"),
            "iteration_in_fold": rec.get("iteration_in_fold"),
            "ts_start": rec.get("ts_start"),
        },
    }


def sample(primary: list[dict], secondary: list[dict], n: int, seed: int, kv_probe: int) -> list[dict]:
    rng = random.Random(seed)
    if len(primary) >= n:
        chosen = rng.sample(primary, n)
    else:
        need = n - len(primary)
        if len(secondary) < need:
            raise SystemExit(f"only {len(primary) + len(secondary)} eligible turns, need {n}")
        chosen = list(primary) + rng.sample(secondary, need)
    chosen = sorted(chosen, key=lambda r: (r["provenance"]["source_group"] != "G0a",
                                           r["provenance"]["session_id"], r["provenance"]["row_id"],
                                           r["provenance"]["seq"]))
    for i, r in enumerate(chosen):
        r["prompt_id"] = f"s0-{i:03d}"
    for r in sorted(chosen, key=lambda r: (-r["max_prompt_tokens"], r["prompt_id"]))[:kv_probe]:
        r["kv_probe"] = True
    return chosen


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs-root", default="/runs")
    ap.add_argument("--primary", nargs="+", required=True, help="G0a smoke session id(s)")
    ap.add_argument("--secondary", nargs="*", default=[], help="top-up session id(s) (A02)")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--kv-probe", type=int, default=6)
    ap.add_argument("--model", default=O120)
    ap.add_argument("--out", required=True)
    ap.add_argument("--note", default="")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--entity", default="PlanBench")
    ap.add_argument("--project", default="RogueOne-AGENT2027-prefreeze")
    ap.add_argument("--artifact", default="stage0-prompts")
    ap.add_argument("--run-name", default="A04__prompt_freeze__stage0prompts__f42__r0")
    ap.add_argument("--image-tag", default="agent2027-pre.3")
    args = ap.parse_args(argv)

    root = Path(args.runs_root)
    prim, prim_stats = candidates(root, args.primary, "G0a", args.model)
    sec, sec_stats = candidates(root, args.secondary, "A02", args.model) if len(prim) < args.n else ([], {})
    chosen = sample(prim, sec, args.n, args.seed, args.kv_probe)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for r in chosen:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_g0a = sum(1 for r in chosen if r["provenance"]["source_group"] == "G0a")
    tokens = [r["max_prompt_tokens"] for r in chosen]
    meta = {
        "n": len(chosen), "seed": args.seed, "model": args.model,
        "primary_sessions": args.primary, "secondary_sessions": args.secondary,
        "eligible_primary": len(prim), "eligible_secondary": len(sec) if sec else None,
        "n_from_G0a": n_g0a, "n_from_A02": len(chosen) - n_g0a, "topped_up": n_g0a < len(chosen),
        "kv_probe_ids": [r["prompt_id"] for r in chosen if r["kv_probe"]],
        "kv_probe_max_prompt_tokens": sorted((r["max_prompt_tokens"] for r in chosen if r["kv_probe"]), reverse=True),
        "max_prompt_tokens_max": max(tokens), "max_prompt_tokens_min": min(tokens),
        "max_prompt_tokens_median": sorted(tokens)[len(tokens) // 2],
        "temperature": 0.7, "eligibility": "ExtractorAgent, no error, >=1 structured tool call, "
        "no text rescue, response_model == model",
        "per_trace": {**prim_stats, **sec_stats}, "note": args.note,
        "tool_names": [t["function"]["name"] for t in chosen[0]["request"].get("tools", [])],
        "sampler": "scripts/agent2027/stage0_prompts.py",
    }
    meta_path = out.with_name(out.stem + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k != "per_trace"}, indent=2))
    for r in chosen:
        p = r["provenance"]
        print(f"{r['prompt_id']}\t{p['source_group']}\t{p['session_id']}/{p['row_id']}\tseq={p['seq']}\t"
              f"it={p['iteration_in_fold']}\ttok={r['max_prompt_tokens']}\tkv={int(r['kv_probe'])}")

    if args.wandb:
        import wandb

        run = wandb.init(entity=args.entity, project=args.project, group="A04", job_type="prompt_freeze",
                         name=args.run_name, tags=["prompt-freeze", args.image_tag],
                         config={k: v for k, v in meta.items() if k != "per_trace"})
        art = wandb.Artifact(args.artifact, type="prompt-set",
                             description="A04-T5 frozen Stage-0 Extractor prompts (50, seed 42)",
                             metadata=meta)
        art.add_file(str(out))
        art.add_file(str(meta_path))
        logged = run.log_artifact(art)
        logged.wait()
        print(f"[stage0_prompts] logged {args.entity}/{args.project}/{args.artifact}:{logged.version}")
        run.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
