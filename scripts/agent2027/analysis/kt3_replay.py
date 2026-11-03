"""E8b / A14-T4 / X9: offline replay of logged P7 search queries through the KT3 TF-IDF index.

Every logged ``_web_search`` tool call (one ``event="query"`` row with
``variant_index`` 0) is replayed exactly as ``web_search._local_search_impl`` would serve
it under ``ROGUE_ONE_KNOWLEDGE_BACKEND=local``: the same ``_sanitize_query`` and query-
variant loop, the first variant with a positive-score hit wins, top 3 hits. For each call
it records the rank of the P7 document (``marenko_index.md``) in ``kt3_correct`` and in
``kt3_wrong``.

A14-T4 criteria (proposed, X10 V.14):
  (1) the P7 document is in the top 3 for >= 50% of the queries that contain the domain
      term (``Marenko``), in ``kt3_correct``;
  (2) the P7 document has the same rank (or is absent) in both corpora for >= 90% of all
      queries.

Input: either retrieval transcripts (``retrieval_transcript.jsonl``; ``--transcript
ROW_ID=PATH``) or the frozen query list ``docs/agent2027/results/E8b_a02_p7_queries.jsonl``
(``--queries``; written with ``--write-queries`` from transcripts). CPU only; developer
machine or inside a ``rogue_one`` container, never host Python on CAIR.

    python scripts/agent2027/analysis/kt3_replay.py \
        --queries docs/agent2027/results/E8b_a02_p7_queries.jsonl \
        --corpora scripts/agent2027/corpora --json out.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# web_search imports the global config, which needs these keys; the replay makes no
# LLM call, so placeholders are enough (never overrides a real value).
for _k, _v in {"EMBEDD_API_KEY": "x", "EMBEDD_MODEL": "x", "EMBEDD_PORT": "1", "LLM_MODEL": "x",
               "LLM_API_KEY": "x", "LLM_PORT": "8010"}.items():
    os.environ.setdefault(_k, _v)

from rogueone.llm.agents.core.knowledge import local_corpus  # noqa: E402
from rogueone.llm.agents.core.knowledge.web_search import _query_variants, _sanitize_query  # noqa: E402

P7_DOC = "marenko_index.md"
DOMAIN_TERM = "marenko"
CORPORA = ("kt3_correct", "kt3_wrong")
TOP3_MIN = 0.50
SAME_RANK_MIN = 0.90


def read_transcript_queries(path: Path, row_id: str) -> list[dict[str, Any]]:
    """One record per tool call: ``event="query"`` rows with ``variant_index`` 0."""
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("event") != "query" or int(r.get("variant_index", 0) or 0) != 0:
                continue
            out.append({"row_id": row_id, "seq": r.get("seq"), "query": r.get("query") or ""})
    return out


def read_queries(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def serve(corpus: local_corpus.LocalCorpus, query: str) -> tuple[Optional[int], list[str], Optional[int]]:
    """``(variant_index used, doc_ids returned, rank of P7 doc 1..3 or None)`` as
    ``_local_search_impl`` would serve ``query``."""
    sanitized = _sanitize_query(query)
    if not sanitized:
        return None, [], None
    for i, variant in enumerate(_query_variants(sanitized)):
        hits = [h for h in corpus.search(variant) if h[1] > 0]
        if hits:
            ids = [h[0] for h in hits]
            return i, ids, (ids.index(P7_DOC) + 1 if P7_DOC in ids else None)
    return None, [], None


def replay(queries: Iterable[dict[str, Any]], corpora: dict[str, local_corpus.LocalCorpus]) -> dict[str, Any]:
    rows = []
    for q in queries:
        rec = {"row_id": q.get("row_id"), "seq": q.get("seq"), "query": q["query"],
               "has_term": DOMAIN_TERM in q["query"].lower()}
        for name, c in corpora.items():
            v, ids, rank = serve(c, q["query"])
            rec[f"{name}_variant"] = v
            rec[f"{name}_doc_ids"] = ids
            rec[f"{name}_p7_rank"] = rank
        rows.append(rec)
    return {"rows": rows, "summary": summarise(rows, list(corpora))}


def _frac(n: int, d: int) -> Optional[float]:
    return None if d == 0 else n / d


def summarise(rows: list[dict[str, Any]], names: list[str]) -> dict[str, Any]:
    term = [r for r in rows if r["has_term"]]
    s: dict[str, Any] = {"n_calls": len(rows), "n_unique_queries": len({r["query"] for r in rows}),
                         "n_calls_with_term": len(term)}
    by_row = Counter(r["row_id"] for r in rows)
    s["n_calls_by_row"] = dict(sorted(by_row.items()))
    s["n_calls_with_term_by_row"] = dict(sorted(Counter(r["row_id"] for r in term).items()))
    for name in names:
        rk = f"{name}_p7_rank"
        s[name] = {
            "term_top1": sum(r[rk] == 1 for r in term),
            "term_top3": sum(r[rk] is not None for r in term),
            "term_top3_frac": _frac(sum(r[rk] is not None for r in term), len(term)),
            "all_top1": sum(r[rk] == 1 for r in rows),
            "all_top3": sum(r[rk] is not None for r in rows),
            "no_term_top3": sum(r[rk] is not None for r in rows if not r["has_term"]),
            "no_results": sum(r[f"{name}_variant"] is None for r in rows),
            "fallback_variant_used": sum((r[f"{name}_variant"] or 0) > 0 for r in rows),
        }
    if len(names) == 2:
        a, b = (f"{n}_p7_rank" for n in names)
        same = sum(r[a] == r[b] for r in rows)
        s["same_rank_both"] = same
        s["same_rank_both_frac"] = _frac(same, len(rows))
        s["same_doc_ids_both"] = sum(r[f"{names[0]}_doc_ids"] == r[f"{names[1]}_doc_ids"] for r in rows)
    first = names[0]
    s["criterion_top3_term"] = {"threshold": TOP3_MIN, "value": s[first]["term_top3_frac"],
                                "pass": (s[first]["term_top3_frac"] or 0) >= TOP3_MIN}
    if "same_rank_both_frac" in s:
        s["criterion_same_rank"] = {"threshold": SAME_RANK_MIN, "value": s["same_rank_both_frac"],
                                    "pass": (s["same_rank_both_frac"] or 0) >= SAME_RANK_MIN}
    return s


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--transcript", nargs="+", metavar="ROW_ID=PATH")
    src.add_argument("--queries", type=Path)
    ap.add_argument("--write-queries", type=Path, help="write the extracted query list (jsonl) and continue")
    ap.add_argument("--corpora", type=Path, default=REPO / "scripts" / "agent2027" / "corpora")
    ap.add_argument("--json", type=Path, help="write rows + summary here")
    a = ap.parse_args(argv)

    if a.transcript:
        queries = []
        for spec in a.transcript:
            row_id, _, path = spec.partition("=")
            queries.extend(read_transcript_queries(Path(path), row_id))
    else:
        queries = read_queries(a.queries)
    if a.write_queries:
        with open(a.write_queries, "w", encoding="utf-8", newline="\n") as fh:
            for q in queries:
                fh.write(json.dumps({"row_id": q["row_id"], "seq": q["seq"], "query": q["query"]},
                                    ensure_ascii=False) + "\n")

    corpora = {n: local_corpus.load_corpus(a.corpora / n) for n in CORPORA}
    res = replay(queries, corpora)
    res["corpus_sha256"] = {n: c.sha256 for n, c in corpora.items()}
    if a.json:
        a.json.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"corpus_sha256": res["corpus_sha256"], **res["summary"]}, indent=1))
    ok = res["summary"]["criterion_top3_term"]["pass"] and res["summary"].get("criterion_same_rank", {}).get("pass", True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
