#!/usr/bin/env python3
"""Rendered-prompt diff and leak scan (AGENT 2027 E5).

Compare two prompt dumps written with ``ROGUE_ONE_PROMPT_DUMP=1``
(``rogueone/utils/prompt_dump.py``), byte for byte, per role / phase /
iteration::

    python scripts/agent2027/prompt_diff.py <dumpA> <dumpB> [--max-lines N]

Exit status 0 = byte-identical, 1 = differences, 2 = usage / input error.
A dump argument may be the ``prompts`` directory itself or its parent (for
example a fetched ``run-output`` artifact containing ``prompts/``).

Leak / hint scan on one dump (E5 done-when 1; A15 leak check)::

    python scripts/agent2027/prompt_diff.py --scan <dump> \
        [--term WORD ...] [--allow-text FILE ...]

``--term`` defaults to the E5 list (torque, CAR, buying, maint, safety, winner,
minimax). Matching is case-insensitive on whole words. Text from each
``--allow-text`` file (e.g. the task's own ``context_and_goal``) is removed
from the prompts before scanning. Exit 1 if any term is found. Extra terms
(P6b names, experiment name, dataset_id) can be passed with ``--term``.
"""
from __future__ import annotations

import argparse
import difflib
import re
import sys
from pathlib import Path

DEFAULT_TERMS = ["torque", "CAR", "buying", "maint", "safety", "winner", "minimax"]
SUFFIXES = (".system.txt", ".user.txt", ".tools.json")
NAME_RE = re.compile(
    r"^(?P<role>.+?)__(?P<phase>.+?)__iter(?P<iter>\d+)__c(?P<seq>\d+)\.(?P<kind>system\.txt|user\.txt|tools\.json)$"
)


def resolve_dump(path: str | Path) -> Path:
    p = Path(path)
    if (p / "prompts").is_dir():
        p = p / "prompts"
    if not p.is_dir():
        raise FileNotFoundError(f"not a directory: {path}")
    return p


def load_dump(path: str | Path) -> dict[str, bytes]:
    root = resolve_dump(path)
    return {
        f.name: f.read_bytes()
        for f in sorted(root.iterdir())
        if f.is_file() and f.name.endswith(SUFFIXES)
    }


def describe(name: str) -> str:
    m = NAME_RE.match(name)
    if not m:
        return name
    return f"role={m['role']} phase={m['phase']} iter={int(m['iter'])} call={int(m['seq'])} part={m['kind']}"


def diff_dumps(a: dict[str, bytes], b: dict[str, bytes], max_lines: int = 20) -> tuple[bool, list[str]]:
    """Return (identical, report_lines)."""
    lines: list[str] = []
    only_a = sorted(set(a) - set(b))
    only_b = sorted(set(b) - set(a))
    changed = sorted(n for n in set(a) & set(b) if a[n] != b[n])
    for n in only_a:
        lines.append(f"ONLY IN A: {describe(n)}")
    for n in only_b:
        lines.append(f"ONLY IN B: {describe(n)}")
    for n in changed:
        da, db = a[n], b[n]
        first = next((i for i in range(min(len(da), len(db))) if da[i] != db[i]), min(len(da), len(db)))
        lines.append(f"CHANGED: {describe(n)} (first differing byte {first}; {len(da)} -> {len(db)} bytes)")
        ta = da.decode("utf-8", "replace").splitlines()
        tb = db.decode("utf-8", "replace").splitlines()
        shown = 0
        for ln in difflib.unified_diff(ta, tb, "A", "B", lineterm="", n=0):
            if shown >= max_lines:
                lines.append("    ... (truncated)")
                break
            lines.append("    " + ln[:300])
            shown += 1
    identical = not (only_a or only_b or changed)
    lines.append(
        f"{'IDENTICAL' if identical else 'DIFFERENT'}: {len(a)} files in A, {len(b)} in B, "
        f"{len(changed)} changed, {len(only_a)} only in A, {len(only_b)} only in B"
    )
    return identical, lines


def scan_dump(dump: dict[str, bytes], terms: list[str], allow_texts: list[str]) -> list[str]:
    hits: list[str] = []
    patterns = [(t, re.compile(r"(?<!\w)" + re.escape(t) + r"(?!\w)", re.IGNORECASE)) for t in terms]
    for name in sorted(dump):
        text = dump[name].decode("utf-8", "replace")
        for allowed in allow_texts:
            if allowed.strip():
                text = text.replace(allowed.strip(), "")
        for term, pat in patterns:
            n = len(pat.findall(text))
            if n:
                hits.append(f"LEAK: '{term}' x{n} in {describe(name)}")
    return hits


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dumps", nargs="*", help="dumpA dumpB (diff mode)")
    ap.add_argument("--max-lines", type=int, default=20, help="diff lines shown per changed file")
    ap.add_argument("--scan", metavar="DUMP", help="scan one dump for forbidden terms")
    ap.add_argument("--term", action="append", help="term to scan for (repeatable; default: E5 list)")
    ap.add_argument("--allow-text", action="append", default=[], metavar="FILE",
                    help="file whose text is exempt (e.g. the task's context_and_goal)")
    args = ap.parse_args(argv)
    try:
        if args.scan:
            allow = [Path(f).read_text(encoding="utf-8") for f in args.allow_text]
            dump = load_dump(args.scan)
            if not dump:
                print("no prompt files found", file=sys.stderr)
                return 2
            hits = scan_dump(dump, args.term or DEFAULT_TERMS, allow)
            print("\n".join(hits) if hits else f"CLEAN: {len(dump)} files scanned")
            return 1 if hits else 0
        if len(args.dumps) != 2:
            ap.error("need exactly two dump directories (or use --scan)")
        a, b = load_dump(args.dumps[0]), load_dump(args.dumps[1])
        if not a and not b:
            print("both dumps are empty", file=sys.stderr)
            return 2
        identical, report = diff_dumps(a, b, args.max_lines)
        print("\n".join(report))
        return 0 if identical else 1
    except (OSError, UnicodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
