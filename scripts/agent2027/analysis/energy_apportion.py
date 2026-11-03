"""Apportion session GPU energy to runs by token share (AGENT 2027 E3 / A23 section 2.3).

Inputs
------
* the sampler CSV of the session (``scripts/agent2027/energy_sampler.py``);
* a GPU -> server map (from the session launch line);
* every run of the session: alive interval (W&B start, start + ``_runtime``) and the
  servers it uses;
* every LLM call of those runs with its server, completion tokens (re-rolls
  included) and, when the trace has them, ``[ts_start, ts_end]``.

Method (A23 2.3)
----------------
1. Integrate each server's power (sum over its GPUs, trapezoid rule) into ``bin_s``
   (60 s) bins aligned to the first sample.
2. Spread each call's completion tokens over its ``[ts_start, ts_end]`` (uniformly) or,
   without ``ts_start``, put them in the bin holding ``ts_end``. Split each bin's server
   energy over runs in proportion to their tokens on that server in that bin.
3. A bin with no tokens on a server is split equally among the runs alive in it that use
   that server.
4. A bin with no run alive is session overhead: reported, not charged.
5. A mixed run (A19) sums its shares over its servers (it simply appears under several
   servers).
6. ``equal_split=True`` is the sensitivity: every bin split equally among alive runs.
7. Coverage of a run = share of its alive seconds in which every GPU of its server(s)
   has a power sample within ``coverage_tol_s`` (2 s). Energy is valid for a run only
   at coverage >= 0.95.

``apportion_energy`` is the single implementation; ``process_metrics.py`` (E11b) imports
it so that every analysis uses the same ``energy_j``. Standard library only.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

COVERAGE_MIN = 0.95


@dataclass
class Run:
    name: str
    t_start: float
    t_end: float
    servers: tuple[str, ...]


@dataclass
class Call:
    run: str
    server: str
    t_end: float
    compl_tokens: float
    t_start: Optional[float] = None


@dataclass
class Result:
    run_energy_j: dict[str, float]
    overhead_j: float
    session_energy_j: float
    coverage: dict[str, float]
    valid: dict[str, bool]
    residual_j: float
    bin_s: float
    equal_split: bool
    per_server_j: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return self.__dict__.copy()


# --------------------------------------------------------------------------- parsing
def read_power_csv(path: str | Path) -> dict[int, list[tuple[float, float]]]:
    """``t_epoch,gpu_index,power_w[,uuid]`` -> ``{gpu: [(t, watts), ...]}`` sorted by time."""
    out: dict[int, list[tuple[float, float]]] = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                out[int(row["gpu_index"])].append((float(row["t_epoch"]), float(row["power_w"])))
            except (KeyError, ValueError):
                continue
    return {g: sorted(s) for g, s in out.items()}


def calls_from_trace(path: str | Path, run: str, role_server: dict[str, str]) -> list[Call]:
    """Calls of one run from its ``llm_traces.jsonl.gz``.

    The record's charged ``role`` picks the server (``role_server``), so web-search calls
    follow their caller's server (E7). Tokens = final completion tokens plus discarded
    re-rolls.
    """
    calls: list[Call] = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            server = role_server.get(rec.get("role"))
            if server is None:
                continue
            usage = rec.get("usage") or {}
            disc = rec.get("discarded_usage") or {}
            tokens = float(usage.get("completion_tokens") or 0) + float(disc.get("completion_tokens") or 0)
            calls.append(
                Call(run=run, server=server, t_end=float(rec["ts_end"]), compl_tokens=tokens,
                     t_start=float(rec["ts_start"]) if rec.get("ts_start") is not None else None)
            )
    return calls


# --------------------------------------------------------------------------- integration
def _bin_range(t: float, t0: float, bin_s: float) -> int:
    return int(math.floor((t - t0) / bin_s))


def gpu_bin_energy(samples: list[tuple[float, float]], t0: float, bin_s: float, nbins: int) -> list[float]:
    """Trapezoid energy (J) of one GPU per bin; segments crossing a bin edge are split
    with linearly interpolated power."""
    out = [0.0] * nbins
    for (ta, pa), (tb, pb) in zip(samples, samples[1:]):
        if tb <= ta:
            continue
        k0, k1 = _bin_range(ta, t0, bin_s), _bin_range(tb, t0, bin_s)
        for k in range(max(k0, 0), min(k1, nbins - 1) + 1):
            lo = max(ta, t0 + k * bin_s)
            hi = min(tb, t0 + (k + 1) * bin_s)
            if hi <= lo:
                continue
            p_lo = pa + (pb - pa) * (lo - ta) / (tb - ta)
            p_hi = pa + (pb - pa) * (hi - ta) / (tb - ta)
            out[k] += 0.5 * (p_lo + p_hi) * (hi - lo)
    return out


def server_bin_energy(
    power: dict[int, list[tuple[float, float]]],
    gpu_to_server: dict[int, str],
    t0: float,
    bin_s: float,
    nbins: int,
) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for gpu, samples in power.items():
        server = gpu_to_server.get(gpu)
        if server is None:
            continue
        e = gpu_bin_energy(samples, t0, bin_s, nbins)
        acc = out.setdefault(server, [0.0] * nbins)
        for k, v in enumerate(e):
            acc[k] += v
    return out


def spread_tokens(call: Call, t0: float, bin_s: float, nbins: int) -> dict[int, float]:
    """Completion tokens of a call per bin (uniform over its interval, else the end bin)."""
    if call.t_start is not None and call.t_end > call.t_start:
        dur = call.t_end - call.t_start
        out: dict[int, float] = {}
        for k in range(max(_bin_range(call.t_start, t0, bin_s), 0), min(_bin_range(call.t_end, t0, bin_s), nbins - 1) + 1):
            lo = max(call.t_start, t0 + k * bin_s)
            hi = min(call.t_end, t0 + (k + 1) * bin_s)
            if hi > lo:
                out[k] = out.get(k, 0.0) + call.compl_tokens * (hi - lo) / dur
        return out
    k = _bin_range(call.t_end, t0, bin_s)
    return {k: call.compl_tokens} if 0 <= k < nbins else {}


# --------------------------------------------------------------------------- coverage
def _merge(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _intersect(x: list[tuple[float, float]], y: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out, i, j = [], 0, 0
    while i < len(x) and j < len(y):
        lo, hi = max(x[i][0], y[j][0]), min(x[i][1], y[j][1])
        if hi > lo:
            out.append((lo, hi))
        if x[i][1] < y[j][1]:
            i += 1
        else:
            j += 1
    return out


def run_coverage(
    run: Run,
    power: dict[int, list[tuple[float, float]]],
    gpu_to_server: dict[int, str],
    tol_s: float = 2.0,
) -> float:
    """Share of the run's alive seconds in which every GPU of its servers has a sample within tol_s."""
    alive = run.t_end - run.t_start
    if alive <= 0:
        return 0.0
    gpus = [g for g, s in gpu_to_server.items() if s in run.servers]
    if not gpus:
        return 0.0
    covered = [(run.t_start, run.t_end)]
    for g in gpus:
        samples = power.get(g, [])
        cov = _merge([(t - tol_s, t + tol_s) for t, _ in samples])
        covered = _intersect(covered, cov)
        if not covered:
            return 0.0
    return sum(b - a for a, b in covered) / alive


# --------------------------------------------------------------------------- apportionment
def apportion_energy(
    power: dict[int, list[tuple[float, float]]],
    gpu_to_server: dict[int, str],
    runs: list[Run],
    calls: Iterable[Call],
    bin_s: float = 60.0,
    equal_split: bool = False,
    coverage_tol_s: float = 2.0,
) -> Result:
    all_t = [t for s in power.values() for t, _ in s]
    if not all_t:
        return Result({r.name: 0.0 for r in runs}, 0.0, 0.0, {r.name: 0.0 for r in runs},
                      {r.name: False for r in runs}, 0.0, bin_s, equal_split)
    t0, t1 = min(all_t), max(all_t)
    nbins = max(int(math.ceil((t1 - t0) / bin_s)), 1)
    srv_e = server_bin_energy(power, gpu_to_server, t0, bin_s, nbins)

    tokens: dict[str, dict[int, dict[str, float]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for c in calls:
        for k, v in spread_tokens(c, t0, bin_s, nbins).items():
            tokens[c.server][k][c.run] += v

    run_energy = {r.name: 0.0 for r in runs}
    overhead = 0.0
    for server, energies in srv_e.items():
        for k, e in enumerate(energies):
            if e <= 0:
                continue
            lo, hi = t0 + k * bin_s, t0 + (k + 1) * bin_s
            alive = [r.name for r in runs if server in r.servers and r.t_start < hi and r.t_end > lo]
            shares = {} if equal_split else dict(tokens[server].get(k, {}))
            total = sum(shares.values())
            if total > 0:
                for name, v in shares.items():
                    run_energy[name] = run_energy.get(name, 0.0) + e * v / total
            elif alive:
                for name in alive:
                    run_energy[name] += e / len(alive)
            else:
                overhead += e

    session = sum(sum(v) for v in srv_e.values())
    coverage = {r.name: run_coverage(r, power, gpu_to_server, coverage_tol_s) for r in runs}
    return Result(
        run_energy_j=run_energy,
        overhead_j=overhead,
        session_energy_j=session,
        coverage=coverage,
        valid={n: c >= COVERAGE_MIN for n, c in coverage.items()},
        residual_j=session - sum(run_energy.values()) - overhead,
        bin_s=bin_s,
        equal_split=equal_split,
        per_server_j={s: sum(v) for s, v in srv_e.items()},
    )


# --------------------------------------------------------------------------- CLI
def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--power-csv", required=True)
    ap.add_argument("--spec", required=True, help="JSON: {gpu_to_server:{gpu:server}, runs:[{name,t_start,t_end,servers,trace,role_server}]}")
    ap.add_argument("--bin-s", type=float, default=60.0)
    ap.add_argument("--equal-split", action="store_true")
    ap.add_argument("--out", default=None, help="write the result JSON here")
    args = ap.parse_args(argv)
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    gpu_to_server = {int(k): v for k, v in spec["gpu_to_server"].items()}
    runs, calls = [], []
    for r in spec["runs"]:
        runs.append(Run(r["name"], float(r["t_start"]), float(r["t_end"]), tuple(r["servers"])))
        if r.get("trace"):
            calls.extend(calls_from_trace(r["trace"], r["name"], r["role_server"]))
    res = apportion_energy(read_power_csv(args.power_csv), gpu_to_server, runs, calls, args.bin_s, args.equal_split)
    text = json.dumps(res.as_dict(), indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
