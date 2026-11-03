#!/usr/bin/env python3
"""GPU power sampler sidecar (AGENT 2027 E3).

Polls NVML power draw at 1 Hz on a session's GPUs, writes a CSV, and (optionally)
logs the same samples to the W&B run ``<SESSION_ID>__energy``. It is started by the
session launcher in its own container (``ro27-<session_id>-energy``) next to the
vLLM servers and stopped (SIGTERM) when the session ends. Run it with every GPU of
the host visible (``--gpus all``) so that ``GPUS`` are the host indices that the
launch line and the apportionment use; ``uuid`` is recorded as a cross-check.

Standard library only; ``pynvml`` (NVML) and ``wandb`` are used when importable.
Backends, in order: ``nvml`` (pynvml), ``smi`` (``nvidia-smi`` subprocess).

CSV columns: ``t_epoch,gpu_index,power_w,uuid`` (one row per GPU per tick).

Examples::

    python scripts/agent2027/energy_sampler.py --session-id S1016s --gpus 0,1,2,3 \\
        --out /workspace/energy/S1016s__energy.csv

Environment fallbacks: ``SESSION_ID`` / ``ROGUE_ONE_SESSION_ID``, ``GPUS``,
``WANDB_PROJECT_NAME`` (default ``RogueOne-AGENT2027``), ``WANDB_ENTITY``
(default ``PlanBench``), ``WANDB_API_KEY``.

This script is written to be read and unit tested offline; the production start
belongs to the session launcher (E1) and is only done on the server.
"""
from __future__ import annotations

import argparse
import csv
import os
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Iterable, Optional

DEFAULT_PROJECT = "RogueOne-AGENT2027"
DEFAULT_ENTITY = "PlanBench"
CSV_HEADER = ["t_epoch", "gpu_index", "power_w", "uuid"]

Sample = tuple[int, float, str]  # (gpu_index, power_w, uuid)


# --------------------------------------------------------------------------- parsing / maths
def parse_gpu_list(text: str) -> list[int]:
    """``"0,1, 3"`` -> ``[0, 1, 3]`` (empty string -> empty list)."""
    return [int(p) for p in text.replace(" ", "").split(",") if p != ""]


def parse_smi_output(text: str) -> list[Sample]:
    """Parse ``nvidia-smi --query-gpu=index,power.draw,uuid --format=csv,noheader,nounits``.

    Lines whose power is not a number (``[N/A]``, ``[Not Supported]``) are skipped.
    """
    out: list[Sample] = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            idx = int(parts[0])
            power = float(parts[1])
        except ValueError:
            continue
        out.append((idx, power, parts[2] if len(parts) > 2 else ""))
    return out


def trapezoid_energy_j(samples: list[tuple[float, float]]) -> float:
    """Trapezoid integral of (t_epoch_s, power_w) samples, in joules."""
    total = 0.0
    for (t0, p0), (t1, p1) in zip(samples, samples[1:]):
        if t1 > t0:
            total += 0.5 * (p0 + p1) * (t1 - t0)
    return total


def sample_interval_stats(times: list[float]) -> dict:
    gaps = [b - a for a, b in zip(times, times[1:])]
    if not gaps:
        return {"median_interval_s": None, "max_interval_s": None}
    return {"median_interval_s": statistics.median(gaps), "max_interval_s": max(gaps)}


# --------------------------------------------------------------------------- backends
class SmiBackend:
    name = "smi"

    def __init__(self, gpus: list[int], runner: Optional[Callable[[list[str]], str]] = None):
        self.gpus = gpus
        self._runner = runner or self._run

    @staticmethod
    def _run(cmd: list[str]) -> str:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=True).stdout

    def read(self) -> list[Sample]:
        cmd = ["nvidia-smi", "--query-gpu=index,power.draw,uuid", "--format=csv,noheader,nounits"]
        if self.gpus:
            cmd += ["-i", ",".join(str(g) for g in self.gpus)]
        return parse_smi_output(self._runner(cmd))

    def close(self) -> None:
        pass


class NvmlBackend:
    name = "nvml"

    def __init__(self, gpus: list[int]):
        import pynvml  # type: ignore

        self.nvml = pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        self.gpus = gpus or list(range(count))
        self.handles = {g: pynvml.nvmlDeviceGetHandleByIndex(g) for g in self.gpus}
        self.uuids = {}
        for g, h in self.handles.items():
            u = pynvml.nvmlDeviceGetUUID(h)
            self.uuids[g] = u.decode() if isinstance(u, bytes) else str(u)

    def read(self) -> list[Sample]:
        out: list[Sample] = []
        for g, h in self.handles.items():
            try:
                out.append((g, self.nvml.nvmlDeviceGetPowerUsage(h) / 1000.0, self.uuids[g]))
            except Exception:
                continue
        return out

    def close(self) -> None:
        try:
            self.nvml.nvmlShutdown()
        except Exception:
            pass


def make_backend(kind: str, gpus: list[int]):
    if kind in ("auto", "nvml"):
        try:
            return NvmlBackend(gpus)
        except Exception as exc:
            if kind == "nvml":
                raise
            print(f"[energy_sampler] NVML unavailable ({exc!r}); using nvidia-smi", file=sys.stderr)
    return SmiBackend(gpus)


# --------------------------------------------------------------------------- W&B
class WandbSink:
    """Logs the samples to ``<SESSION_ID>__energy``; a no-op when wandb is unusable."""

    def __init__(self, session_id: str, gpus: list[int], interval: float, project: str, entity: str):
        self.run = None
        try:
            import wandb  # type: ignore

            self.wandb = wandb
            self.run = wandb.init(
                project=project,
                entity=entity,
                name=f"{session_id}__energy",
                group="energy",
                job_type="energy",
                tags=[session_id, "energy"],
                config={"session_id": session_id, "gpus": gpus, "interval_s": interval},
                settings=wandb.Settings(silent=True, console="off"),
            )
        except Exception as exc:
            print(f"[energy_sampler] W&B disabled: {exc!r}", file=sys.stderr)

    def log(self, t: float, samples: Iterable[Sample]) -> None:
        if self.run is None:
            return
        payload = {"sample_epoch": t}
        total = 0.0
        for g, w, _u in samples:
            payload[f"power_w/gpu{g}"] = w
            total += w
        payload["power_w/total"] = total
        try:
            self.wandb.log(payload)
        except Exception:
            pass

    def finish(self, csv_path: Path, summary: dict) -> None:
        if self.run is None:
            return
        try:
            for k, v in summary.items():
                self.run.summary[k] = v
            art = self.wandb.Artifact(name=f"{self.run.name}-power", type="energy")
            art.add_file(str(csv_path))
            self.wandb.log_artifact(art)
            self.wandb.finish()
        except Exception:
            pass


# --------------------------------------------------------------------------- main loop
def run_sampler(
    backend,
    out_path: Path,
    interval: float = 1.0,
    duration: float = 0.0,
    max_samples: int = 0,
    sink: Optional[WandbSink] = None,
    stop: Optional[Callable[[], bool]] = None,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """Sample until ``stop()``, ``duration`` seconds or ``max_samples`` ticks. Returns a summary."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    per_gpu: dict[int, list[tuple[float, float]]] = {}
    ticks = 0
    start = clock()
    next_t = start
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_HEADER)
        while True:
            if stop is not None and stop():
                break
            if duration and clock() - start >= duration:
                break
            t = clock()
            try:
                samples = backend.read()
            except Exception as exc:  # a missed tick shows up as a coverage gap, not a crash
                print(f"[energy_sampler] read failed: {exc!r}", file=sys.stderr)
                samples = []
            for g, w, u in samples:
                writer.writerow([f"{t:.3f}", g, f"{w:.2f}", u])
                per_gpu.setdefault(g, []).append((t, w))
            fh.flush()
            if sink is not None and samples:
                sink.log(t, samples)
            ticks += 1
            if max_samples and ticks >= max_samples:
                break
            next_t += interval
            delay = next_t - clock()
            if delay > 0:
                sleep(delay)
            else:  # fell behind: do not try to catch up with a burst
                next_t = clock()
    stats = {"samples": ticks, "gpus": sorted(per_gpu)}
    energy = {f"energy_j_gpu{g}": trapezoid_energy_j(s) for g, s in per_gpu.items()}
    stats.update(energy)
    stats["energy_j_total_sampled"] = sum(energy.values())
    all_times = sorted({t for s in per_gpu.values() for t, _ in s})
    stats.update(sample_interval_stats(all_times))
    return stats


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session-id", default=os.getenv("SESSION_ID") or os.getenv("ROGUE_ONE_SESSION_ID"))
    ap.add_argument("--gpus", default=os.getenv("GPUS", ""), help="comma separated host GPU ids")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--out", default=None, help="CSV path (default ./<session>__energy.csv)")
    ap.add_argument("--duration", type=float, default=0.0, help="seconds; 0 = until SIGTERM")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--backend", choices=["auto", "nvml", "smi"], default="auto")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--project", default=os.getenv("WANDB_PROJECT_NAME") or DEFAULT_PROJECT)
    ap.add_argument("--entity", default=os.getenv("WANDB_ENTITY") or DEFAULT_ENTITY)
    args = ap.parse_args(argv)
    if not args.session_id:
        ap.error("--session-id (or SESSION_ID) is required")
    gpus = parse_gpu_list(args.gpus)
    out = Path(args.out or f"{args.session_id}__energy.csv")

    stop_flag = {"stop": False}

    def _stop(*_a):
        stop_flag["stop"] = True

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _stop)
        except Exception:
            pass

    backend = make_backend(args.backend, gpus)
    sink = None if args.no_wandb else WandbSink(args.session_id, gpus, args.interval, args.project, args.entity)
    try:
        summary = run_sampler(
            backend, out, interval=args.interval, duration=args.duration,
            max_samples=args.max_samples, sink=sink, stop=lambda: stop_flag["stop"],
        )
    finally:
        backend.close()
    summary["backend"] = backend.name
    print(f"[energy_sampler] done: {summary}")
    if sink is not None:
        sink.finish(out, summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
