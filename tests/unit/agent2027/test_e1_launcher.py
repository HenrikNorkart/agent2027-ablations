"""E1c: ro27_clients_manifest.sh against a fake ``docker`` (tests/unit/agent2027/shims/docker).

A 3-row session gives unique ``ro27-<session>-<row>`` names; a deliberately failing row
is retried once, alone, after the concurrent pass; a row with ``cap_hours=0.01`` is
stopped and logged as CAP_HIT and not retried. DRY_RUN=1 makes no docker call and
shows the arm-specific mounts only on the rows that need them.
"""

from __future__ import annotations

import csv
import io
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit]

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "agent2027" / "ro27_clients_manifest.sh"
SESSION_LAUNCH = REPO / "scripts" / "agent2027" / "ro27_session_launch.sh"
SHIM = Path(__file__).resolve().parent / "shims" / "docker"
DEFAULTS = REPO / "scripts" / "agent2027" / "v2_defaults.env"

R7 = ("row_id,ablation_id,arm_id,dataset_id,task_type,fold_seed,llm_seed,replicate,is_sentinel,"
      "session_id,order_key,image_tag,rung_S,rung_E,rung_T,port_S,port_E,port_T,env_overrides,"
      "cap_hours,expected_u,notes").split(",")


def _find_bash() -> str | None:
    if os.name == "nt":
        for c in (r"C:\Program Files\Git\bin\bash.exe", r"C:\Program Files\Git\usr\bin\bash.exe"):
            if Path(c).exists():
                return c
        return None  # never fall back to WSL's System32\bash.exe
    return shutil.which("bash")


BASH = _find_bash()
needs_bash = pytest.mark.skipif(BASH is None, reason="no POSIX bash available")


def _posix(p: Path) -> str:
    """Path as Git Bash sees it (C:\\x -> /c/x); unchanged on POSIX."""
    s = str(p)
    if os.name == "nt" and len(s) > 1 and s[1] == ":":
        s = "/" + s[0].lower() + s[2:].replace("\\", "/")
    return s


def _row(rid, *, order, ds="synp_p1", tt="regression", cap="6", ov="{}", sentinel="0",
         session="S1012a", arm="R1", tag="agent2027-pre.1", notes="smoke"):
    return {
        "row_id": rid, "ablation_id": rid[:3], "arm_id": arm, "dataset_id": ds, "task_type": tt,
        "fold_seed": "42", "llm_seed": "42", "replicate": "0", "is_sentinel": sentinel,
        "session_id": session, "order_key": str(order), "image_tag": tag,
        "rung_S": "O120", "rung_E": "O120", "rung_T": "O120",
        "port_S": "8010", "port_E": "8010", "port_T": "8010",
        "env_overrides": ov, "cap_hours": cap, "expected_u": "0.15", "notes": notes,
    }


def _write_session(path: Path, rows: list[dict]) -> None:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=R7, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    path.write_bytes(buf.getvalue().encode())


@pytest.fixture()
def sandbox(tmp_path):
    shim_dir = tmp_path / "shim"
    shim_dir.mkdir()
    shim = shim_dir / "docker"
    shim.write_bytes(SHIM.read_bytes().replace(b"\r\n", b"\n"))
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC)
    for tt, ds in (("regression", "synp_p1"), ("classification", "synp_p3")):
        d = tmp_path / "src" / tt / ds
        d.mkdir(parents=True)
        (d / "config.yml").write_text(f"experiment_name: {ds}\n")
        (d / "data.csv").write_text("id,x\n1,2\n")
        (d / "labels.csv").write_text("id,target\n1,0\n")
    (tmp_path / "state").mkdir()
    script = tmp_path / "ro27_clients_manifest.sh"
    script.write_bytes(SCRIPT.read_bytes().replace(b"\r\n", b"\n"))
    defaults = tmp_path / "v2_defaults.env"
    defaults.write_bytes(DEFAULTS.read_bytes().replace(b"\r\n", b"\n"))
    return tmp_path


_LEAKY = ("ROGUE_ONE_", "WANDB_", "LLM_", "SHIM_")
_LEAKY_EXACT = {"CODE_PHASE", "MODEL", "PORT", "API_KEY", "IMAGE", "IMAGE_RO", "GPUS", "GIT_SHA",
                "PREREG_TAG", "DRY_RUN", "V2_DEFAULTS", "RUN_CAP_HOURS", "PARALLEL", "CPUS",
                "SESSION_ID", "SESSION_FILE", "RUNROOT", "RO27_JOB_LOG", "JOB_LOG", "SERVER_NAME"}


def _clean_env() -> dict:
    """os.environ minus anything the launcher reads (importing rogueone.utils sets e.g.
    WANDB_PROJECT_NAME=rogueone_project in this process)."""
    return {k: v for k, v in os.environ.items()
            if not k.startswith(_LEAKY) and k not in _LEAKY_EXACT}


def _run(sb: Path, session_rows: list[dict], behaviour: str = "", **extra) -> subprocess.CompletedProcess:
    _write_session(sb / "S1012a.csv", session_rows)
    (sb / "behavior").write_text(behaviour)
    env = _clean_env()
    env.update({
        "SHIM_STATE": _posix(sb / "state"), "SHIM_BEHAVIOR": _posix(sb / "behavior"),
        "SESSION_ID": "S1012a", "SESSION_FILE": _posix(sb / "S1012a.csv"),
        "RUNROOT": _posix(sb / "runs"), "TASKS_SRC": _posix(sb / "src"),
        "SERPER_CACHE_DIR": _posix(sb / "cache"), "RT3_FOCUS_DIR": _posix(sb / "focus"),
        "CORPORA_DIR": _posix(sb / "corpora"),
        "V2_DEFAULTS": _posix(sb / "v2_defaults.env"), "WANDB_API_KEY": "dummy-key",
        "IMAGE_RO": "rogue_one:agent2027-pre.1", "PORT": "8010", "API_KEY": "k",
        "MODEL": "openai/gpt-oss-120b", "PARALLEL": "3", "POLL_S": "1",
        "RO27_JOB_LOG": _posix(sb / "job.log"), "GIT_SHA": "abc1234",
    })
    env.update({k: str(v) for k, v in extra.items()})
    cmd = f'export PATH="{_posix(sb / "shim")}:$PATH"; bash "{_posix(sb / "ro27_clients_manifest.sh")}"'
    return subprocess.run([BASH, "-c", cmd], env=env, capture_output=True, text=True, timeout=240)


def _status(sb: Path) -> list[dict]:
    text = (sb / "runs" / "S1012a" / "status.tsv").read_text()
    return list(csv.DictReader(io.StringIO(text), delimiter="\t"))


@needs_bash
def test_bash_syntax():
    for s in (SCRIPT, SESSION_LAUNCH, SHIM):
        r = subprocess.run([BASH, "-n", _posix(s)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr


@needs_bash
def test_three_rows_names_retry_and_cap(sandbox):
    rows = [
        _row("A02-902", order=2, arm="R3"),            # fails once -> solo retry -> OK
        _row("A02-901", order=1),                      # OK
        _row("A02-903", order=3, cap="0.01", notes="cap test, quoted"),  # hangs -> CAP_HIT
    ]
    behaviour = ("ro27-s1012a-a02-901 ok:2\n"
                 "ro27-s1012a-a02-902 fail_once:1\n"
                 "ro27-s1012a-a02-903 hang\n")
    r = _run(sandbox, rows, behaviour)
    out = r.stdout + r.stderr
    assert r.returncode == 2, out  # a CAP_HIT row leaves the session non-green

    calls = (sandbox / "state" / "calls.log").read_text().splitlines()
    runs = [c for c in calls if c.startswith("run -d")]
    names = [c.split("--name ")[1].split()[0] for c in runs]
    # unique, lower-case, per session and row; 902 started twice (first pass + retry)
    assert sorted(set(names)) == ["ro27-s1012a-a02-901", "ro27-s1012a-a02-902", "ro27-s1012a-a02-903"]
    assert names.count("ro27-s1012a-a02-902") == 2
    assert names.count("ro27-s1012a-a02-903") == 1  # CAP_HIT is never retried
    assert names.count("ro27-s1012a-a02-901") == 1
    assert all(n == n.lower() for n in names)

    st = _status(sandbox)
    by = {}
    for s in st:
        by.setdefault(s["row_id"], []).append(s)
    assert [s["status"] for s in by["A02-901"]] == ["OK"]
    assert [(s["attempt"], s["status"]) for s in by["A02-902"]] == [("1", "FAILED"), ("2", "OK")]
    assert [s["status"] for s in by["A02-903"]] == ["CAP_HIT"]
    assert by["A02-903"][0]["reason"] == "cap_hit"
    assert 36 <= int(by["A02-903"][0]["wall_s"]) <= 60  # 0.01 h = 36 s

    # the retry ran alone: it started after every first-pass attempt had ended
    retry = by["A02-902"][1]
    first_pass_end = max(int(s["end_epoch"]) for s in st if s["attempt"] == "1")
    assert int(retry["start_epoch"]) >= first_pass_end
    assert "stop" in " ".join(calls) and "ro27-s1012a-a02-903" in [c.split()[-1] for c in calls if c.startswith("stop")]

    # the job log carries the per-row lines (A05 monitoring reads them there)
    job = (sandbox / "job.log").read_text()
    assert "ROW A02-903 attempt=1 status=CAP_HIT" in job
    assert "ROW A02-902 attempt=2 status=OK" in job
    assert "RO27_CLIENTS_DONE session=S1012a ok=2 failed=0 cap_hit=1" in job


@needs_bash
def test_env_file_layers_identity_and_task_copy(sandbox):
    rows = [_row("A02-901", order=1, ov='{"ROGUE_ONE_EXTRACTOR_ROUNDS":"1","ROGUE_ONE_SMOKE_FOLDS":"1"}'),
            _row("A02-902", order=2, arm="R3", ov='{"ROGUE_ONE_EXTRACTOR_ROUNDS":"3"}')]
    r = _run(sandbox, rows, "")
    assert r.returncode == 0, r.stdout + r.stderr
    env = {}
    for line in (sandbox / "state" / "c" / "ro27-s1012a-a02-901.env.1").read_text().splitlines():
        k, v = line.split("=", 1)
        assert k not in env, f"duplicate key {k} in env file"
        env[k] = v
    assert env["ROGUE_ONE_EXTRACTOR_ROUNDS"] == "1"      # override beats the default (10)
    assert env["ROGUE_ONE_SMOKE_FOLDS"] == "1"
    assert env["ROGUE_ONE_NUM_ITERATIONS"] == "10"       # from v2_defaults.env
    assert env["ROGUE_ONE_DATASETS"] == "synp_p1"
    assert env["ROGUE_ONE_FOLD_SEED"] == "42" and env["ROGUE_ONE_LLM_SEED"] == "42"
    assert env["ROGUE_ONE_TASK_REGISTRY"] == "/workspace/tasks/task_registry.yml"
    assert env["LLM_PORT"] == "8010" and env["LLM_MODEL"] == "openai/gpt-oss-120b"
    assert env["WANDB_PROJECT_NAME"] == "RogueOne-AGENT2027-prefreeze"
    assert env["WANDB_ENTITY"] == "PlanBench"
    assert env["ROGUE_ONE_ABLATION_ID"] == "A02" and env["ROGUE_ONE_ARM_ID"] == "R1"
    assert env["ROGUE_ONE_SESSION_ID"] == "S1012a" and env["ROGUE_ONE_MANIFEST_ROW"] == "A02-901"
    assert env["ROGUE_ONE_CODE_PHASE"] == "smoke" and env["ROGUE_ONE_IMAGE_TAG"] == "agent2027-pre.1"
    assert env["ROGUE_ONE_GIT_SHA"] == "abc1234"
    assert env["ROGUE_ONE_SERVER_TOPOLOGY"] == "uniform:O120@8010"
    assert "WANDB_API_KEY" not in env and "LLM_API_KEY" not in env  # secrets by name only
    # own task copy per row, with output/ pre-created and a one-entry registry
    for rid in ("A02-901", "A02-902"):
        t = sandbox / "runs" / "S1012a" / rid / "tasks"
        assert (t / "regression" / "synp_p1" / "data.csv").exists()
        assert (t / "regression" / "synp_p1" / "output").is_dir()
        assert (t / "task_registry.yml").read_text().strip() == "- /workspace/tasks/regression/synp_p1/config.yml"
    run901 = next(c for c in (sandbox / "state" / "calls.log").read_text().splitlines()
                  if "--name ro27-s1012a-a02-901" in c)
    assert "python3 -u /workspace/rogue_one_main.py" in run901  # explicit command
    assert "--cpus 12" in run901 and "--network host" in run901 and "--ipc=host" in run901


@needs_bash
def test_rc0_without_summary_is_failed_and_retried(sandbox):
    r = _run(sandbox, [_row("A02-901", order=1)], "ro27-s1012a-a02-901 rc0_noresult\n")
    assert r.returncode == 2
    st = _status(sandbox)
    assert [(s["attempt"], s["status"], s["reason"]) for s in st] == [
        ("1", "FAILED", "rc0-no-fold_aggregation_summary"),
        ("2", "FAILED", "rc0-no-fold_aggregation_summary"),
    ]


@needs_bash
def test_dry_run_no_docker_and_arm_mounts(sandbox):
    rows = [
        _row("A09-001", order=1, arm="oracle", ov='{"ROGUE_ONE_PLANNER_MODE":"oracle"}', notes=""),
        _row("A05-001", order=2, arm="full", notes="", tag="agent2027-pre.1"),
        _row("A14-007", order=3, arm="kb_local_correct", notes="",
             ov='{"ROGUE_ONE_KNOWLEDGE_BACKEND":"local","ROGUE_ONE_LOCAL_CORPUS":"kt3_correct"}'),
    ]
    r = _run(sandbox, rows, "", DRY_RUN="1", V2_DEFAULTS="")
    out = r.stdout
    assert r.returncode == 0, out + r.stderr
    assert not (sandbox / "state" / "calls.log").exists()  # no docker call at all
    blocks = {}
    cur = None
    for line in out.splitlines():
        if "DRYRUN ROW" in line:
            cur = line.split("DRYRUN ROW ")[1].split()[0]
            blocks[cur] = ""
        elif cur:
            blocks[cur] += line + "\n"
    assert set(blocks) == {"A09-001", "A05-001", "A14-007"}
    assert "/workspace/focus/focus.json:ro" in blocks["A09-001"]
    assert "ROGUE_ONE_PLANNER_FOCUS_FILE=/workspace/focus/focus.json" in blocks["A09-001"]
    assert "focus.json" not in blocks["A05-001"] and "/workspace/corpora" not in blocks["A05-001"]
    assert "/workspace/corpora/kt3_correct:ro" in blocks["A14-007"]
    assert "kt3_wrong" not in blocks["A14-007"]
    assert "ro27-s1012a-a09-001" in out and "ro27-s1012a-a14-007" in out


@needs_bash
def test_session_launch_dry_run(sandbox):
    launch = sandbox / "ro27_session_launch.sh"
    launch.write_bytes(SESSION_LAUNCH.read_bytes().replace(b"\r\n", b"\n"))
    _write_session(sandbox / "S1012a.csv", [_row("A02-901", order=1)])
    env = _clean_env()
    env.update({
        "SHIM_STATE": _posix(sandbox / "state"), "SESSION_ID": "S1012a",
        "SESSION_FILE": _posix(sandbox / "S1012a.csv"), "RUNROOT": _posix(sandbox / "runs"),
        "CLIENTS_SCRIPT": _posix(sandbox / "ro27_clients_manifest.sh"), "GPUS": "0,1,2,3",
        "IMAGE_RO": "rogue_one:agent2027-pre.1", "DRY_RUN": "1", "JOB_LOG": _posix(sandbox / "job.log"),
        "ROGUEONE_CAIR_PATH": _posix(sandbox / "nocheckout"),
    })
    cmd = f'export PATH="{_posix(sandbox / "shim")}:$PATH"; bash "{_posix(launch)}"'
    r = subprocess.run([BASH, "-c", cmd], env=env, capture_output=True, text=True, timeout=120)
    out = r.stdout
    assert r.returncode == 0, out + r.stderr
    assert "server=rogueone-vllm-o120-s1012a" in out
    assert "ro27-s1012a-energy" in out
    assert "DRYRUN ROW A02-901" in out and "container=ro27-s1012a-a02-901" in out
    assert out.strip().splitlines()[-1].endswith("=== RO27_SESSION_END S1012a rc=0 ===")
    assert not (sandbox / "state" / "calls.log").exists()
    # explicit GPUs are mandatory (01 §5.1 rule 2)
    env.pop("GPUS")
    r = subprocess.run([BASH, "-c", cmd], env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 1 and "GPUS must list explicit ids" in r.stdout
    assert "RO27_SESSION_END S1012a rc=1" in r.stdout


@needs_bash
def test_refuses_covtype_and_foreign_session(sandbox):
    r = _run(sandbox, [_row("A05-001", order=1, ds="covtype", tt="classification")], "")
    assert r.returncode != 0 and "covtype" in r.stdout
    assert not (sandbox / "state" / "calls.log").exists()
    r = _run(sandbox, [_row("A05-001", order=1, session="S1013a")], "")
    assert r.returncode != 0 and "not in session" in r.stdout


@needs_bash
def test_image_mismatch_is_refused_not_retried(sandbox):
    r = _run(sandbox, [_row("A05-001", order=1, tag="agent2027-v2.0", notes="")], "")
    st = _status(sandbox)
    assert [(s["status"], s["rc"]) for s in st] == [("FAILED", "setup")]
    assert "image_tag=agent2027-v2.0" in st[0]["reason"]


@needs_bash
def test_a14_manifest_dry_run_corpus_mounts(sandbox):
    """A14-T9: over the generated A14 manifest the kt3_correct mount is on exactly
    A14-007..012 and 903, kt3_wrong on exactly A14-016..018 and 904, none elsewhere."""
    manifest = REPO / "manifests" / "agent2027" / "A14_KT3-P7.csv"
    rows = list(csv.DictReader(io.StringIO(manifest.read_text(encoding="utf-8"))))
    assert len(rows) == 22
    for i, row in enumerate(rows, 1):
        row["session_id"], row["order_key"] = "S1012a", str(i)
    r = _run(sandbox, rows, "", DRY_RUN="1", V2_DEFAULTS="", IMAGE_RO="rogue_one:agent2027-v2.3")
    assert r.returncode == 0, r.stdout + r.stderr
    blocks, cur = {}, None
    for line in r.stdout.splitlines():
        if "DRYRUN ROW" in line:
            cur = line.split("DRYRUN ROW ")[1].split()[0]
            blocks[cur] = ""
        elif cur:
            blocks[cur] += line + "\n"
    assert len(blocks) == 22
    want_correct = {f"A14-{n:03d}" for n in range(7, 13)} | {"A14-903"}
    want_wrong = {"A14-016", "A14-017", "A14-018", "A14-904"}
    got_correct = {k for k, b in blocks.items() if "/workspace/corpora/kt3_correct:ro" in b}
    got_wrong = {k for k, b in blocks.items() if "/workspace/corpora/kt3_wrong:ro" in b}
    got_any = {k for k, b in blocks.items() if "/workspace/corpora" in b}
    assert got_correct == want_correct and got_wrong == want_wrong
    assert got_any == want_correct | want_wrong


@needs_bash
def test_mixed_rows_global_override_and_routing(sandbox):
    """SR4 (2026-10-19): over the generated A19 manifest and A04 session-C rows, mixed rows get
    per-role models/ports from rung_*/port_*, and the global LLM_MODEL/LLM_PORT from their
    env_overrides (A19: Q122/8040; A04-007/008: the third model), never from the session server."""
    rows = []
    for name in ("A19_CAP-2.csv", "A04_CAP-0.csv"):
        text = (REPO / "manifests" / "agent2027" / name).read_text(encoding="utf-8")
        rows += [r for r in csv.DictReader(io.StringIO(text)) if r["row_id"] in
                 ("A19-001", "A19-019", "A04-007", "A04-008")]
    for i, row in enumerate(rows, 1):
        row["session_id"], row["order_key"] = "S1012a", str(i)
    r = _run(sandbox, rows, "", DRY_RUN="1", V2_DEFAULTS="", IMAGE_RO="rogue_one:agent2027-v2.3",
             PORT="8099", MODEL="some/other-model")
    assert r.returncode == 0, r.stdout + r.stderr
    blocks, cur = {}, None
    for line in r.stdout.splitlines():
        if "DRYRUN ROW" in line:
            cur = line.split("DRYRUN ROW ")[1].split()[0]
            blocks[cur] = {}
        elif cur and line.strip().startswith("env: "):
            k, v = line.strip()[5:].split("=", 1)
            blocks[cur][k] = v
    want = {
        "A19-001": ("Qwen/Qwen3.5-122B-A10B-FP8", "8040", "8040", "8032", "8040"),
        "A19-019": ("Qwen/Qwen3.5-122B-A10B-FP8", "8040", "8030", "8040", "8030"),
        "A04-007": ("Qwen/Qwen3.5-0.8B", "8030", "8040", "8032", "8040"),
        "A04-008": ("Qwen/Qwen3.5-4B", "8032", "8030", "8040", "8030"),
    }
    for rid, (gm, gp, ps, pe, pt) in want.items():
        env = blocks[rid]
        assert (env["LLM_MODEL"], env["LLM_PORT"]) == (gm, gp), rid
        assert (env["ROGUE_ONE_SCIENTIST_PORT"], env["ROGUE_ONE_EXTRACTOR_PORT"],
                env["ROGUE_ONE_TESTER_PORT"]) == (ps, pe, pt), rid
        assert env["ROGUE_ONE_WEBSEARCH_ROUTING"] == "caller"
        assert env["ROGUE_ONE_SERVER_TOPOLOGY"].endswith(f";G={gm}@{gp}")


def test_hold_and_queue_scripts_syntax():
    if BASH is None:
        pytest.skip("no POSIX bash available")
    for name in ("ro27_hold_clients.sh", "ro27_queue.sh"):
        p = REPO / "scripts" / "agent2027" / name
        r = subprocess.run([BASH, "-n", _posix(p)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
