"""A06/A08 manifests (A06-T9, A08-T9), spec parameters, plan-session stamping, X4 inputs, A08-T10 payloads."""

from __future__ import annotations

import ast
import csv
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit]

REPO = Path(__file__).resolve().parents[3]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mm = _load("make_manifest", "scripts/agent2027/make_manifest.py")
mli = _load("make_llmfe_inputs", "scripts/agent2027/llmfe_specs/make_llmfe_inputs.py")
alw = _load("a08_log_wandb", "scripts/agent2027/a08_log_wandb.py")
cfp = _load("check_fold_partition", "scripts/agent2027/check_fold_partition.py")


# --------------------------------------------------------------------------- A06
def test_a06_without_g2_has_only_q4_rows(tmp_path):
    notes = []
    out = mm.make_ablation("A06", None, None, manifests_dir=tmp_path, log=notes.append)
    rows = mm.read_csv(out)
    assert out.name == "A06_M0-S.csv"
    assert [r["row_id"] for r in rows] == [f"A06-{n}" for n in range(101, 106)]
    for k, r in enumerate(rows, start=1):
        assert (r["arm_id"], r["dataset_id"], r["rung_E"], r["port_E"]) == ("sentinel_Q4", "synp_p2", "Q4", "8032")
        assert (r["replicate"], r["llm_seed"], r["is_sentinel"], r["fold_seed"]) == (str(k), str(6000 + k), "1", "42")
        assert r["image_tag"] == "" and r["env_overrides"] == "{}" and r["cap_hours"] == "6"
    assert any("SENTINEL_TASK" in n for n in notes)


@pytest.mark.parametrize("task,ctype", [("synp_p3", "classification"), ("synp_p2", "regression")])
def test_a06_with_g2_task(tmp_path, task, ctype):
    out = mm.make_ablation("A06", None, None, manifests_dir=tmp_path, params={"SENTINEL_TASK": task})
    o120 = [r for r in mm.read_csv(out) if r["arm_id"] == "sentinel"]
    assert [r["row_id"] for r in o120] == [f"A06-{n:03d}" for n in range(1, 13)]
    for k, r in enumerate(o120, start=1):
        assert r["dataset_id"] == task and r["task_type"] == ctype and r["rung_S"] == "O120"
        assert r["llm_seed"] == str(5000 + k) and r["replicate"] == str(k)
        assert r["port_E"] == ("8010" if k % 2 else "8011")
        ov = json.loads(r["env_overrides"])
        assert ov == ({"ROGUE_ONE_STRATIFIED_FOLDS": "True"} if ctype == "classification" else {})


@pytest.mark.parametrize("bad", ["synp_p7", "synp_p0", "covtype", "heart"])
def test_a06_task_must_be_a_synp7_candidate(tmp_path, bad):
    with pytest.raises(mm.ManifestError):
        mm.make_ablation("A06", None, None, manifests_dir=tmp_path, params={"SENTINEL_TASK": bad})


def test_unknown_param_rejected(tmp_path):
    with pytest.raises(mm.ManifestError, match="declares no such parameter"):
        mm.make_ablation("A06", None, None, manifests_dir=tmp_path, params={"NOPE": "1"})


# --------------------------------------------------------------------------- A08
def test_a08_manifest(tmp_path):
    rows = mm.read_csv(mm.make_ablation("A08", None, None, manifests_dir=tmp_path))
    assert len(rows) == 95 and [r["row_id"] for r in rows] == [f"A08-{n:03d}" for n in range(1, 96)]
    cpu, gpu = rows[:81], rows[81:]
    assert {r["arm_id"] for r in cpu} == {"base_no_fe", "openfe", "autofeat"}
    assert {(r["arm_id"], r["dataset_id"], r["fold_seed"]) for r in cpu} == {
        (a, d, s) for a in ("base_no_fe", "openfe", "autofeat") for d in mm.SYNP_ALL for s in ("42", "4", "5")}
    for r in cpu:
        assert r["image_tag"] == "classicfe:4" and r["llm_seed"] == "" and r["rung_E"] == "" and r["port_E"] == ""
        assert r["expected_u"] == "0" and r["replicate"] == "0"
    assert {(r["dataset_id"], r["replicate"]) for r in gpu} == {(d, rep) for d in mm.SYNP7 for rep in ("0", "1")}
    for r in gpu:
        assert (r["arm_id"], r["image_tag"], r["rung_E"], r["port_E"], r["fold_seed"], r["llm_seed"]) == (
            "llmfe", "llmfe:oss120", "O120", "8090", "42", "")
    assert not [r for r in rows if r["arm_id"] == "caafe"]  # CL4
    names = [alw.run_name(r) for r in rows]
    assert len(set(names)) == 95  # unique R12 run names


def test_empty_llm_seed_only_for_comparator_rows():
    row = {c: "" for c in mm.R7_HEADER}
    row.update(row_id="A05-001", ablation_id="A05", arm_id="full", dataset_id="heart", task_type="classification",
               fold_seed="42", llm_seed="", replicate="0", is_sentinel="0", env_overrides="{}", cap_hours="6",
               image_tag="agent2027-v2.0")
    with pytest.raises(mm.ManifestError, match="llm_seed"):
        mm.validate_row(row)
    row.update(image_tag="llmfe:oss120")
    mm.validate_row(row)


def test_plan_session_stamps_ports_and_sentinel_image(tmp_path):
    mm.make_ablation("A08", None, None, manifests_dir=tmp_path)
    mm.make_ablation("A06", None, None, manifests_dir=tmp_path, params={"SENTINEL_TASK": "synp_p4"})
    with pytest.raises(mm.ManifestError, match="sentinel without image_tag"):
        mm.plan_session("S1020l", ["A08-082", "A06-003"], manifests_dir=tmp_path, log=lambda *_: None)
    out = mm.plan_session("S1020l", ["A08-082", "A08-084", "A06-003"], manifests_dir=tmp_path,
                          sentinel_first=True, sentinel_image_tag="agent2027-v2.2", port=8020, proxy_port=8090,
                          log=lambda *_: None)
    rows = {r["row_id"]: r for r in mm.read_csv(out)}
    assert rows["A06-003"]["order_key"] == "1" and rows["A06-003"]["image_tag"] == "agent2027-v2.2"
    assert rows["A06-003"]["port_E"] == "8020" and rows["A08-082"]["port_E"] == "8090"
    assert rows["A08-082"]["image_tag"] == "llmfe:oss120"
    plan = out.with_suffix(".plan.txt").read_text()
    assert "sentinel_image_tag=agent2027-v2.2" in plan and "port=8020" in plan and "proxy_port=8090" in plan


def test_committed_session_files_validate():
    for sid in ("S1019q", "S1020l", "S1020m"):
        path = REPO / "manifests" / "agent2027" / "sessions" / f"{sid}.csv"
        rows = mm.validate_rows(mm.read_csv(path), session=True)
        assert all("covtype" not in r["dataset_id"] for r in rows)
    q4 = mm.read_csv(REPO / "manifests/agent2027/sessions/S1019q.csv")
    assert [(r["row_id"], r["image_tag"], r["port_E"]) for r in q4] == [("A06-101", "agent2027-v2.0", "8022")]
    l_rows = mm.read_csv(REPO / "manifests/agent2027/sessions/S1020l.csv")
    m_rows = mm.read_csv(REPO / "manifests/agent2027/sessions/S1020m.csv")
    a08 = [r for r in l_rows + m_rows if r["ablation_id"] == "A08"]
    assert sorted(r["row_id"] for r in a08) == [f"A08-{n:03d}" for n in range(82, 96)]
    for rows in (l_rows, m_rows):  # the two replicates of a task never share a server
        assert len({r["dataset_id"] for r in rows if r["ablation_id"] == "A08"}) == 7


# --------------------------------------------------------------------------- X4
def _fake_task(root: Path, task: str, ttype: str) -> None:
    d = root / ttype / task
    d.mkdir(parents=True)
    ctx = ("Some records. The table has 4 records, one per thing, described by 3 columns.\n"
           "Columns:\n- a_m: length, metres\n- b: a code\n- c_kg: mass, kg\n"
           "Goal: predict the thing (a numeric target).")
    import yaml
    (d / "config.yml").write_text(yaml.safe_dump({"experiment_name": task, "context_and_goal": ctx,
                                                  "task_type": ttype}), encoding="utf-8")
    (d / "data.csv").write_text("id,a_m,b,c_kg\n3,1.50,x,0.1\n1,2.25,y,1e-3\n2,3,x,7\n0,4.125,z,8\n", encoding="utf-8")
    (d / "labels.csv").write_text("id,target\n0,1\n1,0.5\n2,-1.25\n3,2\n", encoding="utf-8")


def test_x4_specs_and_data(tmp_path):
    tasks = tmp_path / "tasks"
    _fake_task(tasks, "synp_p2", "regression")
    _fake_task(tasks, "synp_p3", "classification")
    specs = tmp_path / "specs"
    assert mli.main(["specs", "--tasks-root", str(tasks), "--out", str(specs), "--tasks", "synp_p2", "synp_p3"]) == 0
    for t, tmpl in (("synp_p2", "specification_crab.txt"), ("synp_p3", "specification_heart.txt")):
        spec = (specs / t / f"specification_{t}.txt").read_text()
        ast.parse(spec)  # still valid Python for LLM-FE's code_manipulation
        assert "Some records." in spec and "Goal: predict the thing" in spec
        assert "df_output = df_input.copy()" in spec and "RestingBP" not in spec and "Weight" not in spec
        up = (mli.UPSTREAM / tmpl).read_text()
        assert spec.split("@evaluate.run")[1].split("@equation.evolve")[0] == \
            up.split("@evaluate.run")[1].split("@equation.evolve")[0]  # evaluate() verbatim
        assert json.loads((specs / t / f"{t}-metadata.json").read_text()) == {
            "a_m": "length, metres", "b": "a code", "c_kg": "mass, kg"}
    out = tmp_path / "out"
    assert mli.main(["data", "--tasks-root", str(tasks), "--specs-dir", str(specs), "--out", str(out),
                     "--tasks", "synp_p2", "synp_p3"]) == 0
    data = (out / "data" / "synp_p2.csv").read_text()
    # data.csv row order, id dropped, values byte-identical, target joined on id, last
    assert data.splitlines() == ["a_m,b,c_kg,target", "1.50,x,0.1,2", "2.25,y,1e-3,0.5", "3,x,7,-1.25", "4.125,z,8,1"]
    assert (out / "alias" / "synp_p2" / "crab.csv").read_text() == data
    assert not (out / "alias" / "synp_p3").exists()
    man = json.loads((out / "MANIFEST.json").read_text())
    assert man["files"]["data/synp_p2.csv"]["rows"] == 4
    # a spec that drifted from config.yml is refused
    p = specs / "synp_p3" / "specification_synp_p3.txt"
    p.write_text(p.read_text().replace("Some records.", "Other records."))
    with pytest.raises(SystemExit):
        mli.main(["data", "--tasks-root", str(tasks), "--specs-dir", str(specs), "--out", str(out), "--tasks", "synp_p3"])


def test_x4_refuses_covtype_and_non_synp(tmp_path):
    for bad in (["covtype"], ["heart"]):
        with pytest.raises(SystemExit):
            mli.main(["specs", "--tasks-root", str(tmp_path), "--out", str(tmp_path / "o"), "--tasks", *bad])


def test_committed_specs_match_their_template():
    for t in mli.SYNP7:
        spec = (mli.HERE / t / f"specification_{t}.txt").read_text(encoding="utf-8").replace("\r\n", "\n")
        ast.parse(spec)
        task_text = mli.spec_task_text(spec)
        assert task_text.startswith(("Trip logs", "Load tests", "Loan", "Readings", "Experiments", "Measurements",
                                     "Anonymised"))
        ttype = "classification" if t in ("synp_p3", "synp_p4", "synp_p5") else "regression"
        assert spec == mli.render_spec(ttype, task_text)
        meta = json.loads((mli.HERE / t / f"{t}-metadata.json").read_text(encoding="utf-8"))
        assert meta == mli.column_descriptions(task_text)


def test_llmfe_split_matches_rogueone_folds(tmp_path):
    """check_fold_partition llmfe mode == sklearn StratifiedKFold / KFold on data.csv row order."""
    pytest.importorskip("sklearn")
    tasks = tmp_path / "tasks"
    for task, ttype in (("synp_p2", "regression"), ("synp_p3", "classification")):
        d = tasks / ttype / task
        d.mkdir(parents=True)
        n = 60
        (d / "data.csv").write_text("id,x\n" + "".join(f"{(i * 7) % n},{i}\n" for i in range(n)))
        (d / "labels.csv").write_text("id,target\n" + "".join(f"{i},{i % 3 if ttype == 'classification' else i * 0.5}\n"
                                                             for i in range(n)))
        (d / "config.yml").write_text("context_and_goal: 'x'\ntask_type: %s\n" % ttype)
        text, _, _ = mli.build_table(d)
        csv_path = tmp_path / f"{task}.csv"
        csv_path.write_text(text)
        args = types.SimpleNamespace(task_dir=str(d), llmfe_csv=str(csv_path),
                                     llmfe_problem_name="crab" if ttype == "regression" else task)
        doc = cfp.run_llmfe(args)
        from sklearn.model_selection import KFold, StratifiedKFold
        ids = [int(line.split(",")[0]) for line in (d / "data.csv").read_text().splitlines()[1:]]
        if ttype == "regression":
            ref = [te for _, te in KFold(5, shuffle=True, random_state=42).split(range(n))]
        else:
            lab = {int(a): b for a, b in (l.split(",") for l in (d / "labels.csv").read_text().splitlines()[1:])}
            ref = [te for _, te in StratifiedKFold(5, shuffle=True, random_state=42).split([[0]] * n, [lab[i] for i in ids])]
        assert doc["test_idx"] == [sorted(int(i) for i in f) for f in ref]
        assert doc["llmfe_is_regression"] == (ttype == "regression")


# --------------------------------------------------------------------------- A08-T10
def test_a08_log_payloads_cpu_and_llmfe(tmp_path):
    rows = mm.read_csv(mm.make_ablation("A08", None, None, manifests_dir=tmp_path))
    cells = tmp_path / "out" / "f42"
    cells.mkdir(parents=True)
    (cells / "arm_synp_p3_raw_stratified.json").write_text(json.dumps(
        {"folds_done": 5, "accw_balanced": 0.71, "accw_balanced_sd": 0.02, "secs": 12.5, "n_features": [0] * 5}))
    eps = tmp_path / "eps.csv"
    with open(eps, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset_id", "arm", "fold_seed", "fold", "status", "rec_any", "rec_sel"])
        for f in range(5):
            w.writerow(["synp_p3", "raw", 42, f, "ok", 0, 0])
    cpu = alw.cpu_payloads(rows, tmp_path / "out", eps)
    assert len(cpu) == 81
    p = next(x for x in cpu if x["row"]["row_id"] == "A08-004")  # base_no_fe synp_p3 f42
    assert p["name"] == "A08__base_no_fe__synp_p3__f42__r0" and p["tags"][0] == "cpu_screen"
    assert p["summary"]["heldout_score"] == 0.71 and p["summary"]["rec_sel"] == 0.0 and p["summary"]["status"] == "OK"
    assert p["config"]["llm_seed"] == "" and p["config"]["rung_E"] == "" and p["config"]["comparator/image"] == "classicfe:4"
    # llmfe: session file rows + status + proxy totals
    sess = mm.plan_session("S1020l", ["A08-082", "A08-084"], manifests_dir=tmp_path, no_sentinel=True,
                           proxy_port=8090, log=lambda *_: None)
    status = tmp_path / "status.tsv"
    status.write_text("row_id\tattempt\tstatus\trc\twall_s\tstart_epoch\tend_epoch\tsplits_ok\tclient_key\treason\n"
                      "A08-082\t1\tFAILED\t1\t5\t0\t5\t0\ta08-llmfe-synp_p1-r0\trc=1\n"
                      "A08-082\t2\tOK\t0\t3600\t9\t3609\t5\ta08-llmfe-synp_p1-r0\tok\n"
                      "A08-084\t1\tCAP_HIT\ttimeout\t21600\t0\t21600\t2\ta08-llmfe-synp_p3-r0\tcap\n")
    totals = tmp_path / "totals.json"
    totals.write_text(json.dumps({"a08-llmfe-synp_p1-r0": {"calls": 63, "prompt_tokens": 100000, "completion_tokens": 50000,
                                                          "error_calls": 0, "http_400": 1, "finish_length": 2,
                                                          "usage_missing": 0}}))
    gp = alw.llmfe_payloads(alw.read_rows(sess), "S1020l", status, totals, None, None, None)
    by = {x["row"]["row_id"]: x for x in gp}
    assert by["A08-082"]["summary"]["status"] == "OK" and by["A08-082"]["summary"]["attempts"] == 2
    assert by["A08-082"]["summary"]["tok_compl_total"] == 50000 and by["A08-082"]["summary"]["context_errors"] == 3
    assert by["A08-084"]["summary"]["cap_hit"] == 1 and by["A08-084"]["summary"]["itt_imputed"] == 1
    assert by["A08-082"]["tags"] == ["S1020l", "v2", "llmfe:oss120"]


def test_a08_log_with_fake_wandb(tmp_path, monkeypatch):
    calls = []

    class Run:
        def __init__(self, **kw):
            calls.append(kw)
            self.summary = {}

        def log(self, *_a, **_k):
            pass

        def log_artifact(self, *_a):
            pass

        def finish(self):
            calls[-1]["summary"] = dict(self.summary)

    fake = types.SimpleNamespace(init=lambda **kw: Run(**kw), Table=lambda **kw: kw,
                                 Artifact=lambda *a, **k: types.SimpleNamespace(add_file=lambda *a, **k: None,
                                                                                add_dir=lambda *a, **k: None))
    monkeypatch.setitem(sys.modules, "wandb", fake)
    rows = mm.read_csv(mm.make_ablation("A08", None, None, manifests_dir=tmp_path))
    alw.log_payloads(alw.cpu_payloads(rows[:2], None, None), "RogueOne-AGENT2027", "PlanBench")
    assert [c["name"] for c in calls] == ["A08__base_no_fe__synp_p0__f42__r0", "A08__base_no_fe__synp_p1__f42__r0"]
    assert calls[0]["group"] == "A08" and calls[0]["job_type"] == "base_no_fe" and calls[0]["resume"] == "allow"
    assert calls[0]["id"] == "a08__base_no_fe__synp_p0__f42__r0" and calls[0]["summary"]["status"] == "MISSING"
