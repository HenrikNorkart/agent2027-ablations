"""E1c make_manifest.py: R7 header, unique row ids, covtype rejection, --plan-session."""

from __future__ import annotations

import csv
import importlib.util
import io
import json
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit]

REPO = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("make_manifest", REPO / "scripts" / "agent2027" / "make_manifest.py")
mm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mm)

R7_DOC = ("row_id, ablation_id, arm_id, dataset_id, task_type, fold_seed, llm_seed, replicate, "
          "is_sentinel, session_id, order_key, image_tag, rung_S, rung_E, rung_T, port_S, port_E, "
          "port_T, env_overrides, cap_hours, expected_u, notes")


def _spec_a02():
    import yaml
    return yaml.safe_load((REPO / "scripts" / "agent2027" / "manifest_specs" / "A02.yaml").read_text())


def _header(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()[0].split(",")


def test_header_is_r7():
    assert mm.R7_HEADER == [c.strip() for c in R7_DOC.split(",")]


def test_a02_manifest(tmp_path):
    out = mm.make_ablation("A02", None, None, manifests_dir=tmp_path, image_tag="agent2027-pre.1")
    assert out.name == "A02_M4.csv"
    assert _header(out) == mm.R7_HEADER
    rows = mm.read_csv(out)
    ids = [r["row_id"] for r in rows]
    assert len(ids) == 26 and len(set(ids)) == 26
    assert ids[:24] == [f"A02-{n:03d}" for n in range(1, 25)] and ids[24:] == ["A02-901", "A02-902"]
    for arm, lo in (("R10", 1), ("R3", 9), ("R1", 17)):
        block = [r for r in rows if r["arm_id"] == arm and r["notes"] != "smoke"]
        assert [r["dataset_id"] for r in block] == list(mm.SYNP8)
        assert all(json.loads(r["env_overrides"])["ROGUE_ONE_EXTRACTOR_ROUNDS"] == arm[1:] for r in block)
    for r in rows:
        ov = json.loads(r["env_overrides"])
        assert (ov.get("ROGUE_ONE_STRATIFIED_FOLDS") == "True") == (r["task_type"] == "classification")
        assert r["fold_seed"] == r["llm_seed"] == "42" and r["image_tag"] == "agent2027-pre.1"
        assert r["session_id"] == "" and r["order_key"] == ""
    smoke = [r for r in rows if r["notes"] == "smoke"]
    assert [json.loads(r["env_overrides"])["ROGUE_ONE_SMOKE_FOLDS"] for r in smoke] == ["1", "1"]
    mm.validate_rows(rows)  # round-trips


def test_duplicate_row_ids_rejected():
    spec = _spec_a02()
    spec["blocks"][1]["start"] = 5  # R3 block now overlaps R10's 005-008
    with pytest.raises(mm.ManifestError, match="duplicate row_id"):
        mm.expand_spec(spec)


@pytest.mark.parametrize("ds", ["covtype", "CovType"])
def test_covtype_rejected_everywhere(tmp_path, ds):
    spec = _spec_a02()
    spec["blocks"][0]["datasets"] = ["synp_p1", ds]
    with pytest.raises(mm.ManifestError, match="covtype"):
        mm.expand_spec(spec)
    row = {c: "" for c in mm.R7_HEADER}
    row.update(row_id="A05-001", ablation_id="A05", arm_id="full", dataset_id=ds,
               task_type="classification", fold_seed="42", llm_seed="42", replicate="0",
               is_sentinel="0", env_overrides="{}", cap_hours="6")
    with pytest.raises(mm.ManifestError, match="covtype"):
        mm.validate_row(row)
    # a hand-edited manifest with covtype cannot be planned into a session either
    mm.write_csv(tmp_path / "A05_M0.csv", [row])
    with pytest.raises(mm.ManifestError, match="covtype"):
        mm.plan_session("S1016a", ["A05-001"], manifests_dir=tmp_path, log=lambda *_: None)


def test_no_dataset_set_contains_covtype():
    for name, members in mm.DATASET_SETS.items():
        assert "covtype" not in members, name
    assert "covtype" not in mm.TASK_TYPES


def test_env_overrides_validation():
    assert mm.canonical_overrides('{"ROGUE_ONE_EXTRACTOR_ROUNDS": 3}') == '{"ROGUE_ONE_EXTRACTOR_ROUNDS":"3"}'
    assert mm.canonical_overrides("") == "{}"
    for bad in ('{"NOT_A_FLAG":"1"}', '{"ROGUE_ONE_DATASETS":"heart"}', '{"ROGUE_ONE_FOLD_SEED":"4"}',
                '{"ROGUE_ONE_PLANNER_MODE":"a\\"b"}', '["x"]', "{nope"):
        with pytest.raises(mm.ManifestError):
            mm.canonical_overrides(bad)


def test_row_token_expansion():
    avail = [f"A02-{n:03d}" for n in range(1, 25)] + ["A02-901", "A07-001", "A07-002", "A06-101"]
    assert mm.expand_row_tokens(["A02-001", "…", "A02-003"], avail) == ["A02-001", "A02-002", "A02-003"]
    assert mm.expand_row_tokens(["A02-022…024", "A07-001...A07-002"], avail) == [
        "A02-022", "A02-023", "A02-024", "A07-001", "A07-002"]
    assert mm.expand_row_tokens(["A02-901 A06-101"], avail) == ["A02-901", "A06-101"]
    with pytest.raises(mm.ManifestError):
        mm.expand_row_tokens(["A02-999"], avail)
    with pytest.raises(mm.ManifestError):
        mm.expand_row_tokens(["A02-001", "A02-001"], avail)


def _sentinel_manifest(tmp_path):
    row = {c: "" for c in mm.R7_HEADER}
    row.update(row_id="A06-001", ablation_id="A06", arm_id="sentinel", dataset_id="synp_p3",
               task_type="classification", fold_seed="42", llm_seed="5001", replicate="1",
               is_sentinel="1", env_overrides="{}", cap_hours="6", image_tag="agent2027-pre.1",
               rung_S="O120", rung_E="O120", rung_T="O120", port_S="8010", port_E="8010", port_T="8010")
    mm.write_csv(tmp_path / "A06_M0-S.csv", [mm.validate_row(row)])


def test_plan_session(tmp_path):
    mm.make_ablation("A02", None, None, manifests_dir=tmp_path, image_tag="agent2027-pre.1")
    _sentinel_manifest(tmp_path)
    logs = []
    out = mm.plan_session("S1013a", ["A02-001", "…", "A02-024", "A06-001"],
                          manifests_dir=tmp_path, log=logs.append)
    assert out == tmp_path / "sessions" / "S1013a.csv"
    assert _header(out) == mm.R7_HEADER
    rows = mm.read_csv(out)
    assert len(rows) == 25 and {r["session_id"] for r in rows} == {"S1013a"}
    keys = [int(r["order_key"]) for r in rows]
    assert keys == sorted(keys) and sorted(keys) == list(range(1, 26))  # file in run order
    assert [r["row_id"] for r in rows] != sorted(r["row_id"] for r in rows)  # interleaved
    mm.validate_rows(rows, session=True)
    # seeded by the session id: re-planning gives the same order; another id differs
    order1 = [r["row_id"] for r in rows]
    mm.plan_session("S1013a", ["A02-001…024", "A06-001"], manifests_dir=tmp_path, log=logs.append)
    assert [r["row_id"] for r in mm.read_csv(out)] == order1
    other = mm.plan_session("S1013b", ["A02-001…024", "A06-001"], manifests_dir=tmp_path, log=logs.append)
    assert [r["row_id"] for r in mm.read_csv(other)] != order1
    plan = (tmp_path / "sessions" / "S1013a.plan.txt").read_text()
    assert f"seed={mm.session_seed('S1013a')}" in plan and "sentinels=A06-001" in plan
    # container names unique per session and row
    names = {mm.container_name("S1013a", r) for r in order1}
    assert len(names) == 25 and "ro27-s1013a-a02-001" in names


def test_plan_session_sentinel_first_and_warning(tmp_path):
    mm.make_ablation("A02", None, None, manifests_dir=tmp_path, image_tag="agent2027-pre.1")
    _sentinel_manifest(tmp_path)
    out = mm.plan_session("S1013a", ["A02-001…024", "A06-001"], sentinel_first=True,
                          manifests_dir=tmp_path, log=lambda *_: None)
    rows = mm.read_csv(out)
    assert rows[0]["row_id"] == "A06-001" and rows[0]["order_key"] == "1"
    logs = []
    mm.plan_session("S1012a", ["A02-901", "A02-902"], manifests_dir=tmp_path, log=logs.append)
    assert any("no sentinel" in m for m in logs)
    logs.clear()
    mm.plan_session("S1012a", ["A02-901", "A02-902"], no_sentinel=True, manifests_dir=tmp_path, log=logs.append)
    assert not any("no sentinel" in m for m in logs)


def test_cli(tmp_path, capsys):
    assert mm.main(["--ablation", "A02", "--manifests-dir", str(tmp_path), "--image-tag", "agent2027-pre.2"]) == 0
    assert mm.main(["--plan-session", "S1012a", "--rows", "A02-901", "A02-902", "--no-sentinel",
                    "--manifests-dir", str(tmp_path)]) == 0
    assert mm.main(["--check", str(tmp_path / "A02_M4.csv"), str(tmp_path / "sessions" / "S1012a.csv")]) == 0
    assert mm.main(["--plan-session", "bad", "--rows", "A02-901", "--manifests-dir", str(tmp_path)]) == 2
    assert "OK" in capsys.readouterr().out


def test_committed_manifests_are_valid():
    for path in sorted((REPO / "manifests" / "agent2027").glob("A??_*.csv")):
        if path.name.startswith("A01_"):
            continue  # A01 runs on the legacy launcher (image nestedcv9, R7 shape only)
        mm.validate_rows(mm.read_csv(path))
    for path in sorted((REPO / "manifests" / "agent2027" / "sessions").glob("*.csv")):
        mm.validate_rows(mm.read_csv(path), session=True)


def _mixed_row(**kw):
    row = {"row_id": "A19-001", "ablation_id": "A19", "arm_id": "mix_esmall", "dataset_id": "synp_p1",
           "task_type": "regression", "fold_seed": "42", "llm_seed": "42", "replicate": "0",
           "is_sentinel": "0", "session_id": "", "order_key": "", "image_tag": "agent2027-v2.3",
           "rung_S": "Q122", "rung_E": "Q4", "rung_T": "Q122", "port_S": "8040", "port_E": "8032",
           "port_T": "8040", "cap_hours": "6", "expected_u": "0.85", "notes": "",
           "env_overrides": '{"LLM_MODEL":"Qwen/Qwen3.5-122B-A10B-FP8","LLM_PORT":"8040"}'}
    row.update(kw)
    return row


def test_mixed_global_only_on_mixed_rows():
    """SR4 (2026-10-19): LLM_MODEL/LLM_PORT in env_overrides only on mixed rows, as a pair."""
    out = mm.validate_row(_mixed_row())
    assert json.loads(out["env_overrides"])["LLM_PORT"] == "8040"
    with pytest.raises(mm.ManifestError, match="mixed-routing"):
        mm.validate_row(_mixed_row(rung_E="Q122", port_E="8040"))
    with pytest.raises(mm.ManifestError, match="together"):
        mm.validate_row(_mixed_row(env_overrides='{"LLM_MODEL":"Qwen/Qwen3.5-4B"}'))
    with pytest.raises(mm.ManifestError, match="port number"):
        mm.validate_row(_mixed_row(env_overrides='{"LLM_MODEL":"m","LLM_PORT":"x80"}'))
    with pytest.raises(mm.ManifestError):  # other launcher-owned keys stay refused
        mm.validate_row(_mixed_row(env_overrides='{"ROGUE_ONE_EXTRACTOR_PORT":"8032"}'))


def test_rp1set_rows_are_mixed_only():
    row = _mixed_row(row_id="A18-001", ablation_id="A18", arm_id="replay_Q0.8", dataset_id="rp1set",
                     task_type="mixed", llm_seed="7042", rung_S="Q0.8", rung_E="Q0.8", rung_T="Q0.8",
                     port_S="8030", port_E="8030", port_T="8030", env_overrides="{}")
    assert mm.validate_row(row)["task_type"] == "mixed"
    with pytest.raises(mm.ManifestError):
        mm.validate_row({**row, "task_type": "regression"})
    with pytest.raises(mm.ManifestError):
        mm.validate_row({**row, "dataset_id": "synp_p1"})


def test_plan_session_port_stamp_skips_mixed_rows(tmp_path):
    mm.make_ablation("A19", None, None, manifests_dir=tmp_path)
    mm.make_ablation("A20", None, None, manifests_dir=tmp_path)
    out = mm.plan_session("S1022a", ["A19-001", "A19-019", "A20-001"], port=8099, no_sentinel=True,
                          manifests_dir=tmp_path, log=lambda *_: None)
    rows = {r["row_id"]: r for r in mm.read_csv(out)}
    assert (rows["A19-001"]["port_S"], rows["A19-001"]["port_E"]) == ("8040", "8032")
    assert (rows["A19-019"]["port_S"], rows["A19-019"]["port_E"]) == ("8030", "8040")
    assert rows["A20-001"]["port_E"] == "8099"  # uniform rows are stamped


def test_sr4_manifests():
    """A17 drops f2_O20 and the CL2 rows; A19 is CL1 (SYN-P only) on the SR4 rungs; A20 on Q4."""
    a17 = {r["row_id"]: r for r in mm.read_csv(REPO / "manifests" / "agent2027" / "A17_CAP-1.csv")}
    assert len(a17) == 68 and not any(r["rung_E"] == "O20" for r in a17.values())
    assert not set(a17) & {"A17-016", "A17-017", "A17-027", "A17-028", "A17-045", "A17-046",
                           "A17-056", "A17-057"}
    anchor = a17["A17-031"]
    assert (anchor["arm_id"], anchor["dataset_id"], anchor["fold_seed"], anchor["replicate"]) == (
        "f2_Q4", "synp_p2", "42", "0")
    a19 = mm.read_csv(REPO / "manifests" / "agent2027" / "A19_CAP-2.csv")
    assert len(a19) == 28 and all(r["dataset_id"].startswith("synp_") for r in a19)
    for r in a19:
        ov = json.loads(r["env_overrides"])
        assert ov["LLM_MODEL"] == "Qwen/Qwen3.5-122B-A10B-FP8" and ov["LLM_PORT"] == "8040"
        assert ov["ROGUE_ONE_WEBSEARCH_ROUTING"] == "caller"
        want = ("Q122", "Q4", "Q122") if r["arm_id"] == "mix_esmall" else ("Q0.8", "Q122", "Q0.8")
        assert (r["rung_S"], r["rung_E"], r["rung_T"]) == want
    a20 = mm.read_csv(REPO / "manifests" / "agent2027" / "A20_CAP-3.csv")
    assert {r["rung_E"] for r in a20} == {"Q4"} and len(a20) == 8
