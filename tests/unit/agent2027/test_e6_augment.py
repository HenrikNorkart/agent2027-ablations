"""E6 (AGENT 2027): augment mode, prune protection, raw_coverage, replay-validity check.

A03-T5 (a)-(g). The replace path must stay byte-identical to pre-E6 (no seeded rows,
no ``Seeded`` column, the pruning tool's return text unchanged).
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys

# rogue_one_main / the Tester need the LLM endpoint keys at import; values are irrelevant.
for _k, _v in {"LLM_API_KEY": "x", "LLM_PORT": "1"}.items():
    os.environ.setdefault(_k, _v)
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from rogueone.utils import flag_registry as fr
from rogueone.utils import instrumentation, rouge_one_cfg
from rogueone.utils.config import rogueoneConfig
from rogueone.utils.fold_safe_features import (
    REPLAY_NAN_FILLS_COUNTER,
    SEEDED_COLUMN,
    build_fold_safe_entities_folds,
    materialize_fold_safe_features,
    raw_coverage,
    seed_raw_feature_rows,
    seeded_attribute_mask,
)

pytestmark = [pytest.mark.unit]

ROOT = Path(__file__).resolve().parents[3]
ATTR_COLUMNS = [
    "Attribute",
    "Description",
    "Pandas Command",
    "Status",
    "Added",
    "Pruned Iteration",
    "Prune Rationale",
]


def _raw() -> pd.DataFrame:
    """Mixed raw frame: ids out of order, NaNs, bool, int, float, strings."""
    return pd.DataFrame(
        {
            "id": [7, 3, 11, 5, 2, 9],
            "age": [61.0, np.nan, 45.5, 50.0, 38.0, 70.0],
            "n_visits": [1, 4, 2, 0, 3, 3],
            "smoker": [True, False, False, True, False, True],
            "colour": ["red", "blue", None, "green", "blue", "red"],
            "o'brien": ["x", "y", "x", "y", "x", "x"],  # quote in the name -> repr()
        }
    )


def _empty_attrs() -> pd.DataFrame:
    return pd.DataFrame(columns=ATTR_COLUMNS)


def _ids(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(df["id"].unique(), columns=["id"])


@pytest.fixture
def pool_mode():
    saved = (rouge_one_cfg.feature_pool_mode, rouge_one_cfg.replay_validity_check)
    yield
    rouge_one_cfg.feature_pool_mode, rouge_one_cfg.replay_validity_check = saved


# --------------------------------------------------------------------------- flags


def test_flags_registered_with_code_defaults(monkeypatch):
    assert fr.FLAG_DEFAULTS["ROGUE_ONE_FEATURE_POOL_MODE"] == "replace"
    assert fr.FLAG_DEFAULTS["ROGUE_ONE_REPLAY_VALIDITY_CHECK"] == "0"
    monkeypatch.delenv("ROGUE_ONE_FEATURE_POOL_MODE", raising=False)
    monkeypatch.delenv("ROGUE_ONE_REPLAY_VALIDITY_CHECK", raising=False)
    cfg = object.__new__(rogueoneConfig)
    cfg.__init__()
    assert cfg.feature_pool_mode == "replace" and cfg.replay_validity_check is False
    monkeypatch.setenv("ROGUE_ONE_FEATURE_POOL_MODE", " Augment ")
    monkeypatch.setenv("ROGUE_ONE_REPLAY_VALIDITY_CHECK", "1")
    cfg.__init__()
    assert cfg.feature_pool_mode == "augment" and cfg.replay_validity_check is True
    monkeypatch.setenv("ROGUE_ONE_FEATURE_POOL_MODE", "both")
    with pytest.raises(ValueError):
        cfg.__init__()


def test_v2_defaults_freeze_augment():
    # E6 missed G-E6; the pre-declared fallback freezes augment by design (pre-reg, 2026-10-14).
    text = (ROOT / "scripts" / "agent2027" / "v2_defaults.env").read_text(encoding="utf-8")
    assert "ROGUE_ONE_FEATURE_POOL_MODE=augment" in text


# --------------------------------------------------------------------------- seeding


def test_seed_covers_every_raw_column_and_matches_prep():
    raw = _raw()
    raw_before = raw.copy()
    ent, attrs = seed_raw_feature_rows(raw, _ids(raw), _empty_attrs())
    pd.testing.assert_frame_equal(raw, raw_before)  # input not mutated
    assert ent["id"].tolist() == raw["id"].tolist()  # never reorders rows
    assert list(ent.columns) == ["id", "age", "n_visits", "smoker", "colour_ord", "o'brien_ord"]
    assert list(attrs["Attribute"]) == list(ent.columns[1:])
    assert set(attrs["Added"]) == {"Seed"} and set(attrs["Status"]) == {"Active"}
    assert attrs[SEEDED_COLUMN].dtype == bool and attrs[SEEDED_COLUMN].all()
    assert list(attrs.columns) == ATTR_COLUMNS + [SEEDED_COLUMN]
    assert raw_coverage(raw, attrs) == 1.0
    # values = base_no_fe input: missing -> 0, alphabetical codes, bool -> 0/1
    assert ent["age"].tolist() == [61.0, 0.0, 45.5, 50.0, 38.0, 70.0]
    assert ent["smoker"].tolist() == [1.0, 0.0, 0.0, 1.0, 0.0, 1.0]
    assert ent["colour_ord"].tolist() == [2.0, 0.0, 0.0, 1.0, 0.0, 2.0]  # blue 0, green 1, red 2
    assert all(ent[c].dtype == float for c in ent.columns[1:])
    sys.path.insert(0, str(ROOT / "scripts" / "agent2027"))
    try:
        import e6_seed_check as sc
    finally:
        sys.path.pop(0)
    ref = sc.emulated_reference(raw)
    for col in [c for c in raw.columns if c != "id"]:
        name = col if col in ent.columns else f"{col}_ord"
        np.testing.assert_allclose(ent[name].to_numpy(np.float32), ref[col].to_numpy(np.float32))


def test_seeded_pool_accepts_string_writes_into_empty_columns():
    # S1015k (2026-10-15): seeding into an empty pool left "Pruned Iteration" all-NaN
    # float64, and the Tester's pruning write of "Trial 0" raised TypeError (pandas 3).
    raw = _raw()
    attrs = _empty_attrs()
    if "Pruned Iteration" not in attrs.columns:
        attrs["Pruned Iteration"] = pd.Series(dtype=object)
    _, seeded = seed_raw_feature_rows(raw, _ids(raw), attrs)
    mask = seeded["Attribute"].isin(["age"])
    seeded.loc[mask, "Status"] = "Pruned"
    seeded.loc[mask, "Pruned Iteration"] = "Trial 0"
    assert seeded.loc[mask, "Pruned Iteration"].tolist() == ["Trial 0"]
    for col in seeded.columns:
        if col != SEEDED_COLUMN:
            assert seeded[col].dtype != float, col


def test_categorical_detection_uses_numeric_dtype_not_object():
    raw = _raw()
    raw["colour"] = raw["colour"].astype("string")  # not object; pandas 3 reads 'str'
    ent, _ = seed_raw_feature_rows(raw, _ids(raw), _empty_attrs())
    assert "colour_ord" in ent.columns and ent["colour_ord"].tolist()[0] == 2.0


def test_seed_commands_replay_identically_on_any_partition():
    raw = _raw()
    ent, attrs = seed_raw_feature_rows(raw, _ids(raw), _empty_attrs())
    before = instrumentation.get_counter(REPLAY_NAN_FILLS_COUNTER)
    for mask in (raw["id"].isin({3, 2}).to_numpy(), ~raw["id"].isin({3, 2}).to_numpy()):
        rep = materialize_fold_safe_features(raw, attrs, mask).set_index("id")
        want = ent.set_index("id").loc[rep.index]
        pd.testing.assert_frame_equal(rep[want.columns], want, check_dtype=False)
    assert instrumentation.get_counter(REPLAY_NAN_FILLS_COUNTER) == before


def test_replay_nan_fill_counter_counts_failed_replays():
    raw = _raw()
    attrs = pd.DataFrame(
        [{"Attribute": "bad", "Pandas Command": "result = df['nope']"},
         {"Attribute": "empty", "Pandas Command": ""}]
    )
    before = instrumentation.get_counter(REPLAY_NAN_FILLS_COUNTER)
    materialize_fold_safe_features(raw, attrs, np.ones(len(raw), dtype=bool))
    assert instrumentation.get_counter(REPLAY_NAN_FILLS_COUNTER) == before + 2


def test_seed_check_script_passes_with_emulated_reference(tmp_path, capsys):
    d = tmp_path / "classification" / "toy"
    d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    n = 60
    pd.DataFrame(
        {
            "id": rng.permutation(n) + 100,
            "x": np.where(rng.random(n) < 0.1, np.nan, rng.normal(size=n)),
            "k": rng.integers(0, 5, size=n),
            "cat": rng.choice(["b", "a", "c", None], size=n),
        }
    ).to_csv(d / "data.csv", index=False)
    sys.path.insert(0, str(ROOT / "scripts" / "agent2027"))
    try:
        import e6_seed_check as sc
    finally:
        sys.path.pop(0)
    rc = sc.main(["--tasks", str(tmp_path), "--datasets", "toy", "--emulate-ref"])
    out = capsys.readouterr().out
    assert rc == 0 and "E6_SEED_CHECK PASS_EMULATED_REF" in out
    # a reference file that disagrees fails
    ref = sc.emulated_reference(pd.read_csv(d / "data.csv"))
    ref["k"] = ref["k"] + 1
    ref.to_csv(tmp_path / "toy_prep.csv", index=False)
    rc = sc.main(["--tasks", str(tmp_path), "--ref", str(tmp_path), "--datasets", "toy"])
    assert rc == 1 and "E6_SEED_CHECK FAIL" in capsys.readouterr().out


# --------------------------------------------------------------------------- fold reset


def _stub_network(raw: pd.DataFrame):
    from rogue_one_main import RogueOneAgentNetwork

    stub = MagicMock(spec=RogueOneAgentNetwork)
    stub.df = raw
    stub.df_entities_attributes = _ids(raw)
    stub.df_attributes_explanations = _empty_attrs()
    stub.df_scientist_feedback_pool = pd.DataFrame()
    stub.focus_history = []
    stub.scientist_agent = MagicMock()
    stub.scientist_agent._note_book = ""
    return stub, RogueOneAgentNetwork


def test_replace_reset_is_id_only_and_byte_identical(pool_mode):
    rouge_one_cfg.feature_pool_mode = "replace"
    raw = _raw()
    stub, cls = _stub_network(raw)
    stub.df_entities_attributes["junk"] = 1.0
    cls._reset_fold_state(stub)
    assert list(stub.df_entities_attributes.columns) == ["id"]
    assert stub.df_entities_attributes["id"].tolist() == raw["id"].tolist()
    assert list(stub.df_attributes_explanations.columns) == ATTR_COLUMNS
    assert stub.df_attributes_explanations.empty
    # byte-identical pool serialisation to the pre-E6 reset
    pre_e6 = pd.DataFrame(columns=ATTR_COLUMNS)
    assert stub.df_attributes_explanations.to_csv(index=False) == pre_e6.to_csv(index=False)
    assert raw_coverage(raw, stub.df_attributes_explanations) == 0.0


def test_augment_reset_reseeds_on_every_call(pool_mode):
    rouge_one_cfg.feature_pool_mode = "augment"
    raw = _raw()
    stub, cls = _stub_network(raw)
    cls._reset_fold_state(stub)
    first = stub.df_entities_attributes.copy(), stub.df_attributes_explanations.copy()
    assert raw_coverage(raw, first[1]) == 1.0
    # mid-fold growth and a (forced) prune, then a later reset (E9a restart)
    stub.df_entities_attributes["agent_feat"] = 1.0
    stub.df_attributes_explanations.loc[0, "Status"] = "Pruned"
    cls._reset_fold_state(stub)
    pd.testing.assert_frame_equal(stub.df_entities_attributes, first[0])
    pd.testing.assert_frame_equal(stub.df_attributes_explanations, first[1])
    assert raw_coverage(raw, stub.df_attributes_explanations) == 1.0


def _net_for_pool(attrs: pd.DataFrame):
    from rogue_one_main import RogueOneAgentNetwork

    net = object.__new__(RogueOneAgentNetwork)
    net.df_attributes_explanations = attrs
    net.df_test_pool = None
    return net


def test_update_attribute_pool_never_prunes_seeded_rows(pool_mode):
    raw = _raw()
    _, attrs = seed_raw_feature_rows(raw, _ids(raw), _empty_attrs())
    net = _net_for_pool(attrs)
    incoming = pd.DataFrame(
        [
            {"Attribute": "age", "Status": "Pruned", "Prune Rationale": "x"},
            {"Attribute": "new_feat", "Description": "d", "Pandas Command": "c",
             "Status": "Active", "Added": "Trial 0"},
        ]
    )
    asyncio.run(net.update_attribute_pool(incoming, source_agent="tester"))
    out = net.df_attributes_explanations
    assert out.loc[out["Attribute"] == "age", "Status"].item() == "Active"
    assert out[SEEDED_COLUMN].dtype == bool
    assert not bool(out.loc[out["Attribute"] == "new_feat", SEEDED_COLUMN].item())
    assert raw_coverage(raw, out) == 1.0


def test_update_attribute_pool_replace_adds_no_seeded_column():
    net = _net_for_pool(_empty_attrs())
    incoming = pd.DataFrame(
        [{"Attribute": "f", "Description": "d", "Pandas Command": "c", "Status": "Active",
          "Added": "Trial 0"}]
    )
    asyncio.run(net.update_attribute_pool(incoming, source_agent="extractor"))
    assert list(net.df_attributes_explanations.columns) == ATTR_COLUMNS
    asyncio.run(net.update_attribute_pool(
        pd.DataFrame([{"Attribute": "f", "Status": "Pruned"}]), source_agent="tester"))
    assert net.df_attributes_explanations["Status"].tolist() == ["Pruned"]


def test_fold_safe_folds_carry_seeded_columns():
    raw = _raw()
    _, attrs = seed_raw_feature_rows(raw, _ids(raw), _empty_attrs())
    labels = pd.DataFrame({"id": raw["id"], "target": [0, 1, 0, 1, 0, 1]})
    folds = build_fold_safe_entities_folds(
        raw, attrs, labels, [{7, 3, 11}, {5, 2, 9}], current_test_fold_id=0
    )
    for f in folds:
        assert {"age", "colour_ord", "smoker"} <= set(f.columns)
        assert not f[list(attrs["Attribute"])].isna().any().any()


# --------------------------------------------------------------------------- Tester


def test_pruning_floor_ignores_seeded_rows():
    import rogueone.llm.agents.core.tester.agent as tst_mod

    names = [f"s{i}" for i in range(10)] + ["a0", "a1", "a2", "a3"]
    df = pd.DataFrame(
        {"Attribute": names, "Status": ["Active"] * 14, SEEDED_COLUMN: [True] * 10 + [False] * 4,
         "Pruned Iteration": pd.NA, "Prune Rationale": pd.NA}
    )
    before = df["Status"].copy()
    df.loc[df["Attribute"].isin(["a0", "a1", "a2"]), "Status"] = "Pruned"
    remaining, restored = tst_mod._enforce_pruning_floor(
        df, before, ["a0", "a1", "a2"], {"a2": 0.9, "a1": 0.5, "a0": 0.1}, {}
    )
    # floor = max(3, ceil(0.2 * 4)) = 3 over the 4 agent rows; 1 left -> restore 2
    assert restored == ["a2", "a1"] and remaining == ["a0"]
    # without the Seeded column the 10 extra rows would count and nothing is restored
    df2 = df.drop(columns=[SEEDED_COLUMN])
    df2.loc[df2["Attribute"].isin(["a1", "a2"]), "Status"] = "Pruned"
    _, restored2 = tst_mod._enforce_pruning_floor(
        df2, before, ["a0", "a1", "a2"], {}, {}
    )
    assert restored2 == []


def _run_tester_prune(tmp_path, monkeypatch, expl, prune):
    from agents.tool_context import ToolContext

    import rogueone.llm.agents.core.tester.agent as tst_mod

    cfg = SimpleNamespace(
        output_dir=Path(tmp_path), experiment_description="d", context_and_goal="g",
        experiment_name="e", task_type="classification", modality="tabular",
    )
    agent = tst_mod.TesterAgent(cfg, include_report=False, enable_wandb=False)
    replies = []

    class FakeRunner:
        @staticmethod
        async def run(a, *args, **kwargs):
            tools = {t.name: t for t in a.tools}
            if "attribute_pruning_tool" in tools:
                payload = json.dumps({"attribute_names_list": prune})
                ctx = ToolContext(context=None, tool_name="attribute_pruning_tool",
                                  tool_call_id="c", tool_arguments=payload)
                replies.append(await tools["attribute_pruning_tool"].on_invoke_tool(ctx, payload))
            return SimpleNamespace(final_output="pruned")

    monkeypatch.setattr(tst_mod, "Runner", FakeRunner)
    feats = list(expl["Attribute"])
    rng = np.random.default_rng(0)
    folds, nid = [], 0
    for _ in range(rouge_one_cfg.k_folds):
        d = pd.DataFrame(rng.normal(size=(12, len(feats))), columns=feats)
        d["target"] = rng.integers(0, 2, size=12)
        d["id"] = range(nid, nid + 12)
        nid += 12
        folds.append(d)
    _, out = asyncio.run(agent.test_hypotheses(folds, expl, step=0, trial_index=0))
    return out, replies


def test_pruning_tool_skips_seeded_rows(tmp_path, monkeypatch):
    names = ["s0", "s1", "f0", "f1", "f2", "f3"]
    expl = pd.DataFrame(
        {"Attribute": names, "Status": ["Active"] * 6,
         "Pruned Iteration": [pd.NA] * 6, "Prune Rationale": [pd.NA] * 6,
         SEEDED_COLUMN: [True, True, False, False, False, False]}
    )
    out, replies = _run_tester_prune(tmp_path, monkeypatch, expl, ["s0", "f0"])
    status = dict(zip(out["Attribute"], out["Status"]))
    assert status == {"s0": "Active", "s1": "Active", "f0": "Pruned",
                      "f1": "Active", "f2": "Active", "f3": "Active"}
    assert replies and "protected seeded raw columns" in replies[0] and "s0" in replies[0]


def test_pruning_tool_reply_unchanged_under_replace(tmp_path, monkeypatch):
    expl = pd.DataFrame(
        {"Attribute": ["f0", "f1", "f2", "f3"], "Status": ["Active"] * 4,
         "Pruned Iteration": [pd.NA] * 4, "Prune Rationale": [pd.NA] * 4}
    )
    out, replies = _run_tester_prune(tmp_path, monkeypatch, expl, ["f0"])
    assert replies == ["Attributes pruned successfully."]
    assert SEEDED_COLUMN not in out.columns


# --------------------------------------------------------------------------- replay check


def _load_diag():
    path = ROOT / "paper" / "scripts" / "diag_finding39_authoring_frame.py"
    spec = importlib.util.spec_from_file_location("_diag39_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _toy_run(tmp_path: Path, *, iterations: int = 2, corrupt_fold: int | None = None):
    """A task folder with an authoring trail per (fold, iteration) and a pool whose
    reported score is the authoring-frame score (or wrong for ``corrupt_fold``)."""
    diag = _load_diag()
    ds = tmp_path / "toy"
    out = tmp_path / "custom_out"
    (out / "intermediate_steps").mkdir(parents=True)
    ds.mkdir()
    rng = np.random.default_rng(1)
    n = 80
    ids = np.arange(n) + 1000
    x = rng.normal(size=n)
    y = (x + 0.3 * rng.normal(size=n) > 0.6).astype(int)  # imbalanced
    pd.DataFrame({"id": ids, "x": x}).to_csv(ds / "data.csv", index=False)
    pd.DataFrame({"id": ids, "target": y}).to_csv(ds / "labels.csv", index=False)
    fixed = [set(ids[i::2].tolist()) for i in range(2)]  # not what _fold_test_ids builds
    rows = []
    for fold in range(2):
        for it in range(iterations):
            d = pd.DataFrame({"id": ids, "x": x})
            d.to_csv(out / "intermediate_steps" / f"df_entities_attributes_trail_{fold * iterations + it}.csv", index=False)
            rows.append({"status": "completed", "test_fold_id": fold, "iteration_in_fold": it,
                         "inner_cv_accuracy": 0.9 if it == 0 else 0.5, "accuracy": np.nan})
    pool = pd.DataFrame(rows)
    pool.to_csv(out / "df_test_pool_final.csv", index=False)
    # fill reported scores with the authoring-frame scores on the run's own folds
    res = diag.analyse(ds, "classification", 2, 42, fixed_test_ids=fixed, out_dir=out,
                       iterations_per_fold=iterations, balanced=rouge_one_cfg.do_balancing,
                       scored_only=True, tester_row_order=True)
    for r in res:
        m = (pool["test_fold_id"] == r["fold"]) & (pool["iteration_in_fold"] == r["selected_iteration"])
        score = r["authoring_frame_score"]
        pool.loc[m, "accuracy"] = score - 0.2 if r["fold"] == corrupt_fold else score
    pool.to_csv(out / "df_test_pool_final.csv", index=False)
    return diag, ds, out, fixed


def test_diag_analyse_optional_args(tmp_path):
    diag, ds, out, fixed = _toy_run(tmp_path, corrupt_fold=1)
    kw = dict(fixed_test_ids=fixed, out_dir=out, iterations_per_fold=2,
              balanced=rouge_one_cfg.do_balancing, scored_only=True, tester_row_order=True)
    rows = diag.analyse(ds, "classification", 2, 42, **kw)
    assert [r["reproduced"] for r in rows] == [True, False]
    # balanced (sample-weighted) scoring differs from plain accuracy on imbalanced labels
    kw["balanced"] = not kw["balanced"]
    other = diag.analyse(ds, "classification", 2, 42, **kw)
    assert [r["authoring_frame_score"] for r in other] != [r["authoring_frame_score"] for r in rows]
    assert all(r["selected_iteration"] == 0 for r in rows)
    # defaults keep the original contract: <ds>/output, which does not exist here
    assert diag.analyse(ds, "classification", 2, 42) == []


def test_network_replay_validity_results(tmp_path, pool_mode):
    from rogue_one_main import RogueOneAgentNetwork

    _, ds, out, fixed = _toy_run(tmp_path, corrupt_fold=0)
    net = object.__new__(RogueOneAgentNetwork)
    net.cfg = SimpleNamespace(output_dir=out, feature_file_path=ds / "data.csv",
                              task_type="classification")
    net._tabular_fixed_test_ids = fixed
    net.enable_wandb = False
    saved = (rouge_one_cfg.k_folds, rouge_one_cfg.num_iterations)
    try:
        rouge_one_cfg.k_folds, rouge_one_cfg.num_iterations = 2, 2
        rouge_one_cfg.replay_validity_check = False
        res = net._replay_validity_results()
        assert set(res) == {"replay_nan_fills"}
        rouge_one_cfg.replay_validity_check = True
        res = net._report_replay_validity()
    finally:
        rouge_one_cfg.k_folds, rouge_one_cfg.num_iterations = saved
    assert res["replay_check_ran"] == 1, res
    assert res["replay_nonrepro_folds"] == 1 and res["replay_checked_folds"] == 2
    assert isinstance(res["replay_nan_fills"], int)
    assert json.loads((out / "replay_validity_summary.json").read_text())["replay_nonrepro_folds"] == 1
    assert (out / "replay_validity.csv").is_file()
