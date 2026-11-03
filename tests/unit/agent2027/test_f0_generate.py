"""X7 / A17-T7 F0 generator: frozen prompt, seed rule, parsing, folds, generation loop."""

import asyncio
import csv
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from rogueone.utils.train_test_splitter import (
    compute_entity_duplicate_groups,
    deterministic_kfold_indices,
)
from scripts.agent2027 import f0_generate as f0

pytestmark = [pytest.mark.unit]

REPO = Path(__file__).resolve().parents[3]
DRAFT = REPO / "docs" / "agent2027" / "X7_F0_prompt_DRAFT.md"
MANIFEST = REPO / "manifests" / "agent2027" / "A17_F0.csv"


def _draft_block(heading: str) -> str:
    text = DRAFT.read_text(encoding="utf-8")
    m = re.search(re.escape(heading) + r"\s*\n\s*```\n(.*?)\n```", text, flags=re.S)
    assert m, heading
    return m.group(1)


# --------------------------------------------------------------------------- frozen text
def test_system_prompt_equals_frozen_draft():
    assert f0.SYSTEM_PROMPT == _draft_block("## System message (DRAFT)")


def test_user_template_equals_frozen_draft():
    draft = _draft_block("## User message (DRAFT)")
    rendered = f0.USER_TEMPLATE.format(context_and_goal="{context_and_goal}", columns="{columns}",
                                       rows="{rows}")
    assert rendered == draft


def test_request_parameters_frozen():
    assert (f0.TEMPERATURE, f0.MAX_TOKENS, f0.N_DRAWS, f0.N_FOLDS) == (0.7, 32768, 128, 5)


# --------------------------------------------------------------------------- seed rule
def test_seed_rule_ranges():
    seeds42 = [f0.seed_for(42, f, j) for f in range(5) for j in range(128)]
    assert seeds42[0] == 42000 and seeds42[-1] == 42639 and len(set(seeds42)) == 640
    seeds4 = [f0.seed_for(4, f, j) for f in range(5) for j in range(128)]
    assert (min(seeds4), max(seeds4)) == (4000, 4639)
    with pytest.raises(ValueError):
        f0.seed_for(42, 5, 0)
    with pytest.raises(ValueError):
        f0.seed_for(42, 0, 128)


# --------------------------------------------------------------------------- parsing
ITEM = {"name": "a", "pandas_command": "result = df_features['id'].map(df.groupby('id')['x'].first())",
        "argumentation": "because"}


def test_parse_plain_list():
    p = f0.parse_answer(json.dumps([ITEM, ITEM]))
    assert p["ok"] and p["n_items"] == 2 and not p["fenced"] and not p["over_limit"]


def test_parse_fenced_list_accepted_and_flagged():
    p = f0.parse_answer("```json\n" + json.dumps([ITEM]) + "\n```")
    assert p["ok"] and p["fenced"] and p["n_items"] == 1
    p2 = f0.parse_answer("```\n" + json.dumps([ITEM]) + "\n```")
    assert p2["ok"] and p2["fenced"]


@pytest.mark.parametrize("text,err", [
    (None, "empty"), ("  ", "empty"), ("Here you go: []", "json"),
    (json.dumps({"name": "a"}), "not a list"), (json.dumps([1]), "not an object"),
    (json.dumps([{"name": "a"}]), "pandas_command"), (json.dumps([{"pandas_command": "x"}]), "name"),
])
def test_parse_failures(text, err):
    p = f0.parse_answer(text)
    assert not p["ok"] and err in p["error"]


def test_parse_over_limit_kept_and_flagged():
    p = f0.parse_answer(json.dumps([ITEM] * 11))
    assert p["ok"] and p["over_limit"] and p["n_items"] == 11


# --------------------------------------------------------------------------- task, rows, folds
def _toy_task(task_type: str):
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame({"id": np.arange(n)[::-1], "a": rng.normal(size=n), "b": rng.integers(0, 5, n),
                       "c": rng.choice(["x", "y"], n)})
    df.loc[[3, 4], ["a", "b", "c"]] = df.loc[[2, 2], ["a", "b", "c"]].to_numpy()  # duplicate raw rows
    y = rng.integers(0, 2, n) if task_type == "classification" else rng.normal(size=n)
    labels = pd.DataFrame({"id": np.arange(n), "target": y})
    return df, labels


@pytest.mark.parametrize("task_type,stratified", [("regression", False), ("classification", True),
                                                  ("classification", False)])
def test_outer_folds_match_independent_splitter_call(task_type, stratified):
    df, labels = _toy_task(task_type)
    folds = f0.outer_folds(df, labels, task_type, 42, stratified)
    ids = df["id"].unique().tolist()
    strat = None
    if stratified:
        strat = pd.DataFrame({"id": ids}).merge(labels, on="id", how="left")["target"].to_numpy()
    groups = compute_entity_duplicate_groups(df, ids, exclude_cols=("id",))
    ref = deterministic_kfold_indices(n_rows=len(ids), n_splits=5, random_state=42,
                                      stratify_labels=strat, dedup_groups=groups)
    assert [set(f["test_ids"]) for f in folds] == [{ids[i] for i in te} for _, te in ref]
    for f in folds:
        assert not set(f["train_ids"]) & set(f["test_ids"])
        assert len(f["train_ids"]) + len(f["test_ids"]) == len(ids)


def test_render_rows_first_five_training_rows_by_id_without_target():
    df, labels = _toy_task("regression")
    train_ids = [7, 3, 50, 1, 9, 22, 0]
    out = f0.render_rows(df, train_ids)
    back = pd.read_csv(pd.io.common.StringIO(out))
    assert back["id"].tolist() == [0, 1, 3, 7, 9]
    assert "target" not in back.columns
    assert f0.render_columns(df).splitlines()[0] == "- a (float64)"
    assert "- id" not in f0.render_columns(df)


def test_build_messages_shape():
    msgs = f0.build_messages("Goal text\n", "- a (float64)", "id,a\n0,1.0")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[1]["content"].startswith("# Task\nGoal text\n\n# Raw table `df`")
    assert "id,a\n0,1.0\n\n# How each feature" in msgs[1]["content"]
    assert msgs[1]["content"].rstrip().endswith('"argumentation": "<why this feature should help predict the target>"}, ...]')


# --------------------------------------------------------------------------- manifest
def test_manifest_rows():
    rows = list(csv.DictReader(open(MANIFEST, newline="", encoding="utf-8")))
    ids = [r["row_id"] for r in rows]
    expected = [f"A17-{n:03d}" for n in range(101, 178) if not 143 <= n <= 149]
    assert ids == expected
    for r in rows:
        assert r["arm_id"] == f"f0pool_{r['rung_S']}" and r["llm_seed"] == r["fold_seed"]
        assert r["image_tag"] == "agent2027-v2.0"
        strat = "ROGUE_ONE_STRATIFIED_FOLDS" in r["env_overrides"]
        assert strat == (r["task_type"] == "classification")
    o120 = [r for r in rows if r["rung_S"] == "O120"]
    assert [r["row_id"] for r in o120] == [f"A17-{n}" for n in range(164, 178)]
    assert f0.load_row(MANIFEST, "A17-164")["dataset_id"] == "synp_p1"


def test_load_row_refuses_non_f0(tmp_path):
    m = tmp_path / "m.csv"
    with open(m, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(f0.R7_HEADER)
        w.writerow(["A17-031", "A17", "f2_Q4", "synp_p2", "regression", 42, 42, 0, 0, "", "",
                    "agent2027-v2.0", "Q4", "Q4", "Q4", 8032, 8032, 8032, "{}", 6, 0.75, ""])
        w.writerow(["A17-164", "A17", "f0pool_O120", "covtype", "classification", 42, 42, 0, 0, "", "",
                    "agent2027-v2.0", "O120", "O120", "O120", 8010, 8010, 8010, "{}", 6, 0.25, ""])
    with pytest.raises(ValueError):
        f0.load_row(m, "A17-031")
    with pytest.raises(ValueError, match="covtype"):
        f0.load_row(m, "A17-164")
    with pytest.raises(KeyError):
        f0.load_row(m, "A17-165")


# --------------------------------------------------------------------------- generation loop
class FakeClient:
    def __init__(self, fail_seeds=(), infra_once=()):
        self.calls = []
        self.fail_seeds = set(fail_seeds)
        self.infra_once = set(infra_once)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, **kw):
        self.calls.append(kw)
        seed = kw["seed"]
        if seed in self.infra_once:
            self.infra_once.discard(seed)
            raise ConnectionError("Connection refused")
        text = "not json" if seed in self.fail_seeds else "```json\n" + json.dumps([ITEM]) + "\n```"
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=10))


def _msgs(n_folds=2):
    return [f0.build_messages("g", "- a (float64)", f"id,a\n{f},1") for f in range(n_folds)]


def test_generate_seeds_parameters_and_summary(tmp_path):
    client = FakeClient(fail_seeds={42001})
    recs = asyncio.run(f0.generate(client, model="m", llm_seed=42, fold_messages=_msgs(),
                                   out=tmp_path, n_draws=3, concurrency=2))
    assert len(recs) == 6 and len(client.calls) == 6
    assert sorted(c["seed"] for c in client.calls) == [42000, 42001, 42002, 42128, 42129, 42130]
    for c in client.calls:
        assert c["temperature"] == 0.7 and c["max_tokens"] == 32768 and "tools" not in c
    s = f0.summarise(recs, n_expected=6, wall_s=1.0)
    assert s["complete"] and s["n_parse_ok"] == 5 and s["f0_parse_rate"] == round(5 / 6, 4)
    assert s["tok_compl_total"] == 60 and s["fenced_share"] == round(5 / 6, 4)
    lines = (tmp_path / "draws.jsonl").read_text().splitlines()
    assert len(lines) == 6


def test_generate_resumes_and_requeues_infra_once(tmp_path):
    client = FakeClient(infra_once={42002})
    recs = asyncio.run(f0.generate(client, model="m", llm_seed=42, fold_messages=_msgs(1),
                                   out=tmp_path, n_draws=3))
    r2 = [r for r in recs if r["draw"] == 2][0]
    assert r2["attempt"] == 2 and r2["parse_ok"] and r2["error_kind"] is None
    assert len(client.calls) == 4
    client2 = FakeClient()
    again = asyncio.run(f0.generate(client2, model="m", llm_seed=42, fold_messages=_msgs(1),
                                    out=tmp_path, n_draws=3))
    assert client2.calls == [] and len(again) == 3


def test_main_dry_run_writes_folds_and_prompts(tmp_path):
    tasks = tmp_path / "tasks" / "regression" / "synp_p1"
    tasks.mkdir(parents=True)
    df, labels = _toy_task("regression")
    df.to_csv(tasks / "data.csv", index=False)
    labels.to_csv(tasks / "labels.csv", index=False)
    (tasks / "config.yml").write_text("context_and_goal: 'Predict y.'\n", encoding="utf-8")
    out = tmp_path / "out"
    rc = f0.main(["--manifest", str(MANIFEST), "--row", "A17-164", "--tasks", str(tmp_path / "tasks"),
                  "--out", str(out), "--dry-run"])
    assert rc == 0
    folds = json.loads((out / "folds.json").read_text())
    assert len(folds["test_ids"]) == 5 and folds["fold_seed"] == 42
    prompts = (out / "prompts.jsonl").read_text().splitlines()
    assert len(prompts) == 5
    assert json.loads(prompts[0])["messages"][1]["content"].startswith("# Task\nPredict y.\n")
    assert not (out / "draws.jsonl").exists()
