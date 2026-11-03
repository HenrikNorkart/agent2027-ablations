"""Offline harness that renders (and optionally drives) one Scientist call (E9a/E9b tests).

``render_scientist`` runs the real ``ScientistAgent.determine_focus`` with ``Runner``
replaced by a fake that records the agent's rendered instructions, first user message
and tools, optionally invokes tools (``tool_script``), and returns a fixed focus. With
``dump_dir`` set the call also writes the E5 prompt dump (``ROGUE_ONE_PROMPT_DUMP=1``)
so ``scripts/agent2027/prompt_diff.py`` can compare two renderings byte for byte.

The module only uses names that exist at ``agent2027-v2.0`` as well, so the same file
renders the pre-E9 golden prompts (``tests/unit/agent2027/fixtures/e9_full_v2``).
Not collected by pytest (no ``test_`` prefix).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable
from unittest.mock import MagicMock, patch

import pandas as pd

FIXED_CONTEXT = (
    "Synthetic harness task. Predict the binary label from the numeric columns a, b, c "
    "and the categorical column cat."
)
FIXED_FOCUS = "Focus: build ratio features between a and b and interact them with cat."


def make_raw_df(n: int = 12) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": list(range(n)),
            "a": [float(i + 1) for i in range(n)],
            "b": [float((i % 4) + 1) for i in range(n)],
            "c": [float(i % 3) for i in range(n)],
            "cat": ["x" if i % 2 else "y" for i in range(n)],
        }
    )


def _new_scientist(output_dir: Path, task_type: str = "classification"):
    from rogueone.llm.agents.core.scientist.agent import ScientistAgent
    from collections import defaultdict

    sci = ScientistAgent.__new__(ScientistAgent)
    sci.cfg_experiment = SimpleNamespace(
        output_dir=Path(output_dir), context_and_goal=FIXED_CONTEXT, task_type=task_type
    )
    sci.knowledge_agent = MagicMock()
    sci.web_search_agent = MagicMock()
    sci._note_book = ""
    sci._tool_call_counter = defaultdict(int)
    sci._tool_call_counter_success = defaultdict(int)
    sci._tool_call_counter_failed = defaultdict(int)
    sci._tool_error_reporter = None
    return sci


async def render_scientist(
    output_dir: Path | str,
    *,
    df_raw: pd.DataFrame | None = None,
    feedback_pool: pd.DataFrame | None = None,
    attributes: pd.DataFrame | None = None,
    focus_history: Iterable[str] = (),
    index: int = 0,
    max_index: int = 10,
    iteration_in_fold: int | None = 0,
    tool_script: Iterable[tuple[str, dict]] = (),
    dump: bool = False,
    scientist: Any = None,
    on_run: Any = None,
) -> dict:
    """Run one ``determine_focus`` offline; return what the model would have seen."""
    from agents.tool_context import ToolContext

    import rogueone.llm.agents.core.scientist.agent as sci_mod
    from rogueone.utils import prompt_dump

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df_raw = make_raw_df() if df_raw is None else df_raw
    if attributes is None:
        attributes = pd.DataFrame(
            columns=[
                "Attribute", "Description", "Pandas Command", "Status", "Added",
                "Pruned Iteration", "Prune Rationale",
            ]
        )
    sci = scientist if scientist is not None else _new_scientist(output_dir)
    seen: dict = {"tool_outputs": []}
    script = list(tool_script)

    class FakeRunner:
        async def run(self, agent, prompt, **kwargs):
            seen["instructions"] = agent.instructions
            seen["prompt"] = prompt
            seen["tools"] = {t.name: t for t in agent.tools}
            if on_run is not None:
                on_run(agent, prompt)
            for name, args in script:
                payload = json.dumps(args)
                ctx = ToolContext(
                    context=None, tool_name=name, tool_call_id="c", tool_arguments=payload
                )
                out = await seen["tools"][name].on_invoke_tool(ctx, payload)
                seen["tool_outputs"].append((name, args, out))
            return SimpleNamespace(final_output=FIXED_FOCUS)

    env = {"ROGUE_ONE_PROMPT_DUMP": "1" if dump else "0"}
    with patch.dict(os.environ, env), patch.object(sci_mod, "Runner", FakeRunner), patch.object(
        sci_mod, "create_openai_client_for_vllm", lambda **_: MagicMock()
    ):
        prompt_dump.reset_counters()
        focus = await sci.determine_focus(
            df_raw,
            feedback_pool,
            attributes,
            list(focus_history),
            index=index,
            max_index=max_index,
            iteration_in_fold=iteration_in_fold,
        )
    seen["focus"] = focus
    seen["note_book"] = sci._note_book
    return seen


def example_feedback_pool(signal: str = "inner_cv_l1") -> pd.DataFrame:
    """Two earlier trials, as ``_build_scientist_feedback_row`` writes them."""
    base = {
        "status": "completed", "dataset_id": "synp_p3", "fold_seed": 42,
        "aborted_fold_id": None, "skip_reason": None,
    }
    rows = []
    for i, (acc, report) in enumerate(
        [(0.731, "SECRET_REPORT_ZERO: a_div_b has SHAP 0.41"),
         (0.874, "SECRET_REPORT_ONE: a_div_b_x_cat dominates")]
    ):
        row = {"id": f"Trial {i}", **base, "feature_assessment": report}
        if signal == "inner_cv_l1":
            row.update({"cv_accuracy": acc, "cv_RMSE": None})
        else:
            row.update({
                "train_accuracy": acc, "train_precision": acc, "train_recall": acc,
                "train_f1": acc, "train_AUROC": acc, "train_MSE": None, "train_MAE": None,
                "train_R2": None,
            })
        rows.append(row)
    return pd.DataFrame(rows)


def example_attributes() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Attribute": "a_div_b", "Description": "ratio a/b", "Pandas Command": "df['a']/df['b']",
             "Status": "Active", "Added": "Trial 0", "Pruned Iteration": None, "Prune Rationale": None},
            {"Attribute": "c_sq", "Description": "c squared", "Pandas Command": "df['c']**2",
             "Status": "Pruned", "Added": "Trial 0", "Pruned Iteration": "Trial 1",
             "Prune Rationale": "SECRET_PRUNE_RATIONALE low MI"},
        ]
    )
