"""Post-freeze loop switches E9a and E9b (AGENT 2027 A10, A11).

Both are default-off: when unset (or empty) the code path is exactly Full-v2's.

``ROGUE_ONE_RESTART_MODE`` (E9a, A10 LF1)
    ``off`` (default): fold state is reset only at ``iteration_in_fold == 0``.
    ``every_iteration``: ``_reset_fold_state`` runs before **every** iteration, so each
    iteration is an independent first-round attempt; ``df_test_pool`` is never reset and
    inner-CV selection picks the best of the fold's restarts. Restarts also get distinct
    LLM seeds (restart seed rule, ``seed = llm_seed + index``; see
    ``rogueone.llm.clients.set_llm_seed_offset``).
``ROGUE_ONE_FEEDBACK_LEVEL`` (E9b, A11 LF2-L0)
    ``full`` (default): the Scientist sees the feedback pool and the full attribute table.
    ``l0``: open-loop planner. The Scientist's copy of the feedback pool keeps only ``id``,
    ``attribute_lookup_tool`` drops ``Status``/``Pruned Iteration``/``Prune Rationale``,
    and the Scientist's rendered prompt gets the seven local edits of
    ``rogueone.llm.agents.core.scientist.agent.L0_PROMPT_EDITS`` /
    ``apply_feedback_level_to_pandas_tool``.

Values are stripped and lower-cased (like ``ROGUE_ONE_SELECTION_SIGNAL``); anything
else raises ``ValueError`` at config load. Standard library only.
"""

from __future__ import annotations

from typing import Optional

RESTART_MODE_ENV = "ROGUE_ONE_RESTART_MODE"
RESTART_OFF = "off"
RESTART_EVERY_ITERATION = "every_iteration"
RESTART_MODES: tuple[str, ...] = (RESTART_OFF, RESTART_EVERY_ITERATION)

FEEDBACK_LEVEL_ENV = "ROGUE_ONE_FEEDBACK_LEVEL"
FEEDBACK_FULL = "full"
FEEDBACK_L0 = "l0"
FEEDBACK_LEVELS: tuple[str, ...] = (FEEDBACK_FULL, FEEDBACK_L0)


def _parse(env_name: str, value: Optional[str], choices: tuple[str, ...]) -> str:
    mode = (value if value is not None else "").strip().lower() or choices[0]
    if mode not in choices:
        raise ValueError(f"{env_name} must be one of {choices}, got {value!r}")
    return mode


def parse_restart_mode(value: Optional[str]) -> str:
    """Validate ``ROGUE_ONE_RESTART_MODE`` (unset/empty = ``off``)."""
    return _parse(RESTART_MODE_ENV, value, RESTART_MODES)


def parse_feedback_level(value: Optional[str]) -> str:
    """Validate ``ROGUE_ONE_FEEDBACK_LEVEL`` (unset/empty = ``full``)."""
    return _parse(FEEDBACK_LEVEL_ENV, value, FEEDBACK_LEVELS)
