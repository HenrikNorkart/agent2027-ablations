"""The one shared ``context_and_goal`` template for all SYN-P tasks (E8a).

The context is the only domain input to the agent prompts.  Every task uses the
same template; only the slots differ.  The context never names phi or its
formula (for P7 it names the fictional domain term, never its definition).
The neutral variant used by the P6b twin fills the *same* template with opaque
slots and is padded to the named variant's length (+-2%).
"""

from __future__ import annotations

from collections.abc import Sequence

CONTEXT_TEMPLATE = (
    "{intro} "
    "The table has {n_rows} records, one per {unit}, described by {n_cols} columns.\n"
    "Columns:\n{column_block}\n"
    "Goal: {goal}"
)

REGRESSION_GOAL = (
    "predict {target} (a numeric target; it has been standardised, so only its "
    "relative scale is meaningful)."
)
CLASSIFICATION_GOAL = "predict {target} ({classes})."

NEUTRAL_INTRO = "Anonymised tabular records from an undisclosed process."
NEUTRAL_UNIT = "observation"
NEUTRAL_TARGET = "the target variable"
NEUTRAL_NUMERIC = "numeric measurement"
NEUTRAL_CATEGORICAL = "categorical attribute"
# padding sentences, appended to the neutral intro (neutral content only)
NEUTRAL_FILLER = (
    "Values are recorded as plain numbers or short codes.",
    "Rows are independent observations.",
    "No units or measurement conventions are disclosed.",
    "Missing values were not imputed.",
    "The records were collected under comparable conditions.",
    "Column order carries no meaning.",
)


def column_block(columns: Sequence[tuple[str, str]]) -> str:
    return "\n".join(f"- {name}: {desc}" for name, desc in columns)


def render_context(
    *,
    intro: str,
    unit: str,
    n_rows: int,
    columns: Sequence[tuple[str, str]],
    task_type: str,
    target: str,
    classes: str = "",
) -> str:
    goal = (
        REGRESSION_GOAL.format(target=target)
        if task_type == "regression"
        else CLASSIFICATION_GOAL.format(target=target, classes=classes)
    )
    return CONTEXT_TEMPLATE.format(
        intro=intro, unit=unit, n_rows=n_rows, n_cols=len(columns),
        column_block=column_block(columns), goal=goal,
    )


def render_neutral_context(
    *,
    n_rows: int,
    columns: Sequence[tuple[str, bool]],
    task_type: str,
    target_length: int,
    tolerance: float = 0.02,
    classes: str = "",
) -> str:
    """Neutral context from the same template, padded to ``target_length``
    characters (within ``tolerance``).  ``columns`` = (opaque name, is_categorical).

    Raises ValueError if even the unpadded text is longer than the allowance."""
    cols = [(n, NEUTRAL_CATEGORICAL if cat else NEUTRAL_NUMERIC) for n, cat in columns]

    def build(extra: str) -> str:
        return render_context(
            intro=(NEUTRAL_INTRO + (" " + extra if extra else "")),
            unit=NEUTRAL_UNIT, n_rows=n_rows, columns=cols, task_type=task_type,
            target=NEUTRAL_TARGET, classes=classes,
        )

    lo, hi = int(target_length * (1 - tolerance)), int(target_length * (1 + tolerance))
    text = build("")
    if len(text) > hi:
        raise ValueError(f"neutral context ({len(text)} chars) exceeds the allowance ({hi})")
    # aim at the exact length: add whole filler sentences, then a trimmed one
    extra = ""
    i = 0
    while len(build(extra)) < target_length and i < 400:
        sentence = NEUTRAL_FILLER[i % len(NEUTRAL_FILLER)]
        cand = (extra + " " + sentence).strip()
        if len(build(cand)) <= hi:
            extra = cand
        else:
            # trim this sentence word by word to land inside the window
            words = sentence.split()
            for k in range(len(words), 0, -1):
                cand = (extra + " " + " ".join(words[:k])).strip()
                if len(build(cand)) <= target_length:
                    extra = cand
                    break
            break
        i += 1
    text = build(extra)
    if not (lo <= len(text) <= hi):
        raise ValueError(f"could not pad neutral context into [{lo}, {hi}] (got {len(text)})")
    return text
