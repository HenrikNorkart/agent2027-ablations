"""SET-phased prompt construction (AGENT 2027 E9f / A16-T6). Pure text, stdlib only.

Construction rule (A16 §5.2 A16-T6; frozen before any core run):

* **Neutral system prompt**, identical for every phase of an iteration
  (:func:`system_prompt`), in this order:

  1. ``AgentRoles.GLOBAL_GOAL`` without its first line ("You are part of a team of
     agents…"), with rows G1/G2 of :data:`SUBSTITUTIONS` applied;
  2. :data:`PHASES_LINE`;
  3. ``# Task context and predictive goal:`` + ``context_and_goal``;
  4. the Extractor's raw-column block (header line + :func:`raw_columns_str`);
  5. "Do not provide long chain-of-thought narratives." (the Scientist's last output rule).

* **Phase messages** (:func:`phase_message`): :data:`PHASE_HEADER` + the role block +
  the phase's trailing input, separated by blank lines. The role block is cut out of the
  role's **rendered Full-v2 instructions** line by line (indentation included) between
  the anchors of :data:`ROLE_BLOCKS`, so it is byte-identical to what Full sends,
  except for the rows of :data:`SUBSTITUTIONS`.

* **Deleted** (outside the anchors, never sent): the identity lines ("You are the
  Scientist Agent…", "You are the Tester Agent. Your current stage is …") with their
  ``# Role`` / "The setting:" headers, the team blocks, the per-role copies of the
  global goal and task context, and the Extractor's raw-column block (moved once into
  the system prompt).

* **Substituted** (:data:`SUBSTITUTIONS`, 23 rows): every role mention in the moved
  text becomes its phase (Scientist -> "the PLAN phase", Extractor -> "the BUILD phase",
  Tester -> "the ASSESS/PRUNE phases"), plus "the raw columns listed below" -> "the raw
  columns listed in the system prompt". Each ``before`` must occur exactly once in its
  block (else :class:`SetPhasedPromptError`).

* **Not changed**: tool names, tool docstrings and schemas.
"""

from __future__ import annotations

from typing import Iterable

from rogueone.llm.prompts.agent_roles import AgentRoles


class SetPhasedPromptError(RuntimeError):
    """A frozen anchor or substitution does not match the rendered Full text."""


#: Phase id -> (k, NAME).
PHASE_NAMES: dict[str, tuple[int, str]] = {
    "S": (1, "PLAN"),
    "E": (2, "BUILD"),
    "T-A": (3, "ASSESS"),
    "T-B": (4, "PRUNE"),
}

PHASE_HEADER = (
    "### PHASE {k}/4 — {name}. From now on only the instructions below and the tools of "
    "this phase apply."
)

PHASES_LINE = (
    "You will work through four phases (PLAN, BUILD, ASSESS, PRUNE); each phase message "
    "states its instructions and tools."
)

CONTEXT_HEADER = "# Task context and predictive goal:"
RAW_COLUMNS_HEADER = (
    "# Raw columns available in `df` (authoritative — these are the ONLY columns you may "
    "read from `df`):"
)
OUTPUT_STYLE_LINE = "Do not provide long chain-of-thought narratives."

#: Phase id -> segments ``(first line starts with, last line starts with)``, matched on
#: the stripped line. Segments are inclusive and joined by one blank line.
ROLE_BLOCKS: dict[str, tuple[tuple[str, str], ...]] = {
    "S": (
        (
            "Your primary responsibility is to determine the focus of the investigation",
            "Use the tools available to you to explore the data, gather information",
        ),
        ("# Objective:", "- Extractor can ONLY add features via `append_new_attribute`."),
        ("# Output requirements:", "- Keep the final answer to a few sentences."),
    ),
    "E": (
        ("# Objective:", "- **The target / label column is HELD BACK from `df`**"),
        ("# Available tools:", "- search_in_literature_tool"),
    ),
    "T-A": (
        (
            "You will assess the quality of the current feature set",
            "- Recommended candidates for pruning (recommendations only, no pruning action)",
        ),
    ),
    "T-B": (
        ("# Hard Constraints", "5. Return a concise markdown summary with rationale"),
    ),
}

PLAN, BUILD, ASSESS_PRUNE = "the PLAN phase", "the BUILD phase", "the ASSESS/PRUNE phases"

#: The frozen before -> after table (23 rows). ``where``: "goal" = GLOBAL_GOAL (system
#: prompt), else the phase whose role block holds the text. Rows are applied in order;
#: each ``before`` must occur exactly once in its text at the time it is applied.
SUBSTITUTIONS: tuple[tuple[str, str, str, str], ...] = (
    # id, where, before, after
    ("G1", "goal", "The overall goal of the team is", "The overall goal is"),
    ("G2", "goal", "evaluated by a Tester Agent using", "evaluated in the ASSESS/PRUNE phases using"),
    ("S1", "S", "provided by the tester agent and", f"provided by {ASSESS_PRUNE} and"),
    ("S2", "S", "extracted by the extractor agent.", f"extracted by {BUILD}."),
    ("S3", "S", "4. Select one focus for the next extractor round", "4. Select one focus for the next BUILD phase round"),
    ("S4", "S", "- Keep focus within what Extractor can implement.", f"- Keep focus within what {BUILD} can implement."),
    ("S5", "S", "transform the Extractor can build via", f"transform {BUILD} can build via"),
    ("S6", "S", "pruning is the Tester's job", f"pruning is {ASSESS_PRUNE}' job"),
    ("S7", "S", "the Extractor has no pruning tool", f"{BUILD} has no pruning tool"),
    ("S8", "S", "the Tester may then prune redundant one-hots", f"{ASSESS_PRUNE} may then prune redundant one-hots"),
    ("S9", "S", "leave the Extractor unsure", f"leave {BUILD} unsure"),
    ("S10", "S", "(the Tester will retire X", f"({ASSESS_PRUNE} will retire X"),
    ("S11", "S", "so the Extractor's job", f"so {BUILD}'s job"),
    ("S12", "S", "features the Tester has previously pruned", f"features {ASSESS_PRUNE} have previously pruned"),
    ("S13", "S", "# Extractor implementation constraints:", "# BUILD phase implementation constraints:"),
    ("S14", "S", "- Extractor can ONLY add features", "- The BUILD phase can ONLY add features"),
    ("S15", "S", "around what the Extractor can do.", f"around what {BUILD} can do."),
    ("E1", "E", "provided focus area from the Scientist Agent.", f"provided focus area from {PLAN}."),
    ("E2", "E", "The Tester will prune it if it is uninformative.", "The ASSESS/PRUNE phases will prune it if it is uninformative."),
    ("E3", "E", "contains ONLY the raw columns listed below.", "contains ONLY the raw columns listed in the system prompt."),
    ("E4", "E", "Build features only from the raw columns listed below.", "Build features only from the raw columns listed in the system prompt."),
    ("T1", "T-A", "produced by the extractor using", f"produced by {BUILD} using"),
    ("T2", "T-A", "will be used by the Scientist Agent to generate", f"will be used by {PLAN} to generate"),
)


def _apply(text: str, where: str) -> str:
    for row_id, w, before, after in SUBSTITUTIONS:
        if w != where:
            continue
        n = text.count(before)
        if n != 1:
            raise SetPhasedPromptError(
                f"substitution {row_id} expects exactly one {before!r} in {where}, found {n}"
            )
        text = text.replace(before, after)
    return text


def neutral_global_goal(global_goal: str | None = None) -> str:
    """GLOBAL_GOAL without its first line, G1/G2 applied, one line per sentence."""
    text = AgentRoles.GLOBAL_GOAL.value if global_goal is None else global_goal
    lines = [ln.strip() for ln in text.strip("\n").split("\n") if ln.strip()]
    if not lines or not lines[0].startswith("You are part of a team of agents"):
        raise SetPhasedPromptError("GLOBAL_GOAL no longer starts with the team line")
    return _apply("\n".join(lines[1:]), "goal")


def raw_columns_str(columns: Iterable) -> str:
    """The raw-column list exactly as the Extractor builds it (``ext/agent.py``)."""
    return ", ".join(f"`{c}`" for c in sorted(map(str, list(columns))))


def system_prompt(context_and_goal: str, raw_columns: str) -> str:
    """The neutral system prompt, identical for every phase of an iteration."""
    return "\n".join(
        [
            neutral_global_goal(),
            PHASES_LINE,
            "",
            CONTEXT_HEADER,
            str(context_and_goal),
            "",
            RAW_COLUMNS_HEADER,
            raw_columns,
            "",
            OUTPUT_STYLE_LINE,
        ]
    )


def _segment(lines: list[str], start: str, end: str, phase: str) -> str:
    starts = [i for i, ln in enumerate(lines) if ln.strip().startswith(start)]
    if len(starts) != 1:
        raise SetPhasedPromptError(
            f"{phase}: start anchor {start!r} found {len(starts)} times in the Full instructions"
        )
    i0 = starts[0]
    for j in range(i0, len(lines)):
        if lines[j].strip().startswith(end):
            return "\n".join(lines[i0 : j + 1])
    raise SetPhasedPromptError(f"{phase}: end anchor {end!r} not found after {start!r}")


def role_block(phase: str, full_instructions: str) -> str:
    """The phase's role block cut from the rendered Full instructions, table applied."""
    lines = str(full_instructions).split("\n")
    raw = "\n\n".join(_segment(lines, a, b, phase) for a, b in ROLE_BLOCKS[phase])
    return _apply(raw, phase)


def phase_header(phase: str) -> str:
    k, name = PHASE_NAMES[phase]
    return PHASE_HEADER.format(k=k, name=name)


def phase_message(phase: str, full_instructions: str, trailing_input: str) -> str:
    """First message of a phase: header, role block, trailing input."""
    return "\n\n".join(
        [phase_header(phase), role_block(phase, full_instructions), str(trailing_input)]
    )
