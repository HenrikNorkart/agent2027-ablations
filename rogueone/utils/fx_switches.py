"""FX switches (AGENT 2027 E9d, A12 §4, A12-T4…T8): four default-off component swaps.

Each switch replaces one LLM component of Full-v2 with a version that makes no LLM
call. With every flag unset (or at its Full-v2 value) nothing in this module changes
a prompt, a tool list, a tool return value, a file or a W&B key: the call sites test
the flag and take the untouched Full path. The flags (01 §7 R4, Full-v2 value first):

* ``ROGUE_ONE_PLANNER_MODE``  ``llm`` / ``none`` (P−, SE|T). ``oracle`` and
  ``misdirected`` belong to E9c and are only *recognised* here (no FX action).
  Under ``none`` the Scientist is not called. In the first Extractor round only, the
  Extractor gets :data:`P_MINUS_PREAMBLE` plus a digest (last feedback-pool row, the
  ``Attribute | Status | Pruned Iteration | Prune Rationale`` table, its own earlier
  ``FOCUS:`` lines); rounds 2..R send ``"System now wants to focus on: <own focus>."``.
  The digest is logged to ``output/fx_digest.jsonl``.
* ``ROGUE_ONE_ASSESSMENT_MODE``  ``llm`` / ``template`` (A−). Stage A is replaced by
  a template report with the same numbers (counts, the ``_metric_profiles_to_markdown``
  table, top-5 ``mean_abs_shap``); a "Pruned this iteration" section is appended after
  Stage B for the planner.
* ``ROGUE_ONE_PRUNING_MODE``  ``llm`` / ``rules`` (J−a). Stage B's VIF/SHAP/PFI rules
  (``tst/agent.py`` "PRUNING ONLY" prompt) run as code (:func:`ja_rules`), followed by
  the same floor and all-pruned guard. The derivative rule is not executed (declared).
* ``ROGUE_ONE_PRUNING_SHADOW``  ``0`` / ``1`` (FX J+ cells). J−a is computed on a copy
  of the table before Stage B and logged to ``output/pruning_shadow.jsonl`` with the
  LLM's prunes; it never mutates the table and never enters a prompt.
* ``ROGUE_ONE_MEMORY_MODE``  ``full`` / ``amnesic`` (M−). The Scientist sees only the
  last focus and its notebook is cleared before every call (A12-T8: the tools stay, so
  the prompt and tool list are byte-identical). Under P− it shortens the digest's
  focus history to the last line.

Readers validate the value and raise :class:`FxFlagError` on anything unknown, so a
typo in a manifest fails the run instead of silently running Full.
"""

from __future__ import annotations

import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from rogueone.utils import instrumentation

# --------------------------------------------------------------------------- flags
PLANNER_FLAG = "ROGUE_ONE_PLANNER_MODE"
MEMORY_FLAG = "ROGUE_ONE_MEMORY_MODE"
ASSESSMENT_FLAG = "ROGUE_ONE_ASSESSMENT_MODE"
PRUNING_FLAG = "ROGUE_ONE_PRUNING_MODE"
SHADOW_FLAG = "ROGUE_ONE_PRUNING_SHADOW"

PLANNER_LLM, PLANNER_NONE = "llm", "none"
PLANNER_VALUES = (PLANNER_LLM, PLANNER_NONE, "oracle", "misdirected")
MEMORY_FULL, MEMORY_AMNESIC = "full", "amnesic"
ASSESS_LLM, ASSESS_TEMPLATE = "llm", "template"
PRUNE_LLM, PRUNE_RULES = "llm", "rules"

#: flag -> (code default = Full-v2 value, allowed values)
FX_FLAGS: dict[str, tuple[str, tuple[str, ...]]] = {
    PLANNER_FLAG: (PLANNER_LLM, PLANNER_VALUES),
    MEMORY_FLAG: (MEMORY_FULL, (MEMORY_FULL, MEMORY_AMNESIC)),
    ASSESSMENT_FLAG: (ASSESS_LLM, (ASSESS_LLM, ASSESS_TEMPLATE)),
    PRUNING_FLAG: (PRUNE_LLM, (PRUNE_LLM, PRUNE_RULES)),
    SHADOW_FLAG: ("0", ("0", "1")),
}

_SHADOW_ALIASES = {"0": "0", "1": "1", "false": "0", "true": "1", "off": "0", "on": "1", "": "0"}


class FxFlagError(ValueError):
    """An FX flag holds a value the code does not implement."""


def _read(flag: str, environ: Optional[Mapping[str, str]] = None) -> str:
    env = os.environ if environ is None else environ
    default, allowed = FX_FLAGS[flag]
    raw = env.get(flag)
    if raw is None:
        return default
    val = str(raw).strip().lower()
    if flag == SHADOW_FLAG:
        if val not in _SHADOW_ALIASES:
            raise FxFlagError(f"{flag}={raw!r}; allowed {allowed}")
        return _SHADOW_ALIASES[val]
    if val == "":
        return default
    if val not in allowed:
        raise FxFlagError(f"{flag}={raw!r}; allowed {allowed}")
    return val


def planner_mode(environ: Optional[Mapping[str, str]] = None) -> str:
    return _read(PLANNER_FLAG, environ)


def memory_mode(environ: Optional[Mapping[str, str]] = None) -> str:
    return _read(MEMORY_FLAG, environ)


def assessment_mode(environ: Optional[Mapping[str, str]] = None) -> str:
    return _read(ASSESSMENT_FLAG, environ)


def pruning_mode(environ: Optional[Mapping[str, str]] = None) -> str:
    return _read(PRUNING_FLAG, environ)


def pruning_shadow(environ: Optional[Mapping[str, str]] = None) -> bool:
    return _read(SHADOW_FLAG, environ) == "1"


def planner_off(environ: Optional[Mapping[str, str]] = None) -> bool:
    """P−: ``ROGUE_ONE_PLANNER_MODE=none``."""
    return planner_mode(environ) == PLANNER_NONE


def amnesic(environ: Optional[Mapping[str, str]] = None) -> bool:
    return memory_mode(environ) == MEMORY_AMNESIC


def assessment_template(environ: Optional[Mapping[str, str]] = None) -> bool:
    return assessment_mode(environ) == ASSESS_TEMPLATE


def pruning_rules(environ: Optional[Mapping[str, str]] = None) -> bool:
    return pruning_mode(environ) == PRUNE_RULES


def shadow_active(environ: Optional[Mapping[str, str]] = None) -> bool:
    """Shadow J−a runs only next to an LLM pruner (under J−a it would be the decision)."""
    return pruning_shadow(environ) and not pruning_rules(environ)


def any_active(environ: Optional[Mapping[str, str]] = None) -> bool:
    """True iff an FX switch departs from Full-v2 (E9c's oracle/misdirected excluded)."""
    return (
        planner_off(environ)
        or amnesic(environ)
        or assessment_template(environ)
        or pruning_rules(environ)
        or pruning_shadow(environ)
    )


def validate_all(environ: Optional[Mapping[str, str]] = None) -> dict[str, str]:
    """Read every FX flag (raises :class:`FxFlagError` on a bad value)."""
    return {flag: _read(flag, environ) for flag in FX_FLAGS}


# --------------------------------------------------------------------------- jsonl
DIGEST_FILE = "fx_digest.jsonl"
SHADOW_FILE = "pruning_shadow.jsonl"


def start_run(environ: Optional[Mapping[str, str]] = None) -> dict[str, str]:
    """Network init: validate every FX flag and empty FX logs left in the output dir
    by an earlier run (only files that exist; nothing is created under Full)."""
    values = validate_all(environ)
    for name in (DIGEST_FILE, SHADOW_FILE):
        try:
            path = instrumentation._out_dir() / name
            if path.exists():
                path.write_text("", encoding="utf-8")
        except Exception:
            pass
    return values


def _log_jsonl(name: str, record: dict) -> None:
    try:
        instrumentation._append_jsonl(name, record)  # same output dir as focus_log.jsonl
    except Exception:  # a side channel never breaks a run
        instrumentation.count("fx_log_errors")


def _iter_ids() -> dict:
    try:
        return dict(instrumentation.current_iteration())
    except Exception:
        return {}


# --------------------------------------------------------------------------- helpers
def _max_chars(environ: Optional[Mapping[str, str]] = None) -> int:
    env = os.environ if environ is None else environ
    try:
        return int(env.get("ROGUE_ONE_TOOL_OUTPUT_MAX_CHARS", 0) or 0)
    except ValueError:
        return 0


def cap_section(text: str, max_chars: int) -> str:
    """Cap one digest section at ``max_chars`` (0 = unbounded)."""
    if max_chars and max_chars > 0 and len(text) > max_chars:
        return text[:max_chars] + f"\n[... section truncated to {max_chars} characters]"
    return text


def _is_missing(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, (list, tuple, dict, set)):
        return False
    try:
        import pandas as pd

        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _cell(v: Any) -> str:
    if _is_missing(v):
        return ""
    return re.sub(r"\s+", " ", str(v)).replace("|", "/").strip()


def _num(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


# --------------------------------------------------------------------------- M−
def scientist_history_view(focus_history: list, environ: Optional[Mapping[str, str]] = None) -> list:
    """The history handed to ``determine_focus``: the same object under Full, the last
    element only under M−."""
    if amnesic(environ):
        return list(focus_history[-1:])
    return focus_history


def reset_scientist_memory(scientist_agent: Any, environ: Optional[Mapping[str, str]] = None) -> None:
    """M−: clear the Scientist notebook before every ``determine_focus`` call."""
    if amnesic(environ):
        scientist_agent._note_book = ""
        instrumentation.count("fx_notebook_resets")


# --------------------------------------------------------------------------- P−
P_MINUS_PREAMBLE = (
    "The Scientist Agent is not available; choose the focus yourself. "
    "Start with one line `FOCUS: <one sentence>`, then build features for it."
)
NO_TRIAL_DIGEST = "No previous trial in this fold."
FOCUS_FALLBACK = "Build the new features most likely to improve predictive performance."
FOCUS_FALLBACK_CHARS = 300
_FOCUS_RE = re.compile(r"^[\s>*_#`-]*FOCUS[*_`]*\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_FEEDBACK_SKIP = {"dataset_id", "fold_seed", "feature_assessment"}


def _feedback_section(feedback_pool: Any) -> str:
    row = feedback_pool.iloc[-1].to_dict()
    lines = ["## Last trial (feedback row)"]
    for k, v in row.items():
        if k in _FEEDBACK_SKIP or _is_missing(v):
            continue
        lines.append(f"- {k}: {v}")
    report = row.get("feature_assessment")
    lines.append("")
    lines.append("### Tester report")
    lines.append("" if _is_missing(report) else str(report))
    return "\n".join(lines)


def _status_section(attribute_table: Any) -> str:
    cols = ["Attribute", "Status", "Pruned Iteration", "Prune Rationale"]
    out = ["## Attribute status", "| " + " | ".join(cols) + " |", "|---|---|---|---|"]
    if attribute_table is None or len(attribute_table) == 0:
        return "\n".join(out + ["(no attributes)"])
    for _, r in attribute_table.iterrows():
        out.append("| " + " | ".join(_cell(r.get(c)) for c in cols) + " |")
    return "\n".join(out)


def _focus_section(own_focus_history: Sequence[str]) -> str:
    lines = ["## Your earlier FOCUS lines in this fold"]
    if not own_focus_history:
        lines.append("(none)")
    for i, f in enumerate(own_focus_history, 1):
        lines.append(f"{i}. {f}")
    return "\n".join(lines)


def build_digest(
    feedback_pool: Any,
    attribute_table: Any,
    focus_history: Sequence[str],
    *,
    environ: Optional[Mapping[str, str]] = None,
) -> tuple[str, dict]:
    """The P− round-1 digest and its per-section character counts.

    ``focus_history`` is the run's own FOCUS lines in this fold; under M− only the
    last one is shown. Each section is capped at ``ROGUE_ONE_TOOL_OUTPUT_MAX_CHARS``.
    """
    if feedback_pool is None or len(feedback_pool) == 0:
        return NO_TRIAL_DIGEST, {"empty": True}
    cap = _max_chars(environ)
    history = list(focus_history[-1:]) if amnesic(environ) else list(focus_history)
    sections = [
        cap_section(_feedback_section(feedback_pool), cap),
        cap_section(_status_section(attribute_table), cap),
        cap_section(_focus_section(history), cap),
    ]
    meta = {
        "empty": False,
        "chars_feedback": len(sections[0]),
        "chars_status": len(sections[1]),
        "chars_focus": len(sections[2]),
        "focus_lines": len(history),
    }
    return "\n\n".join(sections), meta


def parse_focus(texts: Iterable[str]) -> tuple[str, bool]:
    """``(focus, missing)``: the first ``FOCUS:`` line across the round-1 messages, or
    the first 300 characters of the round-1 output (``missing=True``)."""
    texts = [t for t in texts if isinstance(t, str) and t.strip()]
    for t in texts:
        m = _FOCUS_RE.search(t)
        if m and m.group(1).strip(" *_`"):
            return m.group(1).strip(" *_`"), False
    if texts:
        return texts[0].strip()[:FOCUS_FALLBACK_CHARS], True
    return FOCUS_FALLBACK, True


def _result_texts(result: Any) -> list[str]:
    texts: list[str] = []
    for item in getattr(result, "new_items", None) or []:
        if getattr(item, "type", None) != "message_output_item":
            continue
        try:
            from agents import ItemHelpers

            texts.append(ItemHelpers.text_message_output(item))
        except Exception:
            continue
    final = getattr(result, "final_output", None)
    if isinstance(final, str) and final not in texts:
        texts.append(final)
    return texts


@dataclass
class SelfFocus:
    """P− state for one iteration: the round-1 digest message and the parsed focus."""

    message: str
    digest_meta: dict = field(default_factory=dict)
    focus: Optional[str] = None
    focus_line_missing: bool = False
    digest_resends: int = 0
    _observed: bool = False

    def message_for_round(self, round_idx: int) -> str:
        """Round 1 (and any retry before a round returns) carries the digest; later
        rounds send ``"System now wants to focus on: <own focus>."``."""
        if not self._observed:
            if round_idx > 0:  # round 1 raised before returning: the digest is resent
                self.digest_resends += 1
                instrumentation.count("fx_digest_resent")
            return self.message
        return f"System now wants to focus on: {self.focus}."

    def observe(self, round_idx: int, result: Any) -> None:
        """Record the first completed round's output and parse its ``FOCUS:`` line."""
        if self._observed:
            return
        self._observed = True
        self.focus, self.focus_line_missing = parse_focus(_result_texts(result))

    def finalize(self) -> str:
        """The iteration's focus (fallback if no Extractor round completed)."""
        if self.focus is None:
            self.focus, self.focus_line_missing = FOCUS_FALLBACK, True
        if self.focus_line_missing:
            instrumentation.count("focus_line_missing")
        return self.focus


def begin_self_focus(
    feedback_pool: Any,
    attribute_table: Any,
    focus_history: Sequence[str],
    *,
    index: int,
    fold_run_id: int,
    iteration_in_fold: int,
    environ: Optional[Mapping[str, str]] = None,
) -> Optional[SelfFocus]:
    """P− hook for ``_forward_pass``: ``None`` unless ``PLANNER_MODE=none``."""
    if not planner_off(environ):
        return None
    digest, meta = build_digest(feedback_pool, attribute_table, focus_history, environ=environ)
    message = f"{P_MINUS_PREAMBLE}\n\n{digest}"
    instrumentation.count("fx_digests")
    _log_jsonl(
        DIGEST_FILE,
        {
            "index": index,
            "fold_run_id": fold_run_id,
            "iteration_in_fold": iteration_in_fold,
            "digest": digest,
            "message_chars": len(message),
            **meta,
            "ts": time.time(),
        },
    )
    return SelfFocus(message=message, digest_meta=meta)


# --------------------------------------------------------------------------- A−
def _top_shap(profiles: Sequence[Any], k: int = 5) -> list[tuple[str, float]]:
    vals = [
        (str(p.feature_name), _num(getattr(p, "mean_abs_shap", None)))
        for p in profiles
    ]
    vals = [(n, v) for n, v in vals if v is not None]
    vals.sort(key=lambda t: (-t[1], t[0]))
    return vals[:k]


def template_assessment(
    metric_profiles: Sequence[Any],
    metric_table_markdown: str,
    *,
    active_count: int,
    new_count: int,
) -> str:
    """A− sections 1–3 (what Stage B receives): counts, the metric table, top-5 SHAP."""
    top = _top_shap(metric_profiles)
    lines = [
        "# Feature Assessment (template)",
        "",
        "## 1. Feature counts",
        f"- Active features assessed: {active_count}",
        f"- New features this iteration: {new_count}",
        "",
        "## 2. Pre-computed metric table",
        metric_table_markdown,
        "",
        "## 3. Top-5 features by mean_abs_shap",
        "| rank | feature_name | mean_abs_shap |",
        "|---:|---|---:|",
    ]
    lines += [f"| {i} | {n} | {v:.6g} |" for i, (n, v) in enumerate(top, 1)]
    if not top:
        lines.append("| - | (no SHAP values) | - |")
    return "\n".join(lines)


def pruned_section(entries: Sequence[tuple[str, str]]) -> str:
    """A− section 4 (planner only): ``(feature, source)`` with source in
    ``llm`` / ``ja_vif_cluster`` / ``ja_vif_pfi`` / ``floor-restored``."""
    lines = ["", "## 4. Pruned this iteration", "| feature_name | source |", "|---|---|"]
    lines += [f"| {f} | {s} |" for f, s in entries]
    if not entries:
        lines.append("| (none) | - |")
    return "\n".join(lines)


def count_new_features(attribute_table: Any, active_feature_names: Iterable[str], trial_index: Any) -> int:
    """Active features whose ``Added`` is this trial (``Trial <index>``)."""
    try:
        label = f"Trial {trial_index}"
        names = set(map(str, active_feature_names))
        t = attribute_table
        mask = t["Attribute"].astype(str).isin(names) & (t["Added"].astype(str) == label)
        return int(mask.sum())
    except Exception:
        return 0


# --------------------------------------------------------------------------- J−a
VIF_CLUSTER = 1e10
VIF_HIGH = 10.0
SHAP_KEEP = 0.1
RULE_CLUSTER = "ja_vif_cluster"
RULE_PFI = "ja_vif_pfi"


@dataclass(frozen=True)
class JaDecision:
    feature: str
    rule: str


@dataclass
class _Row:
    name: str
    mi: Optional[float]
    vif: Optional[float]
    pfi_mean: Optional[float]
    pfi_std: Optional[float]
    shap: Optional[float]


def _rows(metric_profiles: Sequence[Any]) -> list[_Row]:
    out = []
    for p in metric_profiles:
        out.append(
            _Row(
                name=str(p.feature_name),
                mi=_num(getattr(p, "mutual_information", None)),
                vif=_num(getattr(p, "vif", None)),
                pfi_mean=_num(getattr(p, "permutation_importance_mean", None)),
                pfi_std=_num(getattr(p, "permutation_importance_std", None)),
                shap=_num(getattr(p, "mean_abs_shap", None)),
            )
        )
    return out


def ja_rules(
    metric_profiles: Sequence[Any],
    X_train: Any,
    feature_names: Sequence[str],
    *,
    protected: Iterable[str] = (),
    vif_fn: Optional[Callable[[Any, list[str]], list]] = None,
) -> list[JaDecision]:
    """J−a (A12-T7): Stage B's VIF/SHAP/PFI rules as code. Pure: mutates nothing.

    * ``ja_vif_cluster``: while an active feature has VIF ≥ 1e10, prune the one with the
      lowest SHAP (ties: lowest MI, then name) among those with VIF ≥ 1e10 and SHAP ≤ 0.1,
      then recompute VIF on the remaining active training columns. Stop when no feature
      has VIF ≥ 1e10 or none of them has SHAP ≤ 0.1 (high SHAP is never pruned for VIF).
    * ``ja_vif_pfi``: then prune every feature with table VIF in [10, 1e10),
      |pfi_mean| ≤ pfi_std and SHAP < 0.1.
    * VIF < 10 or a missing value: never pruned. ``protected`` (E6 seeded rows) never.

    ``feature_names`` are the columns of ``X_train`` in order (the assessed active set).
    """
    if vif_fn is None:
        from rogueone.llm.agents.core.tester.metrics import _compute_vif as vif_fn  # noqa: N813
    import numpy as np

    protected = set(map(str, protected))
    rows = {r.name: r for r in _rows(metric_profiles)}
    names = [str(n) for n in feature_names]
    X = np.asarray(X_train, dtype=float) if X_train is not None else None
    active = list(names)
    decisions: list[JaDecision] = []

    def _eligible_cluster(vifs: dict[str, Optional[float]]) -> list[_Row]:
        out = []
        for n in active:
            v, r = vifs.get(n), rows.get(n)
            if r is None or v is None or n in protected:
                continue
            if v >= VIF_CLUSTER and r.shap is not None and r.shap <= SHAP_KEEP:
                out.append(r)
        return out

    vifs: dict[str, Optional[float]] = {n: (rows[n].vif if n in rows else None) for n in active}
    guard = len(active) + 1
    while guard > 0:
        guard -= 1
        if not any(v is not None and v >= VIF_CLUSTER for v in vifs.values()):
            break
        cand = _eligible_cluster(vifs)
        if not cand:
            break
        cand.sort(key=lambda r: (r.shap, r.mi if r.mi is not None else math.inf, r.name))
        victim = cand[0].name
        decisions.append(JaDecision(victim, RULE_CLUSTER))
        active.remove(victim)
        if X is None or len(active) == 0:
            break
        cols = [names.index(n) for n in active]
        new = vif_fn(X[:, cols], list(active))
        vifs = {n: _num(v) for n, v in zip(active, new)}

    cluster_pruned = {d.feature for d in decisions}
    for n in names:
        r = rows.get(n)
        if r is None or n in cluster_pruned or n in protected:
            continue
        if None in (r.vif, r.pfi_mean, r.pfi_std, r.shap):
            continue
        if VIF_HIGH <= r.vif < VIF_CLUSTER and abs(r.pfi_mean) <= r.pfi_std and r.shap < SHAP_KEEP:
            decisions.append(JaDecision(n, RULE_PFI))
    return decisions


@dataclass
class JaOutcome:
    summary: str
    newly_pruned: list[str]
    restored: list[str]
    decisions: list[JaDecision]
    stage_status: str
    records: list[dict]


def apply_ja_pruning(
    df_attribute_explanations: Any,
    decisions: Sequence[JaDecision],
    *,
    trial_label: str,
    floor_fn: Callable[..., tuple[list[str], list[str]]],
    shap_lookup: Mapping[str, float],
    mi_lookup: Mapping[str, float],
    live: bool = True,
) -> JaOutcome:
    """Apply J−a decisions in place, as Stage B would: ``Status``, ``Pruned Iteration``
    (``trial_label``), ``Prune Rationale`` = rule label, then the same floor and the
    all-pruned guard. Mirrors ``generate_pruning``'s bookkeeping. ``live=False`` (the
    shadow, on a copy) leaves every run counter untouched."""
    df = df_attribute_explanations
    status_before = df["Status"].copy()
    rule_of = {d.feature: d.rule for d in decisions}
    targets = [n for n in rule_of]
    if targets:
        mask = df["Attribute"].astype(str).isin(targets)
        df.loc[mask, "Status"] = "Pruned"
    newly_pruned = df[(status_before != "Pruned") & (df["Status"] == "Pruned")]["Attribute"].tolist()
    for name in newly_pruned:
        m = df["Attribute"] == name
        df.loc[m, "Prune Rationale"] = rule_of.get(str(name), RULE_PFI)
        df.loc[m, "Pruned Iteration"] = trial_label

    newly_pruned, restored = floor_fn(
        df_attribute_explanations=df,
        status_before=status_before,
        newly_pruned=newly_pruned,
        shap_lookup=dict(shap_lookup),
        mi_lookup=dict(mi_lookup),
    )
    status = "rules_floor_enforced" if restored else "rules"
    if restored and live:
        instrumentation.count("floor_activations")
        instrumentation.count("floor_restored_features", len(restored))

    if int((df["Status"] != "Pruned").sum()) == 0:
        df.loc[:, "Status"] = status_before
        return JaOutcome(
            summary="Pruning skipped: minimum feature floor reached.",
            newly_pruned=[],
            restored=list(restored),
            decisions=list(decisions),
            stage_status="skipped",
            records=[],
        )

    for name in newly_pruned if live else ():
        instrumentation.count(f"fx_{rule_of.get(str(name), RULE_PFI)}")
    records = [
        {
            "feature": str(name),
            "decision": "prune",
            "evidence_source": "ja_rules",
            "rationale": rule_of.get(str(name), RULE_PFI),
        }
        for name in newly_pruned
    ]
    summary = (
        "J-a rules pruning: "
        + (", ".join(f"{n} ({rule_of.get(str(n))})" for n in newly_pruned) or "none")
        + (f"; floor-restored: {', '.join(map(str, restored))}" if restored else "")
    )
    return JaOutcome(summary, list(newly_pruned), list(restored), list(decisions), status, records)


def shap_mi_lookups(metric_profiles: Sequence[Any]) -> tuple[dict[str, float], dict[str, float]]:
    """The floor's ranking lookups, built exactly as ``generate_pruning`` builds them."""
    shap, mi = {}, {}
    for p in metric_profiles:
        s, m = _num(getattr(p, "mean_abs_shap", None)), _num(getattr(p, "mutual_information", None))
        if s is not None:
            shap[p.feature_name] = s
        if m is not None:
            mi[p.feature_name] = m
    return shap, mi


def seeded_names(df_attribute_explanations: Any) -> set[str]:
    from rogueone.utils.fold_safe_features import seeded_attribute_mask

    t = df_attribute_explanations
    return set(t.loc[seeded_attribute_mask(t), "Attribute"].astype(str))


def ja_for_tester(
    metric_profiles: Sequence[Any],
    X_train: Any,
    feature_names: Sequence[str],
    df_attribute_explanations: Any,
    *,
    floor_fn: Callable[..., tuple[list[str], list[str]]],
    trial_label: str,
    simulate: bool,
) -> JaOutcome:
    """J−a for one Tester call: decisions from the metric profiles, then applied in
    place (``simulate=False``, J−a cells) or to a deep copy (``simulate=True``, shadow)."""
    decisions = ja_rules(
        metric_profiles, X_train, feature_names, protected=seeded_names(df_attribute_explanations)
    )
    shap, mi = shap_mi_lookups(metric_profiles)
    if simulate:
        return simulate_ja(
            df_attribute_explanations, decisions, floor_fn=floor_fn, shap_lookup=shap, mi_lookup=mi
        )
    return apply_ja_pruning(
        df_attribute_explanations,
        decisions,
        trial_label=trial_label,
        floor_fn=floor_fn,
        shap_lookup=shap,
        mi_lookup=mi,
    )


def newly_pruned_since(df_attribute_explanations: Any, status_before: Any) -> list[str]:
    t = df_attribute_explanations
    return t[(status_before != "Pruned") & (t["Status"] == "Pruned")]["Attribute"].astype(str).tolist()


# --------------------------------------------------------------------------- shadow
def simulate_ja(
    df_attribute_explanations: Any,
    decisions: Sequence[JaDecision],
    *,
    floor_fn: Callable[..., tuple[list[str], list[str]]],
    shap_lookup: Mapping[str, float],
    mi_lookup: Mapping[str, float],
) -> JaOutcome:
    """J−a applied to a **deep copy** (the shadow never touches the live table)."""
    return apply_ja_pruning(
        df_attribute_explanations.copy(deep=True),
        decisions,
        trial_label="shadow",
        floor_fn=floor_fn,
        shap_lookup=shap_lookup,
        mi_lookup=mi_lookup,
        live=False,
    )


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    a, b = set(map(str, a)), set(map(str, b))
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def log_shadow(
    *,
    trial_index: Any,
    shadow: JaOutcome,
    llm_pruned: Sequence[str],
    pre_active: int,
) -> float:
    """One ``pruning_shadow.jsonl`` record per Tester call (J+ cells with SHADOW=1)."""
    shadow_pruned = list(shadow.newly_pruned)
    jac = jaccard(llm_pruned, shadow_pruned)
    instrumentation.count("fx_shadow_iterations")
    instrumentation.count("fx_shadow_pruned", len(shadow_pruned))
    instrumentation.count("fx_shadow_jaccard_sum_x1e6", int(round(jac * 1e6)))
    _log_jsonl(
        SHADOW_FILE,
        {
            **_iter_ids(),
            "trial": trial_index,
            "pre_active": pre_active,
            "shadow_decisions": [{"feature": d.feature, "rule": d.rule} for d in shadow.decisions],
            "shadow_pruned": shadow_pruned,
            "shadow_floor_restored": list(shadow.restored),
            "shadow_status": shadow.stage_status,
            "llm_pruned": list(map(str, llm_pruned)),
            "jaccard": jac,
            "ts": time.time(),
        },
    )
    return jac


# --------------------------------------------------------------------------- prune rate
def note_prune_rate(pre_active: int, pruned: int) -> None:
    """Per-iteration prune counts (FX runs only): ``fx_prune_rate`` = pruned / pre-active."""
    instrumentation.count("fx_prune_iterations")
    instrumentation.count("fx_prune_candidates", int(pre_active))
    instrumentation.count("fx_pruned", int(pruned))


def agent_active_count(df_attribute_explanations: Any, status: Any = None) -> int:
    """Active non-seeded rows (the floor's ``pre_active``)."""
    from rogueone.utils.fold_safe_features import seeded_attribute_mask

    st = df_attribute_explanations["Status"] if status is None else status
    return int(((st != "Pruned") & ~seeded_attribute_mask(df_attribute_explanations)).sum())


# --------------------------------------------------------------------------- summary
def summary_keys(counters: Mapping[str, Any], environ: Optional[Mapping[str, str]] = None) -> dict:
    """FX keys for ``compute_summary``; empty unless an FX switch is active, so the
    Full-v2 summary is unchanged."""
    try:
        if not any_active(environ):
            return {}
    except FxFlagError:
        return {}
    c = counters
    g = lambda k: int(c.get(k, 0))  # noqa: E731
    s = {
        "fx_planner_mode": planner_mode(environ),
        "fx_assessment_mode": assessment_mode(environ),
        "fx_pruning_mode": pruning_mode(environ),
        "fx_pruning_shadow": int(pruning_shadow(environ)),
        "fx_memory_mode": memory_mode(environ),
        "focus_line_missing": g("focus_line_missing"),
        "fx_digests": g("fx_digests"),
        "fx_digest_resent": g("fx_digest_resent"),
        "fx_template_reports": g("fx_template_reports"),
        "fx_notebook_resets": g("fx_notebook_resets"),
        "fx_ja_vif_cluster": g(f"fx_{RULE_CLUSTER}"),
        "fx_ja_vif_pfi": g(f"fx_{RULE_PFI}"),
        "fx_prune_iterations": g("fx_prune_iterations"),
        "fx_prune_candidates": g("fx_prune_candidates"),
        "fx_pruned": g("fx_pruned"),
        "fx_shadow_iterations": g("fx_shadow_iterations"),
        "fx_shadow_pruned": g("fx_shadow_pruned"),
    }
    s["fx_prune_rate"] = (s["fx_pruned"] / s["fx_prune_candidates"]) if s["fx_prune_candidates"] else None
    s["fx_shadow_prune_rate"] = (
        s["fx_shadow_pruned"] / s["fx_prune_candidates"]
        if s["fx_shadow_iterations"] and s["fx_prune_candidates"]
        else None
    )
    s["fx_shadow_jaccard_mean"] = (
        g("fx_shadow_jaccard_sum_x1e6") / 1e6 / s["fx_shadow_iterations"]
        if s["fx_shadow_iterations"]
        else None
    )
    return s
