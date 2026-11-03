"""Shared sandboxed-exec helper for user-supplied (LLM-authored) pandas commands.

Lives in `rogueone.utils` rather than alongside the Extractor agent so it can be
reused (e.g. by `rogueone.utils.fold_safe_features`) without pulling in the whole
LLM-agent import chain -- this module has no dependency on `rogueone.llm.*`.
"""

import ast
import contextlib
import io
import re
from textwrap import dedent

import numpy as np
import pandas as pd

from rogueone.utils.sandbox import sandbox_scope


def normalize_user_command(command: str) -> str:
    """Normalize a user-supplied exec() snippet so accidental formatting does not break it.

    The Extractor LLM emits multi-statement commands in several malformed shapes:

    1. **Literal ``\\n`` / ``\\t`` two-char sequences** where a real newline / tab was
       intended. The model double-escapes through the JSON tool-call layer, so
       ``"a = 1\\nb = 2"`` (4 chars between assignments: backslash, n, ...) reaches us
       as Python source, not as a real newline. ``exec`` then sees a line-continuation
       backslash followed by ``n`` and raises SyntaxError. Observed in
       ``agent_toolcall_errors_car:v23`` (run ``09vnwi4s``).
    2. **Continuation lines with leading whitespace after a semicolon**
       (``"a = 1;\\n  b = 2"``) — the existing ``;\\s*\\n[ \\t]+`` regex collapses these.
    3. **Continuation lines with leading whitespace WITHOUT a preceding semicolon**
       (``"a = 1\\n b = 2"``) — these slip past the regex above. ``exec`` raises
       ``IndentationError: unexpected indent (<string>, line 2)``. Also observed in
       ``agent_toolcall_errors_car:v23``.
    4. **Whole-block uniform indentation** — handled by :func:`textwrap.dedent`.

    For (3) we use a probe-and-fall-back strategy: try ``compile``; only if it raises
    ``IndentationError`` do we ``lstrip`` each line, and only when the snippet contains
    no ``:``-terminated lines (which would mean a ``def``/``if``/``for`` block whose
    body indentation is meaningful and must NOT be stripped).
    """
    if not isinstance(command, str):
        return command
    if ("\\n" in command or "\\t" in command) and needs_unescape(command):
        command = command.replace("\\n", "\n").replace("\\t", "\t")
    normalized = dedent(command).strip()
    normalized = re.sub(r";[ \t]*\n[ \t]+", "\n", normalized)
    if "\n" in normalized:
        try:
            compile(normalized, "<extractor-command>", "exec")
        except IndentationError:
            no_blocks = not any(
                ln.rstrip().endswith(":") for ln in normalized.splitlines()
            )
            if no_blocks:
                normalized = "\n".join(
                    ln.lstrip() for ln in normalized.splitlines()
                )
        except SyntaxError:
            pass
    return normalized


def _exec_compat() -> bool:
    from rogueone.utils import rouge_one_cfg

    return rouge_one_cfg.exec_compat


def needs_unescape(command: str) -> bool:
    """Whether literal ``\\n`` / ``\\t`` should become real newlines / tabs.

    Always, by default. Under ``ROGUE_ONE_EXEC_COMPAT`` only when the code does not
    compile as sent: a correctly transported ``print("\\n")`` must keep its escape,
    or the rewrite splits the string literal (A17 Q4 smoke, "unterminated string
    literal").
    """
    if not _exec_compat():
        return True
    try:
        compile(dedent(command).strip(), "<probe>", "exec")
    except SyntaxError:
        return True
    return False


def exec_with_last_expr(code: str, scope: dict) -> None:
    """``exec(code, scope)``; under ``ROGUE_ONE_EXEC_COMPAT`` a trailing bare
    expression (``df.head()``) sets ``scope['result']`` when the code did not."""
    if not _exec_compat():
        exec(code, scope)
        return
    try:
        tree = ast.parse(code)
    except SyntaxError:
        exec(code, scope)  # raises the usual SyntaxError
        return
    if not tree.body or not isinstance(tree.body[-1], ast.Expr):
        exec(code, scope)
        return
    last = tree.body.pop()
    exec(compile(tree, "<string>", "exec"), scope)
    value = eval(compile(ast.Expression(last.value), "<string>", "eval"), scope)
    if scope.get("result") is None:
        scope["result"] = value


_KEYERROR_COLUMN_RE = re.compile(r"['\"]([^'\"]+)['\"]")


def extract_missing_column(
    exc: KeyError,
    df: pd.DataFrame,
    df_features: pd.DataFrame,
) -> tuple[str, str] | None:
    """Return (available_in, intended_for) when a KeyError pinpoints a column
    that lives in the *other* DataFrame, else None.

    pandas KeyErrors come in several shapes:
      ``KeyError: 'safety_ord'``
      ``KeyError: 'Column not found: safety_ord'``
      ``KeyError: "['safety_ord'] not in index"``
    The regex pulls quoted names from the message; we then check membership
    against the two known DataFrames.
    """
    msg = exc.args[0] if exc.args else str(exc)
    if not isinstance(msg, str):
        return None

    candidates: list[str] = []
    candidates.extend(_KEYERROR_COLUMN_RE.findall(msg))
    if msg and msg not in candidates:
        candidates.append(msg)

    df_cols = set(map(str, df.columns))
    df_features_cols = set(map(str, df_features.columns))

    for raw in candidates:
        name = raw.split(":")[-1].strip()
        if not name:
            continue
        if name in df_features_cols and name not in df_cols:
            return ("df_features", "df")
        if name in df_cols and name not in df_features_cols:
            return ("df", "df_features")
    return None


def exec_user_command(
    command: str,
    df: pd.DataFrame,
    df_features: pd.DataFrame,
) -> tuple[object, str]:
    """Execute a user-supplied pandas command and return ``(result, stdout)``.

    The command runs in a fresh ``exec`` scope with ``pd``, ``np``, ``df`` and
    ``df_features`` available. The snippet is normalized via
    :func:`normalize_user_command` so accidental leading indentation is tolerated.

    The scope is a **single dict** used as both globals and locals. This is
    deliberate: when ``exec(code, globals, locals)`` receives separate dicts,
    top-level assignments land in ``locals`` while nested ``lambda`` / ``def``
    bodies look names up in ``globals``. The Extractor frequently emits
    multi-statement code such as ::

        persons_map = {'2': 2, '4': 4, 'more': 5}
        result = df.groupby('id').apply(
            lambda g: persons_map[g['persons'].iloc[0]]
        )

    and would get ``NameError: name 'persons_map' is not defined`` from the
    lambda even though the assignment ran. Sharing one namespace removes the
    skew.

    With ``ROGUE_ONE_SANDBOX=on`` the scope gets restricted builtins and
    I/O-less module proxies (:mod:`rogueone.utils.sandbox`, E13); ``off`` leaves
    it untouched.

    KeyError handling: if the command tried to read a column from ``df`` that
    is missing there but exists on ``df_features`` (or the reverse), the
    KeyError is re-raised with an explanatory hint so the agent can correct
    the next attempt. This addresses the dominant CAR failure mode where the
    Extractor invents derived columns (``safety_ord``, ``persons_ord``, ...),
    stores them in ``df_features``, then immediately tries to read them back
    from ``df`` via ``df.groupby('id')['safety_ord']``. CAR run m4i2r4v9
    repeated this pattern ~12 times across one fold and the dataset aborted.
    """
    normalized = normalize_user_command(command)
    scope: dict = {"pd": pd, "df": df, "np": np, "df_features": df_features}
    try:
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            exec_with_last_expr(normalized, sandbox_scope(scope))
    except KeyError as exc:
        missing = extract_missing_column(exc, df, df_features)
        if missing is not None:
            available_in, intended_for = missing
            raise KeyError(
                f"{exc.args[0] if exc.args else exc} — the column is not in "
                f"`{intended_for}` but it IS available in `{available_in}`. "
                f"Use `{available_in}` instead (engineered columns added via "
                f"append_new_attribute live ONLY in df_features; df is the raw "
                f"input table)."
            ) from exc
        raise
    return scope.get("result", None), buf.getvalue()
