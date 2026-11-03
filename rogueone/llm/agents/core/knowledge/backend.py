"""Knowledge-backend switch (AGENT 2027 E9e) and web-search routing (E7).

Both switches are default-off: with ``ROGUE_ONE_KNOWLEDGE_BACKEND=web`` (default)
and ``ROGUE_ONE_WEBSEARCH_ROUTING=global`` (default) every function here returns
its input unchanged (the same object), so Full-v2's rendered prompts, tool lists
and endpoints are byte-identical to ``prompt-baseline-v2.0``.

``ROGUE_ONE_KNOWLEDGE_BACKEND``
    ``web``        Full-v2: the shared summariser ``_run_search_agent`` searches the web.
    ``off``        A13 / A14 ``kb_off``: the search tool is removed from all three roles
                   and the search instructions are edited (A13 §4 "Off edit list",
                   edits 1-6, pre-registered in X10 V.13). Nothing else changes.
    ``parametric`` A14 ``kb_parametric``: the same summariser with ``tools=[]`` and its
                   tool-referring lines replaced by the pre-registered no-tool text
                   (X10 V.14). The roles' side is unchanged.
    ``local``      A14 ``kb_local_*``: the E8b TF-IDF backend behind the unchanged
                   ``_web_search`` tool. Needs ``local_corpus.py`` (E8b) and a mounted
                   corpus; without them the run refuses to start (:func:`validate`).

``ROGUE_ONE_WEBSEARCH_ROUTING``
    ``global``     the summariser uses ``LLM_MODEL`` / ``LLM_PORT`` (pre-E7 behaviour).
    ``caller``     the summariser uses the calling role's model and endpoint
                   (``ROGUE_ONE_<ROLE>_MODEL`` / ``_PORT``). For a uniform-model config
                   both give the same server and model id, so ``caller`` is a no-op.

This module imports only the standard library, so it is cheap to import from the
agents and from tests.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Optional, Sequence

BACKEND_ENV = "ROGUE_ONE_KNOWLEDGE_BACKEND"
ROUTING_ENV = "ROGUE_ONE_WEBSEARCH_ROUTING"
LOCAL_CORPUS_ENV = "ROGUE_ONE_LOCAL_CORPUS"
CORPORA_ROOT_ENV = "ROGUE_ONE_CORPORA_ROOT"

BACKENDS = ("web", "off", "parametric", "local")
ROUTINGS = ("global", "caller")
DEFAULT_BACKEND = "web"
DEFAULT_ROUTING = "global"
DEFAULT_CORPORA_ROOT = "/workspace/corpora"
LOCAL_CORPORA = ("kt3_correct", "kt3_wrong")

#: Names of the role-side tools that reach the summariser.
SEARCH_TOOL_NAMES = frozenset({"search_tool", "search_in_literature_tool"})

#: Roles whose model/endpoint the summariser can follow under ``caller`` routing.
CALLER_ROLES = ("scientist", "extractor", "tester")


class KnowledgeBackendError(RuntimeError):
    """The configured knowledge backend cannot run (bad value, missing E8b backend)."""


# --------------------------------------------------------------------------- flags

def _read(env: str, default: str, allowed: Sequence[str]) -> str:
    raw = os.getenv(env, "")
    value = (raw or "").strip().lower() or default
    if value not in allowed:
        raise KnowledgeBackendError(f"{env} must be one of {'|'.join(allowed)}, got {raw!r}")
    return value


def backend() -> str:
    """Effective ``ROGUE_ONE_KNOWLEDGE_BACKEND`` (default ``web``)."""
    return _read(BACKEND_ENV, DEFAULT_BACKEND, BACKENDS)


def routing() -> str:
    """Effective ``ROGUE_ONE_WEBSEARCH_ROUTING`` (default ``global``)."""
    return _read(ROUTING_ENV, DEFAULT_ROUTING, ROUTINGS)


# --------------------------------------------------------------------------- off: role side

# A13 §4 Off edit list, located by quoted string (line numbers at 07cbf93 in brackets).
# Each entry is (pattern, replacement); every pattern must match exactly once, so a
# drifted prompt fails loudly instead of leaving a search instruction behind (a call
# to a missing tool would confound A07).
_SCIENTIST_OFF_EDITS: tuple[tuple[str, str], ...] = (
    # edit 2 (sci/agent.py:414)
    (re.escape(", the focus history, and compare with existing information from the web to determine"),
     ", and the focus history to determine"),
    # edit 3 (sci/agent.py:431)
    (re.escape("(search, lookup, quick dataframe checks). Always perform at least one web search."),
     "(lookup, quick dataframe checks)."),
    # edit 4 (sci/agent.py:438): delete the whole line
    (r"\n[ \t]*- Use search_tool actively when uncertain\.[ \t]*(?=\n)", ""),
)

_EXTRACTOR_OFF_EDITS: tuple[tuple[str, str], ...] = (
    # edit 5, prompt half (ext/agent.py:921): drop the tool from "# Available tools:"
    (r"\n[ \t]*- search_in_literature_tool[ \t]*(?=\n)", ""),
)

_OFF_EDITS = {
    "scientist": _SCIENTIST_OFF_EDITS,
    "extractor": _EXTRACTOR_OFF_EDITS,
    "tester": (),  # edit 6 is a tool-list edit only; Stage A's prompt names no search
}


def _apply_edits(text: str, edits: Sequence[tuple[str, str]], where: str) -> str:
    for pattern, repl in edits:
        new, n = re.subn(pattern, lambda _m, r=repl: r, text)
        if n != 1:
            raise KnowledgeBackendError(
                f"{BACKEND_ENV}: edit {pattern!r} matched {n} times in the {where} prompt (expected 1)"
            )
        text = new
    return text


def role_instructions(role: str, instructions: str) -> str:
    """The role's system prompt under the current backend.

    ``off`` applies A13's prompt edits 2-5; every other backend returns
    ``instructions`` unchanged (the role side differs only under ``off``).
    """
    if backend() != "off":
        return instructions
    return _apply_edits(instructions, _OFF_EDITS[role], role)


def role_tools(tools: list) -> list:
    """The role's tool list under the current backend.

    ``off`` drops ``search_tool`` / ``search_in_literature_tool`` (edits 1, 5, 6);
    every other backend returns ``tools`` itself.
    """
    if backend() != "off":
        return tools
    return [t for t in tools if getattr(t, "name", None) not in SEARCH_TOOL_NAMES]


# --------------------------------------------------------------------------- parametric: summariser

#: Pre-registered no-tool text (X10 V.14, A14 §4).
PARAMETRIC_TEXT = (
    "You have no tools and no access to the web or documents. Answer the user's query "
    "from your own knowledge, as specifically as you can; if you do not know, say so."
)

# The summariser's tool-referring lines (kn/web_search.py instructions). The first
# two (setup) are replaced by PARAMETRIC_TEXT; the rest are deleted, together with
# the section headers whose every line is deleted. Workflow step 5 becomes step 2,
# because steps 2-4 are tool steps. Each line must occur exactly once.
_PARAMETRIC_REPLACE_FIRST = (
    "You are a knowledge lookup assistant that helps researchers find relevant information from the web.",
    "You are connected to the web via the web search tool.",
)
_PARAMETRIC_DELETE = (
    "# Your Role and Tasks:",
    "Your goal is to answer the user's query in the best possible way using the information available from the web-search.",
    "You are operating as a RAG (Retrieval-Augmented Generation) agent.",
    "2. For each key concept, use the web search tool to find relevant information from the web.",
    "3. If a search returns no useful results, rephrase and try again with different keywords. Do NOT give up after one failed search.",
    "4. Summarize the information retrieved from the web to answer the user's query.",
    "# CRITICAL — Search Query Rules:",
    "The web search tool sends queries to Google via the Serper API. You MUST follow these rules when crafting queries:",
    "- Use plain natural-language keywords ONLY — like what a human would type into Google.",
    '- NEVER include mathematical notation, formulas, or equations (e.g. "= sum(x) / n", "y = mx + b").',
    "- NEVER include special symbols: =, <, >, ≥, ≤, θ, π, σ, α, β, ∑, ∫, √, ±, ×, ÷, ^, or similar.",
    "- NEVER include programming syntax, code, or brackets.",
    "- Keep queries short: ideally 3-8 words. Longer queries are more likely to fail.",
    '- Use descriptive words instead of symbols: say "variance formula" not "s = sum(x - m) / n".',
    "- If a search fails, simplify the query further and retry with fewer, more common words.",
    'Good query examples: "variance formula statistics explained", "feature engineering classification", "ratio interaction features"',
    'Bad query examples: "mean = sum(x) / n", "F1 > 0.8 classification threshold", "σ = sqrt(variance)"',
    "# Important Guidelines:",
    "- Always provide a final answer based on your search results, even if limited information is found.",
    "- You must only base your summaries on the information found using the web search tool.",
    "- If a search fails, rephrase with simpler keywords and try again. Repeat up to 3-4 times before giving up.",
    '- Only respond with "Unable to provide an explanation at this time." as a LAST RESORT when ALL retry attempts have produced no results.',
)
_PARAMETRIC_RENUMBER = (
    "5. Provide a final response containing a relevant summary to the user based on the user's query.",
    "2. Provide a final response containing a relevant summary to the user based on the user's query.",
)


def parametric_instructions(web_instructions: str) -> str:
    """Summariser prompt for ``parametric``, derived line by line from the web prompt.

    Indentation of kept lines is preserved. Raises if any listed line is not found
    exactly once (the web prompt drifted).
    """
    lines = web_instructions.split("\n")
    stripped = [ln.strip() for ln in lines]

    def _index(text: str) -> int:
        hits = [i for i, s in enumerate(stripped) if s == text]
        if len(hits) != 1:
            raise KnowledgeBackendError(
                f"{BACKEND_ENV}=parametric: summariser line {text!r} found {len(hits)} times (expected 1)"
            )
        return hits[0]

    first, second = (_index(t) for t in _PARAMETRIC_REPLACE_FIRST)
    drop = {second, *(_index(t) for t in _PARAMETRIC_DELETE)}
    renum = _index(_PARAMETRIC_RENUMBER[0])

    out = []
    for i, ln in enumerate(lines):
        if i in drop:
            continue
        indent = ln[: len(ln) - len(ln.lstrip())]
        if i == first:
            out.append(indent + PARAMETRIC_TEXT)
        elif i == renum:
            out.append(indent + _PARAMETRIC_RENUMBER[1])
        else:
            out.append(ln)
    return "\n".join(out)


def summariser_instructions(web_instructions: str) -> str:
    """The summariser's system prompt: unchanged except under ``parametric``.

    ``local`` keeps the web prompt byte-identical (agents still "search the web").
    """
    if backend() == "parametric":
        return parametric_instructions(web_instructions)
    return web_instructions


def answer_channel() -> str:
    """Retrieval-transcript channel of the summariser's ``event="answer"`` rows:
    ``web_search`` under ``web`` (unchanged), else the backend's own name."""
    return {"web": "web_search", "local": "local_corpus"}.get(backend(), backend())


# --------------------------------------------------------------------------- local: E8b wiring point

LOCAL_CORPUS_MODULE = "rogueone.llm.agents.core.knowledge.local_corpus"


def local_corpus_dir() -> Path:
    name = (os.getenv(LOCAL_CORPUS_ENV, "") or "").strip()
    if name not in LOCAL_CORPORA:
        raise KnowledgeBackendError(
            f"{BACKEND_ENV}=local needs {LOCAL_CORPUS_ENV} in {'|'.join(LOCAL_CORPORA)}, got {name!r}"
        )
    root = os.getenv(CORPORA_ROOT_ENV, "") or DEFAULT_CORPORA_ROOT
    return Path(root) / name


def load_local_corpus() -> Any:
    """Load the E8b corpus for ``local``; refuses unless E8b has landed.

    The E8b contract: module :data:`LOCAL_CORPUS_MODULE` exposes
    ``load_corpus(path) -> corpus`` where ``corpus.search(query) -> list of
    (doc_id, score, text)`` (top k, scores > 0, best first; the A14 §4 TF-IDF
    settings) and ``corpus.sha256`` (A14-T1 hash definition). See
    ``web_search._local_search_impl`` for how hits become the tool's return text.
    """
    import importlib

    path = local_corpus_dir()
    if not path.is_dir() or not any(p.is_file() for p in path.rglob("*")):
        raise KnowledgeBackendError(f"{BACKEND_ENV}=local: corpus folder {path} is missing or empty")
    try:
        mod = importlib.import_module(LOCAL_CORPUS_MODULE)
    except ModuleNotFoundError as e:
        raise NotImplementedError(
            f"{BACKEND_ENV}=local: the E8b TF-IDF backend ({LOCAL_CORPUS_MODULE}) is not in this "
            "image; A14's kb_local_* rows cannot run until E8b lands"
        ) from e
    return mod.load_corpus(path)


# --------------------------------------------------------------------------- startup check

def validate() -> None:
    """Fail at run start (not mid-fold) on an unusable configuration.

    Called from ``WebSearchAgent.__init__``, i.e. when the three role agents are
    built, before any LLM call. ``local`` loads the corpus once to prove it works.
    """
    b = backend()
    routing()
    if b == "local":
        load_local_corpus()


# --------------------------------------------------------------------------- E7 routing

def summariser_route(llm_cfg: Any, caller_role: Optional[str]) -> tuple[str, str]:
    """``(endpoint, model)`` for the summariser.

    ``global`` (default) returns ``llm_cfg.endpoint`` / ``llm_cfg.model`` exactly as
    before E7. ``caller`` returns the calling role's endpoint and model; a call with
    no known caller role falls back to the global pair.
    """
    if routing() == "caller" and caller_role in CALLER_ROLES:
        return llm_cfg.endpoint_for(caller_role), llm_cfg.model_for(caller_role)
    return llm_cfg.endpoint, llm_cfg.model
