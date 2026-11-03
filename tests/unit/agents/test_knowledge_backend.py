"""E9e knowledge-backend switch and E7 web-search routing (A13-T7, A14-T7/T8 guards).

Agents run with a fake ``Runner`` (helpers shared with the E2 tests), so the system
prompts and tool lists are the ones a real run would send. The ``web`` goldens are
the SHA-256 of the prompt dumps rendered by ``agent2027/v2`` @ c67ddbd (Full-v2
before E9e/E7) with these same helpers: any change to them under the default (or
the Full-v2 ``ROGUE_ONE_WEBSEARCH_ROUTING=caller``) is a Full-v2 prompt change.
"""

import asyncio
import hashlib
import os
import re
from pathlib import Path
from types import SimpleNamespace

for _k, _v in {"EMBEDD_API_KEY": "x", "EMBEDD_MODEL": "x", "EMBEDD_PORT": "1", "LLM_MODEL": "x",
               "LLM_API_KEY": "x", "LLM_PORT": "8010"}.items():
    os.environ.setdefault(_k, _v)

import pytest

import rogueone.llm.agents.core.knowledge.web_search as ws
import rogueone.llm.agents.core.tester.agent as tst_mod
from rogueone.llm.agents.core.knowledge import backend as kb
from rogueone.utils import instrumentation, retrieval_audit
from rogueone.utils import prompt_dump
from tests.unit.agents import test_tester_e2_fixes as tt
from tests.unit.agents import test_tool_error_feedback as tef

pytestmark = [pytest.mark.unit, pytest.mark.agent]

SEARCH_TOOLS = {"search_tool", "search_in_literature_tool"}
FORBIDDEN = ("web search", "search_tool", "search_in_literature_tool", "information from the web")

# sha256 of the v2.0 (pre-E9e) dumps; see module docstring.
GOLDEN = {
    "scientist__focus__iter012__c01.system.txt": "a326f46194b64d53ca068b9030617a6de6da2a2ab68f5ae4ce1432f64af14cba",
    "scientist__focus__iter012__c01.tools.json": "0e2f782617129dda22806b2ca07f5ff8f4a05785ef5911fdb176316a67512f9f",
    "extractor__append__iter000__c01.system.txt": "5601b35ec57f6cf3cc60d7002b90d4fee23117e9327687841129a523678d17f1",
    "extractor__append__iter000__c01.tools.json": "45c2c67a08492d136b854bc86f223a419a7cbf8fb07d368018ae52f45a8e3bfd",
    "tester__tester-agent-assessment__iter000__c01.system.txt": "94ec31b38c44a8e1eff8f4eb9837dfca17ead33dc927b5168788abf96309a24c",
    "tester__tester-agent-assessment__iter000__c01.tools.json": "d810b65dd8a5b4099fcdccde777e26f0a36e7a1801bd4f2fdc5bb570901f927e",
    "tester__tester-agent-pruning__iter000__c01.system.txt": "80e1c9947b23dce5dea226879862b2ebd039516c8c7cfb7204001f04f92c4a4b",
    "tester__tester-agent-pruning__iter000__c01.tools.json": "862e7595ad1da364dbe9a6e68716875340e90cc85bb291b426896f5a2403e2e2",
    "websearch__search__iter000__c01.system.txt": "62e0d12c041e5682e232b5923361bd78d9c9bf7b153bc3297074d238dcf6aa36",
    "websearch__search__iter000__c01.tools.json": "6e96f0d770a6cf8681e2fb79959be4ff5ade599d639fb1f2d13df80410c4be50",
}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for k in (kb.BACKEND_ENV, kb.ROUTING_ENV, kb.LOCAL_CORPUS_ENV, kb.CORPORA_ROOT_ENV):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("ROGUE_ONE_WEB_SEARCH_PROVIDER", "searxng")  # no Serper key needed
    ws._LOCAL_CORPUS_CACHE.clear()
    prompt_dump.reset_counters()
    yield
    prompt_dump.reset_counters()


# --------------------------------------------------------------------------- rendering

class _WRunner:
    seen: list = []

    @staticmethod
    async def run(a, *args, **kwargs):
        _WRunner.seen.append(a)
        return SimpleNamespace(final_output="answer")


def _render(tmp_path, monkeypatch, *, summariser=True):
    """Run S (iteration 12), E, T (Stage A + B) and W once; return captured agents."""
    out = tmp_path / "dump"
    monkeypatch.setenv(prompt_dump.FLAG, "1")
    monkeypatch.setenv(prompt_dump.DIR_ENV, str(out / "prompts"))
    agents = {}

    sci = tef._make_scientist(tmp_path / "s")
    st = tef._run_scientist(sci, monkeypatch, lambda tools: _keep(agents, "S", tools), 0)
    assert "error" not in st, st.get("error")
    ext = tef._make_extractor(tmp_path / "e")
    st = tef._run_extractor(ext, monkeypatch, lambda tools: _keep(agents, "E", tools))
    assert "error" not in st, st.get("error")

    class TRunner:
        @staticmethod
        async def run(a, *args, **kwargs):
            agents.setdefault("T:" + a.name, a)
            return SimpleNamespace(final_output="ok")

    monkeypatch.setattr(tst_mod, "Runner", TRunner)
    tester = tt._make_tester(tmp_path / "t")
    tester.include_report = True
    asyncio.run(tester.test_hypotheses(tt._make_folds(), tt._expl(), step=0, trial_index=0))

    if summariser:
        _WRunner.seen = []
        monkeypatch.setattr(ws, "Runner", _WRunner)
        asyncio.run(ws._run_search_agent("q"))
        agents["W"] = _WRunner.seen[-1]
    files = {p.name: p.read_bytes() for p in (out / "prompts").iterdir()}
    return agents, files


async def _keep(store, key, tools):
    store[key] = set(tools)
    return {}


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _system(files, stem):
    return files[f"{stem}.system.txt"].decode("utf-8")


# --------------------------------------------------------------------------- flags

def test_defaults_are_web_and_global():
    assert kb.backend() == "web"
    assert kb.routing() == "global"


def test_registry_defaults():
    from rogueone.utils import flag_registry as fr

    assert fr.FLAG_DEFAULTS["ROGUE_ONE_KNOWLEDGE_BACKEND"] == "web"
    assert fr.FLAG_DEFAULTS["ROGUE_ONE_WEBSEARCH_ROUTING"] == "global"


@pytest.mark.parametrize("env,val", [(kb.BACKEND_ENV, "serper"), (kb.ROUTING_ENV, "role")])
def test_bad_values_refuse(monkeypatch, env, val):
    monkeypatch.setenv(env, val)
    with pytest.raises(kb.KnowledgeBackendError):
        kb.validate()


@pytest.mark.parametrize("backend", ["web", "parametric", "local"])
def test_role_side_is_identity_unless_off(monkeypatch, backend):
    monkeypatch.setenv(kb.BACKEND_ENV, backend)
    tools = [SimpleNamespace(name="search_tool"), SimpleNamespace(name="x")]
    instr = "anything at all"
    assert kb.role_tools(tools) is tools
    for role in ("scientist", "extractor", "tester"):
        assert kb.role_instructions(role, instr) is instr


# --------------------------------------------------------------------------- Full-v2 byte identity

@pytest.mark.parametrize("routing", [None, "global", "caller"])
def test_web_prompts_match_v20_golden(tmp_path, monkeypatch, routing):
    """Full-v2 (backend web, routing default/global/caller, uniform model) renders
    exactly the v2.0 prompts and tool lists."""
    if routing:
        monkeypatch.setenv(kb.ROUTING_ENV, routing)
    monkeypatch.setenv(kb.BACKEND_ENV, "web")
    _, files = _render(tmp_path, monkeypatch)
    got = {name: _sha(files[name]) for name in GOLDEN}
    assert got == GOLDEN


# --------------------------------------------------------------------------- off (A13 edit list)

def test_off_removes_search_from_all_roles(tmp_path, monkeypatch):
    monkeypatch.setenv(kb.BACKEND_ENV, "off")
    agents, files = _render(tmp_path, monkeypatch, summariser=False)
    assert "search_tool" not in agents["S"] and "generic_pandas_tool" in agents["S"]
    assert "search_in_literature_tool" not in agents["E"] and "append_new_attribute" in agents["E"]
    stage_a = agents["T:Tester Agent (Assessment)"]
    assert not SEARCH_TOOLS & {t.name for t in stage_a.tools}
    for stem in ("scientist__focus__iter012__c01", "extractor__append__iter000__c01"):
        text = _system(files, stem)
        for word in FORBIDDEN:
            assert word not in text, (stem, word)
    # W is never rendered: nobody can call it.
    assert not any(n.startswith("websearch") for n in files)


def test_off_diff_is_exactly_the_edit_list(tmp_path, monkeypatch):
    import difflib

    monkeypatch.setenv(kb.BACKEND_ENV, "web")
    _, web = _render(tmp_path / "w", monkeypatch, summariser=False)
    prompt_dump.reset_counters()
    monkeypatch.setenv(kb.BACKEND_ENV, "off")
    _, off = _render(tmp_path / "o", monkeypatch, summariser=False)

    def changes(stem):
        a = _system(web, stem).splitlines()
        b = _system(off, stem).splitlines()
        return [l for l in difflib.unified_diff(a, b, lineterm="", n=0)
                if l[:1] in "+-" and not l.startswith(("+++", "---"))]

    def norm(lines):
        return [l[0] + l[1:].strip() for l in lines]

    sci = changes("scientist__focus__iter012__c01")
    assert norm(sci) == [
        "-Your task is to analyze the test results, the attribute explanations, the focus history, and compare with existing information from the web to determine the focus area of the investigation. Find the focus area that will lead to the most useful hypotheses.",
        "+Your task is to analyze the test results, the attribute explanations, and the focus history to determine the focus area of the investigation. Find the focus area that will lead to the most useful hypotheses.",
        "-3. Use tools to gather missing context (search, lookup, quick dataframe checks). Always perform at least one web search.",
        "+3. Use tools to gather missing context (lookup, quick dataframe checks).",
        "-- Use search_tool actively when uncertain.",
    ]
    assert norm(changes("extractor__append__iter000__c01")) == ["-- search_in_literature_tool"]
    for stem in ("tester__tester-agent-assessment__iter000__c01", "tester__tester-agent-pruning__iter000__c01"):
        assert web[f"{stem}.system.txt"] == off[f"{stem}.system.txt"]
    # user messages never change
    for name in web:
        if name.endswith(".user.txt"):
            assert web[name] == off[name], name


def test_off_edit_on_drifted_prompt_fails_loudly(monkeypatch):
    monkeypatch.setenv(kb.BACKEND_ENV, "off")
    with pytest.raises(kb.KnowledgeBackendError):
        kb.role_instructions("scientist", "a prompt without the search lines")


def test_off_summariser_cannot_run(monkeypatch):
    monkeypatch.setenv(kb.BACKEND_ENV, "off")
    with pytest.raises(kb.KnowledgeBackendError):
        asyncio.run(ws._run_search_agent("q"))


# --------------------------------------------------------------------------- parametric

def _no_provider(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("provider or cache touched")

    monkeypatch.setattr(ws, "make_search_client", boom)
    monkeypatch.setattr(ws.search_cache, "get_cache", boom)


def test_parametric_summariser(tmp_path, monkeypatch):
    monkeypatch.setenv(kb.BACKEND_ENV, "parametric")
    _no_provider(monkeypatch)
    agents, files = _render(tmp_path, monkeypatch)
    w = agents["W"]
    assert w.tools == []
    assert kb.PARAMETRIC_TEXT in w.instructions
    rest = w.instructions.replace(kb.PARAMETRIC_TEXT, "")
    for word in ("search", "web", "Serper", "RAG", "Google", "tool"):
        assert not re.search(rf"{word}", rest, re.I), word
    # role side unchanged: same prompts and tools as Full-v2
    for name, digest in GOLDEN.items():
        if not name.startswith("websearch"):
            assert _sha(files[name]) == digest, name


def test_parametric_prompt_is_line_edit_of_web():
    import inspect

    src = inspect.getsource(ws._run_search_agent)
    web = src.split('summariser_instructions(f"""', 1)[1].split('""")', 1)[0]
    par = kb.parametric_instructions(web)
    kept = [l for l in par.split("\n") if kb.PARAMETRIC_TEXT not in l and not l.strip().startswith("2. Provide")]
    web_lines = web.split("\n")
    assert all(l in web_lines for l in kept)  # nothing new except the two edited lines
    assert par.count(kb.PARAMETRIC_TEXT) == 1


# --------------------------------------------------------------------------- local (E8b wiring point)

def test_local_refuses_without_corpus(tmp_path, monkeypatch):
    monkeypatch.setenv(kb.BACKEND_ENV, "local")
    monkeypatch.setenv(kb.LOCAL_CORPUS_ENV, "kt3_correct")
    monkeypatch.setenv(kb.CORPORA_ROOT_ENV, str(tmp_path))
    with pytest.raises(kb.KnowledgeBackendError, match="missing or empty"):
        kb.validate()
    monkeypatch.setenv(kb.LOCAL_CORPUS_ENV, "kt3_other")
    with pytest.raises(kb.KnowledgeBackendError, match="ROGUE_ONE_LOCAL_CORPUS"):
        kb.validate()


def test_local_refuses_without_e8b_backend(tmp_path, monkeypatch):
    """No local_corpus.py in this image: the run refuses to start (WebSearchAgent init)."""
    import importlib.util

    if importlib.util.find_spec(kb.LOCAL_CORPUS_MODULE) is not None:
        pytest.skip("E8b backend present")
    (tmp_path / "kt3_correct").mkdir()
    (tmp_path / "kt3_correct" / "doc.txt").write_text("text")
    monkeypatch.setenv(kb.BACKEND_ENV, "local")
    monkeypatch.setenv(kb.LOCAL_CORPUS_ENV, "kt3_correct")
    monkeypatch.setenv(kb.CORPORA_ROOT_ENV, str(tmp_path))
    with pytest.raises(NotImplementedError, match="E8b"):
        ws.WebSearchAgent(cfg=None)


class _FakeCorpus:
    def __init__(self, hits_for):
        self.hits_for = hits_for
        self.queries = []

    def search(self, query):
        self.queries.append(query)
        return self.hits_for(query)


def test_local_search_impl(monkeypatch):
    rows = []
    monkeypatch.setattr(retrieval_audit, "record", lambda **kw: rows.append(kw))
    before = {k: instrumentation.get_counter(k) for k in ("serper_calls", "search_cache_hits", "search_cache_misses")}
    corpus = _FakeCorpus(lambda q: [("p7.md", 0.9, "P7: doc"), ("d1.md", 0.0, "zero")]
                         if q == "alpha beta gamma" else [("d1.md", 0.0, "zero")])
    out = ws._local_search_impl(corpus, "alpha beta gamma delta epsilon six seven eight nine")
    assert out == "P7: doc"
    assert corpus.queries == ["alpha beta gamma delta epsilon six seven eight nine",
                              "alpha beta gamma delta epsilon six seven eight",
                              "alpha beta gamma delta epsilon", "alpha beta gamma"]
    assert {r["channel"] for r in rows} == {"local_corpus"}
    assert rows[-1]["doc_ids"] == ["p7.md"] and rows[-1]["ok"] is True
    assert rows[0]["ok"] is False and rows[0]["doc_ids"] == []
    assert {k: instrumentation.get_counter(k) for k in before} == before
    assert ws._local_search_impl(_FakeCorpus(lambda q: []), "nothing here") == "No results found."


def test_local_summariser_keeps_web_prompt_and_tool(tmp_path, monkeypatch):
    """Under local the summariser prompt and tool schema equal Full-v2's (A14-T8)."""
    monkeypatch.setenv(kb.BACKEND_ENV, "local")
    _no_provider(monkeypatch)
    corpus = _FakeCorpus(lambda q: [("p7.md", 1.0, "P7: doc")])
    monkeypatch.setattr(ws, "_get_local_corpus", lambda: corpus)
    monkeypatch.setenv(prompt_dump.FLAG, "1")
    monkeypatch.setenv(prompt_dump.DIR_ENV, str(tmp_path / "prompts"))
    _WRunner.seen = []
    monkeypatch.setattr(ws, "Runner", _WRunner)
    asyncio.run(ws._run_search_agent("q"))
    for name in ("websearch__search__iter000__c01.system.txt", "websearch__search__iter000__c01.tools.json"):
        assert _sha((tmp_path / "prompts" / name).read_bytes()) == GOLDEN[name]


# --------------------------------------------------------------------------- E7 routing

def _llm_cfg(roles=None, model="M120", port="8010"):
    roles = roles or {}
    return SimpleNamespace(
        model=model,
        endpoint=f"http://localhost:{port}/v1",
        model_for=lambda r: roles.get(r, (model, port))[0],
        endpoint_for=lambda r: f"http://localhost:{roles.get(r, (model, port))[1]}/v1",
    )


@pytest.mark.parametrize("role", ["scientist", "extractor", "tester", None, "websearch"])
def test_caller_routing_is_noop_for_uniform_models(monkeypatch, role):
    cfg = _llm_cfg()
    monkeypatch.setenv(kb.ROUTING_ENV, "global")
    g = kb.summariser_route(cfg, role)
    monkeypatch.setenv(kb.ROUTING_ENV, "caller")
    assert kb.summariser_route(cfg, role) == g == ("http://localhost:8010/v1", "M120")


def test_caller_routing_follows_role_in_mixed_config(monkeypatch):
    cfg = _llm_cfg({"extractor": ("M20", "8020")})
    monkeypatch.setenv(kb.ROUTING_ENV, "caller")
    assert kb.summariser_route(cfg, "extractor") == ("http://localhost:8020/v1", "M20")
    assert kb.summariser_route(cfg, "scientist") == ("http://localhost:8010/v1", "M120")
    assert kb.summariser_route(cfg, None) == ("http://localhost:8010/v1", "M120")
    monkeypatch.setenv(kb.ROUTING_ENV, "global")
    assert kb.summariser_route(cfg, "extractor") == ("http://localhost:8010/v1", "M120")


def test_summariser_uses_route_end_to_end(monkeypatch):
    cfg = _llm_cfg({"extractor": ("M20", "8020")})
    monkeypatch.setattr(ws, "llm_cfg", SimpleNamespace(api_key="k", **vars(cfg)))
    seen = {}
    real = ws.create_openai_client_for_vllm

    def spy(base_url, **kw):
        seen["endpoint"] = base_url
        return real(base_url=base_url, **kw)

    monkeypatch.setattr(ws, "create_openai_client_for_vllm", spy)
    monkeypatch.setattr(ws, "Runner", _WRunner)
    for routing, want in (("caller", ("http://localhost:8020/v1", "M20")),
                          ("global", ("http://localhost:8010/v1", "M120"))):
        monkeypatch.setenv(kb.ROUTING_ENV, routing)
        _WRunner.seen = []
        with instrumentation.caller_scope("extractor"):
            asyncio.run(ws._run_search_agent("q"))
        assert (seen["endpoint"], _WRunner.seen[-1].model.model) == want, routing


def test_model_w_label(monkeypatch):
    from rogueone.utils.wandb_utils import _summariser_model_label

    uni = _llm_cfg()
    mixed = _llm_cfg({"extractor": ("M20", "8020")})
    monkeypatch.setenv(kb.ROUTING_ENV, "global")
    assert _summariser_model_label(mixed) == "M120"
    monkeypatch.setenv(kb.ROUTING_ENV, "caller")
    assert _summariser_model_label(uni) == "M120"
    assert _summariser_model_label(mixed) == "caller"


@pytest.mark.parametrize("backend,channel", [("web", "web_search"), ("parametric", "parametric"),
                                             ("local", "local_corpus")])
def test_answer_channel(monkeypatch, backend, channel):
    monkeypatch.setenv(kb.BACKEND_ENV, backend)
    assert kb.answer_channel() == channel
