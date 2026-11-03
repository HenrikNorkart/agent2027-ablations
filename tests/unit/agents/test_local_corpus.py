"""E8b local TF-IDF corpus backend (A14-T1, A14-T4, A14-T7).

The KT3 corpora live under ``scripts/agent2027/corpora/`` and are excluded from every
image (``.dockerignore``, A14-T2), so the tests that read them skip inside a container.
"""

import hashlib
import os
import re
import shutil
import subprocess
from pathlib import Path

for _k, _v in {"EMBEDD_API_KEY": "x", "EMBEDD_MODEL": "x", "EMBEDD_PORT": "1", "LLM_MODEL": "x",
               "LLM_API_KEY": "x", "LLM_PORT": "8010"}.items():
    os.environ.setdefault(_k, _v)

import pytest

import rogueone.llm.agents.core.knowledge.web_search as ws
from rogueone.llm.agents.core.knowledge import backend as kb
from rogueone.llm.agents.core.knowledge import local_corpus as lc
from rogueone.utils import instrumentation, retrieval_audit

pytestmark = [pytest.mark.unit, pytest.mark.agent]

REPO = Path(__file__).resolve().parents[3]
CORPORA = REPO / "scripts" / "agent2027" / "corpora"
P7_DOC = "marenko_index.md"
FORMULA_START = "The Marenko index of a cable segment is"
needs_corpora = pytest.mark.skipif(not (CORPORA / "kt3_correct").is_dir(),
                                   reason="KT3 corpora are not in this checkout/image")


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for k in (kb.BACKEND_ENV, kb.ROUTING_ENV, kb.LOCAL_CORPUS_ENV, kb.CORPORA_ROOT_ENV):
        monkeypatch.delenv(k, raising=False)
    ws._LOCAL_CORPUS_CACHE.clear()
    yield
    ws._LOCAL_CORPUS_CACHE.clear()


def _write(root: Path, files: dict) -> Path:
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(text.encode("utf-8"))
    return root


TOY = {
    "p7.md": "# Zorblat index for widgets\n\nThe Zorblat index is the widget load divided by the widget span.\n",
    "d1.md": "# Widget corrosion\n\nCorrosion of widgets depends on humidity and temperature.\n",
    "d2.md": "# Widget vibration\n\nVibration of widgets is measured as an RMS velocity.\n",
    "d3.md": "# Widget records\n\nRecords hold the widget load, the humidity and the operator shift.\n",
    "sub/d4.md": "# Widget anchors\n\nAnchors hold the widget at both ends of the span.\n",
}


# --------------------------------------------------------------------------- hash (A14-T1)

def test_hash_is_sorted_sha256sum_listing(tmp_path):
    root = _write(tmp_path / "c", TOY)
    names = sorted(("./" + r for r in TOY), key=lambda s: s.encode())
    listing = "".join(
        f"{hashlib.sha256((root / n[2:]).read_bytes()).hexdigest()}  {n}\n" for n in names
    )
    assert lc.sha256_listing(root) == listing
    assert lc.corpus_sha256(root) == hashlib.sha256(listing.encode()).hexdigest()
    assert lc.load_corpus(root).sha256 == lc.corpus_sha256(root)


def test_hash_matches_shell_pipeline(tmp_path):
    """The A14-T1 one-liner, where a POSIX shell with coreutils is available."""
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("no bash")
    root = _write(tmp_path / "c", TOY)
    # --text: GNU text-mode "<hex>  <name>" lines (the Linux default; Git for Windows
    # defaults to binary mode "<hex> *<name>", which hashes differently).
    cmd = "find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum --text | sha256sum"
    try:
        out = subprocess.run([bash, "-c", cmd], cwd=root, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as e:
        pytest.skip(f"shell unavailable: {e}")
    if out.returncode != 0 or not re.match(r"^[0-9a-f]{64}\b", out.stdout):
        pytest.skip(f"coreutils unavailable: {out.stderr[:200]}")
    assert out.stdout.split()[0] == lc.corpus_sha256(root)


def test_hash_changes_with_content_and_name(tmp_path):
    a = _write(tmp_path / "a", TOY)
    h = lc.corpus_sha256(a)
    (a / "d1.md").write_bytes(b"# Widget corrosion\n\nchanged\n")
    assert lc.corpus_sha256(a) != h
    b = _write(tmp_path / "b", TOY)
    (b / "d1.md").rename(b / "d1b.md")
    assert lc.corpus_sha256(b) != h


def test_missing_or_empty_folder_refuses(tmp_path):
    with pytest.raises(lc.LocalCorpusError, match="missing"):
        lc.load_corpus(tmp_path / "nope")
    (tmp_path / "empty").mkdir()
    with pytest.raises(lc.LocalCorpusError, match="empty"):
        lc.load_corpus(tmp_path / "empty")


def test_non_utf8_refuses(tmp_path):
    root = _write(tmp_path / "c", TOY)
    (root / "bad.md").write_bytes(b"# T\n\xff\xfe")
    with pytest.raises(lc.LocalCorpusError, match="UTF-8"):
        lc.load_corpus(root)


# --------------------------------------------------------------------------- search (A14 §4)

def test_vectorizer_settings_are_the_preregistered_ones(tmp_path):
    c = lc.load_corpus(_write(tmp_path / "c", TOY))
    p = c.vectorizer.get_params()
    assert (p["lowercase"], p["stop_words"], p["ngram_range"], p["sublinear_tf"]) == (True, "english", (1, 2), True)
    assert lc.TOP_K == 3 and lc.SNIPPET_CHARS == 1500


def test_search_top3_positive_sorted_and_formatted(tmp_path):
    c = lc.load_corpus(_write(tmp_path / "c", TOY))
    hits = c.search("widget load span humidity")
    assert 1 <= len(hits) <= 3
    assert all(s > 0 for _, s, _ in hits)
    assert [s for _, s, _ in hits] == sorted((s for _, s, _ in hits), reverse=True)
    ids = {d.doc_id for d in c.documents}
    assert "sub/d4.md" in ids
    for doc_id, _, text in hits:
        d = next(x for x in c.documents if x.doc_id == doc_id)
        assert text == f"{d.title}: {d.body[:1500]}"
    assert c.search("Zorblat index")[0][0] == "p7.md"


def test_search_snippet_is_cut_at_1500(tmp_path):
    root = _write(tmp_path / "c", {"long.md": "# Long doc\n" + "zorblat " * 600})
    (_, _, text), = lc.load_corpus(root).search("zorblat")
    assert text.startswith("Long doc: ") and len(text) == len("Long doc: ") + 1500


def test_search_no_match_returns_empty(tmp_path):
    c = lc.load_corpus(_write(tmp_path / "c", TOY))
    assert c.search("quasar nebula") == []
    assert c.search("the and of") == []  # stop words only
    assert c.search("") == []


def test_search_is_deterministic_with_ties(tmp_path):
    root = _write(tmp_path / "c", {"b.md": "# Same\nalpha beta\n", "a.md": "# Same\nalpha beta\n"})
    hits = lc.load_corpus(root).search("alpha")
    assert [h[0] for h in hits] == ["a.md", "b.md"]
    assert hits[0][1] == pytest.approx(hits[1][1])


# --------------------------------------------------------------------------- wiring (E9e + E8b)

def _local_env(monkeypatch, root: Path, name: str = "kt3_correct"):
    monkeypatch.setenv(kb.BACKEND_ENV, "local")
    monkeypatch.setenv(kb.LOCAL_CORPUS_ENV, name)
    monkeypatch.setenv(kb.CORPORA_ROOT_ENV, str(root))


def test_backend_loads_local_corpus_and_agent_starts(tmp_path, monkeypatch):
    _write(tmp_path / "kt3_correct", TOY)
    _local_env(monkeypatch, tmp_path)
    corpus = kb.load_local_corpus()
    assert isinstance(corpus, lc.LocalCorpus)
    assert corpus.sha256 == lc.corpus_sha256(tmp_path / "kt3_correct")
    ws.WebSearchAgent(cfg=None)  # validate() passes: no NotImplementedError any more


def test_local_search_impl_with_real_corpus_touches_no_web(tmp_path, monkeypatch):
    _write(tmp_path / "kt3_correct", TOY)
    _local_env(monkeypatch, tmp_path)
    rows = []
    monkeypatch.setattr(retrieval_audit, "record", lambda **kw: rows.append(kw))
    monkeypatch.setattr(ws, "make_search_client", lambda *a, **k: pytest.fail("web client built"))
    monkeypatch.setattr(ws.search_cache, "get_cache", lambda *a, **k: pytest.fail("E4 cache opened"))
    before = {k: instrumentation.get_counter(k) for k in ("serper_calls", "search_cache_hits", "search_cache_misses")}
    out = ws._local_search_impl(ws._get_local_corpus(), "Zorblat index widgets")
    assert out.startswith("Zorblat index for widgets: ")
    assert rows[-1]["channel"] == "local_corpus" and rows[-1]["doc_ids"][0] == "p7.md"
    assert all(s > 0 for s in rows[-1]["doc_scores"])
    assert {k: instrumentation.get_counter(k) for k in before} == before


def test_wandb_config_logs_corpus_hash_only_under_local(tmp_path, monkeypatch):
    from rogueone.utils import wandb_utils

    assert wandb_utils.kt3_corpus_config() == {}
    monkeypatch.setenv(kb.BACKEND_ENV, "parametric")
    assert wandb_utils.kt3_corpus_config() == {}
    _write(tmp_path / "kt3_wrong", TOY)
    _local_env(monkeypatch, tmp_path, "kt3_wrong")
    assert wandb_utils.kt3_corpus_config() == {"kt3_corpus_sha256": lc.corpus_sha256(tmp_path / "kt3_wrong")}
    monkeypatch.setenv(kb.LOCAL_CORPUS_ENV, "kt3_correct")  # not present: logged, not raised
    assert wandb_utils.kt3_corpus_config()["kt3_corpus_sha256"].startswith("ERROR")


# --------------------------------------------------------------------------- the KT3 corpora (A14-T1/T4)

def _formula_paragraph(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    para = next(p for p in text.split("\n\n") if p.startswith(FORMULA_START))
    return para


@needs_corpora
def test_kt3_corpora_differ_only_in_the_p7_formula_paragraph():
    a, b = CORPORA / "kt3_correct", CORPORA / "kt3_wrong"
    files_a = sorted(p.relative_to(a).as_posix() for p in a.rglob("*") if p.is_file())
    files_b = sorted(p.relative_to(b).as_posix() for p in b.rglob("*") if p.is_file())
    assert files_a == files_b and P7_DOC in files_a
    assert 6 <= len(files_a) - 1 <= 10  # 5-10 distractors
    for rel in files_a:
        ba, bb = (a / rel).read_bytes(), (b / rel).read_bytes()
        assert b"\r" not in ba and b"\r" not in bb  # LF bytes, so the hash is machine-independent
        if rel != P7_DOC:
            assert ba == bb, rel
    pa, pb = _formula_paragraph(a / P7_DOC), _formula_paragraph(b / P7_DOC)
    ta, tb = (a / P7_DOC).read_text(encoding="utf-8"), (b / P7_DOC).read_text(encoding="utf-8")
    assert pa != pb and ta.replace(pa, "") == tb.replace(pb, "")
    assert abs(len(tb) - len(ta)) <= 0.10 * len(ta)  # length within 10%
    for t, p in ((ta, pa), (tb, pb)):
        assert t.index(p) + len(p) <= 1000  # formula paragraph within the first 1,000 chars
        doc = lc.read_documents(a if t is ta else b)
        snippet = next(d for d in doc if d.doc_id == P7_DOC).snippet
        assert p in snippet  # and therefore inside the returned hit text
    assert lc.corpus_sha256(a) != lc.corpus_sha256(b)


@needs_corpora
@pytest.mark.parametrize("name", ["kt3_correct", "kt3_wrong"])
@pytest.mark.parametrize("query", ["Marenko index", "Marenko index cable inspection suspension bridge",
                                   "Marenko", "marenko cable"])
def test_p7_document_ranks_first_for_its_domain_term(name, query):
    hits = lc.load_corpus(CORPORA / name).search(query)
    assert hits and hits[0][0] == P7_DOC


@needs_corpora
def test_distractors_never_name_the_domain_term():
    for p in (CORPORA / "kt3_correct").iterdir():
        if p.name != P7_DOC:
            assert "marenko" not in p.read_text(encoding="utf-8").lower(), p.name


@needs_corpora
def test_p7_constants_appear_only_in_the_corpora():
    """A14-T2 leak guard, repo side: the numbers in the two formula paragraphs (P7's
    threshold and phi_wrong's) occur in no other file of the checkout. The constants
    are read from the corpus, so this test does not itself contain them."""
    consts = set()
    for name in ("kt3_correct", "kt3_wrong"):
        consts |= set(re.findall(r"\d+\.\d{3,}", _formula_paragraph(CORPORA / name / P7_DOC)))
    assert len(consts) == 2, consts
    skip_dirs = {".git", "__pycache__", ".venv", "venv", "node_modules", "wandb", ".pytest_cache", "tasks"}
    hits = []
    for dirpath, dirnames, filenames in os.walk(REPO):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        if Path(dirpath).resolve().is_relative_to(CORPORA.resolve()):
            dirnames[:] = []
            continue
        for fn in filenames:
            p = Path(dirpath) / fn
            try:
                if p.stat().st_size > 5_000_000:
                    continue
                data = p.read_bytes()
            except OSError:
                continue
            for c in consts:
                if c.encode() in data:
                    hits.append(f"{p.relative_to(REPO)}: {c}")
    assert not hits, hits
