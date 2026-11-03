"""E8b local corpus backend: TF-IDF retrieval over a mounted folder (AGENT 2027, A14).

Selected by ``ROGUE_ONE_KNOWLEDGE_BACKEND=local`` with ``ROGUE_ONE_LOCAL_CORPUS=
kt3_correct|kt3_wrong`` (see :mod:`backend`). The corpus folder is mounted read-only at
``/workspace/corpora/<ROGUE_ONE_LOCAL_CORPUS>/`` for A14's ``kb_local_*`` rows only;
it is never baked into an image (``.dockerignore``, A14-T2).

Contract used by ``web_search._local_search_impl`` (E9e wiring point):

* :func:`load_corpus` ``(path) -> LocalCorpus``
* ``LocalCorpus.search(query)`` -> at most :data:`TOP_K` hits ``(doc_id, score, text)``,
  scores > 0, best first; ``text`` = ``"<title>: <first 1,500 chars of the body>"``.
  No hit with a positive score -> ``[]`` (the caller then tries the next query variant,
  finally "No results found.").
* ``LocalCorpus.sha256`` -> the A14-T1 corpus hash, logged as ``kt3_corpus_sha256``.

TF-IDF settings (A14 §4, pre-registered in X10 V.14)::

    TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), sublinear_tf=True)

fitted on whole documents (title line + body); score = cosine similarity of the query
vector with each document vector (rows are L2-normalised, so a dot product). Ties are
broken by ``doc_id`` so the ranking is deterministic.

Document format: every regular file in the folder is one UTF-8 document; ``doc_id`` is
its path relative to the folder (``/`` separators). The first non-empty line is the
title (a leading Markdown ``#`` is stripped); the rest, stripped, is the body.

Corpus hash (A14-T1): SHA-256 of the sorted ``sha256sum`` listing of the folder's files
by relative path, i.e. byte-identical to::

    cd <corpus> && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum

with GNU coreutils' default text-mode lines ``<hex>  ./<rel>`` (two spaces), as on Linux
(CAIR containers). Git for Windows' ``sha256sum`` defaults to binary mode
(``<hex> *./<rel>``) and gives a different hash there; use ``sha256sum --text`` in the
inner call on Windows.

This module never touches the web provider, the E4 search cache or the
``serper_calls`` counter.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Union

TOP_K = 3
SNIPPET_CHARS = 1500
VECTORIZER_KWARGS = dict(lowercase=True, stop_words="english", ngram_range=(1, 2), sublinear_tf=True)


class LocalCorpusError(RuntimeError):
    """The corpus folder cannot be served (missing, empty, unreadable, bad file name)."""


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    body: str

    @property
    def text(self) -> str:
        """What TF-IDF indexes: the whole document (title + body)."""
        return f"{self.title}\n{self.body}"

    @property
    def snippet(self) -> str:
        """The tool's hit text: ``<title>: <first 1,500 chars of the body>``."""
        return f"{self.title}: {self.body[:SNIPPET_CHARS]}"


# --------------------------------------------------------------------------- files and hash

def _files(root: Path) -> list[tuple[str, Path]]:
    """``(relative posix path, path)`` of every regular file, in ``LC_ALL=C sort`` order
    of ``./<relative path>`` (byte order), as ``find . -type f | sort`` lists them."""
    root = Path(root)
    if not root.is_dir():
        raise LocalCorpusError(f"corpus folder {root} is missing")
    out = []
    for p in root.rglob("*"):
        if p.is_file():
            rel = p.relative_to(root).as_posix()
            if any(ch in rel for ch in ("\\", "\n", "\r")):
                # sha256sum escapes such names, which would break the hash definition
                raise LocalCorpusError(f"corpus file name not allowed: {rel!r}")
            out.append((rel, p))
    out.sort(key=lambda t: ("./" + t[0]).encode("utf-8"))
    if not out:
        raise LocalCorpusError(f"corpus folder {root} is empty")
    return out


def sha256_listing(path: Union[str, Path]) -> str:
    """The ``sha256sum`` listing of the folder (``<hex>  ./<rel>`` lines, sorted)."""
    lines = []
    for rel, p in _files(Path(path)):
        lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  ./{rel}\n")
    return "".join(lines)


def corpus_sha256(path: Union[str, Path]) -> str:
    """A14-T1 corpus hash (see module docstring)."""
    return hashlib.sha256(sha256_listing(path).encode("utf-8")).hexdigest()


def read_documents(path: Union[str, Path]) -> list[Document]:
    docs = []
    for rel, p in _files(Path(path)):
        try:
            raw = p.read_bytes().decode("utf-8")
        except UnicodeDecodeError as e:
            raise LocalCorpusError(f"corpus file {rel} is not UTF-8") from e
        lines = raw.replace("\r\n", "\n").split("\n")
        while lines and not lines[0].strip():
            lines.pop(0)
        if not lines:
            raise LocalCorpusError(f"corpus file {rel} is empty")
        title = lines[0].strip().lstrip("#").strip()
        body = "\n".join(lines[1:]).strip()
        docs.append(Document(doc_id=rel, title=title, body=body))
    return docs


# --------------------------------------------------------------------------- index

class LocalCorpus:
    """TF-IDF index over one corpus folder (A14 §4 settings)."""

    def __init__(self, documents: list[Document], sha256: str, path: Union[str, Path, None] = None):
        from sklearn.feature_extraction.text import TfidfVectorizer

        if not documents:
            raise LocalCorpusError("corpus has no documents")
        self.documents = list(documents)
        self.sha256 = sha256
        self.path = None if path is None else Path(path)
        self.vectorizer = TfidfVectorizer(**VECTORIZER_KWARGS)
        self.matrix = self.vectorizer.fit_transform([d.text for d in self.documents])

    @property
    def doc_ids(self) -> list[str]:
        return [d.doc_id for d in self.documents]

    def scores(self, query: str) -> list[tuple[str, float]]:
        """Cosine score of ``query`` against every document, in corpus order."""
        q = self.vectorizer.transform([query or ""])
        sims = (self.matrix @ q.T).toarray().ravel()
        return [(d.doc_id, float(s)) for d, s in zip(self.documents, sims)]

    def search(self, query: str, k: int = TOP_K) -> list[tuple[str, float, str]]:
        """Top ``k`` hits with a positive score, best first, ties by ``doc_id``."""
        by_id = {d.doc_id: d for d in self.documents}
        ranked = sorted(
            ((doc_id, s) for doc_id, s in self.scores(query) if s > 0.0),
            key=lambda t: (-t[1], t[0]),
        )
        return [(doc_id, s, by_id[doc_id].snippet) for doc_id, s in ranked[:k]]


def load_corpus(path: Union[str, Path]) -> LocalCorpus:
    """Index the corpus folder at ``path``; refuses a missing or empty folder."""
    path = Path(path)
    sha = corpus_sha256(path)
    return LocalCorpus(read_documents(path), sha, path)
