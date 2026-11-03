"""Web-search query -> result cache (AGENT 2027 E4, 01 §3.3 E4 and §7 R4).

``_web_search`` (``rogueone/llm/agents/core/knowledge/web_search.py``) asks this
cache before it calls the upstream provider (Serper or SearXNG) and stores
every non-empty upstream answer here. The cached value is the exact text the
tool returned to the summariser plus the URLs and engines behind it, so a hit
returns a byte-identical tool result.

Modes (``ROGUE_ONE_SEARCH_CACHE``; unset = ``off``, the legacy behaviour):

``off``
    No cache. Every tool call goes upstream.
``run``
    In-process memo only: an identical (provider, sanitised query) asked twice
    in the same process goes upstream once. Nothing is written to disk.
``shared``
    Full-v2 value. In-process memo plus a SQLite file at
    ``ROGUE_ONE_SEARCH_CACHE_PATH`` shared by every run that mounts it.
    Read-through: a miss goes upstream and the answer is recorded. The first
    writer of a key wins (``INSERT OR IGNORE``), so parallel runs converge on
    one answer per query.
``replay``
    Read-only: answers come only from the SQLite file (and the memo). A miss
    never goes upstream; the tool sees an empty result and moves on to its
    next query variant, as it does for a live empty answer.
``record``
    Always go upstream and overwrite the stored answer (``INSERT OR REPLACE``).
    Used to refresh a cache; lookups are skipped.

Only non-empty upstream answers are stored. Empty answers and errors are not,
because during an engine outage they say more about the outage than about the
query.

The key is ``sha256(provider + "\\x1f" + sanitised query)``. The provider is
part of the key so Serper and SearXNG answers never mix.

Like ``retrieval_audit`` this module is standard-library only and every disk
operation degrades instead of raising: a broken cache file means a cache miss
(counted by the caller), never a failed run. An unknown mode raises
``ValueError``, as an unknown ``ROGUE_ONE_WEB_SEARCH_PROVIDER`` does.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from logging import getLogger
from pathlib import Path
from typing import Optional

logger = getLogger(__name__)

ENV_MODE = "ROGUE_ONE_SEARCH_CACHE"
ENV_PATH = "ROGUE_ONE_SEARCH_CACHE_PATH"

MODES = ("off", "run", "shared", "replay", "record")
DEFAULT_MODE = "off"
#: 01 §7 R4 Full-v2 value (host ``~/rogueone-serper-cache/`` mounted at ``/workspace/cache``).
DEFAULT_PATH = "/workspace/cache/serper_cache.sqlite"

#: Bound on the in-process memo (entries); oldest evicted first.
MEMO_MAX_ENTRIES = 4096
SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS search_cache (
    key        TEXT PRIMARY KEY,
    provider   TEXT NOT NULL,
    query      TEXT NOT NULL,
    result     TEXT NOT NULL,
    urls       TEXT NOT NULL,
    engines    TEXT NOT NULL,
    created_at TEXT NOT NULL,
    schema     INTEGER NOT NULL
)
"""


@dataclass(frozen=True)
class CachedResult:
    text: str
    urls: list = field(default_factory=list)
    engines: list = field(default_factory=list)
    source: str = "memo"  # "memo" or "disk"


def cache_key(provider: str, query: str) -> str:
    return hashlib.sha256(f"{provider}\x1f{query}".encode("utf-8")).hexdigest()


def cache_mode() -> str:
    mode = (os.getenv(ENV_MODE) or DEFAULT_MODE).strip().lower() or DEFAULT_MODE
    if mode not in MODES:
        raise ValueError(f"{ENV_MODE} must be one of {'|'.join(MODES)}, got {mode!r}")
    return mode


def cache_path() -> str:
    return (os.getenv(ENV_PATH) or DEFAULT_PATH).strip() or DEFAULT_PATH


class SearchCache:
    """Query -> result cache; see the module docstring for the modes."""

    def __init__(self, mode: str = DEFAULT_MODE, path: Optional[str] = None):
        if mode not in MODES:
            raise ValueError(f"{ENV_MODE} must be one of {'|'.join(MODES)}, got {mode!r}")
        self.mode = mode
        self.path: Optional[Path] = Path(path) if (path and mode in ("shared", "replay", "record")) else None
        self._memo: "OrderedDict[str, CachedResult]" = OrderedDict()
        self._lock = threading.Lock()
        self._disk_ok = False
        if self.path is not None:
            self._disk_ok = self._init_db()

    # ------------------------------------------------------------------ properties
    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    @property
    def reads_upstream(self) -> bool:
        """False only in ``replay``: a miss must not reach the provider."""
        return self.mode != "replay"

    @property
    def disk_ok(self) -> bool:
        return self._disk_ok

    # ------------------------------------------------------------------ sqlite
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=30.0)
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _init_db(self) -> bool:
        try:
            if self.mode != "replay":
                self.path.parent.mkdir(parents=True, exist_ok=True)
            elif not self.path.is_file():
                logger.warning("Search cache replay file %s does not exist; every lookup misses", self.path)
                return False
            conn = self._connect()
            try:
                if self.mode != "replay":
                    conn.execute("PRAGMA journal_mode=WAL")
                    conn.execute(_SCHEMA)
                    conn.commit()
            finally:
                conn.close()
            logger.info("Search cache (%s): %s", self.mode, self.path)
            return True
        except Exception as exc:  # noqa: BLE001 - a cache must never break a run
            logger.warning("Search cache at %s unavailable (%s); in-process memo only", self.path, exc)
            return False

    def _disk_get(self, key: str) -> Optional[CachedResult]:
        try:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT result, urls, engines FROM search_cache WHERE key = ?", (key,)
                ).fetchone()
            finally:
                conn.close()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Search cache read failed: %s", exc)
            return None
        if row is None:
            return None
        try:
            urls, engines = json.loads(row[1]), json.loads(row[2])
        except Exception:  # noqa: BLE001
            urls, engines = [], []
        return CachedResult(text=row[0], urls=list(urls), engines=list(engines), source="disk")

    def _disk_put(self, key: str, provider: str, query: str, value: CachedResult) -> bool:
        verb = "INSERT OR REPLACE" if self.mode == "record" else "INSERT OR IGNORE"
        try:
            conn = self._connect()
            try:
                conn.execute(
                    f"{verb} INTO search_cache (key, provider, query, result, urls, engines, created_at, schema)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        key,
                        provider,
                        query,
                        value.text,
                        json.dumps(list(value.urls), ensure_ascii=False),
                        json.dumps(list(value.engines), ensure_ascii=False),
                        datetime.now(timezone.utc).isoformat(),
                        SCHEMA_VERSION,
                    ),
                )
                conn.commit()
            finally:
                conn.close()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.debug("Search cache write failed: %s", exc)
            return False

    # ------------------------------------------------------------------ memo
    def _memo_get(self, key: str) -> Optional[CachedResult]:
        with self._lock:
            hit = self._memo.get(key)
            if hit is not None:
                self._memo.move_to_end(key)
            return hit

    def _memo_put(self, key: str, value: CachedResult) -> None:
        with self._lock:
            self._memo[key] = value
            self._memo.move_to_end(key)
            while len(self._memo) > MEMO_MAX_ENTRIES:
                self._memo.popitem(last=False)

    # ------------------------------------------------------------------ public
    def get(self, provider: str, query: str) -> Optional[CachedResult]:
        """Cached answer for (provider, query), or None. ``off``/``record`` never hit."""
        if self.mode in ("off", "record"):
            return None
        key = cache_key(provider, query)
        hit = self._memo_get(key)
        if hit is not None:
            return CachedResult(text=hit.text, urls=hit.urls, engines=hit.engines, source="memo")
        if self._disk_ok and self.mode in ("shared", "replay"):
            hit = self._disk_get(key)
            if hit is not None:
                self._memo_put(key, hit)
                return hit
        return None

    def put(self, provider: str, query: str, text: str, urls=(), engines=()) -> None:
        """Store a non-empty upstream answer (no-op for ``off``/``replay`` and empty text)."""
        if self.mode in ("off", "replay") or not text or not str(text).strip():
            return
        key = cache_key(provider, query)
        value = CachedResult(text=str(text), urls=list(urls or []), engines=list(engines or []), source="memo")
        if self._disk_ok and self.mode in ("shared", "record"):
            self._disk_put(key, provider, query, value)
        if self.mode != "record":
            self._memo_put(key, value)


_active: Optional[SearchCache] = None
_active_cfg: Optional[tuple] = None
_active_lock = threading.Lock()


def get_cache() -> SearchCache:
    """Process-wide cache for the current env (rebuilt if the env flags change)."""
    global _active, _active_cfg
    mode = cache_mode()
    path = cache_path() if mode in ("shared", "replay", "record") else None
    cfg = (mode, path)
    with _active_lock:
        if _active is None or _active_cfg != cfg:
            _active = SearchCache(mode, path)
            _active_cfg = cfg
        return _active


def reset_for_testing() -> None:
    global _active, _active_cfg
    with _active_lock:
        _active, _active_cfg = None, None
