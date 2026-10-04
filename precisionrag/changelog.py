"""Change log for live updates: every insert / update / metadata change / delete is recorded with the full text of
that version. Qdrant always holds ONLY the latest version (so search can never return stale text); this log keeps the
history for audit, "what changed?" diffs and the before/after demo.

Stored in a small SQLite file (Python standard library, no server): paths.changelog in config.yaml.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from .config import ROOT, get_settings

_lock = threading.Lock()
SCHEMA = """CREATE TABLE IF NOT EXISTS changes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    at TEXT NOT NULL,
    collection TEXT NOT NULL,
    doc_id TEXT NOT NULL,
    action TEXT NOT NULL,          -- inserted | updated | metadata_updated | deleted
    version INTEGER,
    text TEXT,
    meta TEXT,                     -- JSON: source, category, topic, source_type, corpus
    index_version INTEGER
);
CREATE INDEX IF NOT EXISTS idx_changes_doc ON changes(collection, doc_id, id);"""


def _path() -> Path:
    p = Path(get_settings()["paths"].get("changelog", "data/changelog.sqlite"))
    return p if p.is_absolute() else ROOT / p


def _conn() -> sqlite3.Connection:
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(path)
    c.executescript(SCHEMA)
    return c


def record(collection: str, doc_id: str, action: str, version: int | None, text: str | None, meta: dict | None,
           index_version: int) -> None:
    with _lock, _conn() as c:
        c.execute("INSERT INTO changes (at, collection, doc_id, action, version, text, meta, index_version) "
                  "VALUES (?,?,?,?,?,?,?,?)",
                  (datetime.now(timezone.utc).isoformat(timespec="seconds"), collection, doc_id, action, version, text,
                   json.dumps(meta or {}), index_version))


def _rows(cur) -> list[dict]:
    cols = [d[0] for d in cur.description]
    out = []
    for r in cur.fetchall():
        d = dict(zip(cols, r))
        d["meta"] = json.loads(d["meta"] or "{}")
        out.append(d)
    return out


def history(collection: str, doc_id: str) -> list[dict]:
    with _lock, _conn() as c:
        return _rows(c.execute("SELECT * FROM changes WHERE collection=? AND doc_id=? ORDER BY id", (collection, doc_id)))


def recent(collection: str, limit: int = 30) -> list[dict]:
    with _lock, _conn() as c:
        return _rows(c.execute("SELECT * FROM changes WHERE collection=? ORDER BY id DESC LIMIT ?", (collection, limit)))
