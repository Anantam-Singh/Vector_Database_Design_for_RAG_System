"""Live index updates: insert, update, metadata-only update, delete — without rebuilding the index.

How stale data is prevented:
  * point ID = UUID5(doc_id), so a new version OVERWRITES the old one (dense vector, BM25 vector and metadata
    together, in one write). The old version cannot be returned any more.
  * content_hash: identical text is skipped (no wasted re-embedding, no duplicate).
  * metadata-only change: payload updated in place, vectors untouched (nothing to re-embed).
  * BM25 statistics (IDF) are recomputed by Qdrant from the live collection; the HNSW graph accepts inserts and
    deletes incrementally — no full rebuild.
  * every write increments INDEX_VERSION, which the result cache uses so it can never serve stale results.
"""
from __future__ import annotations

import threading
import time
import uuid
from datetime import datetime, timezone

from . import changelog
from .data import clean, content_hash
from .encoders import get_dense, get_sparse
from .store import Store, point_id
from .tags import source_type, tags_for

_lock = threading.Lock()
INDEX_VERSION = 0


def _bump() -> int:
    global INDEX_VERSION
    with _lock:
        INDEX_VERSION += 1
        return INDEX_VERSION


def index_version() -> int:
    return INDEX_VERSION


class Updater:
    def __init__(self, store: Store | None = None):
        self.store = store or Store()
        self.dense = get_dense()
        self.sparse = get_sparse()

    def get(self, doc_id: str) -> dict | None:
        pts = self.store.get_doc(doc_id)
        return pts[0].payload if pts else None

    def _log(self, doc_id, action, version, text, payload, iv) -> None:
        try:
            changelog.record(self.store.collection, doc_id, action, version, text,
                             {k: payload.get(k) for k in ("source", "category", "topic", "source_type", "corpus")}, iv)
        except Exception:   # the history is an audit aid — it must never block a write to the index
            pass

    def upsert(self, text: str, doc_id: str | None = None, source: str = "manual", category: str = "description",
               corpus: str = "internal") -> dict:
        t0 = time.perf_counter()
        doc_id = doc_id or f"user-{uuid.uuid4().hex[:12]}"
        text = clean(text)
        if not text:
            raise ValueError("text is empty")
        h = content_hash(text)
        old = self.get(doc_id)
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        prev = {"previous_version": old.get("version"), "previous_text": old.get("text")} if old else {}

        if old and old.get("content_hash") == h:
            if old.get("source") == source and old.get("category") == category:
                return {"status": "unchanged", "doc_id": doc_id, "version": old["version"],
                        "ms": round((time.perf_counter() - t0) * 1000, 1), "index_version": INDEX_VERSION, **prev}
            # metadata-only change: no re-embedding needed
            meta = {"source": source, "category": category, "source_type": source_type(source), "updated_at": now}
            self.store.client.set_payload(self.store.collection, wait=True, payload=meta, points=[point_id(doc_id)])
            iv = _bump()
            self._log(doc_id, "metadata_updated", old["version"], text, {**old, **meta}, iv)
            return {"status": "metadata_updated", "doc_id": doc_id, "version": old["version"],
                    "ms": round((time.perf_counter() - t0) * 1000, 1), "index_version": iv, **prev}

        dv = self.dense.encode_passages([text])
        sv = self.sparse.encode_passages([text])
        payload = {"doc_id": doc_id, "passage_id": old.get("passage_id") if old else None, "chunk": 0,
                   "source": source, "category": category, "version": (old["version"] + 1) if old else 1,
                   "updated_at": now, "content_hash": h, "text": text,
                   **tags_for(dv[0], source, old.get("corpus", corpus) if old else corpus)}
        self.store.upsert([payload], dv, sv, wait=True)        # same ID → overwrites the previous version
        iv = _bump()
        status = "updated" if old else "inserted"
        self._log(doc_id, status, payload["version"], text, payload, iv)
        return {"status": status, "doc_id": doc_id, "version": payload["version"], "topic": payload["topic"],
                "source_type": payload["source_type"], "ms": round((time.perf_counter() - t0) * 1000, 1),
                "index_version": iv, **prev}

    def delete(self, doc_id: str) -> dict:
        t0 = time.perf_counter()
        old = self.get(doc_id)
        if not old:
            return {"status": "not_found", "doc_id": doc_id}
        self.store.delete_doc(doc_id)
        iv = _bump()
        self._log(doc_id, "deleted", old.get("version"), old.get("text"), old, iv)
        return {"status": "deleted", "doc_id": doc_id, "ms": round((time.perf_counter() - t0) * 1000, 1),
                "index_version": iv, "previous_version": old.get("version")}
