"""Small LRU result cache (bonus feature). The key includes index_version, so ANY upsert/delete makes every older
entry unreachable — a cached answer can never be stale. Disabled during benchmarks (no_cache=true)."""
from __future__ import annotations

import threading
from collections import OrderedDict


class ResultCache:
    def __init__(self, max_items: int = 1000):
        self.max_items = max_items
        self._data: OrderedDict = OrderedDict()
        self._lock = threading.Lock()
        self.hits = self.misses = 0

    @staticmethod
    def key(query: str, mode: str, k: int, category, source, index_version: int, *extra) -> tuple:
        """extra = any further filter / index settings; every setting that changes the result must be in the key."""
        return (" ".join(query.lower().split()), mode, k, category, source, index_version, *extra)

    def get(self, key):
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                self.hits += 1
                return self._data[key]
            self.misses += 1
            return None

    def put(self, key, value) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.max_items:
                self._data.popitem(last=False)

    def stats(self) -> dict:
        return {"items": len(self._data), "hits": self.hits, "misses": self.misses}
