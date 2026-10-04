"""Cross-encoder reranker: reads (question, passage) TOGETHER and scores how well the passage answers the question.

Far more accurate than comparing two separate vectors, but too slow for the whole corpus — so it only reorders the
small candidate list produced by hybrid retrieval (top-20 on GPU, top-10 on CPU).
"""
from __future__ import annotations

from functools import lru_cache

from .config import get_settings


class Reranker:
    def __init__(self, device: str | None = None):
        s = get_settings()
        from sentence_transformers import CrossEncoder
        self.device = device or s.device
        self.model = CrossEncoder(s["models"]["reranker"], device=self.device)

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        return self.model.predict([(query, t) for t in texts], batch_size=32, show_progress_bar=False).tolist()


@lru_cache(maxsize=2)
def get_reranker(device: str | None = None) -> Reranker:
    return Reranker(device)
