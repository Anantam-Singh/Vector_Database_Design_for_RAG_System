"""The four retrieval pipelines compared in the ablation, each timed stage by stage.

    dense          query → bge-small vector → Qdrant cosine search                      (challenge Phase 1)
    bm25           query → BM25 sparse vector → Qdrant sparse search                     (ablation baseline)
    hybrid         dense top-N ∥ BM25 top-N (one batched call) → fusion                  (challenge Phase 2)
    hybrid_rerank  hybrid → cross-encoder reorders the top-`depth` candidates            (Phase 2 + bonus)
    dense_rerank   dense candidates only → cross-encoder (ablation: isolates what BM25 adds after reranking)

Metadata filters are passed INTO the Qdrant searches (pre-retrieval), never applied afterwards.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

from .config import get_settings
from .encoders import get_dense, get_sparse
from .fusion import fuse
from .store import Store, make_filter

MODES = ("dense", "bm25", "hybrid", "hybrid_rerank", "dense_rerank")


@dataclass
class Hit:
    rank: int
    passage_id: int | None
    doc_id: str
    text: str
    score: float
    source: str
    category: str
    version: int = 1
    topic: str | None = None
    source_type: str | None = None
    corpus: str | None = None
    updated_at: str | None = None
    dense_rank: int | None = None
    bm25_rank: int | None = None
    fused_score: float | None = None


@dataclass
class SearchResult:
    query: str
    mode: str
    hits: list[Hit]
    timings_ms: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _hit(rank, point, score, **kw) -> Hit:
    p = point.payload or {}
    return Hit(rank=rank, passage_id=p.get("passage_id"), doc_id=p.get("doc_id", str(point.id)), text=p.get("text", ""),
               score=float(score), source=p.get("source", ""), category=p.get("category", ""),
               version=p.get("version", 1), topic=p.get("topic"), source_type=p.get("source_type"),
               corpus=p.get("corpus"), updated_at=p.get("updated_at"), **kw)


class Retriever:
    def __init__(self, store: Store | None = None, device: str | None = None):
        """device: "cuda" or "cpu" (default: the machine's). The hardware profile (rerank depth) follows the device."""
        self.s = get_settings()
        self.store = store or Store()
        self.device = device or self.s.device
        self.profile_name = "gpu" if self.device == "cuda" else "cpu"
        self.profile = self.s["profiles"][self.profile_name]
        self.dense = get_dense(None if self.device == self.s.device else self.device)   # share the default model
        self.sparse = get_sparse()
        self._reranker = None

    @property
    def reranker(self):
        if self._reranker is None:
            from .rerank import get_reranker
            self._reranker = get_reranker(self.device if self.device != self.s.device else None)
        return self._reranker

    def warmup(self, rerank: bool = True) -> None:
        for mode in ("dense", "hybrid") + (("hybrid_rerank",) if rerank else ()):
            self.search("warm up the models", mode=mode)

    def search(self, query: str, mode: str = "hybrid_rerank", k: int | None = None, category: str | None = None,
               source: str | None = None, fusion: str | None = None, weights: dict | None = None,
               depth: int | None = None, prefetch: int | None = None, topic=None, source_type: str | None = None,
               corpus: str | None = None, auto_topic: int = 0) -> SearchResult:
        """auto_topic=N: predict the question's N most likely topics and search only passages tagged with them
        (ignored when a topic is given explicitly)."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        r = self.s["retrieval"]
        k = k or r["top_k"]
        prefetch = prefetch or r["prefetch_limit"]
        t = {}
        t0 = time.perf_counter()
        qv = None
        predicted = None
        if auto_topic and not topic:
            from .tags import query_topics
            a = time.perf_counter(); qv = self.dense.encode_query(query); t["embed_dense"] = time.perf_counter() - a
            a = time.perf_counter(); predicted = query_topics(qv, top=auto_topic)
            topic = [name for name, _ in predicted]; t["route"] = time.perf_counter() - a
        flt = make_filter(category, source, topic, source_type, corpus)

        if mode == "dense":
            if qv is None:
                a = time.perf_counter(); qv = self.dense.encode_query(query); t["embed_dense"] = time.perf_counter() - a
            a = time.perf_counter(); pts = self.store.search_dense(qv, k, flt); t["qdrant"] = time.perf_counter() - a
            hits = [_hit(i, p, p.score, dense_rank=i) for i, p in enumerate(pts, 1)]
            params = {}

        elif mode == "bm25":
            a = time.perf_counter(); qs = self.sparse.encode_query(query); t["embed_bm25"] = time.perf_counter() - a
            a = time.perf_counter(); pts = self.store.search_sparse(qs, k, flt); t["qdrant"] = time.perf_counter() - a
            hits = [_hit(i, p, p.score, bm25_rank=i) for i, p in enumerate(pts, 1)]
            params = {}

        else:
            rerank = mode in ("hybrid_rerank", "dense_rerank")
            fusion = fusion or ("weighted_rrf" if rerank else r["fusion_hybrid"])
            if mode == "dense_rerank":
                weights = {"dense": 1.0, "bm25": 0.0}
            weights = weights or (r["weights_hybrid_rerank"] if rerank else r["weights_hybrid"])
            depth = depth or self.profile["rerank_depth"]
            if qv is None:
                a = time.perf_counter(); qv = self.dense.encode_query(query); t["embed_dense"] = time.perf_counter() - a
            a = time.perf_counter(); qs = self.sparse.encode_query(query); t["embed_bm25"] = time.perf_counter() - a
            a = time.perf_counter(); d_pts, b_pts = self.store.search_both(qv, qs, prefetch, flt)
            t["qdrant"] = time.perf_counter() - a
            a = time.perf_counter()
            d_rank = {p.id: i for i, p in enumerate(d_pts, 1)}
            b_rank = {p.id: i for i, p in enumerate(b_pts, 1)}
            fused = fuse(fusion, {"dense": d_pts, "bm25": b_pts}, weights, r["rrf_k"])
            t["fusion"] = time.perf_counter() - a
            if rerank:
                cand = fused[:depth]
                a = time.perf_counter()
                scores = self.reranker.score(query, [p.payload.get("text", "") for p, _ in cand])
                t["rerank"] = time.perf_counter() - a
                order = sorted(zip(scores, cand), key=lambda x: -x[0])[:k]
                hits = [_hit(i, p, sc, dense_rank=d_rank.get(p.id), bm25_rank=b_rank.get(p.id), fused_score=fs)
                        for i, (sc, (p, fs)) in enumerate(order, 1)]
            else:
                hits = [_hit(i, p, fs, dense_rank=d_rank.get(p.id), bm25_rank=b_rank.get(p.id), fused_score=fs)
                        for i, (p, fs) in enumerate(fused[:k], 1)]
            params = {"fusion": fusion, "weights": weights, "prefetch": prefetch, "rrf_k": r["rrf_k"],
                      **({"rerank_depth": depth} if rerank else {})}

        t["total"] = time.perf_counter() - t0
        params.update({"k": k, "category": category, "source": source, "topic": topic, "source_type": source_type,
                       "corpus": corpus, **({"auto_topics": predicted} if predicted else {})})
        if qv is not None:   # topic suggestions for the user to click (measured: a correct, user-chosen topic helps)
            from .tags import query_topics
            params["suggested_topics"] = query_topics(qv, top=3)
        return SearchResult(query=query, mode=mode, hits=hits,
                            timings_ms={n: round(v * 1000, 2) for n, v in t.items()}, params=params)
