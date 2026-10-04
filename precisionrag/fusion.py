"""Combine ranked lists from different retrievers (dense, BM25) into one ranking.

Both methods allowed by rule FR-3 are implemented:
  * RRF / weighted RRF — uses only RANKS: score(d) = Σ w_i / (k + rank_i(d)). Robust because cosine and BM25 scores
    live on different scales and need no normalisation. Plain RRF = all weights 1.0.
  * Linear — weighted sum of min-max normalised SCORES. Sensitive to score outliers, kept for comparison.
"""
from __future__ import annotations


def rrf(ranked: dict[str, list], weights: dict[str, float], k: int = 60) -> list[tuple[object, float]]:
    """ranked: {"dense": [point, ...], "bm25": [...]} best-first. Returns [(point, fused_score)] best-first."""
    score, keep = {}, {}
    for name, points in ranked.items():
        w = weights.get(name, 1.0)
        if w == 0:
            continue
        for rank, p in enumerate(points, start=1):
            score[p.id] = score.get(p.id, 0.0) + w / (k + rank)
            keep.setdefault(p.id, p)
    return sorted(((keep[i], s) for i, s in score.items()), key=lambda x: -x[1])


def linear(ranked: dict[str, list], weights: dict[str, float]) -> list[tuple[object, float]]:
    score, keep = {}, {}
    for name, points in ranked.items():
        w = weights.get(name, 1.0)
        if w == 0 or not points:
            continue
        raw = [p.score for p in points]
        lo, hi = min(raw), max(raw)
        for p in points:
            norm = (p.score - lo) / (hi - lo) if hi > lo else 1.0
            score[p.id] = score.get(p.id, 0.0) + w * norm
            keep.setdefault(p.id, p)
    return sorted(((keep[i], s) for i, s in score.items()), key=lambda x: -x[1])


def fuse(method: str, ranked: dict[str, list], weights: dict[str, float], k: int = 60):
    if method == "rrf":
        return rrf(ranked, {n: 1.0 for n in ranked}, k)
    if method == "weighted_rrf":
        return rrf(ranked, weights, k)
    if method == "linear":
        return linear(ranked, weights)
    raise ValueError(f"unknown fusion method: {method}")
