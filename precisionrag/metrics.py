"""Label-based retrieval metrics (MS MARCO human labels; binary relevance). No API, fully deterministic.

    hit@k        1 if ANY relevant passage is in the top k
    recall@k     share of the relevant passages found in the top k
    precision@k  share of the top k that is relevant  (max ≈ 0.2 at k=5 here: most questions have 1 labelled passage)
    mrr@10       1 / rank of the first relevant passage (0 if not in top 10) — MS MARCO's official metric
    ndcg@10      ranking quality with a log position discount, normalised to the ideal ranking
"""
from __future__ import annotations

import math
import random
import statistics


def per_query(ranked_ids: list, relevant: set, k_small: int = 5, k_big: int = 10) -> dict:
    top_s, top_b = ranked_ids[:k_small], ranked_ids[:k_big]
    found_s = len(relevant & set(top_s))
    first = next((i for i, pid in enumerate(top_b, 1) if pid in relevant), None)
    dcg = sum(1 / math.log2(i + 1) for i, pid in enumerate(top_b, 1) if pid in relevant)
    idcg = sum(1 / math.log2(i + 1) for i in range(1, min(len(relevant), k_big) + 1))
    return {
        f"hit@{k_small}": float(found_s > 0),
        f"recall@{k_small}": found_s / len(relevant),
        f"precision@{k_small}": found_s / k_small,
        f"hit@{k_big}": float(any(pid in relevant for pid in top_b)),
        f"mrr@{k_big}": 1 / first if first else 0.0,
        f"ndcg@{k_big}": dcg / idcg if idcg else 0.0,
    }


def summarize(rows: list[dict], keys: list[str]) -> dict:
    return {k: round(statistics.mean(r[k] for r in rows), 4) for k in keys}


def paired_bootstrap(a: list[float], b: list[float], n: int = 2000, seed: int = 42) -> dict:
    """95% confidence interval for mean(b - a) on the SAME queries. If the interval excludes 0, the gain is real."""
    diffs = [y - x for x, y in zip(a, b)]
    k = len(diffs)
    rng = random.Random(seed)
    # plain float mean (statistics.mean uses exact fractions: ~3x slower, same results to 4 decimals on all our data)
    means = sorted(sum(rng.choices(diffs, k=k)) / k for _ in range(n))
    return {"mean_diff": round(statistics.mean(diffs), 4), "ci95_low": round(means[int(0.025 * n)], 4),
            "ci95_high": round(means[int(0.975 * n)], 4),
            "wins": sum(d > 0 for d in diffs), "ties": sum(d == 0 for d in diffs), "losses": sum(d < 0 for d in diffs)}
