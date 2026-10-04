"""Experiment: should the reranker see the UNION of dense and BM25 candidates instead of a fused top-20?

Motivation (measured): at 500k, dense top-20 ∪ BM25 top-20 contains the correct passage 93.1% of the time vs 89.9% for
dense top-20 alone — but RRF squeezes both lists into one top-20, so some of BM25's unique finds never reach the
reranker. Tuning questions only (dev_large); a "dense top-30" control has a similar reranking budget.

    python -m scripts.exp_union_rerank --collection msmarco_500k
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import pandas as pd

from precisionrag.config import get_settings
from precisionrag.fusion import fuse
from precisionrag.metrics import per_query
from precisionrag.rerank import get_reranker
from precisionrag.results import save_csv
from precisionrag.retriever import Retriever
from precisionrag.store import Store


def union(*lists):
    seen, out = set(), []
    for lst in lists:
        for p in lst:
            if p.id not in seen:
                seen.add(p.id)
                out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default=None)
    ap.add_argument("--split", default="dev_large")
    args = ap.parse_args()
    s = get_settings()
    rows = [json.loads(l) for l in (s.eval_dir / f"{args.split}.jsonl").read_text(encoding="utf-8").splitlines() if l]
    r = Retriever(Store(args.collection))
    rr = get_reranker()
    cands = [(row, *r.store.search_both(r.dense.encode_query(row["query"]), r.sparse.encode_query(row["query"]), 50))
             for row in rows]

    configs = {
        "dense20": lambda d, b: d[:20],
        "dense30": lambda d, b: d[:30],
        "rrf20 (current)": lambda d, b: [p for p, _ in fuse("rrf", {"dense": d, "bm25": b}, {})][:20],
        "union d20+b10": lambda d, b: union(d[:20], b[:10]),
        "union d20+b20": lambda d, b: union(d[:20], b[:20]),
        "union d10+b10": lambda d, b: union(d[:10], b[:10]),
    }
    table = []
    for name, pick in configs.items():
        mets, sizes, ms = [], [], []
        for row, d, b in cands:
            pool = pick(d, b)
            t = time.perf_counter()
            sc = rr.score(row["query"], [p.payload["text"] for p in pool])
            ms.append((time.perf_counter() - t) * 1000)
            ranked = [p.payload["passage_id"] for _, p in sorted(zip(sc, pool), key=lambda x: -x[0])]
            mets.append(per_query(ranked[:10], set(row["relevant"])))
            sizes.append(len(pool))
        agg = {k: round(statistics.mean(m[k] for m in mets), 4) for k in ("mrr@10", "recall@5", "ndcg@10")}
        table.append({"config": name, "avg_pool": round(statistics.mean(sizes), 1),
                      "rerank_ms_median": round(statistics.median(ms), 1), **agg})
        print(f"{name:16s} pool {table[-1]['avg_pool']:5.1f} | rerank {table[-1]['rerank_ms_median']:5.1f} ms | "
              + " | ".join(f"{k} {v:.3f}" for k, v in agg.items()), flush=True)
    save_csv(f"experiments/union_rerank_{args.split}_{r.store.collection}.csv", pd.DataFrame(table),
             split=args.split, collection=r.store.collection)


if __name__ == "__main__":
    main()
