"""Tune fusion (and rerank depth) on the DEV set only. Candidates are fetched once per question, then every setting
is scored offline on the same candidates — fast and a fair comparison.

Also measures CANDIDATE RECALL: is a correct passage anywhere in the pool the fusion/reranker gets to work with?

    python -m scripts.tune_fusion                 # fusion grid (no reranker)
    python -m scripts.tune_fusion --rerank        # + reranker: weights x depth grid
"""
from __future__ import annotations

import argparse
import json
import statistics

import pandas as pd

from precisionrag.config import get_settings
from precisionrag.fusion import fuse
from precisionrag.metrics import per_query
from precisionrag.results import save_csv, save_json
from precisionrag.retriever import Retriever


def load(split):
    return [json.loads(l) for l in (get_settings().eval_dir / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "dev_large", "test"])
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--pool", type=int, default=get_settings()["retrieval"]["prefetch_limit"])
    args = ap.parse_args()
    s = get_settings()
    k_rrf = s["retrieval"]["rrf_k"]

    rows = load(args.split)
    r = Retriever()
    cands = []
    for row in rows:
        qv, qs = r.dense.encode_query(row["query"]), r.sparse.encode_query(row["query"])
        d, b = r.store.search_both(qv, qs, args.pool)
        cands.append((row, d, b))

    def pid(p):
        return p.payload["passage_id"]

    # ---------------- candidate recall: does the pool contain a correct passage at all?
    pool = []
    for depth in (10, 20, 50):
        hits = {"dense": 0, "bm25": 0, "union": 0}
        for row, d, b in cands:
            rel = set(row["relevant"])
            dd, bb = {pid(p) for p in d[:depth]}, {pid(p) for p in b[:depth]}
            hits["dense"] += bool(rel & dd); hits["bm25"] += bool(rel & bb); hits["union"] += bool(rel & (dd | bb))
        pool.append({"depth": depth, **{f"recall_{k}": round(v / len(cands), 4) for k, v in hits.items()}})
    print("CANDIDATE RECALL (correct passage anywhere in the pool):")
    for p in pool:
        print(f"  top-{p['depth']:<3} dense {p['recall_dense']:.3f} | bm25 {p['recall_bm25']:.3f} | union {p['recall_union']:.3f}")

    # ---------------- fusion grid
    configs = [("dense_only", "weighted_rrf", 1.0, 0.0), ("bm25_only", "weighted_rrf", 0.0, 1.0), ("rrf", "rrf", 1, 1)]
    configs += [(f"wrrf_{w}", "weighted_rrf", 1.0, w) for w in (0.1, 0.2, 0.3, 0.5, 0.7)]
    configs += [(f"linear_{w}", "linear", 1.0, w) for w in (0.1, 0.2, 0.3, 0.5)]
    depths = (10, 20, 30) if args.rerank else (None,)
    if args.rerank:
        configs = [c for c in configs if c[0] in ("dense_only", "rrf", "wrrf_0.3", "wrrf_0.5")]

    table = []
    for name, method, wd, wb in configs:
        for depth in depths:
            mets = []
            for row, d, b in cands:
                fused = [p for p, _ in fuse(method, {"dense": d, "bm25": b}, {"dense": wd, "bm25": wb}, k_rrf)]
                if depth:
                    cand = fused[:depth]
                    sc = r.reranker.score(row["query"], [p.payload["text"] for p in cand])
                    fused = [p for _, p in sorted(zip(sc, cand), key=lambda x: -x[0])]
                mets.append(per_query([pid(p) for p in fused[:10]], set(row["relevant"])))
            agg = {k: round(statistics.mean(m[k] for m in mets), 4) for k in ("mrr@10", "ndcg@10", "recall@5", "hit@5")}
            table.append({"config": name, "fusion": method, "w_dense": wd, "w_bm25": wb, "rerank_depth": depth or 0, **agg})
            print(f"  {name:12s} depth={depth or '-':<3} " + " | ".join(f"{k} {v:.3f}" for k, v in agg.items()), flush=True)

    tag = "rerank" if args.rerank else "fusion"
    save_csv(f"tuning/{args.split}_{tag}_grid.csv", pd.DataFrame(table), split=args.split, pool=args.pool)
    save_json(f"tuning/{args.split}_candidate_recall.json", {"pool": pool}, split=args.split)
    best = max(table, key=lambda t: t["mrr@10"])
    print(f"BEST on {args.split} by MRR@10: {best}")


if __name__ == "__main__":
    main()
