"""Experiment: does BM25 get better when avg_len matches the real corpus (42.8 tokens) instead of FastEmbed's 256?

Builds temporary sparse-only collections (one per avg_len), then compares BM25-only and hybrid fusion on a tuning
split. Dense candidates come from the main collection, which is not touched. Temporary collections are deleted.

    python -m scripts.exp_bm25_avglen --split dev_large --avg-lens 256 42.8
"""
from __future__ import annotations

import argparse
import json
import statistics

import pandas as pd
from qdrant_client import models

from precisionrag.config import get_settings
from precisionrag.data import build_corpus
from precisionrag.encoders import SparseEncoder, get_dense
from precisionrag.fusion import fuse
from precisionrag.metrics import per_query
from precisionrag.results import save_csv
from precisionrag.store import SPARSE, Store, point_id, to_sparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev_large")
    ap.add_argument("--avg-lens", nargs="+", type=float, default=[256.0, 42.8])
    ap.add_argument("--size", type=int, default=100_000)
    args = ap.parse_args()
    s = get_settings()
    rows = [json.loads(l) for l in (s.eval_dir / f"{args.split}.jsonl").read_text(encoding="utf-8").splitlines() if l]
    main_store = Store()
    dense = get_dense()
    dense_cands = {r["qid"]: main_store.search_dense(dense.encode_query(r["query"]), 50) for r in rows}
    corpus = build_corpus(args.size)

    table = []
    for avg in args.avg_lens:
        enc = SparseEncoder(avg_len=avg)
        tmp = Store(collection=f"exp_bm25_avg{str(avg).replace('.', '_')}")
        if tmp.exists():
            tmp.client.delete_collection(tmp.collection)
        tmp.client.create_collection(tmp.collection, vectors_config={},
                                     sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)})
        for lo in range(0, len(corpus.texts), 8192):
            texts = corpus.texts[lo:lo + 8192]
            sv = enc.encode_passages(texts)
            tmp.client.upload_points(tmp.collection, [
                models.PointStruct(id=point_id(corpus.meta[lo + i]["doc_id"]), vector={SPARSE: to_sparse(v)},
                                   payload={"passage_id": corpus.meta[lo + i]["passage_id"]})
                for i, v in enumerate(sv)], batch_size=512, wait=True)
        tmp.wait_ready()

        configs = [("bm25_only", 0.0, 1.0)] + [(f"wrrf_{w}", 1.0, w) for w in (0.1, 0.2, 0.3, 0.5, 1.0)]
        for name, wd, wb in configs:
            mets = []
            for r in rows:
                b = tmp.search_sparse(enc.encode_query(r["query"]), 50)
                fused = [p for p, _ in fuse("weighted_rrf", {"dense": dense_cands[r["qid"]], "bm25": b},
                                            {"dense": wd, "bm25": wb})]
                mets.append(per_query([p.payload["passage_id"] for p in fused[:10]], set(r["relevant"])))
            agg = {k: round(statistics.mean(m[k] for m in mets), 4) for k in ("mrr@10", "recall@5", "ndcg@10")}
            table.append({"avg_len": avg, "config": name, "w_bm25": wb, **agg})
            print(f"avg_len={avg:<6} {name:10s} " + " | ".join(f"{k} {v:.3f}" for k, v in agg.items()), flush=True)
        tmp.client.delete_collection(tmp.collection)

    dmets = [per_query([p.payload["passage_id"] for p in dense_cands[r["qid"]][:10]], set(r["relevant"])) for r in rows]
    print("dense_only           " + " | ".join(f"{k} {statistics.mean(m[k] for m in dmets):.3f}"
                                               for k in ("mrr@10", "recall@5", "ndcg@10")))
    save_csv(f"experiments/bm25_avglen_{args.split}.csv", pd.DataFrame(table), split=args.split, size=args.size)


if __name__ == "__main__":
    main()
