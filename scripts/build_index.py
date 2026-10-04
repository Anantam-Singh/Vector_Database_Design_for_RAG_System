"""Build the Qdrant index: MS MARCO → dedup + metadata → dense + BM25 vectors → Qdrant.

Resumable: progress is saved after every chunk, so a crash restarts where it stopped (use --resume).

    python -m scripts.build_index                  # size from the hardware profile (GPU 500k / CPU 100k)
    python -m scripts.build_index --size 100000
    python -m scripts.build_index --size 100000 --resume
    python -m scripts.build_index --size 500000 --collection msmarco_500k
"""
from __future__ import annotations

import argparse
import json
import time

import psutil

from precisionrag.config import get_settings
from precisionrag.data import build_corpus
from precisionrag.encoders import get_dense, get_sparse
from precisionrag.results import results_path, save_json
from precisionrag.store import Store


def ram_mb() -> dict:
    qdrant = sum(p.memory_info().rss for p in psutil.process_iter(["name"])
                 if p.info["name"] and "qdrant" in p.info["name"].lower())
    return {"qdrant_mb": round(qdrant / 2**20), "python_mb": round(psutil.Process().memory_info().rss / 2**20),
            "system_free_gb": round(psutil.virtual_memory().available / 2**30, 1)}


def main():
    s = get_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=s.profile["corpus_size"])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--collection", help="default: qdrant.collection in config.yaml")
    ap.add_argument("--tag", default="", help="suffix for the result file, e.g. cpu (index_build_100000_cpu.json)")
    args = ap.parse_args()

    t0 = time.time()
    corpus = build_corpus(args.size)
    t_corpus = time.time() - t0
    n = len(corpus.texts)
    print(f"corpus: {n:,} unique passages ({corpus.n_validation:,} validation) in {t_corpus:.0f}s | "
          f"profile={s.profile_name} device={s.device}", flush=True)

    store = Store(args.collection)
    ckpt = results_path(f"checkpoints/index_{store.collection}.json")
    start = 0
    if args.resume and ckpt.exists() and store.exists():
        state = json.loads(ckpt.read_text())
        if state["size"] == n:
            start = state["next"]
            print(f"resuming at passage {start:,}", flush=True)
    if start == 0:
        store.recreate()

    dense, sparse = get_dense(), get_sparse()
    chunk = s["index"]["chunk"]
    t_dense = t_sparse = t_upload = 0.0
    for lo in range(start, n, chunk):
        hi = min(lo + chunk, n)
        texts = corpus.texts[lo:hi]
        t = time.time(); dv = dense.encode_passages(texts); t_dense += time.time() - t
        t = time.time(); sv = sparse.encode_passages(texts); t_sparse += time.time() - t
        payloads = [{**corpus.meta[i], "text": corpus.texts[i]} for i in range(lo, hi)]
        t = time.time(); store.upsert(payloads, dv, sv, batch_size=s["index"]["upload_batch"]); t_upload += time.time() - t
        ckpt.write_text(json.dumps({"size": n, "next": hi}))
        if (hi // chunk) % 4 == 0 or hi == n:
            print(f"  {hi:,}/{n:,}  dense {t_dense / 60:.1f} min | bm25 {t_sparse / 60:.1f} min | "
                  f"upload {t_upload / 60:.1f} min", flush=True)
    t_hnsw = store.wait_ready()
    total = time.time() - t0
    count = store.count()

    summary = {
        "passages_indexed": count, "validation_passages": corpus.n_validation,
        "labelled_validation_queries": len(corpus.qrels), "resumed_from": start,
        "seconds": {"corpus_build": round(t_corpus, 1), "dense_embed": round(t_dense, 1),
                    "bm25_embed": round(t_sparse, 1), "upload": round(t_upload, 1),
                    "hnsw_finish": round(t_hnsw, 1), "total": round(total, 1)},
        "total_minutes": round(total / 60, 2), "limit_minutes": 120, "pass": total < 7200 and count == n,
        "ram_after": ram_mb(),
    }
    path = save_json(f"index_build_{n}{'_' + args.tag if args.tag else ''}.json", summary, collection=store.collection)
    print(f"DONE: {count:,} points in {total / 60:.1f} min (limit 120) -> {'PASS' if summary['pass'] else 'FAIL'} "
          f"| saved {path.name}", flush=True)


if __name__ == "__main__":
    main()
