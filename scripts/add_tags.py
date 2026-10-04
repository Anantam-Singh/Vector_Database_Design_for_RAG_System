"""Adds topic / source_type / corpus tags to every passage of an existing collection — without re-embedding or
re-indexing. Reads the dense vectors already stored in Qdrant, classifies them, and writes only the payload.

    python -m scripts.add_tags                          # official 100k collection
    python -m scripts.add_tags --collection msmarco_500k
"""
from __future__ import annotations

import argparse
import collections
import time

import numpy as np
from qdrant_client import models

from precisionrag.results import save_json
from precisionrag.store import DENSE, Store, ensure_payload_indexes
from precisionrag.tags import TOPIC_NAMES, assign_topics, source_type

BATCH = 4000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection")
    args = ap.parse_args()
    store = Store(args.collection)
    c = store.client
    t0 = time.time()
    ensure_payload_indexes(c, store.collection)
    topics, stypes, corpora = collections.Counter(), collections.Counter(), collections.Counter()
    margins, nxt, n = [], None, 0
    while True:
        pts, nxt = c.scroll(store.collection, limit=BATCH, offset=nxt, with_vectors=[DENSE],
                            with_payload=["source", "doc_id"])
        if not pts:
            break
        vecs = np.array([p.vector[DENSE] for p in pts], dtype=np.float32)
        ops = []
        for p, (topic, conf) in zip(pts, assign_topics(vecs)):
            corpus = "web" if str(p.payload.get("doc_id", "")).startswith("msmarco-") else "internal"
            st = source_type(p.payload.get("source", ""))
            ops.append(models.SetPayloadOperation(set_payload=models.SetPayload(
                payload={"topic": topic, "topic_conf": conf, "source_type": st, "corpus": corpus}, points=[p.id])))
            topics[topic] += 1; stypes[st] += 1; corpora[corpus] += 1; margins.append(conf)
        c.batch_update_points(store.collection, update_operations=ops, wait=nxt is None)
        n += len(pts)
        print(f"  tagged {n:,} passages ({time.time() - t0:.0f}s)", flush=True)
        if nxt is None:
            break
    secs = round(time.time() - t0, 1)
    summary = {"collection": store.collection, "passages": n, "seconds": secs,
               "topics": dict(topics.most_common()), "source_types": dict(stypes.most_common()),
               "corpus": dict(corpora), "topic_margin_median": round(float(np.median(margins)), 4),
               "n_topics": len(TOPIC_NAMES)}
    save_json(f"tags/summary_{store.collection}.json", summary)
    print(f"done: {n:,} passages tagged in {secs}s — no re-embedding, no re-index")
    for k, v in topics.most_common():
        print(f"  {k:24s} {v:7,d}  ({v / n:.1%})")
    print("  source types:", dict(stypes.most_common()))


if __name__ == "__main__":
    main()
