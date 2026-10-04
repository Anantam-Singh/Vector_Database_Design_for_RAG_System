"""Does filtering by topic make retrieval better or faster?

For each question we compare: no topic filter · auto-topic (the question's 1 / 2 / 3 most likely topics, predicted
from its vector) · oracle (the topic of the labelled correct passage — an upper bound a real user only reaches if they
pick the right topic). Choose settings on dev_large, confirm once on test.

    python -m scripts.eval_topics --split dev_large
    python -m scripts.eval_topics --split test --configs none auto2 oracle
"""
from __future__ import annotations

import argparse
import json

import pandas as pd
from qdrant_client import models

from precisionrag.metrics import paired_bootstrap, per_query
from precisionrag.results import save_csv, save_json
from precisionrag.retriever import Retriever
from scripts.eval_ir import load_split

CONFIGS = ["none", "auto1", "auto2", "auto3", "oracle"]


def gold_topics(store, rows) -> dict:
    ids = sorted({pid for r in rows for pid in r["relevant"]})
    out = {}
    for i in range(0, len(ids), 500):
        chunk = [f"msmarco-{p}" for p in ids[i:i + 500]]
        pts, _ = store.client.scroll(store.collection, limit=len(chunk), with_payload=["passage_id", "topic"],
                                     scroll_filter=models.Filter(must=[models.FieldCondition(
                                         key="doc_id", match=models.MatchAny(any=chunk))]))
        out.update({p.payload["passage_id"]: p.payload.get("topic") for p in pts})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev_large", choices=["dev", "dev_large", "test"])
    ap.add_argument("--modes", nargs="+", default=["hybrid_rerank", "dense"])
    ap.add_argument("--configs", nargs="+", default=CONFIGS, choices=CONFIGS)
    ap.add_argument("--tag", default="", help="suffix for result file names, e.g. cpu (run with DEVICE=cpu)")
    args = ap.parse_args()
    tag = f"_{args.tag}" if args.tag else ""
    rows = load_split(args.split)
    retr = Retriever()
    retr.warmup()
    gold = gold_topics(retr.store, rows)
    per, summary = [], {}
    for mode in args.modes:
        for cfg in args.configs:
            for r in rows:
                kw = {}
                if cfg.startswith("auto"):
                    kw["auto_topic"] = int(cfg[-1])
                elif cfg == "oracle":
                    kw["topic"] = sorted({gold.get(p) for p in r["relevant"] if gold.get(p)}) or None
                res = retr.search(r["query"], mode=mode, k=10, **kw)
                ranked = [h.passage_id for h in res.hits]
                m = per_query(ranked, set(r["relevant"]))
                routed = res.params.get("auto_topics")
                hit_route = None
                if routed:
                    hit_route = any(gold.get(p) in [t for t, _ in routed] for p in r["relevant"])
                per.append({"qid": r["qid"], "mode": mode, "config": cfg, "total_ms": res.timings_ms["total"],
                            "routed": json.dumps(routed) if routed else "", "route_correct": hit_route, **m})
            print(f"  {args.split} {mode:14s} {cfg:7s} done", flush=True)
    df = pd.DataFrame(per)
    for mode in args.modes:
        base = df[(df["mode"] == mode) & (df["config"] == "none")].sort_values("qid")
        for cfg in args.configs:
            d = df[(df["mode"] == mode) & (df["config"] == cfg)].sort_values("qid")
            item = {"mrr@10": round(d["mrr@10"].mean(), 6), "recall@5": round(d["recall@5"].mean(), 6),
                    "hit@5": round(d["hit@5"].mean(), 6), "median_ms": round(d["total_ms"].median(), 1), "n": len(d)}
            if cfg.startswith("auto"):
                item["route_accuracy"] = round(d["route_correct"].astype(float).mean(), 6)
            if cfg != "none" and len(base):
                item["mrr_vs_none"] = paired_bootstrap(base["mrr@10"].tolist(), d["mrr@10"].tolist(), n=2000)
            summary[f"{mode}|{cfg}"] = item
    # save BEFORE printing anything (a console encoding error must never lose a 10-minute run)
    save_csv(f"tags/topic_filter_{args.split}{tag}_perquery.csv", df)
    save_json(f"tags/topic_filter_{args.split}{tag}.json", summary, split=args.split, index_points=retr.store.count())
    for key, item in summary.items():
        bs = item.get("mrr_vs_none")
        print(f"{key:24s} MRR {item['mrr@10']:.3f} R@5 {item['recall@5']:.3f} median {item['median_ms']} ms"
              + (f" | route acc {item['route_accuracy']:.1%}" if "route_accuracy" in item else "")
              + (f" | dMRR {bs['mean_diff']:+.3f} [{bs['ci95_low']:+.3f}, {bs['ci95_high']:+.3f}]" if bs else ""), flush=True)


if __name__ == "__main__":
    main()
