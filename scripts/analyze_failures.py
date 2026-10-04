"""Failure analysis on the saved per-question test results (no models, no API — reads results/ir/*.csv).

Answers: WHERE does each component help or hurt, and WHY?
  * breakdown by question type (numeric / entity / person / location / description), length, contains-a-number
  * case counts: dense fails & hybrid fixes, hybrid fails, reranker fixes the order, reranker hurts, BM25 rescues
  * real examples (question + the passage each method put first + the labelled correct passage)

    python -m scripts.analyze_failures                       # 100k, GPU profile files
    python -m scripts.analyze_failures --suffix _cpu         # 100k, CPU profile files
    QDRANT_COLLECTION=msmarco_500k python -m scripts.analyze_failures --suffix _msmarco_500k
"""
from __future__ import annotations

import argparse
import json
import re

import pandas as pd

from precisionrag.config import get_settings
from precisionrag.metrics import paired_bootstrap
from precisionrag.results import save_json
from precisionrag.store import Store, point_id

MODES = ["dense", "bm25", "hybrid", "dense_rerank", "hybrid_rerank"]


def first_rank(v):
    return 99 if pd.isna(v) else int(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="", help="per-query file suffix, e.g. _cpu or _msmarco_500k")
    sfx = ap.parse_args().suffix
    s = get_settings()
    q = {r["qid"]: r for r in map(json.loads, (s.eval_dir / "test.jsonl").read_text(encoding="utf-8").splitlines())}
    frames = {m: pd.read_csv(s.results_dir / "ir" / f"test_{m}{sfx}_perquery.csv").set_index("qid") for m in MODES}
    df = pd.DataFrame({f"{m}_rank": frames[m]["first_relevant_rank"].map(first_rank) for m in MODES})
    for m in MODES:
        df[f"{m}_mrr"] = frames[m]["mrr@10"]
        df[f"{m}_top1"] = frames[m]["top10"].map(lambda t: json.loads(t)[0] if json.loads(t) else None)
    df["query"] = [q[i]["query"] for i in df.index]
    df["query_type"] = [q[i]["query_type"] for i in df.index]
    df["relevant"] = [q[i]["relevant"] for i in df.index]
    df["has_number"] = df["query"].str.contains(r"\d")
    df["n_words"] = df["query"].str.split().map(len)
    df["length"] = pd.cut(df["n_words"], [0, 4, 8, 100], labels=["short (≤4 words)", "medium (5–8)", "long (9+)"])

    # ---------------- breakdowns
    def breakdown(col):
        g = df.groupby(col, observed=True)
        out = {}
        for key, part in g:
            row = {"n": len(part)}
            for m in MODES:
                row[f"{m}_mrr"] = round(part[f"{m}_mrr"].mean(), 4)
            row["hybrid_vs_dense"] = paired_bootstrap(part["dense_mrr"].tolist(), part["hybrid_mrr"].tolist(), n=1000)
            row["rerank_bm25_value"] = paired_bootstrap(part["dense_rerank_mrr"].tolist(),
                                                        part["hybrid_rerank_mrr"].tolist(), n=1000)
            out[str(key)] = row
        return out

    breakdowns = {"query_type": breakdown("query_type"), "has_number": breakdown("has_number"),
                  "length": breakdown("length")}

    # ---------------- case counts (rank 99 = not in top 10)
    cases = {
        "dense_misses_top10": df["dense_rank"] > 10,
        "dense_fails_hybrid_fixes(top5)": (df["dense_rank"] > 5) & (df["hybrid_rank"] <= 5),
        "dense_ok_hybrid_breaks(top5)": (df["dense_rank"] <= 5) & (df["hybrid_rank"] > 5),
        "bm25_finds_dense_misses(top10)": (df["dense_rank"] > 10) & (df["bm25_rank"] <= 10),
        "rerank_fixes_order(to #1)": (df["hybrid_rank"] > 1) & (df["hybrid_rank"] <= 10) & (df["hybrid_rerank_rank"] == 1),
        "rerank_hurts(#1 lost)": (df["hybrid_rank"] == 1) & (df["hybrid_rerank_rank"] > 1),
        "all_methods_fail(top10)": df[[f"{m}_rank" for m in MODES]].min(axis=1) > 10,
    }
    counts = {k: int(v.sum()) for k, v in cases.items()}

    # ---------------- examples with texts (looked up in Qdrant by deterministic point ID)
    store = Store()

    def text(pid):
        if pid is None or pd.isna(pid):
            return None
        pts = store.client.retrieve(store.collection, [point_id(f"msmarco-{int(pid)}")], with_payload=["text"])
        return pts[0].payload["text"][:300] if pts else None

    examples = {}
    for name in ["dense_fails_hybrid_fixes(top5)", "rerank_fixes_order(to #1)", "rerank_hurts(#1 lost)",
                 "all_methods_fail(top10)"]:
        rows = df[cases[name]].head(3)
        examples[name] = [{
            "qid": int(i), "query": r["query"], "query_type": r["query_type"],
            "ranks": {m: (None if r[f"{m}_rank"] == 99 else int(r[f"{m}_rank"])) for m in MODES},
            "dense_top1": text(r["dense_top1"]), "hybrid_rerank_top1": text(r["hybrid_rerank_top1"]),
            "labelled_correct": text(r["relevant"][0]),
        } for i, r in rows.iterrows()]

    save_json(f"analysis/failure_analysis{sfx}.json", {"n_questions": len(df), "case_counts": counts,
                                                 "breakdowns": breakdowns, "examples": examples})
    print("CASE COUNTS (1,000 test questions):")
    for k, v in counts.items():
        print(f"  {k:36s} {v}")
    print("\nMRR@10 BY QUESTION TYPE:")
    for k, row in breakdowns["query_type"].items():
        print(f"  {k:12s} n={row['n']:<4} " + " | ".join(f"{m} {row[f'{m}_mrr']:.3f}" for m in MODES))
    print("\nBY 'CONTAINS A NUMBER':")
    for k, row in breakdowns["has_number"].items():
        print(f"  {k:12s} n={row['n']:<4} dense {row['dense_mrr']:.3f} | hybrid {row['hybrid_mrr']:.3f} | "
              f"hybrid-vs-dense {row['hybrid_vs_dense']['mean_diff']:+.3f} "
              f"[{row['hybrid_vs_dense']['ci95_low']:+.3f}, {row['hybrid_vs_dense']['ci95_high']:+.3f}]")


if __name__ == "__main__":
    main()
