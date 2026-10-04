"""Label-based evaluation (MS MARCO human labels): MRR@10, nDCG@10, Hit/Recall/Precision@5 for each mode.

Saves one row PER QUESTION (needed for confidence intervals and failure analysis) plus a summary.

    python -m scripts.eval_ir --modes dense bm25 hybrid hybrid_rerank --split test
    python -m scripts.eval_ir --modes hybrid --split dev --fusion weighted_rrf --w-bm25 0.3 --tag w03
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from precisionrag.config import get_settings
from precisionrag.metrics import per_query, summarize
from precisionrag.results import save_csv, save_json
from precisionrag.retriever import MODES, Retriever
from precisionrag.store import Store

KEYS = ["mrr@10", "ndcg@10", "hit@5", "recall@5", "precision@5", "hit@10"]


def load_split(name: str) -> list[dict]:
    path = get_settings().eval_dir / f"{name}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


CATEGORIES = ["description", "entity", "location", "numeric", "person"]


def pick_category(query_type: str, filter_mode: str) -> str | None:
    """oracle = the question's own type (an UPPER BOUND: real users must choose it); wrong = a different type."""
    if filter_mode == "oracle":
        return query_type
    if filter_mode == "wrong":
        return CATEGORIES[(CATEGORIES.index(query_type) + 1) % len(CATEGORIES)]
    return None


def run(retriever: Retriever, rows: list[dict], mode: str, filter_mode: str = "none", **search_kw) -> pd.DataFrame:
    out = []
    for r in rows:
        cat = pick_category(r["query_type"], filter_mode)
        res = retriever.search(r["query"], mode=mode, k=10, category=cat, **search_kw)
        ranked = [h.passage_id for h in res.hits]
        m = per_query(ranked, set(r["relevant"]))
        out.append({"qid": r["qid"], "query_type": r["query_type"], "mode": mode,
                    "first_relevant_rank": next((i for i, pid in enumerate(ranked, 1) if pid in set(r["relevant"])), None),
                    "top10": json.dumps(ranked), "total_ms": res.timings_ms["total"], **m})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    ap.add_argument("--split", default="test", choices=["dev", "dev_large", "test"])
    ap.add_argument("--fusion", choices=["rrf", "weighted_rrf", "linear"])
    ap.add_argument("--w-bm25", type=float, help="BM25 weight (dense weight stays 1.0)")
    ap.add_argument("--depth", type=int, help="rerank depth")
    ap.add_argument("--tag", default="", help="suffix for result file names")
    ap.add_argument("--collection", help="evaluate another collection (e.g. msmarco_500k)")
    ap.add_argument("--filter-mode", default="none", choices=["none", "oracle", "wrong"],
                    help="category filter experiment (pre-retrieval)")
    args = ap.parse_args()

    rows = load_split(args.split)
    retriever = Retriever(Store(args.collection) if args.collection else None)
    retriever.warmup(rerank=any(m.endswith("rerank") for m in args.modes))
    kw = {k: v for k, v in {"fusion": args.fusion, "depth": args.depth,
                             "weights": {"dense": 1.0, "bm25": args.w_bm25} if args.w_bm25 is not None else None}.items()
          if v is not None}
    tag = f"_{args.tag}" if args.tag else ""
    if args.collection:                        # never overwrite the official 100k results
        tag += f"_{args.collection}"
    if args.filter_mode != "none":
        tag += f"_filter-{args.filter_mode}"
    n_points = retriever.store.count()

    summary = {}
    for mode in args.modes:
        df = run(retriever, rows, mode, args.filter_mode, **kw)
        save_csv(f"ir/{args.split}_{mode}{tag}_perquery.csv", df, split=args.split, mode=mode, search=kw,
                 index_points=n_points, filter_mode=args.filter_mode)
        summary[mode] = {**summarize(df.to_dict("records"), KEYS), "n_queries": len(df)}
        print(f"{mode:14s} " + " | ".join(f"{k} {summary[mode][k]:.3f}" for k in KEYS), flush=True)
    save_json(f"ir/{args.split}_summary{tag}.json", summary, split=args.split, search=kw, index_points=n_points,
              filter_mode=args.filter_mode)


if __name__ == "__main__":
    main()
