"""RAGAS Context Precision + Context Recall for retrieval modes, on the frozen 50-question RAGAS set.

* The SAME judge model is used for every compared mode (default: config groq.judge_model).
* Results are appended to CSV every few questions; re-running the same command resumes where it stopped.
* When one Groq key reaches its free-tier limit it moves to the next key in .env; stops safely (and says so) when
  every key is used up.

    python -m scripts.eval_ragas --modes dense                      # official judge
    python -m scripts.eval_ragas --modes dense --judge dev --limit 5  # quick test on the dev model's quota
"""
from __future__ import annotations

import argparse
import json
import time
import warnings

import pandas as pd

from precisionrag.config import get_settings
from precisionrag.groq_quota import LATEST, KeyPool, QuotaExhausted, check, http_clients
from precisionrag.results import results_path, save_json
from precisionrag.retriever import MODES, Retriever

warnings.filterwarnings("ignore", category=DeprecationWarning)
BATCH = 5
PAUSE = 20   # seconds between batches


def make_judge(model: str, api_key: str):
    from langchain_groq import ChatGroq
    from ragas.llms import LangchainLLMWrapper
    sync_c, async_c = http_clients()
    llm = ChatGroq(model=model, temperature=0, max_tokens=2000, reasoning_effort="low",
                   api_key=api_key, http_client=sync_c, http_async_client=async_c)
    return LangchainLLMWrapper(llm)


def main():
    s = get_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["dense"], choices=MODES)
    ap.add_argument("--judge", default="official", help="'official', 'dev', or a Groq model id")
    ap.add_argument("--limit", type=int, help="only the first N questions (quick tests)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--key", default="auto",
                    help="'auto' = start with GROQ_API_KEY and move to GROQ_API_KEY_2, _3, ... when a key hits its "
                         "limit; '2' = use only GROQ_API_KEY_2 (from .env)")
    args = ap.parse_args()
    pool = KeyPool(args.key)
    print(f"Groq keys in use: {', '.join('#' + k[0] for k in pool.keys)}", flush=True)

    from ragas import EvaluationDataset, RunConfig, evaluate
    from ragas.metrics import LLMContextPrecisionWithReference, LLMContextRecall

    model = {"official": s["groq"]["judge_model"], "dev": s["groq"]["dev_model"]}.get(args.judge, args.judge)
    judge = make_judge(model, pool.key)
    metrics = [LLMContextPrecisionWithReference(), LLMContextRecall()]
    rows = [json.loads(l) for l in (s.eval_dir / "ragas.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = rows[:args.limit] if args.limit else rows
    retriever = Retriever()
    retriever.warmup(rerank="hybrid_rerank" in args.modes)
    tag = f"_{args.tag}" if args.tag else ""
    judge_tag = model.split("/")[-1]

    summary = {}
    for mode in args.modes:
        out = results_path(f"ragas/{mode}_{judge_tag}{tag}.csv")
        done = pd.read_csv(out) if out.exists() else pd.DataFrame()
        done_ids = set(done["qid"]) if len(done) else set()
        todo = [r for r in rows if r["qid"] not in done_ids]
        print(f"[{mode}] judge={model} | {len(done_ids)} done, {len(todo)} to go", flush=True)
        try:
            i = 0
            while i < len(todo):
                try:
                    check(model, s["groq"]["min_remaining_requests"])
                except QuotaExhausted as e:
                    if not pool.rotate(model, str(e).split(":")[0]):
                        raise
                    judge = make_judge(model, pool.key)
                    continue
                batch = todo[i:i + BATCH]
                samples = []
                for r in batch:
                    res = retriever.search(r["query"], mode=mode)
                    samples.append({"user_input": r["query"], "retrieved_contexts": [h.text for h in res.hits],
                                    "reference": r["answer"]})
                t = time.time()
                scores = evaluate(EvaluationDataset.from_list(samples), metrics=metrics, llm=judge,
                                  show_progress=False, run_config=RunConfig(max_workers=2, timeout=240, max_retries=10, max_wait=90))
                df = scores.to_pandas()
                df.insert(0, "qid", [r["qid"] for r in batch])
                df.insert(1, "query_type", [r["query_type"] for r in batch])
                df["mode"], df["judge"] = mode, model
                df = df.drop(columns=[c for c in ("retrieved_contexts",) if c in df.columns])
                # a NaN score means the judge call failed (usually Groq's 200k tokens/day limit). Don't save it:
                # the question stays "to do" and is scored on a later run, instead of silently lowering n.
                failed = df[["llm_context_precision_with_reference", "context_recall"]].isna().any(axis=1)
                ok = df[~failed]
                if len(ok):
                    ok.to_csv(out, mode="a", header=not out.exists(), index=False)
                q = LATEST.get(model, {})
                print(f"  +{len(ok)} scored, {int(failed.sum())} failed (left for later) in {time.time() - t:.0f}s | "
                      f"key #{pool.label} | Groq requests left today: "
                      f"{q.get('remaining_requests', '?')}/{q.get('limit_requests', '?')}", flush=True)
                if failed.sum() == len(batch):
                    # nothing was saved, so the same batch is retried with the next key
                    if pool.rotate(model, "all judge calls in a batch failed, most likely 200k tokens/day"):
                        judge = make_judge(model, pool.key)
                        continue
                    raise QuotaExhausted(f"all {len(batch)} judge calls failed for {model} on every key — most "
                                         "likely the 200k tokens/day limit. Re-run the same command after the reset.")
                i += BATCH
                time.sleep(PAUSE)   # let Groq's 8,000 tokens/minute window refill (a few failures are per-minute limits)
        except QuotaExhausted as e:
            print(f"STOPPED: {e}", flush=True)
        if not out.exists():
            print(f"[{mode}] nothing scored yet", flush=True)
            continue
        final = pd.read_csv(out).drop_duplicates("qid")   # two overlapping runs must not count a question twice
        cp, cr = "llm_context_precision_with_reference", "context_recall"
        summary[mode] = {"n_questions": len(final), "context_precision": round(final[cp].mean(), 6),
                         "context_recall": round(final[cr].mean(), 6), "judge": model,
                         "nan_scores": int(final[[cp, cr]].isna().sum().sum())}
        print(f"[{mode}] Context Precision {summary[mode]['context_precision']:.3f} | "
              f"Context Recall {summary[mode]['context_recall']:.3f} | n={len(final)}", flush=True)
    save_json(f"ragas/summary_{judge_tag}{tag}.json", summary, judge=model, groq_quota=LATEST,
              index_points=retriever.store.count())


if __name__ == "__main__":
    main()
