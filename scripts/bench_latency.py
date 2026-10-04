"""Latency benchmark (rules NFR-3 / C-05): 100 CONSECUTIVE different questions against the running API.

* 5 warm-up questions first (reported separately, not counted).
* Cache disabled (the API is called with no_cache=true once caching exists).
* Records the client-side end-to-end time (HTTP round trip) AND the server's per-stage timings for every query.
* Reports p50 / p95 / p99 per mode. Raw per-query logs are saved — no self-reported estimates.

Run with nothing else using the GPU/CPU (e.g. no RAGAS run), or the numbers are not representative.
    python -m scripts.bench_latency --modes dense hybrid hybrid_rerank --filtered
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import httpx
import pandas as pd

from precisionrag.config import get_settings
from precisionrag.results import save_csv, save_json

API = os.getenv("PRECISIONRAG_API", "http://127.0.0.1:8765")


def pct(values, p):
    v = sorted(values)
    return v[min(len(v) - 1, int(round(p / 100 * (len(v) - 1))))]


def main():
    s = get_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["dense", "hybrid", "hybrid_rerank"])
    ap.add_argument("--filtered", action="store_true", help="also run hybrid_rerank with each question's category filter")
    ap.add_argument("--tag", default="")
    ap.add_argument("--index", default="100k", help="which index the API should search: 100k or 500k")
    args = ap.parse_args()

    rows = [json.loads(l) for l in (s.eval_dir / "latency.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    warm_n = s["eval"]["latency_warmup"]
    warm, timed = rows[:warm_n], rows[warm_n:warm_n + s["eval"]["latency_queries"]]
    client = httpx.Client(base_url=API, timeout=60)
    health = client.get("/health").json()

    runs = [(m, False) for m in args.modes] + ([("hybrid_rerank", True)] if args.filtered else [])
    summary = {}
    for mode, filtered in runs:
        label = mode + ("+filter" if filtered else "")
        params = lambda r: {"q": r["query"], "mode": mode, "no_cache": "true", "index": args.index,
                            **({"category": r["query_type"]} if filtered else {})}
        warm_ms = []
        for r in warm:
            t = time.perf_counter(); client.get("/search", params=params(r)).raise_for_status()
            warm_ms.append((time.perf_counter() - t) * 1000)
        log = []
        for i, r in enumerate(timed, 1):
            t = time.perf_counter()
            resp = client.get("/search", params=params(r))
            e2e = (time.perf_counter() - t) * 1000
            resp.raise_for_status()
            body = resp.json()
            log.append({"i": i, "qid": r["qid"], "mode": label, "e2e_ms": round(e2e, 2),
                        **{f"server_{k}_ms": v for k, v in body["timings_ms"].items()}})
        df = pd.DataFrame(log)
        e2e = df["e2e_ms"].tolist()
        stages = {c.replace("server_", "").replace("_ms", ""): round(float(df[c].median()), 2)
                  for c in df.columns if c.startswith("server_")}
        summary[label] = {"n": len(df), "p50_ms": round(pct(e2e, 50), 1), "p95_ms": round(pct(e2e, 95), 1),
                          "p99_ms": round(pct(e2e, 99), 1), "max_ms": round(max(e2e), 1),
                          "mean_ms": round(statistics.mean(e2e), 1), "pass_p95_lt_300": pct(e2e, 95) < 300,
                          "warmup_ms": [round(w, 1) for w in warm_ms], "median_stage_ms": stages}
        save_csv(f"latency/{label}{('_' + args.tag) if args.tag else ''}.csv", df, mode=label,
                 index=args.index)
        x = summary[label]
        print(f"{label:22s} p50 {x['p50_ms']:6.1f} | p95 {x['p95_ms']:6.1f} | p99 {x['p99_ms']:6.1f} ms "
              f"-> {'PASS' if x['pass_p95_lt_300'] else 'FAIL'} | stages {stages}", flush=True)
    serving = (health.get("serving") or {}).get(args.index, {})
    save_json(f"latency/summary{('_' + args.tag) if args.tag else ''}.json", summary,
              index_points=(health.get("indexes") or {}).get(args.index, health["points"]), index=args.index,
              serving=serving, api=API, cache="off")


if __name__ == "__main__":
    main()
