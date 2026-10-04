"""Collects every number the web dashboard shows from results/ — nothing is typed by hand.
A missing file gives None, which the page shows as "not measured"."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

from .config import get_settings
from .metrics import paired_bootstrap

CP, CR = "llm_context_precision_with_reference", "context_recall"
IR_MODES = ("dense", "bm25", "hybrid", "dense_rerank", "hybrid_rerank")


def _json(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _bs(a, b) -> dict:
    r = paired_bootstrap(list(a), list(b), n=2000)
    return {k: round(v, 6) if isinstance(v, float) else v for k, v in r.items()}


def _ragas(R: Path, judge: str, suffix: str = "", shared: tuple = ()) -> dict | None:
    """shared: modes whose retrieved contexts were checked identical to the base (no-suffix) run, so its scores apply"""
    files = {m: R / "ragas" / f"{m}_{judge}{'' if m in shared else suffix}.csv" for m in IR_MODES}
    if not files["dense"].exists():
        return None
    df = {m: pd.read_csv(f).drop_duplicates("qid").set_index("qid") for m, f in files.items() if f.exists()}
    out = {"judge": judge, "modes": {m: {"n": len(d), "context_precision": round(d[CP].mean(), 6),
                                         "context_recall": round(d[CR].mean(), 6)} for m, d in df.items()},
           "paired": {}, "shared_with_gpu": [m for m in shared if m in df]}   # each mode vs Phase 1 dense
    for m, d in df.items():
        if m == "dense":
            continue
        common = sorted(set(df["dense"].index) & set(d.index))
        out["paired"][m] = {"n": len(common),
                            "context_precision": _bs(df["dense"].loc[common, CP], d.loc[common, CP]),
                            "context_recall": _bs(df["dense"].loc[common, CR], d.loc[common, CR])}
    return out


def _ir(R: Path, suffix: str = "") -> dict | None:
    files = {m: R / "ir" / f"test_{m}{suffix}_perquery.csv" for m in IR_MODES}
    if not files["dense"].exists():
        return None
    base = pd.read_csv(files["dense"]).sort_values("qid")
    out = {}
    for m, f in files.items():
        if not f.exists():
            continue
        d = pd.read_csv(f).sort_values("qid")
        out[m] = {k: round(d[k].mean(), 6) for k in ("mrr@10", "ndcg@10", "hit@5", "recall@5", "precision@5", "hit@10")}
        out[m]["n"] = len(d)
        if "total_ms" in d:     # in-process search time per question (no HTTP), used for the CPU-profile card
            out[m]["p50_ms"] = round(float(d["total_ms"].quantile(0.5)), 1)
            out[m]["p95_ms"] = round(float(d["total_ms"].quantile(0.95)), 1)
        if m != "dense":
            out[m]["vs_dense"] = _bs(base["mrr@10"], d["mrr@10"])
    return out


def _tuning(R: Path) -> dict | None:
    out = {"split": "dev_large", "curve": None, "bm25_only": None, "rerank": None}
    f = R / "tuning" / "dev_large_fusion_grid.csv"
    if f.exists():
        d = pd.read_csv(f)
        pts = lambda rows: [{"w_bm25": float(r.w_bm25), "mrr": round(r["mrr@10"], 6), "recall5": round(r["recall@5"], 6)}
                            for _, r in rows.sort_values("w_bm25").iterrows()]
        dense = d[d["config"] == "dense_only"]
        out["curve"] = pts(pd.concat([dense, d[d["config"].str.startswith("wrrf") | (d["config"] == "rrf")]]))
        out["curve_linear"] = pts(pd.concat([dense, d[d["fusion"] == "linear"]]))
        bm25 = d[d["config"] == "bm25_only"]
        out["bm25_only"] = round(bm25["mrr@10"].iloc[0], 6) if len(bm25) else None
    f = R / "tuning" / "dev_large_rerank_grid.csv"
    if f.exists():
        d = pd.read_csv(f)
        out["rerank"] = [{"config": r.config, "w_bm25": float(r.w_bm25), "depth": int(r.rerank_depth),
                          "mrr": round(r["mrr@10"], 6)} for _, r in d.iterrows()]
    return out if out["curve"] or out["rerank"] else None


def _query_types(R: Path, suffix: str = "") -> dict | None:
    d = _json(R / "analysis" / f"failure_analysis{suffix}.json")
    if not d:
        return None
    qt = d["data"]["breakdowns"]["query_type"]
    keep = ("n", "dense_mrr", "bm25_mrr", "hybrid_mrr", "dense_rerank_mrr", "hybrid_rerank_mrr")
    return {"types": {t: {k: v[k] for k in keep if k in v} for t, v in qt.items()},
            "cases": d["data"]["case_counts"], "n_questions": d["data"]["n_questions"]}


def _latency(R: Path, name: str) -> dict | None:
    d = _json(R / "latency" / name)
    if not d:
        return None
    return {"modes": {m: {k: x[k] for k in ("n", "p50_ms", "p95_ms", "p99_ms", "max_ms", "pass_p95_lt_300",
                                            "median_stage_ms")} for m, x in d["data"].items()},
            "timestamp": d["meta"]["timestamp"], "gpu": d["meta"]["hardware"].get("gpu", "CPU")}


def _shared_contexts(R: Path) -> tuple:
    c = _json(R / "ragas" / "cpu_vs_gpu_contexts_100k.json")
    return tuple(m for m, x in (c or {}).get("data", {}).items() if x["identical_top5"] == x["n"])


def _build_stats(R: Path, n: int, suffix: str = "") -> dict | None:
    b = _json(R / f"index_build_{n}{suffix}.json")
    if not b:
        return None
    d = b["data"]
    out = {"passages": d["passages_indexed"], "minutes": d["total_minutes"], "seconds": d["seconds"],
           "ram_after": d.get("ram_after"), "timestamp": b["meta"]["timestamp"],
           "gpu": b["meta"]["hardware"].get("gpu", "CPU"), "device": b["meta"].get("device")}
    # the first build of this size (idle machine) is kept in its log when a later rebuild overwrote the JSON
    log = R / f"build_index_{n // 1000}k{suffix}.log"
    if log.exists():
        import re
        m = re.search(r"DONE: [\d,]+ points in ([\d.]+) min", log.read_text(encoding="utf-8", errors="ignore"))
        if m and abs(float(m.group(1)) - d["total_minutes"]) > 0.5:
            out["first_build_minutes"] = float(m.group(1))
    return out


def _tags(R: Path) -> dict:
    out = {"distribution": {}, "validation": None, "filter": {}}
    for name in ("msmarco", "msmarco_500k"):
        d = _json(R / "tags" / f"summary_{name}.json")
        if d:
            out["distribution"][name] = d["data"]
    v = _json(R / "tags" / "topic_validation_summary.json")
    out["validation"] = v and v["data"]
    for split in ("dev_large", "test", "test_cpu"):
        f = _json(R / "tags" / f"topic_filter_{split}.json")
        if f:
            out["filter"][split] = f["data"]
    return out


@lru_cache(maxsize=1)
def _build(stamp: tuple) -> dict:
    s = get_settings()
    R = s.results_dir
    build = _json(R / "index_build_100000.json")
    ret = s["retrieval"]
    judge = s["groq"]["judge_model"].split("/")[-1]
    bs500 = _json(R / "ir" / "test_bootstrap_msmarco_500k.json")
    b100, b100c = _build_stats(R, 100000), _build_stats(R, 100000, "_cpu")
    if b100 and b100c:   # the CPU timing run used a temporary collection next to the others; RAM is the 100k index's own
        b100c["ram_after"] = b100["ram_after"]
    return {
        "ragas_500k": _ragas(R, judge, "_500k"),
        "scale": {"100k": b100, "100k_cpu": b100c, "500k": _build_stats(R, 500000),
                  "candidate_recall_500k": bs500 and bs500["data"].get("candidate_recall")},
        "tags": _tags(R),
        "ragas": _ragas(R, judge),
        # 100k on the CPU profile: dense and hybrid return identical top-5 contexts on CPU and GPU for all 50 RAGAS
        # questions (results/ragas/cpu_vs_gpu_contexts_100k.json), so those scores apply; the reranker modes were re-run
        "ragas_cpu": _ragas(R, judge, "_cpu", shared=_shared_contexts(R)),
        "cpu_contexts": (_json(R / "ragas" / "cpu_vs_gpu_contexts_100k.json") or {}).get("data"),
        "ir": _ir(R),
        "ir_500k": _ir(R, "_msmarco_500k"),
        "ir_cpu": _ir(R, "_cpu"),          # CPU profile (DEVICE=cpu): 100k, rerank top-10, no GPU
        "profiles": s["profiles"], "serving": s["serving"]["devices"],
        "ir_gpu_rerank_depth": s["profiles"]["gpu"]["rerank_depth"], "ir_cpu_rerank_depth": s["profiles"]["cpu"]["rerank_depth"],
        "latency": _latency(R, "summary.json"),
        "latency_500k": _latency(R, "summary_500k.json"),
        "latency_cpu": _latency(R, "summary_cpu.json"),   # 100k index served on the CPU profile
        "latency_cpu_recheck": _latency(R, "summary_cpu_recheck.json"),   # same, re-measured with the laptop busy
        "tuning": _tuning(R),
        "query_types": _query_types(R),
        "query_types_cpu": _query_types(R, "_cpu"),
        "query_types_500k": _query_types(R, "_msmarco_500k"),
        "candidate_recall": (_json(R / "tuning" / "dev_large_candidate_recall.json") or {}).get("data"),
        "index_build": build and {"passages": build["data"]["passages_indexed"],
                                  "minutes": build["data"]["total_minutes"], "timestamp": build["meta"]["timestamp"]},
        "updates_demo": _json(R / "updates_demo.json") is not None,
        "config": {"fusion_hybrid": ret["fusion_hybrid"], "weights_hybrid": ret["weights_hybrid"], "weights_hybrid_rerank": ret["weights_hybrid_rerank"],
                   "rrf_k": ret["rrf_k"], "prefetch": ret["prefetch_limit"], "top_k": ret["top_k"],
                   "rerank_depth": s.profile["rerank_depth"], "dense_model": s["models"]["dense"],
                   "reranker": s["models"]["reranker"], "judge": s["groq"]["judge_model"]},
    }


def dashboard() -> dict:
    """Rebuilt only when a file in results/ changes (the bootstraps take ~1 s)."""
    R = get_settings().results_dir
    stamp = tuple(sorted((str(p), p.stat().st_mtime) for p in R.rglob("*") if p.suffix in (".csv", ".json")))
    return _build(stamp)
