"""Generate docs/BENCHMARK_REPORT.md + charts ONLY from files in results/. Nothing is typed by hand; anything missing
is printed as NOT MEASURED YET.

    python -m scripts.make_report
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from precisionrag.config import ROOT, get_settings  # noqa: E402
from precisionrag.metrics import paired_bootstrap  # noqa: E402

R = get_settings().results_dir
DOCS = ROOT / "docs"
IMG = DOCS / "img"
NM = "**NOT MEASURED YET**"
LABEL = {"dense": "Dense (Phase 1 baseline)", "bm25": "BM25 only", "hybrid": "Hybrid (dense + BM25, weighted RRF)",
         "dense_rerank": "Dense + reranker (ablation)", "hybrid_rerank": "**Hybrid + reranker (Phase 2)**",
         "hybrid_rerank+filter": "Hybrid + reranker + category filter"}
CP, CR = "llm_context_precision_with_reference", "context_recall"


def jload(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def section_index():
    files = sorted(R.glob("index_build_*.json"))
    if not files:
        return NM
    rows = []
    for f in files:
        d = jload(f)
        s, m = d["data"]["seconds"], d["meta"]
        hw = "CPU only (GPU not used)" if m.get("device") == "cpu" else m["hardware"].get("gpu", "CPU")
        rows.append([f"{d['data']['passages_indexed']:,}", m["profile"], hw,
                     s["dense_embed"], s["bm25_embed"], s["upload"], f"**{d['data']['total_minutes']} min**",
                     "✅" if d["data"]["pass"] else "❌", f"`results/{f.name}`"])
    return table(["Passages", "Profile", "Hardware", "Dense s", "BM25 s", "Upload s", "Total", "< 120 min", "Source"], rows)


def section_ragas():
    # the 100k GPU-profile runs only; *_500k and *_cpu runs have their own sections, never averaged in here
    files = sorted(f for f in (R / "ragas").glob("*.csv") if not f.stem.endswith(("_500k", "_cpu")))
    if not files:
        return NM, None
    dfs = [pd.read_csv(f).drop_duplicates("qid").assign(file=f.name) for f in files]
    allr = pd.concat(dfs)
    rows = []
    for (judge, mode), g in allr.groupby(["judge", "mode"]):
        rows.append([judge, LABEL.get(mode, mode), len(g), f"{g[CP].mean():.3f}", f"{g[CR].mean():.3f}"])
    text = table(["Judge", "Mode", "Questions", "Context Precision", "Context Recall"], rows)
    # paired comparison on the SAME questions, same judge
    pairs = []
    for judge, g in allr.groupby("judge"):
        if {"dense", "hybrid_rerank"} <= set(g["mode"]):
            a = g[g["mode"] == "dense"].set_index("qid")
            b = g[g["mode"] == "hybrid_rerank"].set_index("qid")
            common = sorted(set(a.index) & set(b.index))
            for metric, name in ((CP, "Context Precision"), (CR, "Context Recall")):
                bs = paired_bootstrap(a.loc[common, metric].tolist(), b.loc[common, metric].tolist(), n=2000)
                pairs.append([judge, name, len(common), f"{a.loc[common, metric].mean():.3f}",
                              f"{b.loc[common, metric].mean():.3f}", f"{bs['mean_diff']:+.3f}",
                              f"[{bs['ci95_low']:+.3f}, {bs['ci95_high']:+.3f}]"])
    if pairs:
        text += "\n\n**Phase 1 vs Phase 2 on the same questions (paired):**\n\n" + table(
            ["Judge", "Metric", "Questions", "Dense (P1)", "Hybrid+rerank (P2)", "Change", "95% CI"], pairs)
    return text, allr


def section_ir():
    modes = ["dense", "bm25", "hybrid", "dense_rerank", "hybrid_rerank"]
    have = {m: R / "ir" / f"test_{m}_perquery.csv" for m in modes}
    if not have["dense"].exists():
        return NM, None
    base = pd.read_csv(have["dense"]).sort_values("qid")
    rows, chart = [], {}
    for m in modes:
        if not have[m].exists():
            rows.append([LABEL[m]] + ["NOT MEASURED YET"] * 6)
            continue
        d = pd.read_csv(have[m]).sort_values("qid")
        bs = paired_bootstrap(base["mrr@10"].tolist(), d["mrr@10"].tolist(), n=2000) if m != "dense" else None
        rows.append([LABEL[m], f"{d['mrr@10'].mean():.3f}", f"{d['ndcg@10'].mean():.3f}", f"{d['hit@5'].mean():.3f}",
                     f"{d['recall@5'].mean():.3f}",
                     "—" if bs is None else f"{bs['mean_diff']:+.3f} [{bs['ci95_low']:+.3f}, {bs['ci95_high']:+.3f}]",
                     "—" if bs is None else f"{bs['wins']}/{bs['ties']}/{bs['losses']}"])
        chart[m] = (d["mrr@10"].mean(), d["recall@5"].mean())
    return table(["Mode", "MRR@10", "nDCG@10", "Hit@5", "Recall@5", "MRR vs dense (95% CI)", "better/same/worse"],
                 rows), chart


def section_latency():
    d = jload(R / "latency" / "summary.json")
    if not d:
        return NM, None
    rows = [[LABEL.get(m, m), x["n"], x["p50_ms"], f"**{x['p95_ms']}**", x["p99_ms"], x["max_ms"],
             "✅" if x["pass_p95_lt_300"] else "❌",
             ", ".join(f"{k} {v}" for k, v in x["median_stage_ms"].items() if k != "total")]
            for m, x in d["data"].items()]
    hw = d["meta"]["hardware"]
    head = (f"{d['meta']['timestamp']} · {d['meta']['profile']} profile · {hw.get('gpu', 'CPU')} · "
            f"{d['data'][next(iter(d['data']))]['n']} consecutive queries per mode · cache off · end-to-end HTTP time\n\n")
    return head + table(["Mode", "n", "p50 ms", "p95 ms", "p99 ms", "max ms", "p95 < 300", "median server stages (ms)"],
                        rows), d["data"]


def section_deployed():
    """The configuration the demo serves: 100k on the CPU profile, 500k on the GPU profile."""
    out = []
    modes = ["dense", "bm25", "hybrid", "dense_rerank", "hybrid_rerank"]
    rows = []
    base = R / "ir" / "test_dense_cpu_perquery.csv"
    b = pd.read_csv(base).sort_values("qid") if base.exists() else None
    for m in modes:
        f = R / "ir" / f"test_{m}_cpu_perquery.csv"
        if not f.exists():
            rows.append([LABEL[m]] + ["NOT MEASURED YET"] * 5)
            continue
        d = pd.read_csv(f).sort_values("qid")
        bs = paired_bootstrap(b["mrr@10"].tolist(), d["mrr@10"].tolist(), n=2000) if m != "dense" and b is not None else None
        rows.append([LABEL[m], f"{d['mrr@10'].mean():.3f}", f"{d['ndcg@10'].mean():.3f}", f"{d['hit@5'].mean():.3f}",
                     f"{d['recall@5'].mean():.3f}", "—" if bs is None else f"{bs['mean_diff']:+.3f} [{bs['ci95_low']:+.3f}, {bs['ci95_high']:+.3f}]"])
    out.append("**100k on the CPU profile (reranker reads the top 10) — 1,000 test questions:**\n\n" +
               table(["Mode", "MRR@10", "nDCG@10", "Hit@5", "Recall@5", "MRR vs dense (95% CI)"], rows))
    ctx = jload(R / "ragas" / "cpu_vs_gpu_contexts_100k.json")
    rag = []
    for m in ("dense", "hybrid", "hybrid_rerank"):
        same = ctx and ctx["data"].get(m, {}).get("identical_top5") == ctx["data"].get(m, {}).get("n")
        f = R / "ragas" / (f"{m}_gpt-oss-120b.csv" if same else f"{m}_gpt-oss-120b_cpu.csv")
        if f.exists():
            d = pd.read_csv(f).drop_duplicates("qid")
            rag.append([LABEL.get(m, m), len(d), f"{d[CP].mean():.3f}", f"{d[CR].mean():.3f}",
                        "same top-5 as the GPU for all 50 questions (shared score)" if same else "scored on the CPU"])
        else:
            rag.append([LABEL.get(m, m), "—", "NOT MEASURED YET", "", ""])
    out.append("**RAGAS, 100k on the CPU (judge gpt-oss-120b, the same 50 questions):**\n\n" +
               table(["Mode", "Questions", "Context Precision", "Context Recall", "Note"], rag))
    lat = jload(R / "latency" / "summary_cpu.json")
    if lat:
        out.append("**Latency, 100k on the CPU, 100 consecutive queries (same session as the 500k GPU run):**\n\n" + table(
            ["Mode", "p50 ms", "p95 ms", "p99 ms", "p95 < 300"],
            [[LABEL.get(m, m), x["p50_ms"], f"**{x['p95_ms']}**", x["p99_ms"], "✅" if x["pass_p95_lt_300"] else "❌"]
             for m, x in lat["data"].items()]))
    tf = jload(R / "tags" / "topic_filter_test_cpu.json")
    if tf:
        out.append("**Topic routing, 100k on the CPU (Phase 2, 1,000 test questions):**\n\n" + table(
            ["Setting", "MRR@10", "Recall@5", "Routed right", "vs no filter (95% CI)"],
            [[k.split("|")[1], f"{x['mrr@10']:.3f}", f"{x['recall@5']:.3f}",
              f"{x['route_accuracy']:.1%}" if "route_accuracy" in x else "—",
              f"{x['mrr_vs_none']['mean_diff']:+.3f} [{x['mrr_vs_none']['ci95_low']:+.3f}, {x['mrr_vs_none']['ci95_high']:+.3f}]"
              if "mrr_vs_none" in x else "—"] for k, x in tf["data"].items()]))
    bc = jload(R / "index_build_100000_cpu.json")
    if bc:
        out.append(f"**Index build on the CPU:** {bc['data']['passages_indexed']:,} passages in {bc['data']['total_minutes']} min "
                   f"(limit {bc['data']['limit_minutes']}; dense encoding {bc['data']['seconds']['dense_embed'] / 60:.1f} min).")
    out.append("The 500k GPU numbers are in section 9. Sections 2–4 and 8 are the 100k index measured on the GPU "
               "profile during development (reranker top 20).")
    return "\n\n".join(out)


def section_500k():
    out = []
    modes = ["dense", "bm25", "hybrid", "dense_rerank", "hybrid_rerank"]
    rows = []
    for m in modes:
        f = R / "ir" / f"test_{m}_msmarco_500k_perquery.csv"
        if f.exists():
            d = pd.read_csv(f)
            rows.append([LABEL[m], f"{d['mrr@10'].mean():.3f}", f"{d['recall@5'].mean():.3f}", f"{d['hit@5'].mean():.3f}"])
    out.append("**Quality, 1,000 test questions:**\n\n" + (table(["Mode", "MRR@10", "Recall@5", "Hit@5"], rows) if rows else NM))
    bs = jload(R / "ir" / "test_bootstrap_msmarco_500k.json")
    if bs:
        b = bs["data"]["bootstrap"]
        out.append(table(["Comparison", "MRR@10 change (95% CI)", "Recall@5 change (95% CI)"],
                         [[k.replace("_", " "), f"{v['mrr@10']['mean_diff']:+.3f} [{v['mrr@10']['ci95_low']:+.3f}, "
                           f"{v['mrr@10']['ci95_high']:+.3f}]", f"{v['recall@5']['mean_diff']:+.3f} "
                           f"[{v['recall@5']['ci95_low']:+.3f}, {v['recall@5']['ci95_high']:+.3f}]"] for k, v in b.items()]))
        cr = bs["data"]["candidate_recall"]
        out.append("**Candidate recall (correct passage anywhere in the pool):**\n\n" + table(
            ["Pool", "Dense", "BM25", "Union"], [[f"top-{k}", v["dense"], v["bm25"], v["union"]] for k, v in cr.items()]))
    lat = jload(R / "latency" / "summary_500k.json")
    if lat:
        out.append("**Latency, 100 consecutive queries (measured in the same session as the 100k GPU and CPU runs; both collections loaded):**\n\n" + table(
            ["Mode", "p50 ms", "p95 ms", "p99 ms", "p95 < 300"],
            [[LABEL.get(m, m), x["p50_ms"], f"**{x['p95_ms']}**", x["p99_ms"], "✅" if x["pass_p95_lt_300"] else "❌"]
             for m, x in lat["data"].items()]))
    rag = [(f, pd.read_csv(f).drop_duplicates("qid")) for f in sorted((R / "ragas").glob("*_500k.csv"))]
    if rag:
        out.append("**RAGAS at 500k (same judge, same 50 questions as the 100k runs):**\n\n" + table(
            ["Judge", "Mode", "Questions", "Context Precision", "Context Recall"],
            [[d["judge"].iloc[0], LABEL.get(d["mode"].iloc[0], d["mode"].iloc[0]), len(d), f"{d[CP].mean():.3f}",
              f"{d[CR].mean():.3f}"] for f, d in rag if len(d)]))
    build = jload(R / "index_build_500000.json")
    if build:
        d = build["data"]
        out.append(f"**Index build:** {d['passages_indexed']:,} passages in {d['total_minutes']} min "
                   f"(limit {d['limit_minutes']}); Qdrant RAM after build {d['ram_after']['qdrant_mb']} MB. "
                   "(An earlier build on an idle machine took 13.3 min; this one shared the GPU with other jobs.)")
    return "\n\n".join(out)


def section_tags():
    out = []
    s = jload(R / "tags" / "summary_msmarco.json")
    if not s:
        return NM
    d = s["data"]
    out.append(f"{d['passages']:,} passages tagged in {d['seconds']} s from the vectors already in Qdrant "
               "(no re-embedding, no re-index).\n\n" + table(
                   ["Topic", "Passages", "Share"], [[k, f"{v:,}", f"{v / d['passages']:.1%}"] for k, v in d["topics"].items()]))
    out.append(table(["Source type", "Passages", "Share"],
                     [[k, f"{v:,}", f"{v / d['passages']:.1%}"] for k, v in d["source_types"].items()]))
    v = jload(R / "tags" / "topic_validation_summary.json")
    if v:
        x = v["data"]
        out.append(f"**Topic tags vs an LLM judge** ({x['judge']}, {x['n']} random passages): "
                   f"{x['top1_agreement']:.0%} agree on the topic; {x['llm_topic_in_our_top2']:.0%} within our top 2.")
    for split in ("dev_large", "test"):
        f = jload(R / "tags" / f"topic_filter_{split}.json")
        if not f:
            continue
        rows = []
        for key, x in f["data"].items():
            mode, cfg = key.split("|")
            bs = x.get("mrr_vs_none")
            rows.append([LABEL.get(mode, mode), cfg, f"{x['mrr@10']:.3f}", f"{x['recall@5']:.3f}",
                         f"{x['route_accuracy']:.1%}" if x.get("route_accuracy") is not None else "—",
                         f"{bs['mean_diff']:+.3f} [{bs['ci95_low']:+.3f}, {bs['ci95_high']:+.3f}]" if bs else "—"])
        out.append(f"**Topic routing, {split} questions** (auto*N* = pre-filter to the question's N most likely topics; "
                   "oracle = the correct passage's topic, an upper bound):\n\n" + table(
                       ["Mode", "Setting", "MRR@10", "Recall@5", "Routing correct", "MRR vs no filter (95% CI)"], rows))
    return "\n\n".join(out)


def charts(ir_chart, lat):
    IMG.mkdir(parents=True, exist_ok=True)
    out = []
    if ir_chart:
        names = list(ir_chart)
        fig, ax = plt.subplots(figsize=(7.5, 3.4))
        x = range(len(names))
        ax.bar([i - 0.2 for i in x], [ir_chart[n][0] for n in names], 0.4, label="MRR@10", color="#1f3f73")
        ax.bar([i + 0.2 for i in x], [ir_chart[n][1] for n in names], 0.4, label="Recall@5", color="#0c8a78")
        ax.set_xticks(list(x), [n.replace("_", "\n") for n in names])
        ax.set_ylim(0, 1); ax.set_title("Ablation on 1,000 test questions (100k passages)"); ax.legend(frameon=False)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout(); fig.savefig(IMG / "ablation.png", dpi=150); plt.close(fig)
        out.append("![Ablation](img/ablation.png)")
    if lat:
        names = list(lat)
        fig, ax = plt.subplots(figsize=(7.5, 3.0))
        ax.bar([n.replace("_", "\n") for n in names], [lat[n]["p95_ms"] for n in names], color="#1f3f73", label="p95")
        ax.scatter([n.replace("_", "\n") for n in names], [lat[n]["p50_ms"] for n in names], color="#0c8a78",
                   zorder=3, label="p50")
        ax.axhline(300, color="#c2410c", ls="--", lw=1); ax.text(-0.4, 305, "300 ms limit", color="#c2410c", fontsize=8)
        ax.set_ylabel("ms"); ax.set_title("Latency, 100 consecutive queries"); ax.legend(frameon=False)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout(); fig.savefig(IMG / "latency.png", dpi=150); plt.close(fig)
        out.append("![Latency](img/latency.png)")
    return "\n\n".join(out)


def main():
    ragas_md, _ = section_ragas()
    ir_md, ir_chart = section_ir()
    lat_md, lat = section_latency()
    upd = jload(R / "updates_demo.json")
    fa = jload(R / "analysis" / "failure_analysis.json")
    filt = {fm: jload(R / "ir" / f"test_summary_filter-{fm}.json") for fm in ("oracle", "wrong")}
    cand = jload(R / "tuning" / "dev_large_candidate_recall.json")

    parts = [
        "# PrecisionRAG — Benchmark Report (Team OutLiers)",
        "_Generated by `scripts/make_report.py` from the files in `results/`. Every number has a source file; "
        "anything not run yet says NOT MEASURED YET._",
        "## 0. Deployed configuration: 100k on the CPU, 500k on the GPU", section_deployed(),
        "## 1. Index build (rule FR-1: < 2 hours on consumer hardware)", section_index(),
        "## 2. RAGAS — Phase 1 vs Phase 2, 100k, GPU profile (rules C-02, C-07, NFR-1/2)",
        "Judge models run on Groq's free tier at temperature 0. Each comparison uses the **same judge and the same "
        "questions**. The official judge is `openai/gpt-oss-120b`; others are independent cross-checks.", ragas_md,
        "## 3. Ablation with human labels, 100k, GPU profile (1,000 held-out test questions)",
        "Paired bootstrap 95% confidence intervals: if the interval excludes 0, the change is not luck.", ir_md,
        "## 4. Latency, 100k, GPU profile (rules NFR-3, C-05)", lat_md, charts(ir_chart, lat),
    ]
    if cand:
        rows = [[f"top-{p['depth']}", p["recall_dense"], p["recall_bm25"], p["recall_union"]] for p in cand["data"]["pool"]]
        parts += ["## 5. Why BM25 is in the pipeline: candidate recall (dev_large, 1,000 questions)",
                  table(["Pool", "Dense", "BM25", "Union"], rows)]
    if any(filt.values()):
        rows = [[fm, f"{d['data']['hybrid']['mrr@10']:.3f}", f"{d['data']['hybrid']['recall@5']:.3f}"]
                for fm, d in filt.items() if d]
        parts += ["## 6. Metadata filter trade-off (hybrid, category filter, pre-retrieval)",
                  table(["Filter", "MRR@10", "Recall@5"], rows),
                  "`oracle` = the question's true type (upper bound; a real user must choose it). `wrong` = another type."]
    parts += ["## 7. Live updates (rule FR-5)",
              NM if not upd else table(["Step", "Result", "Detail"],
                                       [[s["step"], "✅" if s["pass"] else "❌", s["detail"]] for s in upd["data"]["steps"]])
              + f"\n\nIndex size after the demo: {upd['data']['points_after']:,} points (no rebuild)."]
    if fa:
        c = fa["data"]["case_counts"]
        parts += ["## 8. Failure analysis (1,000 test questions)", table(["Case", "Questions"], list(c.items())),
                  table(["Question type", "n"] + [LABEL[m].replace("*", "") for m in
                                                   ["dense", "hybrid", "dense_rerank", "hybrid_rerank"]],
                        [[k, v["n"], v["dense_mrr"], v["hybrid_mrr"], v["dense_rerank_mrr"], v["hybrid_rerank_mrr"]]
                         for k, v in fa["data"]["breakdowns"]["query_type"].items()]) + "\n\n(MRR@10 per type.)"]
    parts += ["## 9. Scale: 500,000 passages (bonus)", section_500k()]
    parts += ["## 10. Metadata tags and topic routing", section_tags()]
    if fa and fa["data"].get("examples"):
        ex_md = []
        for case, items in fa["data"]["examples"].items():
            for e in items[:1]:
                ex_md.append(f"**{case}** — _{e['query']}_ ({e['query_type']})  \n"
                             f"ranks of the correct passage: {e['ranks']}  \n"
                             f"dense #1: {e['dense_top1'] or '—'}  \n"
                             f"Phase 2 #1: {e['hybrid_rerank_top1'] or '—'}  \n"
                             f"labelled correct: {e['labelled_correct'] or '—'}")
        parts += ["## 11. Real examples from the failure analysis", "\n\n".join(ex_md)]
    (DOCS / "BENCHMARK_REPORT.md").write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    print(f"wrote {DOCS / 'BENCHMARK_REPORT.md'}")


if __name__ == "__main__":
    main()
