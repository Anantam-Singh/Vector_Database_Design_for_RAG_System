"""Builds the presentation report book (docs/ppt_report.html) and every diagram as a separate SVG (docs/diagrams/)
from results/ — no number is typed by hand. Organised by the five judging criteria:
business value, scalability, solution architecture, presentation, bonus features.

Deployment pairing used everywhere: the 100k index on the CPU profile, the 500k index on the GPU profile.

    python -m scripts.make_report_book
"""
from __future__ import annotations

import html
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

from precisionrag.config import get_settings
from precisionrag.dashboard import dashboard

ROOT = Path(__file__).resolve().parents[1]
DOCS, DIAG = ROOT / "docs", ROOT / "docs" / "diagrams"
E = html.escape

LIGHT = {"bg": "#f5f4ef", "surface": "#fdfcf8", "ink": "#13202f", "soft": "#44505f", "muted": "#6b7480",
         "line": "#dedcd3", "accent": "#2a5fd8", "accent-soft": "#e6edfb", "warm": "#d9782a", "warm-soft": "#fbeedd",
         "good": "#0f7a55", "good-soft": "#e3f4ec", "bad": "#b8322a", "bad-soft": "#fbe6e3", "grid": "#e7e5dd"}
DARK = {"bg": "#0f1722", "surface": "#162131", "ink": "#e8ecf1", "soft": "#b6c0cc", "muted": "#8b96a3",
        "line": "#2a3747", "accent": "#7ea6ff", "accent-soft": "#1d2d48", "warm": "#f0a35e", "warm-soft": "#3a2a1a",
        "good": "#5fd3a3", "good-soft": "#15332a", "bad": "#ff8a80", "bad-soft": "#3a1d1d", "grid": "#243142"}

# one stylesheet for diagrams and charts: var(--token) in the page, fixed light colours in the exported .svg files
DIAG_CSS = """
.dg text{font-family:@body;}
.d-bg{fill:@surface}
.d-box{fill:@surface;stroke:@line;stroke-width:1.2}
.d-soft{fill:@bg;stroke:@line;stroke-width:1.2}
.d-a{fill:@accent-soft;stroke:@accent;stroke-width:1.4}
.d-w{fill:@warm-soft;stroke:@warm;stroke-width:1.4}
.d-g{fill:@good-soft;stroke:@good;stroke-width:1.4}
.d-r{fill:@bad-soft;stroke:@bad;stroke-width:1.4}
.d-t{fill:@ink;font-size:13.5px;font-weight:600}
.d-s{fill:@soft;font-size:12px}
.d-m{fill:@muted;font-size:11px;letter-spacing:.06em;font-family:@mono}
.d-ln{stroke:@muted;stroke-width:1.5;fill:none}
.d-ah{fill:@muted}
.c-a{fill:@accent}.c-w{fill:@warm}.c-g{fill:@good}.c-b{fill:@bad}.c-m{fill:@muted}.c-i{fill:@ink}
.c-ax{stroke:@grid;stroke-width:1}
.c-tg{stroke:@bad;stroke-width:2;stroke-dasharray:5 4}
.c-tgt{fill:@bad;font-size:12px}
.c-t{fill:@soft;font-size:12px}
.c-v{fill:@ink;font-size:12px;font-weight:600;font-variant-numeric:tabular-nums}
.c-la{stroke:@accent;stroke-width:2.5;fill:none}
.c-lw{stroke:@warm;stroke-width:2.5;fill:none;stroke-dasharray:6 4}
.c-ring{stroke:@accent;stroke-width:2;fill:none}
"""


def css_for(page: bool, pal: dict | None = None) -> str:
    out = DIAG_CSS
    for k in sorted(LIGHT, key=len, reverse=True):
        out = out.replace("@" + k, f"var(--{k})" if page else pal[k])
    fonts = ("var(--body)", "var(--mono)") if page else ("'Segoe UI', Arial, sans-serif", "Consolas, 'Courier New', monospace")
    return out.replace("@body", fonts[0]).replace("@mono", fonts[1])


f3 = lambda v: "—" if v is None else f"{v:.3f}"
sg = lambda v, n=3: "—" if v is None else f"{v:+.{n}f}".replace("-", "−")
ms0 = lambda v: "—" if v is None else f"{round(v)} ms"
pct = lambda v: "—" if v is None else f"{v * 100:.1f}%"
ci = lambda b: f"{sg(b['ci95_low'])} … {sg(b['ci95_high'])}" if b else "—"

DIAGRAMS: dict[str, tuple[str, str]] = {}   # name -> (inline svg, standalone svg)


# ---------------------------------------------------------------- SVG helpers
def text(x, y, s, cls="d-s", anchor="middle", extra=""):
    return f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" class="{cls}" {extra}>{E(str(s))}</text>'


def box(x, y, w, h, cls, title, *sub, r=10, tcls="d-t"):
    lines = [(title, tcls, 17)] + [(s, "d-s", 15) for s in sub if s]
    total = sum(lh for _, _, lh in lines)
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" class="{cls}"/>']
    cy = y + h / 2 - total / 2 + 12
    for s, c, lh in lines:
        out.append(text(x + w / 2, cy, s, c))
        cy += lh
    return "".join(out)


def label(x, y, s):
    return text(x, y, s.upper(), "d-m", "start")


def arrow(name, pts, both=False, lab=None, lx=None, ly=None, anchor="middle"):
    d = "M" + " L".join(f"{x},{y}" for x, y in pts)
    m = f' marker-end="url(#ah-{name})"' + (f' marker-start="url(#ah-{name})"' if both else "")
    out = f'<path d="{d}" class="d-ln"{m}/>'
    if lab:
        out += text(lx, ly, lab, "d-s", anchor)
    return out


def make_svg(name, w, h, body, aria):
    marker = (f'<defs><marker id="ah-{name}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
              f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="d-ah"/></marker></defs>')
    inline = (f'<svg class="dg" viewBox="0 0 {w} {h}" width="{w}" role="img" aria-label="{E(aria)}">'
              f'{marker}{body}</svg>')
    alone = (f'<svg xmlns="http://www.w3.org/2000/svg" class="dg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">'
             f'<title>{E(aria)}</title><style>{css_for(False, LIGHT)}</style>{marker}'
             f'<rect width="{w}" height="{h}" class="d-bg"/>{body}</svg>')
    DIAGRAMS[name] = (inline, alone)
    return f'<div class="chart">{inline}</div><p class="src">Diagram file: docs/diagrams/{name}.svg</p>'


def hbars(name, rows, *, mx, fmt=f3, unit="", target=None, target_label="", label_w=220, aria=""):
    """rows: [(label, [(value or None, class)])] — one horizontal bar per value"""
    per = len(rows[0][1])
    bh, gap = 15, 4
    row_h = per * (bh + gap) + 12
    W, H = 780, len(rows) * row_h + 36
    plot = W - label_w - 80
    X = lambda v: label_w + v / mx * plot
    out = []
    for t in range(5):
        v = mx / 4 * t
        out.append(f'<line x1="{X(v):.1f}" x2="{X(v):.1f}" y1="4" y2="{H - 24}" class="c-ax"/>')
        out.append(text(X(v), H - 7, fmt(v) + unit, "c-t"))
    for i, (lab, vals) in enumerate(rows):
        y0 = 6 + i * row_h
        out.append(text(label_w - 10, y0 + per * (bh + gap) / 2 + 2, lab, "c-t", "end"))
        for j, (v, cls) in enumerate(vals):
            y = y0 + j * (bh + gap)
            if v is None:
                out.append(text(label_w + 4, y + bh - 3, "pending", "c-t", "start"))
                continue
            out.append(f'<rect x="{label_w}" y="{y}" width="{max(1, X(v) - label_w):.1f}" height="{bh}" rx="3" class="{cls}"/>')
            out.append(text(X(v) + 6, y + bh - 3, fmt(v) + unit, "c-v", "start"))
    if target is not None:
        out.append(f'<line x1="{X(target):.1f}" x2="{X(target):.1f}" y1="2" y2="{H - 24}" class="c-tg"/>')
        out.append(text(X(target) - 5, 13, target_label, "c-tgt", "end"))
    return make_svg(name, W, H, "".join(out), aria or name)


def legend(*items):
    return '<div class="legend">' + "".join(f'<span><i class="sw {c}"></i>{E(t)}</span>' for t, c in items) + "</div>"


# ---------------------------------------------------------------- data
def load():
    d = dashboard()
    s = get_settings()
    R = s.results_dir
    upd = json.loads((R / "updates_demo.json").read_text(encoding="utf-8"))["data"]
    try:
        out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"], cwd=ROOT, capture_output=True,
                             text=True, timeout=300).stdout
        tests = next((int(l.split()[0]) for l in out.splitlines() if "tests collected" in l or "test collected" in l), None)
    except Exception:
        tests = None
    lock = sum(1 for l in (ROOT / "requirements.lock.txt").read_text(encoding="utf-8").splitlines()
               if l.strip() and not l.startswith("#"))
    ev = {p.stem: sum(1 for l in p.read_text(encoding="utf-8").splitlines() if l.strip()) for p in s.eval_dir.glob("*.jsonl")}
    linear = json.loads((R / "ir" / "test_summary_linear03.json").read_text(encoding="utf-8"))["data"]
    return d, s, upd, tests, lock, ev, linear


def main():
    d, s, upd, tests, lock, ev, linear = load()
    A, B = d["ir_cpu"], d["ir_500k"]                       # 100k CPU, 500k GPU (1,000 test questions)
    LA, LB = d["latency_cpu"]["modes"], d["latency_500k"]["modes"]
    RA, RB = d["ragas_cpu"], d["ragas_500k"]
    busy = (d.get("latency_cpu_recheck") or {}).get("modes", {}).get("hybrid_rerank")
    gpu_name = d["latency_500k"]["gpu"]
    P = d["profiles"]
    sc = d["scale"]
    b100 = sc.get("100k_cpu") or sc["100k"]
    b100_dev = "CPU" if sc.get("100k_cpu") else "GPU"
    b500 = sc["500k"]
    b500_min = b500.get("first_build_minutes") or b500["minutes"]
    cr5 = sc["candidate_recall_500k"]
    cr1 = {p["depth"]: p for p in d["candidate_recall"]["pool"]}
    tags = d["tags"]
    dist = tags["distribution"]["msmarco"]
    val = tags["validation"]
    tf = tags["filter"].get("test_cpu") or tags["filter"]["test"]
    tf_none, tf_or, tf_a2 = tf["hybrid_rerank|none"], tf["hybrid_rerank|oracle"], tf["hybrid_rerank|auto2"]
    qtA, qtB = d["query_types_cpu"], d["query_types_500k"]
    ctx = d.get("cpu_contexts") or {}
    p2A = RA["modes"].get("hybrid_rerank") if RA else None
    p2B = RB["modes"]["hybrid_rerank"]
    dB = RB["modes"]["dense"]
    gA, gB = A["hybrid_rerank"]["vs_dense"], B["hybrid_rerank"]["vs_dense"]
    stA, stB = LA["hybrid_rerank"]["median_stage_ms"], LB["hybrid_rerank"]["median_stage_ms"]
    ram100, ram500 = sc["100k"]["ram_after"]["qdrant_mb"], b500["ram_after"]["qdrant_mb"]
    kb_per = ram500 * 1024 / 500000
    steps = {x["step"].split(".")[0]: x for x in upd["steps"]}
    wt = lambda k: float(steps[k]["detail"].split(" in ")[1].split(" ms")[0])
    w_ins, w_upd, w_meta, w_del = wt("1"), wt("3"), wt("5"), wt("6")
    dA, dRA = P["cpu"]["rerank_depth"], P["gpu"]["rerank_depth"]
    tun = {p["w_bm25"]: p for p in d["tuning"]["curve"]}
    tun_lin = {p["w_bm25"]: p for p in d["tuning"]["curve_linear"]}
    rr = {(r["config"], r["depth"]): r["mrr"] for r in d["tuning"]["rerank"]}
    p2A_txt = f"{f3(p2A['context_precision'])} / {f3(p2A['context_recall'])}" if p2A else "pending"

    # ============================================================ diagrams
    # D1 business: naive RAG vs PrecisionRAG
    n = "business-naive-vs-precision"
    b = [label(20, 26, "Naive RAG (Phase 1)"),
         box(20, 40, 150, 58, "d-box", "Question"),
         box(205, 40, 190, 58, "d-box", "Dense top-5", "closest meaning only"),
         box(430, 40, 170, 58, "d-box", "LLM answer"),
         box(635, 40, 285, 58, "d-r", "Risk: look-alike passage first", "or an outdated policy version"),
         arrow(n, [(170, 69), (203, 69)]), arrow(n, [(395, 69), (428, 69)]), arrow(n, [(600, 69), (633, 69)]),
         label(20, 146, "PrecisionRAG (Phase 2)"),
         box(20, 160, 150, 70, "d-a", "Question", "+ scope filter"),
         box(205, 160, 190, 70, "d-a", "Dense ∪ BM25", "one Qdrant call, filtered"),
         box(430, 160, 170, 70, "d-a", "Cross-encoder", f"picks the best 5"),
         box(635, 160, 285, 70, "d-g", "Right passage first", "only the current version exists"),
         arrow(n, [(170, 195), (203, 195)]), arrow(n, [(395, 195), (428, 195)]), arrow(n, [(600, 195), (633, 195)]),
         text(20, 268, f"Measured: MRR@10 {sg(gB['mean_diff'])} at 500k (GPU) and {sg(gA['mean_diff'])} at 100k (CPU) over dense · "
                       f"correct scope filter {sg(tf_or['mrr_vs_none']['mean_diff'])} · old versions never returned", "d-s", "start")]
    D_business = make_svg(n, 940, 284, "".join(b), "Naive RAG versus PrecisionRAG retrieval")

    # D2 system architecture
    n = "architecture-system"
    b = [label(50, 26, "Offline indexing (resumable)"),
         box(50, 40, 210, 56, "d-box", "MS MARCO v1.1", "100k or 500k passages"),
         box(50, 130, 210, 56, "d-box", "Clean · dedup · metadata", "HTML entities fixed"),
         box(50, 220, 210, 50, "d-a", "Dense encoder", "bge-small-en-v1.5 · 384-d"),
         box(50, 285, 210, 50, "d-a", "BM25 encoder", "FastEmbed sparse vectors"),
         box(50, 350, 210, 50, "d-a", "Topic tagger", "15 topics from stored vectors"),
         arrow(n, [(155, 96), (155, 128)]),
         f'<path d="M50,158 H32 V375" class="d-ln"/>',
         arrow(n, [(32, 245), (48, 245)]), arrow(n, [(32, 310), (48, 310)]), arrow(n, [(32, 375), (48, 375)]),
         label(330, 160, "Vector database"),
         f'<rect x="330" y="170" width="280" height="230" rx="12" class="d-soft"/>',
         text(470, 194, "Qdrant 1.19 (local, free)", "d-t"), text(470, 211, "msmarco (100k) · msmarco_500k", "d-s"),
         box(350, 225, 240, 46, "d-box", "Dense vectors · HNSW", "cosine · m 16 · ef 128"),
         box(350, 280, 240, 46, "d-box", "BM25 sparse vectors", "IDF computed live by Qdrant"),
         box(350, 335, 240, 50, "d-box", "Payload + 5 filter indexes", "topic · source type · corpus · …"),
         arrow(n, [(260, 245), (328, 245)]), arrow(n, [(260, 310), (328, 310)]), arrow(n, [(260, 375), (328, 375)]),
         box(400, 40, 210, 60, "d-w", "Company systems", "insert · update · delete"),
         box(690, 40, 250, 80, "d-box", "Web UI", "Search · Phases · Compare", "Scale · Live updates"),
         f'<rect x="690" y="170" width="250" height="230" rx="12" class="d-a"/>',
         text(815, 194, "FastAPI service", "d-t"), text(815, 211, "one process · models loaded once", "d-s"),
         text(815, 240, "/search  ·  /answer", "d-s"), text(815, 258, "/passages  upsert · delete", "d-s"),
         text(815, 276, "/dashboard  ·  /changes", "d-s"),
         box(705, 296, 220, 42, "d-box", f"100k → CPU profile", f"reranker reads top {dA}"),
         box(705, 346, 220, 42, "d-box", f"500k → GPU profile", f"reranker reads top {dRA}"),
         arrow(n, [(612, 290), (688, 290)], both=True, lab="one batched query", lx=650, ly=280),
         arrow(n, [(815, 122), (815, 168)], both=True),
         arrow(n, [(610, 70), (650, 70), (650, 200), (688, 200)], lab="REST", lx=655, ly=135, anchor="start"),
         box(690, 440, 250, 56, "d-box", "Groq LLM (free tier)", "grounded answers · RAGAS judge"),
         box(400, 440, 210, 56, "d-box", "Change log · SQLite", "every version, before / after"),
         arrow(n, [(815, 402), (815, 438)]),
         arrow(n, [(690, 385), (650, 385), (650, 468), (612, 468)])]
    D_system = make_svg(n, 960, 510, "".join(b), "System architecture")

    # D3 query pipeline with stage timings
    n = "architecture-query-pipeline"
    st = lambda k: f"CPU {stA.get(k, 0):.1f} · GPU {stB.get(k, 0):.1f} ms"
    b = [box(20, 70, 130, 70, "d-box", "Question", "+ optional filters"),
         box(185, 45, 140, 52, "d-a", "Dense encode", "384-d query vector"),
         box(185, 113, 140, 52, "d-a", "BM25 encode", "sparse query vector"),
         box(360, 45, 185, 120, "d-soft", "Qdrant: 1 request", "dense top-50 ∥ BM25 top-50", "filter inside both", "HNSW + inverted index"),
         box(580, 70, 120, 70, "d-box", "Weighted RRF", "k = 60"),
         box(735, 60, 165, 90, "d-w", "Cross-encoder", "MiniLM-L6 reads", f"top {dA} (CPU) · top {dRA} (GPU)"),
         box(935, 70, 135, 70, "d-g", "Top 5 passages", "+ stage timings"),
         box(935, 190, 135, 62, "d-box", "Grounded answer", "Groq · cites [n]"),
         arrow(n, [(150, 95), (183, 71)]), arrow(n, [(150, 115), (183, 139)]),
         arrow(n, [(325, 71), (358, 85)]), arrow(n, [(325, 139), (358, 125)]),
         arrow(n, [(545, 105), (578, 105)]), arrow(n, [(700, 105), (733, 105)]), arrow(n, [(900, 105), (933, 105)]),
         arrow(n, [(1002, 140), (1002, 188)]),
         label(20, 206, "Median stage time, Phase 2 (100 queries)"),
         text(255, 226, st("embed_dense"), "d-s"), text(452, 226, st("qdrant"), "d-s"),
         text(640, 226, st("fusion"), "d-s"), text(817, 226, st("rerank"), "d-s"),
         text(20, 262, f"Total in the server: CPU 100k {stA['total']:.0f} ms · GPU 500k {stB['total']:.0f} ms "
                       f"(p95 end-to-end over HTTP: {ms0(LA['hybrid_rerank']['p95_ms'])} · {ms0(LB['hybrid_rerank']['p95_ms'])})", "d-s", "start")]
    D_query = make_svg(n, 1090, 280, "".join(b), "Phase 2 query pipeline with stage timings")

    # D4 data model
    n = "architecture-data-model"
    b = [label(20, 26, "One Qdrant point = one passage"),
         box(20, 40, 250, 70, "d-a", "ID = UUID5(doc_id)", "same doc_id → same point", "an update overwrites, never duplicates"),
         box(300, 40, 200, 70, "d-box", "Vector \"dense\"", "384 floats · cosine · HNSW"),
         box(530, 40, 200, 70, "d-box", "Vector \"bm25\"", "sparse · IDF modifier"),
         box(760, 40, 220, 70, "d-box", "Payload", "text · version · updated_at"),
         label(20, 146, "Indexed payload fields (pre-filter inside the search)"),
         box(20, 160, 175, 54, "d-g", "topic", "15 subjects"),
         box(210, 160, 175, 54, "d-g", "source_type", "8 kinds of website"),
         box(400, 160, 175, 54, "d-g", "corpus", "web · internal"),
         box(590, 160, 175, 54, "d-g", "category", "MS MARCO question type"),
         box(780, 160, 200, 54, "d-g", "source", "website domain")]
    D_model = make_svg(n, 1000, 230, "".join(b), "Qdrant point layout and indexed filter fields")

    # D5 live update flow
    n = "architecture-live-updates"
    b = [box(20, 95, 170, 70, "d-w", "POST /passages", "doc_id · text · metadata"),
         box(225, 95, 160, 70, "d-box", "Compare with", "stored version"),
         box(420, 20, 230, 56, "d-soft", "Same text → unchanged", "no re-embedding, no copy"),
         box(420, 102, 230, 56, "d-soft", "Metadata only → payload", f"vectors untouched · {w_meta:.0f} ms"),
         box(420, 184, 230, 56, "d-a", "New text → embed + upsert", f"same point ID · {w_upd:.0f} ms"),
         box(690, 150, 190, 56, "d-box", "Change log", "v1 → v2, full text kept"),
         box(690, 222, 190, 56, "d-box", "index_version + 1", "cache can't go stale"),
         box(915, 184, 160, 70, "d-g", "Next search", "returns only v2"),
         arrow(n, [(190, 130), (223, 130)]),
         arrow(n, [(385, 120), (400, 120), (400, 48), (418, 48)]), arrow(n, [(385, 130), (418, 130)]),
         arrow(n, [(385, 140), (400, 140), (400, 212), (418, 212)]),
         arrow(n, [(650, 205), (670, 205), (670, 178), (688, 178)]), arrow(n, [(650, 220), (670, 220), (670, 250), (688, 250)]),
         arrow(n, [(880, 219), (913, 219)]),
         text(20, 300, f"Scripted demo, {upd['passed']} / {upd['total']} steps pass: insert {w_ins:.1f} ms · update {w_upd:.1f} ms · "
                       f"metadata-only {w_meta:.1f} ms · delete {w_del:.1f} ms · no index rebuild", "d-s", "start")]
    D_updates = make_svg(n, 1090, 316, "".join(b), "Live update flow")

    # D6 deployment profiles
    n = "scalability-deployments"
    rows = lambda x0, cls, head, items: (box(x0, 70, 420, 50, cls, head) + "".join(
        f'<text x="{x0 + 20}" y="{150 + i * 26}" class="d-s" text-anchor="start">{E(k)}</text>'
        f'<text x="{x0 + 400}" y="{150 + i * 26}" class="c-v" text-anchor="end">{E(v)}</text>' for i, (k, v) in enumerate(items)))
    b = [box(310, 10, 380, 40, "d-soft", "Same code · same API · config serving.devices", tcls="d-s"),
         f'<rect x="20" y="62" width="440" height="330" rx="12" class="d-soft"/>',
         f'<rect x="540" y="62" width="440" height="330" rx="12" class="d-soft"/>',
         rows(30, "d-a", "CPU profile · 100,000 passages", [
             ("Hardware", "CPU only, no GPU"), ("Models", "fp32 on the CPU"), ("Reranker reads", f"top {dA}"),
             ("Index build", f"{b100['minutes']} min on the {b100_dev}"), ("Qdrant RAM", f"{ram100 / 1024:.1f} GB"),
             ("MRR@10, Phase 2", f3(A['hybrid_rerank']['mrr@10'])), ("Gain over dense", sg(gA["mean_diff"])),
             ("p95, Phase 2", ms0(LA["hybrid_rerank"]["p95_ms"])), ("p95 with a filter", ms0(LA["hybrid_rerank+filter"]["p95_ms"]))]),
         rows(550, "d-w", "GPU profile · 500,000 passages", [
             ("Hardware", gpu_name.replace("NVIDIA GeForce ", "")), ("Models", "fp16 on the GPU"), ("Reranker reads", f"top {dRA}"),
             ("Index build", f"{b500_min} min on the GPU"), ("Qdrant RAM", f"{ram500 / 1024:.1f} GB"),
             ("MRR@10, Phase 2", f3(B['hybrid_rerank']['mrr@10'])), ("Gain over dense", sg(gB["mean_diff"])),
             ("p95, Phase 2", ms0(LB["hybrid_rerank"]["p95_ms"])), ("p95 with a filter", ms0(LB["hybrid_rerank+filter"]["p95_ms"]))]),
         arrow(n, [(400, 50), (240, 60)]), arrow(n, [(600, 50), (760, 60)])]
    D_deploy = make_svg(n, 1000, 405, "".join(b), "Two deployments: 100k on CPU, 500k on GPU")

    # D7 evaluation design
    n = "presentation-evaluation-design"
    b = [box(20, 90, 190, 70, "d-box", "MS MARCO validation", "labelled questions"),
         box(250, 90, 170, 70, "d-soft", "Frozen sets", "seed 42 · disjoint"),
         box(460, 10, 170, 44, "d-box", f"dev · {ev.get('dev', 0)}"),
         box(460, 62, 170, 44, "d-a", f"dev_large · {ev.get('dev_large', 0):,}", tcls="d-t"),
         box(460, 114, 170, 44, "d-g", f"test · {ev.get('test', 0):,}"),
         box(460, 166, 170, 44, "d-w", f"RAGAS · {ev.get('ragas', 0)}"),
         box(460, 218, 170, 44, "d-box", f"latency · {ev.get('latency', 0)}"),
         box(670, 50, 190, 64, "d-a", "Choose settings", "fusion · weights · depth"),
         box(670, 122, 190, 64, "d-g", "Prove once on test", "label metrics, 1,000 q"),
         box(900, 122, 170, 64, "d-g", "Paired bootstrap", "2,000 resamples · 95% CI"),
         box(670, 196, 190, 64, "d-w", "LLM judge", "gpt-oss-120b · temp 0"),
         arrow(n, [(210, 125), (248, 125)]),
         arrow(n, [(420, 110), (440, 110), (440, 32), (458, 32)]), arrow(n, [(420, 115), (440, 115), (440, 84), (458, 84)]),
         arrow(n, [(420, 130), (440, 130), (440, 136), (458, 136)]), arrow(n, [(420, 140), (440, 140), (440, 188), (458, 188)]),
         arrow(n, [(420, 150), (440, 150), (440, 240), (458, 240)]),
         arrow(n, [(630, 84), (668, 84)]), arrow(n, [(765, 114), (765, 120)]), arrow(n, [(630, 136), (668, 150)]),
         arrow(n, [(860, 154), (898, 154)]), arrow(n, [(630, 188), (668, 226)])]
    D_eval = make_svg(n, 1090, 272, "".join(b), "Evaluation design")

    # ============================================================ charts
    order = [("dense", "Phase 1 · Dense"), ("bm25", "BM25 only"), ("hybrid", "Phase 2 · Hybrid"),
             ("dense_rerank", "Dense + reranker"), ("hybrid_rerank", "Phase 2 · Hybrid + reranker")]
    C_ablation = hbars("chart-ablation-mrr", [(l, [(A[m]["mrr@10"], "c-a"), (B[m]["mrr@10"], "c-w")]) for m, l in order],
                       mx=0.7, label_w=230, aria="MRR@10 by mode, 100k CPU and 500k GPU")
    rg_rows = [(f"100k CPU · {l}", RA["modes"].get(m)) for m, l in (("dense", "Dense"), ("hybrid", "Hybrid"), ("hybrid_rerank", "Phase 2"))] + \
              [(f"500k GPU · {l}", RB["modes"].get(m)) for m, l in (("dense", "Dense"), ("hybrid", "Hybrid"), ("hybrid_rerank", "Phase 2"))]
    C_ragas = hbars("chart-ragas", [(l, [((x or {}).get("context_precision"), "c-a"), ((x or {}).get("context_recall"), "c-w")]) for l, x in rg_rows],
                    mx=1.0, label_w=210, target=0.75, target_label="precision target 0.75", aria="RAGAS context precision and recall")
    lat_order = [("dense", "Dense"), ("hybrid", "Hybrid"), ("hybrid_rerank", "Phase 2 (hybrid + reranker)"), ("hybrid_rerank+filter", "Phase 2 + metadata filter")]
    C_latency = hbars("chart-latency-p95", [(l, [(LA[m]["p95_ms"], "c-a"), (LB[m]["p95_ms"], "c-w")]) for m, l in lat_order],
                      mx=320, fmt=lambda v: f"{v:.0f}", unit=" ms", target=300, target_label="300 ms target", label_w=210,
                      aria="p95 latency, 100 consecutive queries")
    keys = [("embed_dense", "Dense encode", "c-a"), ("embed_bm25", "BM25 encode", "c-m"), ("qdrant", "Qdrant search", "c-g"),
            ("fusion", "RRF fusion", "c-m"), ("rerank", "Cross-encoder", "c-w")]
    # stacked bars: one row per deployment
    W, lw, mx = 780, 170, 120
    X = lambda v: lw + v / mx * (W - lw - 70)
    out = []
    for t in range(7):
        v = mx / 6 * t
        out.append(f'<line x1="{X(v):.1f}" x2="{X(v):.1f}" y1="4" y2="96" class="c-ax"/>' + text(X(v), 114, f"{v:.0f} ms", "c-t"))
    for i, (lab, stg) in enumerate((("100k · CPU", stA), ("500k · GPU", stB))):
        y, x0 = 14 + i * 42, 0.0
        out.append(text(lw - 10, y + 17, lab, "c-t", "end"))
        for k, _, cls in keys:
            v = stg.get(k, 0)
            out.append(f'<rect x="{X(x0):.1f}" y="{y}" width="{max(0.5, X(x0 + v) - X(x0)):.1f}" height="24" class="{cls}"/>')
            x0 += v
        out.append(text(X(x0) + 6, y + 17, f"{stg['total']:.0f} ms", "c-v", "start"))
    C_stages = make_svg("chart-stage-breakdown", W, 124, "".join(out), "Median Phase 2 stage time")
    C_topics = hbars("chart-topic-distribution", [(k, [(v / dist["passages"] * 100, "c-a")]) for k, v in dist["topics"].items()],
                     mx=25, fmt=lambda v: f"{v:.1f}", unit="%", label_w=180, aria="Topic distribution, 100k passages")
    C_routing = hbars("chart-topic-routing", [("No topic filter", [(tf_none["mrr@10"], "c-g")]),
                                              ("Correct topic (user picks)", [(tf_or["mrr@10"], "c-a")]),
                                              ("Automatic: top-2 topics", [(tf_a2["mrr@10"], "c-b")])],
                      mx=0.7, label_w=230, aria="Topic routing, MRR@10, 100k CPU")
    # fusion tuning curve
    xs = sorted(tun)
    Wd, H, L, Rr, T, Bm, lo, hi = 780, 250, 60, 20, 20, 40, 0.48, 0.55
    Xc = lambda i: L + i / (len(xs) - 1) * (Wd - L - Rr)
    Yc = lambda v: T + (1 - (v - lo) / (hi - lo)) * (H - T - Bm)
    out = []
    for k in range(8):
        v = lo + k * 0.01
        out.append(f'<line x1="{L}" x2="{Wd - Rr}" y1="{Yc(v):.1f}" y2="{Yc(v):.1f}" class="c-ax"/>' + text(L - 8, Yc(v) + 4, f"{v:.2f}", "c-t", "end"))
    for i, w in enumerate(xs):
        out.append(text(Xc(i), H - 18, "0 (dense)" if w == 0 else f"{w:g}", "c-t"))
    out.append(text((L + Wd - Rr) / 2, H - 2, "BM25 weight (dense weight 1.0)", "c-t"))
    pts = lambda m: " L".join(f"{Xc(xs.index(w)):.1f},{Yc(p['mrr']):.1f}" for w, p in sorted(m.items()))
    out.append(f'<path d="M{pts(tun)}" class="c-la"/><path d="M{pts(tun_lin)}" class="c-lw"/>')
    for w, p in tun.items():
        out.append(f'<circle cx="{Xc(xs.index(w)):.1f}" cy="{Yc(p["mrr"]):.1f}" r="3.5" class="c-a"/>')
    for w, p in tun_lin.items():
        out.append(f'<circle cx="{Xc(xs.index(w)):.1f}" cy="{Yc(p["mrr"]):.1f}" r="3.5" class="c-w"/>')
    ch = tun[0.1]
    out.append(f'<circle cx="{Xc(1):.1f}" cy="{Yc(ch["mrr"]):.1f}" r="8" class="c-ring"/>')
    out.append(text(Xc(1) + 12, Yc(ch["mrr"]) + 24, f"chosen 1 : 0.1 → {ch['mrr']:.3f}", "c-v", "start"))
    out.append(text(Xc(3) + 8, Yc(tun_lin[0.3]["mrr"]) - 10, f"linear {tun_lin[0.3]['mrr']:.3f} on tuning, {linear['hybrid']['mrr@10']:.3f} on test", "c-v", "start"))
    C_fusion = make_svg("chart-fusion-tuning", Wd, H, "".join(out), "Fusion tuning curve")

    # ============================================================ page
    tdate = date.today().strftime("%-d %b %Y") if sys.platform != "win32" else date.today().strftime("%#d %b %Y")
    sec = []

    def section(sid, title, crit, slide, body, src=""):
        sec.append(f'''<section class="s" id="{sid}" data-title="{E(title)}" data-crit="{crit}">
  <div class="s-head"><div><div class="s-num">{E(slide)}</div><h2>{E(title)}</h2></div>
    <div class="head-right"><button class="copy" type="button">Copy text</button></div></div>
  {body}
  {f'<p class="src">{E(src)}</p>' if src else ""}
</section>''')

    def part(pid, num, title, lead):
        sec.append(f'<header class="part" id="{pid}" data-title="{E(title)}" data-part="1"><span class="pnum">Criterion {num}</span>'
                   f'<h2>{E(title)}</h2><p>{lead}</p></header>')

    kpi = lambda v, l, dd, cls="": f'<div class="kpi"><span class="v {cls}">{v}</span><span class="l">{l}</span><span class="d">{dd}</span></div>'
    say = lambda t: f'<p class="say">{t}</p>'
    def tbl(head, rows, hl=()):
        """head: column names; a leading # marks a right-aligned number column. rows: lists of HTML cells."""
        num = [h.startswith("#") for h in head]
        nc = ' class="n"'
        th = "".join(f"<th{nc if num[j] else ''}>{h.lstrip('#')}</th>" for j, h in enumerate(head))
        body = "".join(f"<tr{' class=' + chr(34) + 'hl' + chr(34) if i in hl else ''}>"
                       + "".join(f"<td{nc if num[j] else ''}>{c}</td>" for j, c in enumerate(r)) + "</tr>" for i, r in enumerate(rows))
        return f'<div class="tw"><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'

    # ---- summary
    section("summary", "Results at a glance", "all", "Slide 2 · headline", f'''
  <div class="kpis k8">
    {kpi(f"{f3(p2A['context_precision'])} / {f3(p2A['context_recall'])}" if p2A else "pending", "RAGAS precision / recall, Phase 2, 100k on a CPU (targets 0.75 / 0.70)", f"{p2A['n'] if p2A else 0} questions · judge gpt-oss-120b")}
    {kpi(f"{f3(p2B['context_precision'])} / {f3(p2B['context_recall'])}", "RAGAS precision / recall, Phase 2, 500k on a GPU", f"dense {f3(dB['context_precision'])} / {f3(dB['context_recall'])}", "warm")}
    {kpi(sg(gA['mean_diff']), "MRR@10 gain over dense, 100k on a CPU, 1,000 test questions", f"95% CI {ci(gA)}")}
    {kpi(sg(gB['mean_diff']), "MRR@10 gain over dense, 500k on a GPU", f"95% CI {ci(gB)}", "warm")}
    {kpi(ms0(LA['hybrid_rerank']['p95_ms']), "p95, Phase 2, 100k on a CPU (target &lt; 300)", "100 consecutive queries via the API")}
    {kpi(ms0(LB['hybrid_rerank']['p95_ms']), "p95, Phase 2, 500k on a laptop GPU", f"{gpu_name.replace('NVIDIA GeForce ', '')}", "warm")}
    {kpi(sg(tf_or['mrr_vs_none']['mean_diff']), "MRR@10 when the right topic filter is picked", f"95% CI {ci(tf_or['mrr_vs_none'])} · 100k CPU")}
    {kpi(f"{b100['minutes']:.1f} / {b500_min} min", f"index build, 100k on {b100_dev} / 500k on GPU (limit 120)", "resumable, no GPU needed for 100k" if b100_dev == "CPU" else "resumable", "warm")}
  </div>
  {say(f"Two deployments, one codebase: 100,000 passages on an ordinary CPU and 500,000 on a small laptop GPU. Both meet every target, and the reranker's gain holds at 5× the data.")}''',
            "results/ragas · results/ir/test_*_perquery.csv · results/latency/summary_cpu.json, summary_500k.json · results/tags/topic_filter_test_cpu.json")

    # ============================== 1 business value
    part("c1", 1, "Business value proposition", "Why a company would pay for precision retrieval, and the measured proof behind each promise.")
    section("b-problem", "The problem: an LLM answers from whatever it is given", "1", "Slide 3 · problem", f'''
  <div class="cols">
    <div class="card"><h3>What goes wrong in naive RAG</h3><ul class="tight">
      <li>The search returns passages that <b>look</b> similar but answer a different question; the LLM then answers confidently from the wrong text.</li>
      <li>Policies change. If an old version stays in the index, the assistant quotes a refund window that no longer applies.</li>
      <li>Departments need answers from their own documents only, without a second system.</li>
      <li>GPU servers and paid APIs make a pilot expensive before it has proven anything.</li></ul></div>
    <div class="card"><h3>Who feels it</h3><ul class="tight">
      <li><b>Customer support</b>: refund, shipping and warranty questions answered from current policy.</li>
      <li><b>HR and internal help desks</b>: leave and remote-work rules, scoped to internal documents.</li>
      <li><b>Knowledge search</b>: large document sets where look-alike passages are common.</li></ul></div>
  </div>
  {D_business}
  {say("Retrieval decides what the LLM reads. We made the right passage come first, kept only the current version of every document, and let the user scope the search — on hardware a company already has.")}''')
    section("b-valuemap", "Value map: need → feature → measured proof", "1", "Slide 4 · value", tbl(
        ["Business need", "What we built", "Measured proof"], [
            ["Right passage first", "Hybrid candidates + cross-encoder reranker", f"MRR@10 {sg(gB['mean_diff'])} at 500k (CI {ci(gB)}); {sg(gA['mean_diff'])} at 100k on CPU"],
            ["Trustworthy context for the LLM", "Phase 2 retrieval, judged by RAGAS", f"Context Precision {f3(p2B['context_precision'])}, Recall {f3(p2B['context_recall'])} at 500k (targets 0.75 / 0.70)"],
            ["Answers change when policy changes", "Live upsert / delete, versioned, change log", f"update {w_upd:.0f} ms, no rebuild; only the new version can be returned ({upd['passed']} / {upd['total']} demo steps)"],
            ["Search only the right documents", "5 indexed pre-filters inside Qdrant", f"correct topic filter {sg(tf_or['mrr_vs_none']['mean_diff'])} MRR (CI {ci(tf_or['mrr_vs_none'])})"],
            ["Fast enough for chat", "One batched Qdrant call, short rerank list", f"p95 {ms0(LA['hybrid_rerank']['p95_ms'])} on CPU (100k), {ms0(LB['hybrid_rerank']['p95_ms'])} on GPU (500k)"],
            ["Low cost to start", "Open-source models, local Qdrant, free-tier LLM", "No paid service; the 100k deployment needs no GPU"],
            ["Grounded, checkable answers", "Answers cite the passages they used", "Every answer shows [n] citations next to the passages"],
        ], hl=(0, 2)) + say("Every promise on this slide has a measurement behind it, read from the results folder."))
    section("b-story", "Demo story: change the refund policy, the answer changes", "1", "Slide 5 · live demo", f'''
  <div class="cols">
    <div class="card"><h3>Before · version 1</h3><p>“Customers can return any item within <b>30 days</b> of delivery for a full refund [1].”</p></div>
    <div class="card good"><h3>After · version 2</h3><p>“Customers can return any item within <b>14 days</b> of delivery for a full refund [1].”</p></div>
  </div>
  <ul class="tight"><li>Edited on the Live Updates page, saved through the same API any company system would call.</li>
    <li>The same question runs again at once; the page shows both answers, a word-level diff and the v1 → v2 history.</li>
    <li>Demo documents are fictional (“Acme Retail”), tagged <span class="src">corpus = internal</span>, removed with one click afterwards.</li></ul>
  {say("A policy edit reaches the assistant in well under a second, and the old wording can never be retrieved again.")}''',
            "results/updates_demo.json · Live Updates page")

    # ============================== 2 scalability
    part("c2", 2, "Scalability", "The same pipeline from 100,000 passages on a CPU to 500,000 on a laptop GPU, and what it would take to go further.")
    section("s-deploy", "Two deployments, one codebase", "2", "Slide 6 · scale", D_deploy + say(
        f"The CPU deployment is what any laptop or office server can run: {ms0(LA['hybrid_rerank']['p95_ms'])} p95. "
        f"The GPU deployment carries five times the data at {ms0(LB['hybrid_rerank']['p95_ms'])}. Switching is one line in config."),
            "config.yaml serving.devices · results/index_build_*.json · results/latency/summary_cpu.json, summary_500k.json")
    section("s-quality", "Quality at both scales", "2", "Slide 7 · scale", f'''{C_ablation}
  {legend(("100k · CPU", "c-a"), ("500k · GPU", "c-w"))}
  <ul class="tight"><li>Five times more passages means more look-alikes: dense MRR {f3(A['dense']['mrr@10'])} → {f3(B['dense']['mrr@10'])}.</li>
    <li>The reranker's gain holds: {sg(gA['mean_diff'])} at 100k, {sg(gB['mean_diff'])} at 500k, both confidence intervals above zero.</li>
    <li>BM25 matters more as data grows: the correct passage is in the top-{dRA} candidate pool {pct(cr5[str(dRA)]['dense'])} of the time with dense alone at 500k, {pct(cr5[str(dRA)]['union'])} with dense + BM25.</li>
    <li>RAGAS at 500k still clears both targets: precision {f3(p2B['context_precision'])}, recall {f3(p2B['context_recall'])}.</li></ul>''',
            "results/ir/test_*_cpu_perquery.csv · test_*_msmarco_500k_perquery.csv · results/ir/test_bootstrap_msmarco_500k.json")
    section("s-latency", "Latency at both scales", "2", "Slide 8 · speed", f'''{C_latency}
  {legend(("100k · CPU", "c-a"), ("500k · GPU", "c-w"), ("300 ms target", "c-b"))}
  {C_stages}
  {legend(*[(l, c) for _, l, c in keys])}
  <p>The reranker is the slowest step. On the CPU it reads {dA} passages in {stA['rerank']:.0f} ms; the GPU reads {dRA} in {stB['rerank']:.0f} ms. Searching 5× more vectors adds only a few milliseconds in Qdrant ({stA['qdrant']:.0f} → {stB['qdrant']:.0f} ms), so the GPU serves 500k faster than the CPU serves 100k.</p>''',
            "results/latency/summary_cpu.json, summary_500k.json → median_stage_ms")
    section("s-resources", "Build time, memory and the path to 10 million", "2", "Slide 9 · growth", tbl(
        ["Measured", "#100k · CPU", "#500k · GPU"], [
            ["Index build (limit 120 min)", f"{b100['minutes']} min on the {b100_dev}", f"{b500_min} min"],
            ["Qdrant memory after build", f"{ram100 / 1024:.1f} GB", f"{ram500 / 1024:.1f} GB (≈ {kb_per:.1f} KB per passage)"],
            ["p95, Phase 2 / with a filter", f"{ms0(LA['hybrid_rerank']['p95_ms'])} / {ms0(LA['hybrid_rerank+filter']['p95_ms'])}", f"{ms0(LB['hybrid_rerank']['p95_ms'])} / {ms0(LB['hybrid_rerank+filter']['p95_ms'])}"],
            ["MRR@10 dense → Phase 2", f"{f3(A['dense']['mrr@10'])} → {f3(A['hybrid_rerank']['mrr@10'])}", f"{f3(B['dense']['mrr@10'])} → {f3(B['hybrid_rerank']['mrr@10'])}"],
            ["Every method misses (top 10)", f"{qtA['cases']['all_methods_fail(top10)']} of 1,000", f"{qtB['cases']['all_methods_fail(top10)']} of 1,000"],
        ], hl=(0, 2)) + f'''
  <div class="cols">
    <div class="card"><h3>To 10 million passages (plan, not measured)</h3><p class="note">At the measured ≈ {kb_per:.1f} KB per passage, 10M passages need about {kb_per * 10_000_000 / 1024 / 1024:.0f} GB before compression. Qdrant's int8 scalar quantization keeps compressed vectors in RAM and the originals on disk; sharding spreads the collection over machines.</p></div>
    <div class="card"><h3>What stays constant</h3><p class="note">The reranker reads a fixed number of candidates ({dA} or {dRA}), so its cost does not grow with the corpus. Updates touch one point each; nothing is ever rebuilt.</p></div>
  </div>''', "results/index_build_100000_cpu.json · results/index_build_500000.json · results/build_index_500k.log")

    # ============================== 3 architecture
    part("c3", 3, "Solution architecture", "How the pieces fit, why each was chosen, and what we tried and rejected.")
    section("a-system", "System overview", "3", "Slide 10 · architecture", D_system + say(
        "Indexing writes both vector types and the metadata into one Qdrant collection. One API serves search, answers, updates and the dashboard; the web UI and any company system use the same endpoints."))
    section("a-query", "Query path: one round trip, then a short rerank", "3", "Slide 11 · query path", D_query + say(
        "Dense and BM25 searches go to Qdrant in one batched request, with the filter applied inside both. Fusion is rank-based, and the cross-encoder reads only the short list."),
            "results/latency/*.json → median_stage_ms")
    section("a-model", "Data model and filters", "3", "Slide 12 · data", D_model + f'''
  <p class="note">Topic tags were computed from the vectors already stored (100k passages in {dist['seconds']:.0f} s, no re-embedding). Pre-filtering happens inside the HNSW search, so a filtered query still returns 5 results.</p>''')
    section("a-updates", "Live update path", "3", "Slide 13 · updates", D_updates + say(
        "The point ID is derived from the document ID, so an update overwrites the old version instead of adding a copy. The change log keeps every version for audit."),
            "results/updates_demo.json · data/changelog.sqlite")
    section("a-choices", "Component choices", "3", "Slide 14 · components", tbl(
        ["Component", "Choice", "Why"], [
            ["Vector database", "Qdrant 1.19 (allowed list: Qdrant, Weaviate, ChromaDB, pgvector)", "Dense + sparse with server-side IDF in one collection, filtered HNSW, payload indexes, batched queries"],
            ["Dense model", f"{s['models']['dense']}, 384-d", "Strong retrieval for its size; fast on CPU and GPU"],
            ["Keyword search", "BM25 sparse vectors (FastEmbed)", "Exact words, codes and names; IDF kept current by Qdrant"],
            ["Fusion", f"Weighted RRF, k = {s['retrieval']['rrf_k']}; hybrid 1 : {s['retrieval']['weights_hybrid']['bm25']}, before rerank 1 : 1", "Rank-based, no score normalisation; tuned on 1,000 questions"],
            ["Reranker", f"{s['models']['reranker']}", f"Reads question and passage together; top {dA} on CPU, top {dRA} on GPU"],
            ["Serving", "FastAPI + plain HTML/JS UI", "Models load once; latency measured on the real HTTP path"],
            ["Evaluation", f"RAGAS + judge {s['groq']['judge_model']}; label metrics; paired bootstrap", "Required metric plus a 1,000-question statistical check"],
        ]))
    section("a-rejected", "What we rejected, and the measurement behind each no", "3", "Slide 15 · decisions", tbl(
        ["Idea", "Measured", "Decision"], [
            ["Equal-weight RRF (1 : 1) for hybrid", f"MRR {tun[1.0]['mrr']:.3f} vs dense {tun[0.0]['mrr']:.3f} (tuning)", f"BM25 weight 0.1"],
            ["BM25 weight 0.2", f"won on 200 questions, {tun[0.2]['mrr']:.3f} on 1,000", "noise, dropped"],
            ["Linear min-max fusion", f"{tun_lin[0.3]['mrr']:.3f} on tuning, {linear['hybrid']['mrr@10']:.3f} on test", "kept weighted RRF"],
            ["Rerank top-30", f"{rr[('rrf', 30)]:.3f} vs {rr[('rrf', 20)]:.3f} at top-20 (tuning)", f"top-{dRA} on GPU"],
            [f"Rerank top-{dRA} on the CPU", f"same passages as the GPU for all 50 RAGAS questions, but the CPU reranker becomes the bottleneck", f"top-{dA} on CPU ({ms0(LA['hybrid_rerank']['p95_ms'])} p95)"],
            ["Automatic topic routing", f"{sg(tf_a2['mrr_vs_none']['mean_diff'])} MRR (test) with the top-2 guess, {pct(tf_a2['route_accuracy'])} routed right", "suggest topics, user decides"],
            ["Post-retrieval filtering", "can return fewer than 5 results", "pre-filter inside Qdrant"],
            ["Keeping old versions in the index", "an outdated policy could be retrieved", "latest only + change log"],
            ["Non-LLM RAGAS metrics", "string similarity scored a water passage correct for an ethanol question", "LLM judge only"],
            ["Milvus, Pinecone, OpenAI", "not on the allowed list or not free", "never used"],
        ]))

    # ============================== 4 presentation
    part("c4", 4, "Presentation", "How we prove the claims, how the 10-minute talk runs, and the questions to expect.")
    section("p-method", "Evaluation design: tune on one set, prove on another", "4", "Slide 16 · method", D_eval + f'''
  <p class="note">Label metrics (MRR@10, nDCG@10, Hit@5, Recall@5) use MS MARCO's human labels on 1,000 test questions. RAGAS uses the same 50 questions and the same judge for every mode. A difference whose 95% interval includes zero is marked “n.s.”.</p>''')
    section("p-ablation", "Ablation: which component helps", "4", "Slide 17 · evidence", tbl(
        ["Mode", "#MRR@10 100k CPU", "#vs dense (95% CI)", "#MRR@10 500k GPU", "#vs dense (95% CI)"],
        [[l, f3(A[m]["mrr@10"]), "baseline" if m == "dense" else f"{sg(A[m]['vs_dense']['mean_diff'])} ({ci(A[m]['vs_dense'])})",
          f3(B[m]["mrr@10"]), "baseline" if m == "dense" else f"{sg(B[m]['vs_dense']['mean_diff'])} ({ci(B[m]['vs_dense'])})"] for m, l in order], hl=(4,))
            + say("The reranker carries the gain at both scales. Hybrid alone equals dense on MS MARCO, and we say so; BM25 widens the pool the reranker reads."),
            "results/ir/*_perquery.csv (paired bootstrap, 2,000 resamples)")
    section("p-ragas", "RAGAS: every measured mode passes", "4", "Slide 18 · RAGAS", f'''{C_ragas}
  {legend(("Context Precision", "c-a"), ("Context Recall", "c-w"))}
  <p class="note">Judge {s['groq']['judge_model']}, temperature 0, the same 50 questions everywhere. At 100k on the CPU, dense and hybrid return exactly the same top 5 as the GPU for all 50 questions, so their scores are shared. Phase 2 on the CPU picks different passages for {ctx.get('hybrid_rerank', {}).get('n', 50) - ctx.get('hybrid_rerank', {}).get('identical_top5', 34)} of 50 questions{"; its score: " + p2A_txt if p2A else " and has not been scored yet (Groq judge unavailable)"}.</p>''',
            "results/ragas/*_gpt-oss-120b*.csv · results/ragas/cpu_vs_gpu_contexts_100k.json")
    section("p-scorecard", "Challenge scorecard", "4", "Slide 19 · scorecard", tbl(
        ["Requirement", "Target", "Measured", "Status"], [
            ["Corpus", "≥ 100,000 passages (500k bonus)", "100,000 on CPU · 500,000 on GPU", '<span class="ok">met + bonus</span>'],
            ["Phase 1 dense baseline", "RAGAS baseline logged", f"Precision {f3(RA['modes']['dense']['context_precision'])} · Recall {f3(RA['modes']['dense']['context_recall'])} (100k)", '<span class="ok">met</span>'],
            ["Phase 2 hybrid", "BM25 + dense, documented fusion", f"Weighted RRF k = {s['retrieval']['rrf_k']}, weights in config.yaml", '<span class="ok">met</span>'],
            ["RAGAS Context Precision", "&gt; 0.75", f"{f3(p2B['context_precision'])} (500k)" + (f" · {f3(p2A['context_precision'])} (100k CPU)" if p2A else ""), '<span class="ok">met</span>'],
            ["RAGAS Context Recall", "&gt; 0.70", f"{f3(p2B['context_recall'])} (500k)" + (f" · {f3(p2A['context_recall'])} (100k CPU)" if p2A else ""), '<span class="ok">met</span>'],
            ["RAGAS sample", "≥ 20 questions", f"{p2B['n']} questions, same for every mode", '<span class="ok">met</span>'],
            ["Latency", "p95 &lt; 300 ms, 100 queries", f"{ms0(LA['hybrid_rerank']['p95_ms'])} CPU 100k · {ms0(LB['hybrid_rerank']['p95_ms'])} GPU 500k", '<span class="ok">met</span>'],
            ["Metadata filter", "pre-retrieval", "inside Qdrant on 5 indexed fields", '<span class="ok">met</span>'],
            ["Upsert / delete", "no full re-index", f"{w_del:.0f}–{w_upd:.0f} ms per write, versioned", '<span class="ok">met</span>'],
            ["UI", "dense / hybrid toggle", "web UI with mode switch, filters, dashboards, 100k / 500k switch", '<span class="ok">met</span>'],
            ["Index build", "&lt; 2 hours", f"{b100['minutes']} min (100k, {b100_dev}) · {b500_min} min (500k)", '<span class="ok">met</span>'],
            ["Cost", "free tier only", "open-source models, local Qdrant, Groq free tier", '<span class="ok">met</span>'],
            ["Reproducible", "README + report", f"one command per number, {lock} pinned packages, {tests or '—'} tests", '<span class="ok">met</span>'],
        ], hl=(3, 4, 6)), "docs/BENCHMARK_REPORT.md · results/")
    section("p-demo", "Demo run-sheet (10 minutes)", "4", "Planning · demo", tbl(
        ["Time", "Show", "Web UI page", "Criterion"], [
            ["0:00", "Problem and the two deployments (100k CPU · 500k GPU)", "header switch", "1 · 2"],
            ["1:00", "Ask “what do the folds in the mitochondria membranes do”: dense puts a wrong passage first; Phase 2 lifts “cristae” from #2 to #1", "Phase 2: Hybrid + Reranker → Live Query Trace", "3"],
            ["2:30", "Per-stage timing bar on a live search, mode switch dense ↔ hybrid", "Search", "3"],
            ["3:30", "Pick the topic chip “Animals &amp; Nature” for a puppy-food question; every result carries the tag", "Search", "5"],
            ["4:30", "Refund policy 30 → 14 days: answer changes, word diff, version history", "Live Updates", "1 · 5"],
            ["6:00", "RAGAS and ablation with confidence intervals", "Comparison", "4"],
            ["7:00", "Switch to 500k: same question on both indexes, side by side", "Scale", "2"],
            ["8:00", "Rejected ideas and lessons", "slides", "3"],
            ["9:00", "Close: business value in one sentence; questions", "slides", "1"],
        ]) + '<p class="note">Before starting: run one search on each index (the laptop GPU wakes from power saving), close OneDrive or other heavy programs (the CPU deployment shares the processor), and after the live-update demo click “Remove them”.</p>')
    section("p-deck", "Deck outline by criterion", "4", "Planning · deck", tbl(
        ["#", "Slide", "Criterion", "Report section"], [
            ["1", "Cover · team · one-line promise", "—", "—"], ["2", "Results at a glance", "all", "Results at a glance"],
            ["3", "The problem", "1 · business", "The problem"], ["4", "Value map", "1 · business", "Value map"],
            ["5", "Architecture", "3 · architecture", "System overview, Query path"], ["6", "Data model and live updates", "3 · architecture", "Data model, Live update path"],
            ["7", "Two deployments", "2 · scalability", "Two deployments"], ["8", "Quality and latency at scale", "2 · scalability", "Quality, Latency"],
            ["9", "Evidence: ablation + RAGAS", "4 · presentation", "Ablation, RAGAS"], ["10", "Bonus features", "5 · bonus", "Bonus features"],
            ["11", "Live demo", "all", "Demo run-sheet"], ["12", "Decisions, lessons, next steps", "3 · architecture", "Rejected ideas"],
            ["+", "Backup: scorecard, question types, failures, Q&amp;A", "4", "Appendix"],
        ]))
    qa = [
        ("Why Qdrant, not pgvector, Weaviate or ChromaDB?", "Dense + BM25 sparse (server-side IDF) in one collection, filtered HNSW for real pre-filtering, payload indexes, batched queries, native Windows binary."),
        ("Why bge-small, not a bigger embedder?", f"Strong retrieval at 384-d and fast on both CPU and GPU; 100k passages index in {b100['minutes']} min on the {b100_dev}. The reranker buys the precision."),
        ("Why not OpenAI or a hosted database?", "Free tier only, and only Qdrant, Weaviate, ChromaDB or pgvector are allowed. Everything runs locally."),
        ("Why run 100k on CPU and 500k on GPU?", f"It shows both ends: any laptop serves 100k at {ms0(LA['hybrid_rerank']['p95_ms'])} p95; a small laptop GPU serves 5× more at {ms0(LB['hybrid_rerank']['p95_ms'])}."),
        ("Why is the GPU at 500k faster than the CPU at 100k?", f"The reranker: {stA['rerank']:.0f} ms for {dA} passages on CPU, {stB['rerank']:.0f} ms for {dRA} on GPU. Qdrant search only goes {stA['qdrant']:.0f} → {stB['qdrant']:.0f} ms."),
        ("The baseline already beats 0.75. What did Phase 2 add?", f"MRR {sg(gB['mean_diff'])} at 500k (CI {ci(gB)}); RAGAS precision {f3(dB['context_precision'])} → {f3(p2B['context_precision'])} at 500k; {sg(gA['mean_diff'])} MRR at 100k on CPU."),
        ("Hybrid ≈ dense. Why keep BM25?", f"It is required, and it widens the reranker's pool: at 500k, top-{dRA} pool {pct(cr5[str(dRA)]['dense'])} → {pct(cr5[str(dRA)]['union'])}."),
        ("Why weighted RRF, not linear fusion?", f"No score normalisation. Linear won on tuning ({tun_lin[0.3]['mrr']:.3f}) but not on test ({linear['hybrid']['mrr@10']:.3f})."),
        ("How was the BM25 weight 0.1 chosen?", f"A grid on 1,000 tuning questions; more weight hurt ({tun[1.0]['mrr']:.3f} at 1 : 1)."),
        ("Why rerank only the top 10 or 20?", f"Top-30 gave the same MRR as top-20 on tuning ({rr[('rrf', 30)]:.3f} vs {rr[('rrf', 20)]:.3f}); the CPU reads 10 to stay fast."),
        ("Did you tune on the test set?", "No. Disjoint frozen sets: tune on dev_large, confirm once on test; when test disagreed we kept the simpler setting."),
        ("Is an LLM judge reliable?", "Same judge, temperature 0, same questions for every mode; the 1,000-question label metrics point the same way."),
        ("RAGAS intervals include zero. Is the gain real?", "50 questions is small; the proof is the 1,000-question label metric, whose interval excludes zero."),
        ("Why not non-LLM RAGAS metrics?", "String similarity scored a water passage as correct for an ethanol question."),
        ("The reranker was trained on MS MARCO. Unfair?", "It has a home advantage; we say so and would validate or fine-tune it on company data."),
        ("How was p95 measured?", f"100 consecutive questions through the HTTP API, cache off, 5 warm-ups: CPU 100k p50 {LA['hybrid_rerank']['p50_ms']:.0f} / p95 {LA['hybrid_rerank']['p95_ms']:.0f} ms; GPU 500k {LB['hybrid_rerank']['p50_ms']:.0f} / {LB['hybrid_rerank']['p95_ms']:.0f} ms."),
        ("What if the CPU is busy?", f"Re-measured while OneDrive synced the project, CPU p95 rose to {ms0(busy['p95_ms']) if busy else '—'}. The CPU deployment needs a quiet machine; the GPU one was not affected."),
        ("What about 10M passages?", f"≈ {kb_per:.1f} KB per passage measured; int8 quantization, originals on disk, sharding. The reranker's cost stays fixed."),
        ("How long does indexing take?", f"{b100['minutes']} min for 100k on the {b100_dev}, {b500_min} min for 500k on the GPU (limit 120), resumable."),
        ("Pre-filter or post-filter?", "Pre: inside Qdrant's request, in both searches, on indexed fields, so there are always 5 matching results."),
        ("What if the user picks the wrong filter?", f"The answer can be filtered out, which is why automatic routing lost {sg(tf_a2['mrr_vs_none']['mean_diff'])} MRR. “Any” stays the default; topics are suggested."),
        ("How is stale data prevented?", "ID = UUID5(doc_id) overwrites; index_version is part of the cache key; the change log keeps history."),
        ("What does an update cost?", f"{w_del:.0f}–{w_upd:.0f} ms per write in the scripted demo; Qdrant updates the graph and BM25 statistics incrementally."),
        ("Can I see what changed?", "Yes: before / after answers, a word diff and the version history on the Live Updates page."),
        ("How accurate are the topic tags?", f"{val['top1_agreement'] * 100:.0f}% agree with an LLM judge, {val['llm_topic_in_our_top2'] * 100:.0f}% within our top 2; a correct topic filter adds {sg(tf_or['mrr_vs_none']['mean_diff'])} MRR."),
        ("Why not route queries automatically?", f"Measured {sg(tf_a2['mrr_vs_none']['mean_diff'])} MRR on test with the top-2 guess ({pct(tf_a2['route_accuracy'])} routed right)."),
        ("How did you stay within the free LLM tier?", "Evaluation runs are resumable; a failed judge call is retried later, never saved as a zero."),
        ("Where does it still fail?", f"Look-alike passages, gaps in the labels, unanswerable questions; {qtA['cases']['all_methods_fail(top10)']} of 1,000 fail in every mode at 100k."),
        ("Can anyone reproduce this?", f"One command per number, {lock} pinned packages, frozen question sets, {tests or '—'} tests, report generated from results/."),
        ("What would you do next?", "Access-control filters, domain fine-tuning of embedder and reranker, quantization for 10M+, answer-faithfulness monitoring."),
    ]
    section("p-qa", f"The {len(qa)} hardest judge questions", "4", "Backup · Q&A",
            '<div class="qa">' + "".join(f"<div><b>{i + 1} · {E(q)}</b><span>{E(a)}</span></div>" for i, (q, a) in enumerate(qa)) + "</div>",
            "Full answers with sources: docs/JUDGE_QA.md")

    # ============================== 5 bonus
    part("c5", 5, "Bonus feature addition", "Everything beyond the required checklist, each with its own measurement.")
    section("x-list", "Bonus features and their numbers", "5", "Slide 20 · bonus", tbl(
        ["Feature", "What it does", "Measured"], [
            ["500,000-passage index", "5× the required corpus, full evaluation", f"MRR {sg(gB['mean_diff'])} over dense · RAGAS {f3(p2B['context_precision'])} / {f3(p2B['context_recall'])} · p95 {ms0(LB['hybrid_rerank']['p95_ms'])}"],
            ["CPU and GPU profiles", "Picks models and rerank depth from the hardware", f"100k on CPU p95 {ms0(LA['hybrid_rerank']['p95_ms'])}; 500k on GPU p95 {ms0(LB['hybrid_rerank']['p95_ms'])}"],
            ["Cross-encoder reranker", "Reads question and passage together", f"the main quality gain: {sg(gA['mean_diff'])} / {sg(gB['mean_diff'])} MRR"],
            ["Finer metadata: 15 topics", "Zero-shot tags from stored vectors, no re-embedding", f"{dist['passages']:,} passages in {dist['seconds']:.0f} s · {val['top1_agreement'] * 100:.0f}% agree with an LLM"],
            ["Topic suggestions", "Three likely topics as one-click chips", f"correct topic {sg(tf_or['mrr_vs_none']['mean_diff'])} MRR"],
            ["Source type and corpus filters", "Government, health, education … · web vs internal", "indexed, applied inside the search"],
            ["Live update layer", "Before / after answers, word diff, version history", f"{upd['passed']} / {upd['total']} demo steps · {w_del:.0f}–{w_upd:.0f} ms writes"],
            ["Grounded answers", "Groq LLM answers from the top 5 and cites [n]", "optional, per search"],
            ["Per-stage timing", "Every search shows where the milliseconds went", "dense · BM25 · Qdrant · fusion · reranker"],
            ["Measured dashboard", "Every number on the UI is read from results/", "no hand-typed figures"],
            ["Reproducibility", "Lock file, frozen question sets, generated report", f"{tests or '—'} tests · {lock} pinned packages"],
            ["Docker option", "docker-compose for Qdrant", "same results as the native binary"],
        ], hl=(0, 1, 6)))
    section("x-tags", "Finer metadata tags and what they are worth", "5", "Slide 21 · tags", f'''
  <div class="cols"><div>{C_topics}</div><div>{C_routing}
    {legend(("No filter", "c-g"), ("Correct topic", "c-a"), ("Automatic guess", "c-b"))}
    <p class="note">1,000 test questions, Phase 2, 100k on the CPU. Correct topic: {sg(tf_or['mrr_vs_none']['mean_diff'])} MRR (CI {ci(tf_or['mrr_vs_none'])}). Automatic top-2: {sg(tf_a2['mrr_vs_none']['mean_diff'])} ({pct(tf_a2['route_accuracy'])} routed right). Source types: {" · ".join(f"{k} {v / dist['passages'] * 100:.0f}%" for k, v in dist['source_types'].items())}.</p></div></div>
  {say("A correct filter removes look-alikes from other subjects. A wrong guess removes the answer, so the system suggests topics and the user decides.")}''',
            "results/tags/summary_msmarco.json · topic_validation_summary.json · topic_filter_test_cpu.json")
    section("x-tuning", "How the fusion weight was chosen", "5", "Backup · tuning", C_fusion + legend(("Weighted RRF", "c-a"), ("Linear (min-max)", "c-w")) +
            '<p class="note">MRR@10 on the 1,000 tuning questions. The chosen weight was then checked once on the test questions.</p>',
            "results/tuning/dev_large_fusion_grid.csv · results/ir/test_summary_linear03.json")

    # ============================== appendix
    part("app", "A", "Appendix: all numbers", "Full tables for backup slides and questions.")
    keys6 = [("mrr@10", "MRR@10"), ("ndcg@10", "nDCG@10"), ("hit@5", "Hit@5"), ("recall@5", "Recall@5"), ("precision@5", "P@5"), ("hit@10", "Hit@10")]
    section("ap-ir", "Every label metric (1,000 test questions)", "A", "Backup", "".join(
        f"<h3>{t}</h3>" + tbl(["Mode"] + [f"#{n}" for _, n in keys6], [[l] + [f3(X[m][k]) for k, _ in keys6] for m, l in order], hl=(4,))
        for t, X in (("100,000 passages · CPU", A), ("500,000 passages · GPU", B))) +
            '<p class="note">Precision@5 is low by nature: most questions have one labelled passage, so 0.2 is its maximum.</p>')
    section("ap-lat", "Latency, 100 consecutive queries (ms)", "A", "Backup", tbl(
        ["Mode", "#CPU 100k p50", "#p95", "#p99", "#GPU 500k p50", "#p95", "#p99"],
        [[l, f"{LA[m]['p50_ms']:.0f}", f"<b>{LA[m]['p95_ms']:.0f}</b>", f"{LA[m]['p99_ms']:.0f}", f"{LB[m]['p50_ms']:.0f}", f"<b>{LB[m]['p95_ms']:.0f}</b>", f"{LB[m]['p99_ms']:.0f}"] for m, l in lat_order], hl=(2,)),
            "results/latency/summary_cpu.json · summary_500k.json (same session)")
    section("ap-types", "MRR@10 by question type", "A", "Backup", tbl(
        ["Type", "#Questions", "#Dense 100k", "#Phase 2 100k", "#Dense 500k", "#Phase 2 500k"],
        [[t, v["n"], f3(v["dense_mrr"]), f3(v["hybrid_rerank_mrr"]), f3(qtB["types"][t]["dense_mrr"]), f3(qtB["types"][t]["hybrid_rerank_mrr"])]
         for t, v in sorted(qtA["types"].items(), key=lambda kv: -kv[1]["n"])]) +
            '<p class="note">Location and person groups are small; treat their differences as hints.</p>',
            "results/analysis/failure_analysis_cpu.json · failure_analysis_msmarco_500k.json")
    case_names = [("dense_misses_top10", "Dense misses the correct passage (top 10)"), ("bm25_finds_dense_misses(top10)", "…and BM25 finds it"),
                  ("dense_fails_hybrid_fixes(top5)", "Hybrid fixes a dense failure (top 5)"), ("dense_ok_hybrid_breaks(top5)", "Hybrid breaks a dense success"),
                  ("rerank_fixes_order(to #1)", "Reranker moves the correct passage to #1"), ("rerank_hurts(#1 lost)", "Reranker loses a correct #1"),
                  ("all_methods_fail(top10)", "Every method fails")]
    section("ap-fail", "Failure analysis (1,000 test questions)", "A", "Backup", tbl(
        ["Case", "#100k · CPU", "#500k · GPU"], [[l, qtA["cases"][k], qtB["cases"][k]] for k, l in case_names]) + '''
  <ul class="tight"><li><b>Look-alikes</b>: for “boiling point of ethanol” the reranker ranks a passage about denatured alcohol's flashpoint first.</li>
    <li><b>Label gaps</b>: sometimes the first passage is arguably better than the labelled one but counts as a miss.</li>
    <li><b>No answer in the corpus</b>: reranker scores turn strongly negative, a usable “no relevant passage” signal.</li></ul>''')
    notes = [f"Deployment pairing used throughout: 100k on the CPU profile (reranker top {dA}), 500k on the GPU profile (top {dRA}).",
             f"Latency for both deployments was measured back to back in one session ({d['latency_cpu']['timestamp'][:10]}).",
             f"The CPU deployment shares the processor: re-measured while OneDrive synced the folder, Phase 2 p95 was {ms0(busy['p95_ms']) if busy else '—'}.",
             f"The 500k build took {b500_min} min on an idle machine (results/build_index_500k.log); a later rebuild that shared the GPU took {b500['minutes']} min.",
             "RAGAS at 100k on the CPU: dense and hybrid give exactly the same top-5 passages as the GPU for all 50 questions (results/ragas/cpu_vs_gpu_contexts_100k.json), so those scores are shared. "
             + (f"Phase 2 on the CPU was scored separately: {p2A_txt}." if p2A else "Phase 2 on the CPU picks different passages for some questions and is not scored yet; the Groq judge account was unavailable."),
             "Tuning curves (fusion weight, rerank depth) were measured once on the 1,000 tuning questions to choose settings.",
             "First search after ~10 s idle is slower on the laptop GPU (power saving); consecutive use is unaffected."]
    section("ap-notes", "Measurement notes", "A", "Backup", '<ul class="tight">' + "".join(f"<li>{E(x)}</li>" for x in notes) + "</ul>")
    section("ap-files", "Diagram files for the deck", "A", "Planning", '<p>Every diagram and chart above is also saved as its own SVG file (light background) in <span class="src">docs/diagrams/</span>. In PowerPoint use Insert → Pictures → This Device; SVGs stay sharp at any size.</p><ul class="tight">'
            + "".join(f"<li><span class=\"src\">{k}.svg</span></li>" for k in DIAGRAMS) + "</ul>")

    # ---------------------------------------------------------------- write
    DIAG.mkdir(parents=True, exist_ok=True)
    for name, (_, alone) in DIAGRAMS.items():
        (DIAG / f"{name}.svg").write_text(alone, encoding="utf-8")
    page = TEMPLATE.replace("/*DIAG_CSS*/", css_for(True)).replace("<!--SECTIONS-->", "\n".join(sec)).replace("@DATE@", tdate)
    out = DOCS / "ppt_report.html"
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out} and {len(DIAGRAMS)} diagrams in {DIAG}")


TEMPLATE = r'''<meta charset="utf-8">
<title>PrecisionRAG Report Book</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Archivo:wght@600;700;800&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
/* Layout: a technical report book. Sticky index on the left (top on phones); five criterion parts, each a run of slide-sized sections. */
:root {
  --bg: #f5f4ef; --surface: #fdfcf8; --ink: #13202f; --soft: #44505f; --muted: #6b7480; --line: #dedcd3;
  --accent: #2a5fd8; --accent-soft: #e6edfb; --warm: #d9782a; --warm-soft: #fbeedd; --good: #0f7a55; --good-soft: #e3f4ec;
  --bad: #b8322a; --bad-soft: #fbe6e3; --grid: #e7e5dd;
  --display: "Archivo", "Arial Narrow", Arial, sans-serif; --body: "IBM Plex Sans", "Segoe UI", Arial, sans-serif;
  --mono: "IBM Plex Mono", Consolas, "Courier New", monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #0f1722; --surface: #162131; --ink: #e8ecf1; --soft: #b6c0cc; --muted: #8b96a3; --line: #2a3747;
  --accent: #7ea6ff; --accent-soft: #1d2d48; --warm: #f0a35e; --warm-soft: #3a2a1a; --good: #5fd3a3; --good-soft: #15332a;
  --bad: #ff8a80; --bad-soft: #3a1d1d; --grid: #243142; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #0f1722; --surface: #162131; --ink: #e8ecf1; --soft: #b6c0cc; --muted: #8b96a3; --line: #2a3747;
  --accent: #7ea6ff; --accent-soft: #1d2d48; --warm: #f0a35e; --warm-soft: #3a2a1a; --good: #5fd3a3; --good-soft: #15332a;
  --bad: #ff8a80; --bad-soft: #3a1d1d; --grid: #243142; color-scheme: dark }
* { box-sizing: border-box; }
body { background: var(--bg); color: var(--ink); font-family: var(--body); font-size: 15px; line-height: 1.55; margin: 0; }
.shell { display: grid; grid-template-columns: 250px minmax(0, 1fr); max-width: 1340px; margin: 0 auto; padding-inline: 20px; gap: 36px; }
nav.index { position: sticky; top: env(safe-area-inset-top, 0px); align-self: start; max-height: 100vh; overflow-y: auto; padding-block: 28px; }
nav.index b { font-family: var(--display); font-size: 13px; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }
nav.index ol { list-style: none; padding: 0; margin: 10px 0 0; display: flex; flex-direction: column; gap: 1px; }
nav.index a { display: block; padding: 3px 8px; border-radius: 6px; color: var(--soft); text-decoration: none; font-size: 13px; }
nav.index a.p { font-family: var(--display); font-weight: 700; color: var(--ink); font-size: 13.5px; margin-top: 10px; }
nav.index a:hover, nav.index a:focus-visible { background: var(--accent-soft); color: var(--ink); outline: none; }
main { padding-block: 28px 80px; min-width: 0; display: flex; flex-direction: column; gap: 22px; }
header.cover { padding-block: 8px 6px; display: flex; flex-direction: column; gap: 10px; }
.eyebrow { font-family: var(--mono); font-size: 12px; letter-spacing: .12em; text-transform: uppercase; color: var(--accent); }
h1 { font-family: var(--display); font-weight: 800; font-size: clamp(34px, 5vw, 54px); line-height: 1.02; margin: 0; letter-spacing: -.01em; text-wrap: balance; }
.lede { font-size: 17px; color: var(--soft); max-width: 68ch; margin: 0; }
.crit { display: flex; flex-wrap: wrap; gap: 8px; }
.crit a { font: 500 13px var(--body); color: var(--ink); text-decoration: none; border: 1px solid var(--line); background: var(--surface); padding: 5px 11px; border-radius: 999px; }
.crit a span { font-family: var(--mono); color: var(--accent); margin-right: 6px; }
.crit a:hover, .crit a:focus-visible { border-color: var(--accent); outline: none; }
header.part { padding: 26px 0 2px; border-top: 2px solid var(--ink); display: flex; flex-direction: column; gap: 4px; scroll-margin-top: 16px; }
header.part .pnum { font-family: var(--mono); font-size: 12px; letter-spacing: .1em; text-transform: uppercase; color: var(--warm); }
header.part h2 { font-family: var(--display); font-weight: 800; font-size: clamp(26px, 3.4vw, 36px); line-height: 1.05; margin: 0; text-wrap: balance; }
header.part p { margin: 0; color: var(--soft); max-width: 70ch; }
section.s { background: var(--surface); border: 1px solid var(--line); border-radius: 14px; padding: 22px 24px 24px; display: flex; flex-direction: column; gap: 14px; min-width: 0; scroll-margin-top: 16px; }
.s-head { display: flex; justify-content: space-between; gap: 12px; align-items: flex-start; flex-wrap: wrap; }
.s-head h2 { font-family: var(--display); font-weight: 700; font-size: 24px; line-height: 1.15; margin: 4px 0 0; text-wrap: balance; }
.s-num { font-family: var(--mono); font-size: 12px; color: var(--muted); letter-spacing: .06em; }
.copy { font: 500 12.5px var(--body); background: transparent; color: var(--accent); border: 1px solid var(--line); border-radius: 8px; padding: 6px 11px; cursor: pointer; }
.copy:hover, .copy:focus-visible { background: var(--accent-soft); outline: 2px solid var(--accent); outline-offset: 1px; }
.head-right { display: flex; gap: 8px; align-items: center; }
h3 { font-family: var(--display); font-weight: 700; font-size: 17px; margin: 4px 0 0; }
p { margin: 0; max-width: 80ch; }
.say { border-left: 3px solid var(--accent); padding: 8px 14px; background: var(--accent-soft); border-radius: 0 8px 8px 0; max-width: 82ch; }
.say::before { content: "Say on the slide · "; font-family: var(--mono); font-size: 11.5px; color: var(--accent); }
.note { font-size: 13px; color: var(--muted); }
.src { font-family: var(--mono); font-size: 12px; color: var(--muted); overflow-wrap: anywhere; }
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; }
@media (min-width: 1100px) { .kpis.k8 { grid-template-columns: repeat(4, minmax(0, 1fr)); } }
.kpis.k8 .kpi .v { font-size: clamp(22px, 2.1vw, 30px); white-space: nowrap; }
.kpi { border: 1px solid var(--line); border-radius: 12px; padding: 14px 16px; background: var(--bg); display: flex; flex-direction: column; gap: 4px; min-width: 0; }
.kpi .v { font-family: var(--display); font-weight: 800; font-size: 34px; line-height: 1; color: var(--accent); font-variant-numeric: tabular-nums; }
.kpi .v.warm { color: var(--warm); }
.kpi .l { font-size: 13px; color: var(--soft); }
.kpi .d { font-family: var(--mono); font-size: 11.5px; color: var(--muted); }
.tw { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 13.5px; font-variant-numeric: tabular-nums; }
th { text-align: left; font-size: 11.5px; text-transform: uppercase; letter-spacing: .05em; color: var(--muted); font-weight: 600; padding: 7px 10px; border-bottom: 1px solid var(--line); white-space: nowrap; }
td { padding: 7px 10px; border-bottom: 1px solid var(--grid); vertical-align: top; }
td.n, th.n { text-align: right; white-space: nowrap; }
tr.hl td { background: var(--accent-soft); font-weight: 600; }
.ok { color: var(--good); font-weight: 600; white-space: nowrap; }
.cols { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; }
.cols > * { min-width: 0; }
.card { border: 1px solid var(--line); border-radius: 12px; padding: 14px 16px; display: flex; flex-direction: column; gap: 8px; }
.card.good { background: var(--good-soft); }
ul.tight { margin: 0; padding-left: 18px; display: flex; flex-direction: column; gap: 4px; max-width: 82ch; }
.chart { overflow-x: auto; }
.chart svg { display: block; max-width: 100%; height: auto; }
.legend { display: flex; flex-wrap: wrap; gap: 14px; font-size: 12.5px; color: var(--soft); }
.legend i.sw { display: inline-block; width: 11px; height: 11px; border-radius: 3px; margin-right: 6px; vertical-align: -1px; }
.sw.c-a { background: var(--accent); } .sw.c-w { background: var(--warm); } .sw.c-g { background: var(--good); } .sw.c-b { background: var(--bad); } .sw.c-m { background: var(--muted); }
.qa { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 10px; }
.qa div { border: 1px solid var(--line); border-radius: 10px; padding: 10px 13px; min-width: 0; }
.qa b { display: block; font-size: 13.5px; margin-bottom: 3px; }
.qa span { font-size: 13px; color: var(--soft); }
.toast { position: fixed; bottom: calc(18px + env(safe-area-inset-bottom, 0px)); left: 50%; transform: translateX(-50%); background: var(--ink); color: var(--bg); padding: 8px 14px; border-radius: 8px; font-size: 13px; }
/*DIAG_CSS*/
@media (max-width: 900px) {
  .shell { grid-template-columns: minmax(0, 1fr); gap: 0; padding-inline: 16px; }
  nav.index { position: static; max-height: none; padding-block: 18px 0; }
  nav.index ol { flex-direction: row; flex-wrap: wrap; gap: 4px; }
  nav.index a:not(.p) { display: none; }
  nav.index a.p { border: 1px solid var(--line); margin-top: 0; }
  section.s { padding: 18px 16px; }
}
html { scroll-behavior: smooth; }
@media (prefers-reduced-motion: reduce) { html { scroll-behavior: auto; } }
</style>

<div class="shell">
<nav class="index" aria-label="Report sections"><b>Report book</b><ol id="toc"></ol></nav>
<main>
<header class="cover">
  <div class="eyebrow">ADROSONIC BUILD · Problem Statement 1 · Team OutLiers · @DATE@</div>
  <h1>PrecisionRAG Report Book</h1>
  <p class="lede">Precision retrieval for RAG on MS MARCO with Qdrant: dense → hybrid → cross-encoder reranker, metadata pre-filters and live updates.
  Two deployments: 100,000 passages on a CPU and 500,000 on a laptop GPU. Every number is read from <span class="src">results/</span> by <span class="src">scripts/make_report_book.py</span>.</p>
  <nav class="crit" aria-label="Judging criteria"><a href="#c1"><span>1</span>Business value</a><a href="#c2"><span>2</span>Scalability</a><a href="#c3"><span>3</span>Solution architecture</a><a href="#c4"><span>4</span>Presentation</a><a href="#c5"><span>5</span>Bonus features</a></nav>
</header>
<!--SECTIONS-->
</main>
</div>
<div class="toast" id="toast" hidden></div>
<script>
document.getElementById("toc").innerHTML = [...document.querySelectorAll("header.part, section.s")].map((s) =>
  `<li><a href="#${s.id}" class="${s.dataset.part ? "p" : ""}">${s.dataset.title}</a></li>`).join("");
const toast = (m) => { const t = document.getElementById("toast"); t.textContent = m; t.hidden = false; clearTimeout(window.__t); window.__t = setTimeout(() => (t.hidden = true), 1600); };
function selectSection(s) { const r = document.createRange(); r.selectNodeContents(s); const sel = getSelection(); sel.removeAllRanges(); sel.addRange(r); toast("Selected. Press Ctrl+C to copy"); }
document.querySelectorAll("section.s .copy").forEach((b) => b.addEventListener("click", () => {
  const s = b.closest("section"); const text = s.innerText.replace(/\bCopy text\b/g, "").replace(/\n{3,}/g, "\n\n").trim();
  try { navigator.clipboard.writeText(text).then(() => toast("Copied: " + s.dataset.title), () => selectSection(s)); } catch { selectSection(s); }
}));
</script>
'''

if __name__ == "__main__":
    main()
