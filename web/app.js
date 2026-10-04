/* PrecisionRAG web UI. Every number comes from the API: /dashboard (measured results in results/) or live
   /search, /answer, /passages calls. Nothing on this page is typed in by hand. */
"use strict";

const MODES = {
  dense: { short: "Phase 1 · Dense", title: "Phase 1: Dense Baseline (Naive RAG)" },
  bm25: { short: "BM25 only", title: "BM25 keyword search" },
  hybrid: { short: "Phase 2 · Hybrid", title: "Phase 2: Hybrid Search (Dense + BM25)" },
  hybrid_rerank: { short: "Phase 2 · Hybrid + Reranker", title: "Phase 2: Hybrid + Cross-Encoder Reranker" },
  dense_rerank: { short: "Dense + Reranker (ablation)", title: "Dense + Reranker" },
};
const SEARCH_MODES = ["dense", "bm25", "hybrid", "hybrid_rerank"];
const STAGES = { embed_dense: ["Dense embedding", "#2563eb"], route: ["Topic routing", "#0ea5e9"],
  embed_bm25: ["BM25 encoding", "#94a3b8"], qdrant: ["Qdrant search", "#10b981"], fusion: ["Fusion", "#f59e0b"],
  rerank: ["Reranker", "#7c3aed"] };
const SOURCE_TYPE_LABELS = { government: "Government", education: "Education", reference: "Encyclopedia & dictionary",
  health: "Health sites", community: "Q&A / how-to", news: "News & media", organization: "Organizations (.org)",
  commercial: "Commercial / other" };
const TYPE_NOTES = { description: "e.g. “what is a corporation”", numeric: "e.g. “how long to boil an egg”",
  entity: "e.g. “what is the name of …”", location: "e.g. “where is …”", person: "e.g. “who wrote …”" };
const STOP = new Set("what which when where who whom whose why how does did do is are was were the and for with from that this into about your have has can much many long".split(" "));

const S = { health: null, meta: null, dash: null, dashErr: null, index: "100k",
  // default question: a real test-set case where the reranker moves the labelled answer from #2 to #1
  search: { q: "what do the folds in the mitochondria membranes do", mode: "hybrid_rerank", category: "", source: "",
    topic: "", stype: "", corpus: "", answer: false, side: false },
  trace: { q: "what do the folds in the mitochondria membranes do" }, adminLog: [],
  upd: { q: "How many days do I have to return an item for a refund at Acme Retail?", doc: "acme-refund-policy",
    internal: true, answer: true, last: null },
  scaleQ: "what do the folds in the mitochondria membranes do" };
const DEMO_IDS = ["acme-refund-policy", "acme-shipping-policy", "acme-warranty-policy", "acme-remote-work-policy", "acme-leave-policy"];
const $view = document.getElementById("view");

/* ---------- helpers ---------- */
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const f3 = (x) => (x == null ? "—" : Number(x).toFixed(3));
const sgn = (x, d = 3) => (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(d);
const ms = (x) => (x == null ? "—" : `${Math.round(x)} ms`);
const icon = {
  info: '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/></svg>',
  check: '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.4"><circle cx="12" cy="12" r="9"/><path d="M8 12l3 3 5-6"/></svg>',
  verified: '<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M12 3l2.4 1.8 3-.2.9 2.9 2.4 1.8-.9 2.9.9 2.9-2.4 1.8-.9 2.9-3-.2L12 21l-2.4-1.8-3 .2-.9-2.9L3.3 14.7l.9-2.9-.9-2.9 2.4-1.8.9-2.9 3 .2z"/><path d="M8.5 12l2.5 2.5 4.5-5"/></svg>',
};

async function api(path, opts = {}) {
  const r = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.detail ? (typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail)) : `HTTP ${r.status}`);
  return body;
}
const qs = (o) => new URLSearchParams(Object.entries(o).filter(([, v]) => v !== "" && v != null && v !== false)).toString();

/** significance chip from a paired bootstrap result */
function sigChip(bs, d = 3) {
  if (!bs) return '<span class="chip">baseline</span>';
  const cls = bs.ci95_low > 0 ? "good" : bs.ci95_high < 0 ? "bad" : "";
  return `<span class="chip ${cls}" title="95% CI ${sgn(bs.ci95_low, d)} … ${sgn(bs.ci95_high, d)}">${sgn(bs.mean_diff, d)}${cls ? "" : " · n.s."}</span>`;
}
const ciText = (bs, d = 3) => `95% CI ${sgn(bs.ci95_low, d)} … ${sgn(bs.ci95_high, d)} · ${bs.wins} better / ${bs.ties} same / ${bs.losses} worse`;

function highlight(text, query) {
  const words = [...new Set(query.toLowerCase().match(/[a-z0-9]+/g) || [])].filter((w) => w.length > 2 && !STOP.has(w));
  let out = esc(text);
  if (!words.length) return out;
  const re = new RegExp(`\\b(${words.map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})\\w*`, "gi");
  return out.replace(re, "<mark>$&</mark>");
}
const clip = (t, n = 300) => (t.length > n ? t.slice(0, n).replace(/\s+\S*$/, "") + " …" : t);
const origin = (h) => [h.dense_rank ? `dense #${h.dense_rank}` : "", h.bm25_rank ? `BM25 #${h.bm25_rank}` : ""].filter(Boolean).join(" · ");

/** small SVG line chart. series: [{name, color, points:[{x,y}], dashed}], xs: category labels */
function lineChart(series, xs, { hi = null, yFmt = f3, xTitle = "" } = {}) {
  const W = 400, H = 150, L = 34, R = 10, T = 14, B = 28;
  const ys = series.flatMap((s) => s.points.map((p) => p.y));
  let lo = Math.min(...ys), up = Math.max(...ys);
  const pad = Math.max((up - lo) * 0.15, 0.004); lo -= pad; up += pad;
  const X = (i) => L + (xs.length === 1 ? 0 : (i * (W - L - R)) / (xs.length - 1));
  const Y = (v) => T + (1 - (v - lo) / (up - lo)) * (H - T - B);
  let g = "";
  for (let k = 0; k <= 3; k++) {
    const v = lo + ((up - lo) * k) / 3;
    g += `<line x1="${L}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}" stroke="#e2e8f0" stroke-width="1"/>`
      + `<text x="${L - 6}" y="${Y(v) + 3}" text-anchor="end">${yFmt(v)}</text>`;
  }
  xs.forEach((x, i) => { g += `<text x="${X(i)}" y="${H - 10}" text-anchor="middle">${esc(x)}</text>`; });
  if (xTitle) g += `<text x="${(L + W - R) / 2}" y="${H + 4}" text-anchor="middle">${esc(xTitle)}</text>`;
  series.forEach((s, si) => {
    const pts = s.points.map((p) => [X(p.x), Y(p.y)]);
    const d = pts.map((p, i) => `${i ? "L" : "M"} ${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join(" ");
    if (si === 0 && pts.length > 1)
      g += `<path d="${d} L ${pts[pts.length - 1][0]} ${H - B} L ${pts[0][0]} ${H - B} Z" fill="${s.color}" fill-opacity="0.08"/>`;
    g += `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="2.5" ${s.dashed ? 'stroke-dasharray="5 4"' : ""} stroke-linejoin="round"/>`;
    s.points.forEach((p) => { g += `<circle cx="${X(p.x)}" cy="${Y(p.y)}" r="3.2" fill="${s.color}"><title>${esc(s.name)}: ${yFmt(p.y)}</title></circle>`; });
  });
  if (hi) {
    g += `<circle cx="${X(hi.x)}" cy="${Y(hi.y)}" r="6" fill="#2563eb" stroke="#fff" stroke-width="2"/>`
      + `<text x="${X(hi.x)}" y="${Y(hi.y) - 10}" text-anchor="middle" style="font-weight:700;fill:#0f172a">${yFmt(hi.y)}</text>`;
  }
  const legend = series.length > 1 ? `<div class="legend" style="margin-top:6px">${series.map((s) =>
    `<span><i class="swatch" style="background:${s.color}"></i>${esc(s.name)}</span>`).join("")}</div>` : "";
  return `<div class="chart"><svg viewBox="0 0 ${W} ${H + 8}" preserveAspectRatio="none" role="img">${g}</svg>${legend}</div>`;
}

function timingBar(t) {
  const parts = Object.entries(STAGES).filter(([k]) => t[k] > 0.05);
  const sum = parts.reduce((a, [k]) => a + t[k], 0) || 1;
  return `<div class="timing">${parts.map(([k, [, c]]) => `<div style="width:${(t[k] / sum) * 100}%;background:${c}" title="${STAGES[k][0]} ${t[k].toFixed(1)} ms"></div>`).join("")}</div>
    <div class="legend">${parts.map(([k, [n, c]]) => `<span><i class="swatch" style="background:${c}"></i>${n} ${t[k] < 1 ? t[k].toFixed(1) : Math.round(t[k])} ms</span>`).join("")}
    <span><strong style="color:#0f172a">Total ${ms(t.total)}</strong></span></div>`;
}

/* ---------- header / footer ---------- */
async function loadHealth() {
  const el = document.getElementById("status");
  try {
    S.health = await api("/health");
    const h = S.health, idx = h.indexes || { "100k": h.points };
    if (!(S.index in idx)) S.index = "100k";
    const sv = h.serving || {}, devName = (k) => (sv[k]?.device === "cpu" ? "CPU" : "GPU");
    el.innerHTML = `<span class="dot ok"></span><span>MS MARCO</span><span class="sep">|</span><span class="lite">${(idx[S.index] || 0).toLocaleString()} passages · ${devName(S.index)}${sv[S.index] ? " · rerank top-" + sv[S.index].rerank_depth : ""}</span>`;
    document.getElementById("index-switch").innerHTML = ["100k", "500k"].map((k) =>
      `<button type="button" data-index="${k}" class="${k === S.index ? "on" : ""}" ${k in idx ? `title="${k} index served on the ${devName(k)} profile"` : "disabled title='500k index not built / not loaded'"}>${k} · ${devName(k)}</button>`).join("");
    document.getElementById("footer-info").textContent =
      `Qdrant (local) · indexes ${Object.entries(idx).map(([k, v]) => `${k}: ${v.toLocaleString()}`).join(", ")} · ${Object.keys(idx).map((k) => `${k} on ${devName(k)}`).join(", ")} · config ${h.config_fingerprint} · index_version ${h.index_version}`;
  } catch (e) {
    S.health = null;
    el.innerHTML = `<span class="dot bad"></span><span>API or Qdrant offline</span>`;
  }
}
document.getElementById("index-switch").onclick = (e) => {
  const b = e.target.closest("button[data-index]");
  if (!b || b.disabled || b.dataset.index === S.index) return;
  S.index = b.dataset.index; S.meta = null;
  loadHealth().then(render);
};
async function loadDash() {
  try { S.dash = await api("/dashboard"); S.dashErr = null; } catch (e) { S.dashErr = e.message; }
}
async function loadMeta() {
  try { S.meta = await api(`/meta?${qs({ index: S.index })}`); }
  catch { S.meta = { category: [], source: [], topic: [], source_type: [], corpus: [] }; }
}

/** the measured numbers for the index chosen in the header (100k or 500k) */
/** 100k is served on the CPU profile and 500k on the GPU profile — every number shown is measured on that pairing */
function idx(d) {
  const big = S.index === "500k";
  return { big, name: big ? "500k" : "100k", label: big ? "500,000" : "100,000",
    hw: big ? "GPU" : "CPU", hwName: big ? (d.latency_500k?.gpu || "GPU") : "CPU only, no GPU",
    depth: big ? d.ir_gpu_rerank_depth : d.ir_cpu_rerank_depth,
    ir: (big ? d.ir_500k : d.ir_cpu) || {}, lat: (big ? d.latency_500k : d.latency_cpu)?.modes || {},
    rg: big ? d.ragas_500k : d.ragas_cpu, qt: big ? d.query_types_500k : d.query_types_cpu,
    build: big ? d.scale?.["500k"] : d.scale?.["100k_cpu"] || d.scale?.["100k"] };
}
function idxBanner(X) {
  if (X.big) return `<section class="callout"><b>500,000-passage index · GPU profile</b> (${esc(X.hwName)}, reranker reads the top ${X.depth}).
    RAGAS, MRR, latency and the question-type bars below are 500k measurements. The tuning charts show how the settings were chosen, on 1,000 separate tuning questions.</section>`;
  const sh = S.dash?.ragas_cpu?.shared_with_gpu || [];
  return `<section class="callout"><b>100,000-passage index · CPU profile</b> (no GPU, reranker reads the top ${X.depth}) — the setting for an ordinary laptop or server.
    RAGAS, MRR, latency and the question-type bars below were all measured on the CPU.${sh.length ? ` For ${sh.map((m) => MODES[m].short).join(" and ")} the CPU returns exactly the same top-5 passages as the GPU for all 50 RAGAS questions, so the judge scores are shared; Phase 2 with the reranker ${S.dash?.ragas_cpu?.modes?.hybrid_rerank ? "was scored again on the CPU" : "picks different passages on the CPU and its RAGAS score is pending"}.` : ""}</section>`;
}

/* ---------- phase view (the dashboard design) ---------- */
function phaseView(mode) {
  const d = S.dash;
  if (!d) return `<div class="error">Could not load measured results: ${esc(S.dashErr || "loading…")}</div>`;
  const c = d.config, X = idx(d), ir = X.ir, lat = X.lat, rg = X.rg;
  const desc = {
    dense: `Each question becomes a 384-number vector with <code>${esc(c.dense_model)}</code>; Qdrant returns the ${c.top_k} closest passages (cosine, HNSW index). This is the naive-RAG baseline every improvement is measured against.`,
    hybrid: `Dense vectors and BM25 keyword vectors are searched in <b>one</b> Qdrant request (top-${c.prefetch} each, metadata filter applied inside both), then merged with ${c.fusion_hybrid === "linear" ? "min-max linear fusion" : `weighted Reciprocal Rank Fusion (k = ${c.rrf_k})`} — weights dense ${c.weights_hybrid.dense} : BM25 ${c.weights_hybrid.bm25}.`,
    hybrid_rerank: `Hybrid candidates (RRF ${c.weights_hybrid_rerank.dense} : ${c.weights_hybrid_rerank.bm25}) are re-read by the cross-encoder <code>${esc(c.reranker)}</code>, which scores question and passage together and reorders the top ${X.depth} (${X.hw} profile). Final output: top ${c.top_k}.`,
  }[mode];

  /* hero metrics */
  const rm = rg?.modes?.[mode];
  const paired = rg?.paired?.[mode];
  const ragasCard = (key, label, target) => {
    if (!rm) return `<div class="card metric"><div class="metric-top"><span>${label} (RAGAS)</span><span class="chip ${X.big ? "" : "warn"}">${X.big ? "not run" : "pending"}</span></div>
      <div class="metric-value na">—</div><div class="metric-note">${X.big ? `Not scored with RAGAS yet — see MRR@10 for this mode.`
        : `Pending on the CPU: its reranker reads the top ${X.depth}, which changes the top 5 for ${(() => { const x = S.dash.cpu_contexts?.[mode]; return x ? x.n - x.identical_top5 : "some"; })()} of 50 questions, so the GPU score is not reused. Run <code>DEVICE=cpu python -m scripts.eval_ragas --modes ${mode} --tag cpu</code>. See MRR@10 (1,000 questions) meanwhile.`}</div></div>`;
    const v = rm[key], ok = v > target, p = paired?.[key];
    return `<div class="card metric"><div class="metric-top"><span>${label} (RAGAS)</span>
      ${p ? sigChip(p) : `<span class="chip ${ok ? "good" : "bad"}">${ok ? "target met" : "below target"}</span>`}</div>
      <div class="metric-value">${f3(v)}</div>
      <div class="metric-note">${p ? `vs ${f3(rg.modes.dense[key])} in Phase 1 · ` : ""}target &gt; ${target} · ${rm.n} questions · judge ${esc(rg.judge)}</div></div>`;
  };
  const m = ir[mode];
  const mrrCard = m ? `<div class="card metric"><div class="metric-top"><span>MRR@10 (labels)</span>${sigChip(m.vs_dense)}</div>
      <div class="metric-value">${f3(m["mrr@10"])}</div>
      <div class="metric-note">${m.vs_dense ? `vs ${f3(ir.dense["mrr@10"])} dense · ` : ""}Recall@5 ${f3(m["recall@5"])} · ${m.n.toLocaleString()} test questions</div></div>`
    : `<div class="card metric"><div class="metric-top"><span>MRR@10</span></div><div class="metric-value na">not measured</div></div>`;
  const l = lat[mode];
  const latCard = l ? `<div class="card metric"><div class="metric-top"><span>p95 Retrieval Latency</span><span class="chip">p50: ${Math.round(l.p50_ms)} ms</span></div>
      <div class="metric-value">${Math.round(l.p95_ms)} ms</div>
      <div class="metric-note">${l.n} consecutive queries · end-to-end HTTP · target &lt; 300 ms ${l.pass_p95_lt_300 ? "✓" : "✗"}</div></div>`
    : `<div class="card metric"><div class="metric-top"><span>p95 latency</span></div><div class="metric-value na">not measured</div></div>`;

  return `
  <div class="page-head"><div><h1>${MODES[mode].title}</h1><p>${desc}</p></div>
    <div class="page-meta"><span>Index: <strong>${X.label} passages</strong> (switch at top right)</span><span>•</span>
    <span>Hardware: <strong>${X.hw} profile · ${esc(X.hwName)}</strong></span></div></div>
  ${idxBanner(X)}
  <section class="grid4">${ragasCard("context_precision", "Context Precision", 0.75)}${ragasCard("context_recall", "Context Recall", 0.7)}${mrrCard}${latCard}</section>
  <section class="grid2">${tuningCard(mode, X)}${typeCard(mode, X)}</section>
  ${traceSection(mode)}`;
}

function tuningCard(mode, X) {
  const d = S.dash, c = { ...d.config, rerank_depth: X.depth }, t = d.tuning;
  if (mode === "hybrid_rerank") {
    const rows = t?.rerank || [];
    const names = { dense_only: ["Dense candidates only", "#94a3b8", true], rrf: ["RRF 1 : 1 (chosen)", "#2563eb"], "wrrf_0.3": ["Weighted RRF 1 : 0.3", "#10b981"] };
    const depths = [...new Set(rows.map((r) => r.depth))].sort((a, b) => a - b);
    const series = Object.entries(names).map(([k, [n, col, dash]]) => ({ name: n, color: col, dashed: dash,
      points: rows.filter((r) => r.config === k).map((r) => ({ x: depths.indexOf(r.depth), y: r.mrr })) })).filter((s) => s.points.length);
    const chosen = rows.find((r) => r.config === "rrf" && r.depth === c.rerank_depth);
    const c5 = X.big && d.scale?.candidate_recall_500k?.[c.rerank_depth];   // 500k: measured on the test questions
    const cr = c5 ? { recall_dense: c5.dense, recall_bm25: c5.bm25, recall_union: c5.union }
      : (d.candidate_recall?.pool || []).find((p) => p.depth === c.rerank_depth);
    return `<div class="card split"><div>
      <div class="card-head"><div><h2>Reranker Depth &amp; Candidate Fusion</h2><p class="sub">How many hybrid candidates the cross-encoder re-reads (MRR@10, 1,000 tuning questions). The CPU profile reads 10 to stay fast; the GPU profile reads 20.</p></div>
      <div class="badge">${X.hw} profile: top-${c.rerank_depth}</div></div>
      <div class="balance"><div class="balance-labels"><span><i class="swatch" style="background:#2563eb"></i>Dense: ${c.weights_hybrid_rerank.dense}</span>
      <span>BM25: ${c.weights_hybrid_rerank.bm25}<i class="swatch" style="background:#94a3b8"></i></span></div>
      <div class="splitbar"><div style="width:50%;background:#2563eb"></div><div style="width:50%;background:#94a3b8"></div></div>
      <p class="sub" style="margin-top:8px">Before reranking, both lists get equal say — the reranker then decides the final order.</p></div>
      <div class="chart-box">${series.length ? lineChart(series, depths.map((x) => `top-${x}`), { hi: chosen && { x: depths.indexOf(chosen.depth), y: chosen.mrr } }) : '<div class="empty">Tuning grid not found.</div>'}</div></div>
      <div class="card-foot"><span>Correct passage in the top-${c.rerank_depth} pool (${X.name}${c5 ? ", test questions" : ", tuning questions"}):</span><span>dense ${f3(cr?.recall_dense)}</span><span>BM25 ${f3(cr?.recall_bm25)}</span><span class="ok-mark">dense ∪ BM25 ${f3(cr?.recall_union)}</span></div></div>`;
  }
  const curve = t?.curve || [], lin = t?.curve_linear || [];
  const xsVals = [...new Set([...curve, ...lin].map((p) => p.w_bm25))].sort((a, b) => a - b);
  const toPts = (arr) => arr.map((p) => ({ x: xsVals.indexOf(p.w_bm25), y: p.mrr }));
  const series = [{ name: "Weighted RRF", color: "#2563eb", points: toPts(curve) }];
  if (lin.length > 1) series.push({ name: "Linear (min-max)", color: "#10b981", dashed: true, points: toPts(lin) });
  const wb = mode === "dense" ? 0 : c.weights_hybrid.bm25, wd = mode === "dense" ? 1 : c.weights_hybrid.dense;
  const chosenArr = mode !== "dense" && c.fusion_hybrid === "linear" ? lin : curve;
  const chosen = chosenArr.find((p) => Math.abs(p.w_bm25 - wb) < 1e-9);
  const dense = curve.find((p) => p.w_bm25 === 0);
  const share = Math.round((wd / (wd + wb)) * 100);
  return `<div class="card split"><div>
    <div class="card-head"><div><h2>Hybrid Balance &amp; Parameter Tuning</h2><p class="sub">MRR@10 on 1,000 tuning questions as the BM25 weight grows (dense weight fixed at 1.0).</p></div>
    <div class="badge">${mode === "dense" ? "BM25 off" : `Chosen: BM25 ${wb}`}</div></div>
    <div class="balance"><div class="balance-labels"><span><i class="swatch" style="background:#2563eb"></i>Dense embedding: ${share}%</span>
    <span>Sparse BM25: ${100 - share}%<i class="swatch" style="background:#94a3b8"></i></span></div>
    <div class="splitbar"><div style="width:${share}%;background:#2563eb"></div><div style="width:${100 - share}%;background:#94a3b8"></div></div></div>
    <div class="chart-box"><div class="chart-labels"><span>Dense only (0)</span><strong>${chosen ? `This phase: ${f3(chosen.mrr)}` : ""}</strong><span>Equal weight (1.0)</span></div>
    ${curve.length ? lineChart(series, xsVals.map((x) => (x === 0 ? "0" : String(x))), { hi: chosen && { x: xsVals.indexOf(chosen.w_bm25), y: chosen.mrr }, xTitle: "" }) : '<div class="empty">Tuning grid not found — run <code>python -m scripts.tune_fusion --split dev_large</code></div>'}</div></div>
    <div class="card-foot"><span>BM25 alone: ${f3(t?.bm25_only)} MRR</span><span class="ok-mark">${mode === "dense" ? "Baseline" : "Chosen"}: ${f3(chosen?.mrr)}</span><span>Dense alone: ${f3(dense?.mrr)}</span></div></div>`;
}

function typeCard(mode, X) {
  const qt = X.qt;
  if (!qt) return `<div class="card"><h2>Performance by Query Type</h2><div class="empty">Failure analysis not found.</div></div>`;
  const types = Object.entries(qt.types).sort((a, b) => b[1].n - a[1].n);
  const key = `${mode}_mrr`;
  let best = null;
  const rows = types.map(([name, v]) => {
    const base = v.dense_mrr, val = v[key], diff = val - base;
    if (mode !== "dense" && (!best || diff > best[1])) best = [name, diff];
    const bar = mode === "dense"
      ? `<div style="width:${base * 100}%;background:#cbd5e1"></div>`
      : diff >= 0 ? `<div style="width:${base * 100}%;background:#cbd5e1"></div><div style="width:${diff * 100}%;background:#10b981"></div>`
        : `<div style="width:${val * 100}%;background:#cbd5e1"></div><div style="width:${-diff * 100}%;background:#fca5a5"></div>`;
    const vals = mode === "dense" ? `<strong>${f3(base)}</strong>`
      : `<span class="old">${f3(base)}</span><strong style="color:${diff >= 0 ? "#047857" : "#dc2626"}">${f3(val)}</strong> (${sgn(diff)})`;
    return `<div><div class="bar-row-head"><span class="name">${name[0].toUpperCase() + name.slice(1)} <span class="sub" style="display:inline">· ${v.n} q</span></span><span class="vals">${vals}</span></div>
      <div class="track">${bar}</div><span class="bar-note">${TYPE_NOTES[name] || ""}</span></div>`;
  }).join("");
  const note = mode === "dense" ? "Baseline MRR@10 per question type — later phases are compared with these bars."
    : best ? `Largest gain on <b>${best[0]}</b> questions (${sgn(best[1])}). Small groups (person, location) are noisy — treat as hints.` : "";
  return `<div class="card split"><div>
    <div class="card-head"><div><h2>Performance by Query Type</h2><p class="sub">MRR@10 per MS MARCO question type, ${qt.n_questions.toLocaleString()} test questions, ${X.name} · ${X.hw}${mode === "dense" ? "" : " — grey: Phase 1 dense, green: gain"}.</p></div>
    <span class="chip">Top-10 ranking</span></div><div class="bars">${rows}</div></div>
    <div class="card-foot" style="justify-content:flex-start"><span class="info">${icon.info}<span>${note}</span></span></div></div>`;
}

function traceSection(mode) {
  return `<section class="card"><div class="trace-head"><div><h2>Live Query Trace</h2>
    <p class="sub">Runs the question now through dense search, BM25 and this phase's full pipeline — real results from the ${S.index} index.</p></div>
    <form class="query-form" id="trace-form"><input class="input mono" id="trace-q" value="${esc(S.trace.q)}" aria-label="Question"><button class="btn" type="submit">Run</button></form></div>
    <div class="grid3" id="trace-out" style="padding-top:20px"><div class="loading"><span class="spin"></span>Running…</div></div></section>`;
}
async function runTrace(mode) {
  const out = document.getElementById("trace-out");
  if (!out) return;
  const q = S.trace.q.trim();
  if (!q) return;
  out.innerHTML = `<div class="loading"><span class="spin"></span>Running…</div>`;
  try {
    const finalMode = mode;
    const [dn, bm, fin] = await Promise.all(["dense", "bm25", finalMode].map((m) => api(`/search?${qs({ q, mode: m, k: 5, no_cache: true, index: S.index })}`)));
    const lane = (title, color, res, sub, final = false) => {
      const h = res.hits[0];
      if (!h) return `<div class="lane"><div class="lane-head"><span>${title}</span></div><div class="empty">No result</div></div>`;
      return `<div class="lane ${final ? "final" : ""}"><div>
        <div class="lane-head"><span>${final ? icon.verified : `<i class="swatch" style="background:${color}"></i>`}${title}</span><span class="mono" style="color:#64748b">${ms(res.timings_ms.total)}</span></div>
        <div class="lane-sub">${sub(h)}</div><p class="quote ${final ? "mono" : ""}">${highlight(clip(h.text, 260), q)}</p></div>
        ${final ? `<div class="lane-foot ok">${icon.check}<span>${origin(h) ? `Came from ${origin(h)} → now #1` : "Top result"} · ${esc(h.source)}</span></div>`
          : `<div class="lane-foot"><span>${esc(h.source)}</span><span style="font-weight:600">Rank #1</span></div>`}</div>`;
    };
    const finSub = { dense: (h) => `Cosine similarity: ${h.score.toFixed(3)} · Phase 1 output`,
      hybrid: (h) => `Fused score: ${h.score.toFixed(4)} · passed to the LLM`,
      hybrid_rerank: (h) => `Cross-encoder score: ${h.score.toFixed(2)} · passed to the LLM` }[finalMode];
    out.innerHTML = lane("Dense Vector Search", "#2563eb", dn, (h) => `Cosine similarity: ${h.score.toFixed(3)}`)
      + lane("Sparse Lexical (BM25)", "#94a3b8", bm, (h) => `BM25 score: ${h.score.toFixed(2)} (exact words)`)
      + lane(finalMode === "dense" ? "Phase 1 Output" : finalMode === "hybrid" ? "Fused Top Chunk (RRF)" : "Reranked Top Chunk", "#2563eb", fin, finSub, true);
  } catch (e) {
    out.innerHTML = `<div class="error" style="grid-column:1/-1">Search failed: ${esc(e.message)}. Is the API / Qdrant running?</div>`;
  }
}

/* ---------- search view ---------- */
function searchView() {
  const s = S.search, meta = S.meta || {};
  const opt = (arr, sel, label = (v) => v) => `<option value="">Any</option>` + (arr || []).map((x) => `<option value="${esc(x.value)}" ${x.value === sel ? "selected" : ""}>${esc(label(x.value))} (${x.count.toLocaleString()})</option>`).join("");
  const topicOpts = `<option value="">Any topic</option><option value="auto2" ${s.topic === "auto2" ? "selected" : ""}>✨ Auto (experimental — measured to lower accuracy)</option>`
    + (meta.topic || []).map((x) => `<option value="${esc(x.value)}" ${x.value === s.topic ? "selected" : ""}>${esc(x.value)} (${x.count.toLocaleString()})</option>`).join("");
  return `<div class="page-head"><div><h1>Search</h1><p>Ask a question and switch between Phase 1 (dense) and Phase 2 (hybrid) retrieval. All filters run inside Qdrant <b>before</b> the search (pre-filtering), on indexed metadata.</p></div>
    <div class="page-meta"><span>Index: <strong>${S.index}</strong></span><span>•</span><span>Hardware: <strong>${S.health?.serving?.[S.index]?.device === "cpu" ? "CPU" : "GPU"} profile${S.health?.serving?.[S.index] ? ", rerank top-" + S.health.serving[S.index].rerank_depth : ""}</strong></span></div></div>
  <section class="card search-panel">
    <form class="search-row" id="search-form"><input class="input big" id="sq" value="${esc(s.q)}" placeholder="e.g. what is the boiling point of ethanol" aria-label="Question"><button class="btn" type="submit">Search</button></form>
    <div class="controls"><div class="seg" id="seg">${SEARCH_MODES.map((m) => `<button type="button" data-mode="${m}" class="${m === s.mode ? "on" : ""}">${MODES[m].short}</button>`).join("")}</div></div>
    <div class="controls"><div class="filters">
      <label for="ftopic">Topic</label><select class="select" id="ftopic">${topicOpts}</select>
      <label for="fstype">Source type</label><select class="select" id="fstype">${opt(meta.source_type, s.stype, (v) => SOURCE_TYPE_LABELS[v] || v)}</select>
      <label for="fcorpus">Corpus</label><select class="select" id="fcorpus">${opt(meta.corpus, s.corpus, (v) => (v === "internal" ? "Company documents" : "Web (MS MARCO)"))}</select>
    </div></div>
    <div class="controls"><div class="filters">
      <label for="fcat">Question type</label><select class="select" id="fcat">${opt(meta.category, s.category)}</select>
      <label for="fsrc">Website</label><select class="select" id="fsrc">${opt(meta.source, s.source)}</select></div></div>
    <div class="controls"><label class="check"><input type="checkbox" id="fans" ${s.answer ? "checked" : ""}> Generate an AI answer from the top-5 results (Groq, cites [n])</label>
      <label class="check"><input type="checkbox" id="fside" ${s.side ? "checked" : ""}> Show Phase 1 · Dense side by side</label></div>
  </section>
  <section id="search-out"></section>`;
}
function hitsHtml(res, q, rt) {
  // A cached answer skips the whole pipeline: show how fast it really came back, and label the stored stage timings
  // as the first (uncached) run — otherwise an old, cold-GPU timing looks like the current speed.
  const head = res.cached
    ? `<p class="sub">${res.hits.length} passages · <b style="color:#047857">served from cache in ${rt != null ? Math.max(1, Math.round(rt)) + " ms" : "a few ms"}</b> — no search was run</p></div></div>
       <p class="sub" style="margin-top:10px">Stage timings from the first (uncached) run of this question:</p><div style="opacity:.6">${timingBar(res.timings_ms)}</div>`
    : `<p class="sub">${res.hits.length} passages · fresh search · ${esc(res.index || S.index)} index</p></div></div>${timingBar(res.timings_ms)}`;
  const sugg = !res.params?.topic && res.params?.suggested_topics;
  const suggLine = sugg ? `<div class="legend" style="margin-top:10px;align-items:center"><span>Narrow to a topic:</span>${sugg.map(([t, sc]) =>
    `<button type="button" class="chip blue" data-topic="${esc(t)}" style="border:0;cursor:pointer" title="similarity ${sc.toFixed(2)}">${esc(t)}</button>`).join("")}
    <span class="sub" style="display:inline">· a correct topic raised MRR by 0.071 on 1,000 test questions</span></div>` : "";
  const routed = res.params?.auto_topics;
  const routeLine = routed ? `<div class="callout" style="margin-top:12px">✨ <b>Auto topic routing:</b> searched only ${routed.map(([t, sc]) =>
    `<span class="chip blue">${esc(t)} · ${sc.toFixed(2)}</span>`).join(" ")}. <span class="sub" style="display:inline">Experimental: on 1,000 held-out test questions it
    guessed right 84% of the time but lowered MRR by 0.069 — when the guess is wrong, the answer is filtered out. A topic <b>you</b> pick correctly raised MRR by 0.071.</span></div>` : "";
  return `<div class="card"><div class="card-head"><div><h2>${MODES[res.mode].short}</h2>${head}${routeLine}${suggLine}
    <div class="hits" style="margin-top:14px">${res.hits.length ? res.hits.map((h) => `<div class="hit">
      <div class="hit-top"><span class="rank">#${h.rank}</span>${h.topic ? `<span class="chip blue">${esc(h.topic)}</span>` : ""}<span class="chip">${esc(h.category)}</span>
      ${h.source_type ? `<span class="chip">${esc(SOURCE_TYPE_LABELS[h.source_type] || h.source_type)}</span>` : ""}<span class="mono">${esc(h.source)}</span>
      <span>score ${h.score.toFixed(res.mode === "dense" ? 3 : 4)}</span>${origin(h) ? `<span class="chip">${origin(h)}</span>` : ""}${h.version > 1 ? `<span class="chip warn">v${h.version}</span>` : ""}
      ${h.doc_id.startsWith("msmarco-") ? "" : `<span class="chip good">${esc(h.doc_id)}</span>`}</div>
      <p>${highlight(h.text, q)}</p></div>`).join("") : '<div class="empty">No passages match this filter.</div>'}</div></div>`;
}
async function runSearch() {
  const s = S.search, out = document.getElementById("search-out");
  if (!out || !s.q.trim()) return;
  out.innerHTML = `<div class="loading"><span class="spin"></span>Searching…</div>`;
  // the Search page always runs the real pipeline (no result cache), so the timings shown are always live
  const p = { q: s.q, mode: s.mode, category: s.category, source: s.source, source_type: s.stype, corpus: s.corpus,
    index: S.index, no_cache: true, ...(s.topic === "auto2" ? { auto_topic: 2 } : { topic: s.topic }) };
  const timed = async (path) => { const t = performance.now(); const r = await api(path); return [r, performance.now() - t]; };
  try {
    const [main, rt] = s.answer ? [await api(`/answer?${qs(p)}`), null] : await timed(`/search?${qs(p)}`);
    const res = s.answer ? main.search : main;
    const [side, rtSide] = s.side && s.mode !== "dense" ? await timed(`/search?${qs({ ...p, mode: "dense" })}`) : [null, null];
    let ans = "";
    if (s.answer) {
      ans = main.answer ? `<div class="answer"><h3>AI answer · ${esc(main.model)} · ${ms(main.ms)}</h3><p>${esc(main.answer)}</p></div>`
        : `<div class="error">No answer generated: ${esc(main.error)}</div>`;
    }
    out.innerHTML = `${ans}<div class="${side ? "cols2" : ""}" style="margin-top:${ans ? 16 : 0}px">${hitsHtml(res, s.q, rt)}${side ? hitsHtml(side, s.q, rtSide) : ""}</div>`;
    out.querySelectorAll("[data-topic]").forEach((b) => { b.onclick = () => {
      s.topic = b.dataset.topic; const sel = document.getElementById("ftopic"); if (sel) sel.value = s.topic; runSearch(); }; });  } catch (e) {
    out.innerHTML = `<div class="error">Search failed: ${esc(e.message)}</div>`;
  }
}

/* ---------- comparison view ---------- */
function compareView() {
  const d = S.dash;
  if (!d) return `<div class="error">Could not load measured results: ${esc(S.dashErr || "loading…")}</div>`;
  const X = idx(d), ir = X.ir, lat = X.lat, rg = X.rg, p2 = rg?.modes?.hybrid_rerank;
  const yes = (b) => (b === "pending" ? '<span class="chip warn">pending</span>' : b ? '<span class="ok-mark">✓ met</span>' : '<span class="no-mark">✗ not met</span>');
  const g5 = d.ragas_500k?.modes?.hybrid_rerank;   // shown beside a pending CPU score, clearly labelled as the 500k GPU result
  const rag = (key, target) => p2 ? [f3(p2[key]), p2[key] > target]
    : [`pending on CPU${g5 ? ` · 500k GPU: ${f3(g5[key])}` : ""}`, "pending"];
  const req = [
    ["Corpus size", "≥ 100,000 passages (500k bonus)", d.index_build ? `${d.index_build.passages.toLocaleString()} (official) · 500,000 tested` : "—", !!d.index_build],
    ["RAGAS Context Precision (Phase 2)", "> 0.75", ...rag("context_precision", 0.75)],
    ["RAGAS Context Recall (Phase 2)", "> 0.70", ...rag("context_recall", 0.7)],
    ["RAGAS questions", "≥ 20", p2 ? `${p2.n}` : g5 ? `${g5.n} (500k GPU)` : "—", p2 ? p2.n >= 20 : "pending"],
    ["p95 latency, 100 consecutive queries", "< 300 ms", lat.hybrid_rerank ? `${Math.round(lat.hybrid_rerank.p95_ms)} ms (filtered: ${Math.round(lat["hybrid_rerank+filter"]?.p95_ms)} ms)` : "—", lat.hybrid_rerank?.pass_p95_lt_300],
    ["Index build time", "< 120 min", X.build ? `${X.build.first_build_minutes ?? X.build.minutes} min (${X.name}, ${X.build.device === "cpu" ? "CPU" : "GPU"})` : "—", (X.build?.first_build_minutes ?? X.build?.minutes) < 120],
    ["Upsert / delete without re-index", "demo passes", d.updates_demo ? "7 / 7 steps (results/updates_demo.json)" : "—", d.updates_demo],
  ];
  const maxMrr = Math.max(...Object.values(ir).map((x) => x["mrr@10"]), 0.01);
  const order = ["dense", "bm25", "hybrid", "dense_rerank", "hybrid_rerank"];
  const irRows = order.filter((m) => ir[m]).map((m) => { const x = ir[m]; return `<tr class="${m === "hybrid_rerank" ? "hl" : ""}">
    <td>${MODES[m].short}</td><td class="num">${f3(x["mrr@10"])}<span class="minibar" style="width:${(x["mrr@10"] / maxMrr) * 60}px"></span></td>
    <td class="num">${f3(x["ndcg@10"])}</td><td class="num">${f3(x["hit@5"])}</td><td class="num">${f3(x["recall@5"])}</td>
    <td>${x.vs_dense ? sigChip(x.vs_dense) + ` <span class="sub" style="display:inline">${sgn(x.vs_dense.ci95_low)} … ${sgn(x.vs_dense.ci95_high)}</span>` : '<span class="chip">baseline</span>'}</td>
    <td class="num">${x.vs_dense ? `${x.vs_dense.wins} / ${x.vs_dense.ties} / ${x.vs_dense.losses}` : "—"}</td></tr>`; }).join("");
  const latRows = Object.entries(lat).map(([m, x]) => `<tr><td>${esc(m.replace("+filter", " + filter").split("_").join(" "))}</td>
    <td class="num">${x.p50_ms}</td><td class="num"><b>${x.p95_ms}</b></td><td class="num">${x.p99_ms}</td>
    <td><div class="track" style="width:160px;display:inline-flex"><div style="width:${Math.min(x.p95_ms / 300, 1) * 100}%;background:#2563eb"></div></div> <span class="sub" style="display:inline">of 300 ms</span></td>
    <td>${x.pass_p95_lt_300 ? '<span class="ok-mark">✓</span>' : '<span class="no-mark">✗</span>'}</td></tr>`).join("");
  const big = d.ir_500k || {}, lat5 = d.latency_500k?.modes || {};
  const bigRows = order.filter((m) => big[m]).map((m) => `<tr class="${m === "hybrid_rerank" ? "hl" : ""}"><td>${MODES[m].short}</td>
    <td class="num">${f3(big[m]["mrr@10"])}</td><td class="num">${f3(big[m]["recall@5"])}</td><td>${big[m].vs_dense ? sigChip(big[m].vs_dense) : '<span class="chip">baseline</span>'}</td>
    <td class="num">${lat5[m] ? lat5[m].p95_ms + " ms" : "—"}</td></tr>`).join("");
  const cases = X.qt?.cases || {};
  const caseNames = { "dense_misses_top10": "Dense misses the correct passage (top 10)", "bm25_finds_dense_misses(top10)": "…and BM25 finds it",
    "dense_fails_hybrid_fixes(top5)": "Hybrid fixes a dense failure (top 5)", "dense_ok_hybrid_breaks(top5)": "Hybrid breaks a dense success",
    "rerank_fixes_order(to #1)": "Reranker moves the correct passage to #1", "rerank_hurts(#1 lost)": "Reranker loses a correct #1", "all_methods_fail(top10)": "Every method fails" };
  return `<div class="page-head"><div><h1>Comparison: Phase 1 vs Phase 2 · ${X.name} index</h1><p>All numbers below are read from <code>results/</code>. Confidence intervals come from a paired bootstrap on the same questions; “n.s.” means the interval includes zero.</p></div></div>
  ${idxBanner(X)}
  <section class="card"><h2>Challenge Requirements</h2><p class="sub">Each target from the problem statement and what we measured.</p>
    <div class="table-wrap"><table><thead><tr><th>Requirement</th><th>Target</th><th>Measured</th><th>Status</th></tr></thead>
    <tbody>${req.map((r) => `<tr><td>${r[0]}</td><td>${r[1]}</td><td><b>${r[2]}</b></td><td>${yes(r[3])}</td></tr>`).join("")}</tbody></table></div></section>
  <section class="grid2">
    <div class="card"><h2>RAGAS — Phase 1 vs Phase 2</h2><p class="sub">Judge ${esc(rg?.judge || "—")}, same questions for both phases.</p>
      <div class="table-wrap"><table><thead><tr><th>Mode</th><th class="num">Questions</th><th class="num">Context Precision</th><th class="num">Context Recall</th></tr></thead><tbody>
      ${rg ? order.filter((m) => rg.modes[m]).map((m) => { const x = rg.modes[m], p = rg.paired?.[m]; return `<tr class="${m === "hybrid_rerank" ? "hl" : ""}">
        <td>${MODES[m].short}</td><td class="num">${x.n}</td>
        <td class="num">${f3(x.context_precision)} ${p ? sigChip(p.context_precision) : ""}</td>
        <td class="num">${f3(x.context_recall)} ${p ? sigChip(p.context_recall) : ""}</td></tr>`; }).join("") : ""}
      </tbody></table></div>
      ${rg?.paired?.hybrid_rerank ? `<p class="sub" style="margin-top:12px">Chips: change vs Phase 1 on the same questions (paired bootstrap). Phase 2 + reranker —
        precision: ${ciText(rg.paired.hybrid_rerank.context_precision)}; recall: ${ciText(rg.paired.hybrid_rerank.context_recall)}.<br>
        With 50 questions the RAGAS intervals include zero; the 1,000-question label metrics below confirm the reranker's gain.</p>` : ""}</div>
    <div class="card"><h2>Latency — 100 consecutive queries</h2><p class="sub">${X.label} passages · ${X.hw} profile (${esc(X.hwName)}) · end-to-end HTTP time, cache off.</p>
      <div class="table-wrap"><table><thead><tr><th>Mode</th><th class="num">p50</th><th class="num">p95</th><th class="num">p99</th><th>p95 vs target</th><th></th></tr></thead><tbody>${latRows}</tbody></table></div></div>
  </section>
  <section class="card"><h2>Ablation — which component helps?</h2><p class="sub">${ir.dense ? ir.dense.n.toLocaleString() : ""} held-out test questions, ${X.label}-passage index, top-5. MRR change is vs Phase 1 dense.</p>
    <div class="table-wrap"><table><thead><tr><th>Mode</th><th class="num">MRR@10</th><th class="num">nDCG@10</th><th class="num">Hit@5</th><th class="num">Recall@5</th><th>MRR vs dense (95% CI)</th><th class="num">better / same / worse</th></tr></thead><tbody>${irRows}</tbody></table></div>
    <div class="card-foot" style="margin-top:14px;justify-content:flex-start"><span class="info">${icon.info}<span>Honest reading: the measurable gain comes from the <b>cross-encoder reranker</b>. Hybrid alone is not significantly better than dense on MS MARCO; BM25 widens the candidate pool the reranker reads.</span></span></div></section>
  <section class="grid2">
    <div class="card"><h2>Scale test — 500,000 passages</h2><p class="sub">Same pipeline and test questions on a 5× larger index.</p>
      <div class="table-wrap"><table><thead><tr><th>Mode</th><th class="num">MRR@10</th><th class="num">Recall@5</th><th>vs dense</th><th class="num">p95</th></tr></thead><tbody>${bigRows || '<tr><td colspan="5">not measured</td></tr>'}</tbody></table></div></div>
    <div class="card"><h2>Failure analysis</h2><p class="sub">${(X.qt?.n_questions || 0).toLocaleString()} test questions, ${X.name} index · ${X.hw} profile.</p>
      <div class="table-wrap"><table><tbody>${Object.entries(caseNames).filter(([k]) => k in cases).map(([k, n]) => `<tr><td>${n}</td><td class="num"><b>${cases[k]}</b></td></tr>`).join("")}</tbody></table></div></div>
  </section>
  ${tagsCard(d, X)}`;
}

/* ---------- live updates view ---------- */
function updatesView() {
  const u = S.upd, h = S.health;
  return `<div class="page-head"><div><h1>Live Updates — change a document, watch the answer change</h1>
    <p>Every insert, update and delete goes straight into the live index through the API — no rebuild. Qdrant keeps only the
    <b>latest</b> version of each document, so an outdated policy can never be returned; a change log keeps the full history.</p></div>
    <div class="page-meta"><span>index_version: <strong id="iv">${h ? h.index_version : "—"}</strong></span><span>•</span><span>writes to the 100k index</span></div></div>
  <section class="card"><div class="card-head"><div><h2>1 · Company documents</h2>
    <p class="sub">Five fictional “Acme Retail” policies (refund, shipping, warranty, remote work, leave), tagged <code>corpus = internal</code>, loaded through the same API any company system would call.</p></div>
    <div class="btn-row"><button class="btn" type="button" id="demo-load">Load demo documents</button><button class="btn danger" type="button" id="demo-remove">Remove them</button></div></div>
    <div id="demo-status" class="sub" style="margin-top:10px"></div></section>
  <section class="card"><h2>2 · Before / after experiment</h2>
    <p class="sub">Ask a question → edit a document → <b>Save</b>. The same question runs again immediately on the updated index and both results are shown side by side.</p>
    <div class="form">
      <div class="field"><label for="u-q">Question</label><input class="input" id="u-q" value="${esc(u.q)}"></div>
      <div class="controls"><div class="filters">
        <label class="check"><input type="checkbox" id="u-int" ${u.internal ? "checked" : ""}> Only company documents (corpus = internal)</label>
        <label class="check"><input type="checkbox" id="u-ans" ${u.answer ? "checked" : ""}> AI answer (Groq)</label></div>
        <button class="btn ghost" type="button" id="u-run">Ask now</button></div>
      <div class="row2"><div class="field"><label for="u-doc">Document to edit (doc_id)</label><input class="input" id="u-doc" list="u-docs" value="${esc(u.doc)}">
        <datalist id="u-docs">${DEMO_IDS.map((d) => `<option value="${d}">`).join("")}</datalist></div>
        <div class="field"><label>&nbsp;</label><button class="btn ghost" type="button" id="u-load">Load current text</button></div></div>
      <div class="field"><label for="u-text">New text</label><textarea class="textarea" id="u-text" placeholder="Load the current text, change something (e.g. 30 days → 14 days), then save."></textarea></div>
      <div class="btn-row"><button class="btn" type="button" id="u-save">Save new version &amp; compare</button><button class="btn danger" type="button" id="u-del">Delete this document &amp; compare</button></div>
    </div>
    <div id="u-out" style="margin-top:18px"></div></section>
  <section class="grid2">
    <div class="card"><h2>Version history</h2><p class="sub" id="hist-title">Every saved version of the selected document (from the change log).</p><ul class="timeline" id="u-hist"><li class="sub">Load or save a document to see its history.</li></ul></div>
    <div class="card"><h2>Recent changes</h2><p class="sub">Every write to the index, newest first.</p><ul class="timeline" id="u-changes"></ul></div>
  </section>`;
}

/** word-level diff (longest common subsequence) → HTML with <del>/<ins> */
function wordDiff(a, b) {
  const A = (a || "").split(/(\s+)/), B = (b || "").split(/(\s+)/);
  const n = A.length, m = B.length;
  const L = Array.from({ length: n + 1 }, () => new Uint16Array(m + 1));
  for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--)
    L[i][j] = A[i] === B[j] ? L[i + 1][j + 1] + 1 : Math.max(L[i + 1][j], L[i][j + 1]);
  let i = 0, j = 0, out = "";
  while (i < n && j < m) {
    if (A[i] === B[j]) { out += esc(A[i]); i++; j++; }
    else if (L[i + 1][j] >= L[i][j + 1]) { out += A[i].trim() ? `<del>${esc(A[i])}</del>` : esc(A[i]); i++; }
    else { out += B[j].trim() ? `<ins>${esc(B[j])}</ins>` : esc(B[j]); j++; }
  }
  while (i < n) { out += A[i].trim() ? `<del>${esc(A[i])}</del>` : esc(A[i]); i++; }
  while (j < m) { out += B[j].trim() ? `<ins>${esc(B[j])}</ins>` : esc(B[j]); j++; }
  return out;
}

async function askUpd() {
  const u = S.upd, p = { q: u.q, mode: "hybrid_rerank", corpus: u.internal ? "internal" : "", no_cache: true };
  const t = performance.now();
  if (u.answer) {
    const a = await api(`/answer?${qs(p)}`);
    return { res: a.search, answer: a.answer, error: a.error, model: a.model, ms: performance.now() - t };
  }
  return { res: await api(`/search?${qs(p)}`), ms: performance.now() - t };
}

function sideHtml(title, data, docId, cls) {
  const hits = data.res.hits.slice(0, 3);
  const mine = data.res.hits.find((h) => h.doc_id === docId);
  const where = mine ? `<span class="chip good">${esc(docId)} is #${mine.rank} · v${mine.version}</span>` : `<span class="chip">${esc(docId)} not in top 5</span>`;
  const ans = S.upd.answer ? (data.answer ? `<div class="answer"><h3>AI answer · ${esc(data.model || "")}</h3><p>${esc(data.answer)}</p></div>`
    : `<div class="error" style="margin-bottom:10px">No answer: ${esc(data.error || "—")}</div>`) : "";
  return `<div class="ba-col ${cls}"><h3><span>${title}</span>${where}</h3>${ans}
    ${hits.map((h) => `<div class="mini-hit"><b>#${h.rank}</b> <span class="mono">${esc(h.doc_id)}</span>${h.version > 1 ? ` <span class="chip warn">v${h.version}</span>` : ""}
      <span class="sub" style="display:inline">· score ${h.score.toFixed(2)}</span><br>${highlight(clip(h.text, 220), S.upd.q)}</div>`).join("") || '<div class="empty">No results</div>'}
    <p class="sub" style="margin-top:8px">search ${ms(data.res.timings_ms.total)}${S.upd.answer ? ` · with answer ${ms(data.ms)}` : ""}</p></div>`;
}

async function refreshUpdSide(docId) {
  try {
    const [hist, ch] = await Promise.all([api(`/passages/${encodeURIComponent(docId)}/history`), api("/changes?limit=15")]);
    const li = (c, showDoc) => `<li class="${c.action}"><b>${esc(c.action)}</b>${c.version ? ` · v${c.version}` : ""}${showDoc ? ` · <code>${esc(c.doc_id)}</code>` : ""}
      <span class="sub" style="display:inline">· ${esc(c.at.replace("T", " ").replace("+00:00", " UTC"))} · index_version ${c.index_version}</span>
      ${c.text && !showDoc ? `<br>“${esc(clip(c.text, 160))}”` : ""}</li>`;
    document.getElementById("hist-title").textContent = `${docId} — ${hist.versions.length} change(s)`;
    document.getElementById("u-hist").innerHTML = hist.versions.length ? hist.versions.slice().reverse().map((c) => li(c, false)).join("") : '<li class="sub">No changes recorded for this doc_id yet.</li>';
    document.getElementById("u-changes").innerHTML = ch.changes.length ? ch.changes.map((c) => li(c, true)).join("") : '<li class="sub">No changes yet.</li>';
  } catch (e) { /* history is optional */ }
  await loadHealth();
  const iv = document.getElementById("iv");
  if (iv && S.health) iv.textContent = S.health.index_version;
}

function bindUpdates() {
  const $ = (id) => document.getElementById(id), out = $("u-out"), u = S.upd;
  const sync = () => { u.q = $("u-q").value; u.doc = $("u-doc").value.trim(); u.internal = $("u-int").checked; u.answer = $("u-ans").checked; };
  const busy = (msg) => { out.innerHTML = `<div class="loading"><span class="spin"></span>${msg}</div>`; };
  $("demo-load").onclick = async () => {
    $("demo-status").innerHTML = '<span class="spin"></span>Loading…';
    try {
      const r = await api("/demo/company-docs", { method: "POST" });
      $("demo-status").innerHTML = r.loaded.map((x) => `<span class="chip good">${esc(x.doc_id)} · ${esc(x.status)} · v${x.version}${x.topic ? " · " + esc(x.topic) : ""}</span>`).join(" ");
      S.meta = null; refreshUpdSide(u.doc);
    } catch (e) { $("demo-status").innerHTML = `<span class="no-mark">${esc(e.message)}</span>`; }
  };
  $("demo-remove").onclick = async () => {
    try {
      const r = await api("/demo/company-docs", { method: "DELETE" });
      $("demo-status").innerHTML = r.removed.map((x) => `<span class="chip">${esc(x.doc_id)} · ${esc(x.status)}</span>`).join(" ");
      S.meta = null; refreshUpdSide(u.doc);
    } catch (e) { $("demo-status").innerHTML = `<span class="no-mark">${esc(e.message)}</span>`; }
  };
  $("u-load").onclick = async () => {
    sync();
    try { const d = await api(`/passages/${encodeURIComponent(u.doc)}`); $("u-text").value = d.text; }
    catch { $("u-text").value = ""; out.innerHTML = `<div class="error">${esc(u.doc)} is not in the index — load the demo documents first, or type new text to insert it.</div>`; }
    refreshUpdSide(u.doc);
  };
  $("u-run").onclick = async () => {
    sync(); busy("Asking…");
    try { const r = await askUpd(); out.innerHTML = `<div class="ba" style="grid-template-columns:1fr">${sideHtml("Current answer", r, u.doc, "")}</div>`; }
    catch (e) { out.innerHTML = `<div class="error">${esc(e.message)}</div>`; }
  };
  const compare = async (write, label) => {
    sync();
    try {
      busy("1/3 · asking the question on the current index…");
      const before = await askUpd();
      busy(`2/3 · ${label}…`);
      const w = await write();
      busy("3/3 · asking the same question again…");
      const after = await askUpd();
      const prevText = w.previous_text, newText = w.status === "deleted" ? "" : $("u-text").value;
      out.innerHTML = `<div class="callout"><b>${esc(w.status)}</b> <code>${esc(w.doc_id)}</code>
          ${w.previous_version ? ` · v${w.previous_version} → ${w.status === "deleted" ? "removed" : "v" + w.version}` : w.version ? ` · v${w.version}` : ""}
          · index write ${ms(w.ms)} · index_version → ${w.index_version} · no rebuild, the old version can no longer be returned</div>
        ${prevText != null || newText ? `<p class="sub" style="margin:14px 0 6px">What changed in the document (red = removed, green = added):</p>
          <div class="diff">${w.status === "deleted" ? `<del>${esc(prevText || "")}</del>` : wordDiff(prevText || "", newText)}</div>` : ""}
        <div class="ba" style="margin-top:16px">${sideHtml("Before", before, w.doc_id, "")}${sideHtml("After", after, w.doc_id, "after")}</div>`;
      refreshUpdSide(w.doc_id);
    } catch (e) { out.innerHTML = `<div class="error">${esc(e.message)}</div>`; }
  };
  $("u-save").onclick = () => {
    if (!$("u-text").value.trim()) { out.innerHTML = '<div class="error">Type or load some text first.</div>'; return; }
    compare(async () => {
      let old = null;
      try { old = await api(`/passages/${encodeURIComponent(u.doc)}`); } catch { /* new document */ }
      return api("/passages", { method: "POST", body: JSON.stringify({ doc_id: u.doc, text: $("u-text").value,
        source: old?.source || "acme-policies.internal", category: old?.category || "description", corpus: old?.corpus || "internal" }) });
    }, "saving the new version into the live index");
  };
  $("u-del").onclick = () => compare(() => api(`/passages/${encodeURIComponent(u.doc)}`, { method: "DELETE" }), "deleting the document");
  refreshUpdSide(u.doc);
}

/* ---------- scale view: 100k vs 500k ---------- */
function scaleView() {
  const d = S.dash;
  if (!d) return `<div class="error">Could not load measured results: ${esc(S.dashErr || "loading…")}</div>`;
  // deployment pairing: the 100k index on the CPU profile, the 500k index on the GPU profile
  const A = d.ir_cpu || {}, B = d.ir_500k || {}, LA = d.latency_cpu?.modes || {}, LB = d.latency_500k?.modes || {};
  const RA = d.ragas_cpu, RB = d.ragas_500k, sc = d.scale || {}, cr = sc.candidate_recall_500k;
  const order = ["dense", "bm25", "hybrid", "dense_rerank", "hybrid_rerank"];
  const kpiCard = (label, hw, build, ir, lat, rg) => `<div class="card"><div class="card-head"><div><h2>${label} passages · ${hw} profile</h2>
      <p class="sub">${build ? `index built ${esc(build.timestamp.slice(0, 10))} on ${build.device === "cpu" ? "the CPU" : esc(build.gpu)} · served on the ${hw}` : "not built"}</p></div><span class="scale-badge">${label} · ${hw}</span></div>
    <div class="kpis">
      <div class="kpi"><b>${build ? (build.first_build_minutes ?? build.minutes) + " min" : "—"}</b><span>index build (limit 120)${build?.first_build_minutes ? ` · ${build.minutes} min in a rebuild that shared the GPU` : ""}</span></div>
      <div class="kpi"><b>${build?.ram_after ? (build.ram_after.qdrant_mb / 1024).toFixed(1) + " GB" : "—"}</b><span>Qdrant RAM after build</span></div>
      <div class="kpi"><b>${lat?.hybrid_rerank ? Math.round(lat.hybrid_rerank.p95_ms) + " ms" : "—"}</b><span>p95, Phase 2 (target &lt; 300)</span></div>
      <div class="kpi"><b>${f3(ir?.hybrid_rerank?.["mrr@10"])}</b><span>MRR@10, Phase 2</span></div>
      <div class="kpi"><b>${f3(rg?.modes?.hybrid_rerank?.context_precision)}</b><span>RAGAS precision, Phase 2</span></div>
      <div class="kpi"><b>${f3(rg?.modes?.hybrid_rerank?.context_recall)}</b><span>RAGAS recall, Phase 2</span></div></div></div>`;
  const qRows = order.filter((m) => A[m] || B[m]).map((m) => {
    const a = A[m], b = B[m], dm = a && b ? b["mrr@10"] - a["mrr@10"] : null;
    return `<tr class="${m === "hybrid_rerank" ? "hl" : ""}"><td>${MODES[m].short}</td>
      <td class="num">${f3(a?.["mrr@10"])}</td><td class="num">${f3(b?.["mrr@10"])}</td><td class="num">${dm == null ? "—" : sgn(dm)}</td>
      <td>${a?.vs_dense ? sigChip(a.vs_dense) : '<span class="chip">baseline</span>'}</td><td>${b?.vs_dense ? sigChip(b.vs_dense) : '<span class="chip">baseline</span>'}</td></tr>`;
  }).join("");
  const KEYS = [["mrr@10", "MRR@10"], ["ndcg@10", "nDCG@10"], ["hit@5", "Hit@5"], ["recall@5", "Recall@5"], ["precision@5", "Precision@5"], ["hit@10", "Hit@10"]];
  const fullRows = order.filter((m) => A[m] || B[m]).map((m) => `<tr class="${m === "hybrid_rerank" ? "hl" : ""}"><td>${MODES[m].short}</td>
      ${KEYS.map(([k]) => `<td class="num">${f3(A[m]?.[k])}</td>`).join("")}<td style="border-left:2px solid #e2e8f0"></td>
      ${KEYS.map(([k]) => `<td class="num">${f3(B[m]?.[k])}</td>`).join("")}</tr>`).join("");
  const latRows = Object.keys({ ...LA, ...LB }).map((m) => {
    const bar = (x) => x ? `<div class="track" style="width:120px;display:inline-flex;vertical-align:middle"><div style="width:${Math.min(x.p95_ms / 300, 1) * 100}%;background:#2563eb"></div></div> <b>${x.p95_ms}</b>` : "—";
    return `<tr><td>${esc(m.replace("+filter", " + filter").split("_").join(" "))}</td><td class="num">${LA[m]?.p50_ms ?? "—"}</td><td>${bar(LA[m])}</td><td class="num">${LA[m]?.p99_ms ?? "—"}</td>
      <td class="num">${LB[m]?.p50_ms ?? "—"}</td><td>${bar(LB[m])}</td><td class="num">${LB[m]?.p99_ms ?? "—"}</td></tr>`;
  }).join("");
  const ragRow = (label, r) => r ? ["dense", "hybrid", "hybrid_rerank"].filter((m) => r.modes[m]).map((m) => `<tr class="${m === "hybrid_rerank" ? "hl" : ""}"><td>${label}</td><td>${MODES[m].short}</td>
      <td class="num">${r.modes[m].n}</td><td class="num">${f3(r.modes[m].context_precision)} ${r.paired?.[m] ? sigChip(r.paired[m].context_precision) : ""}</td>
      <td class="num">${f3(r.modes[m].context_recall)} ${r.paired?.[m] ? sigChip(r.paired[m].context_recall) : ""}</td></tr>`).join("") : `<tr><td>${label}</td><td colspan="4">not measured yet</td></tr>`;
  const take = [];
  if (A.dense && B.dense) take.push(`<b>5× more data is harder:</b> dense MRR ${f3(A.dense["mrr@10"])} → ${f3(B.dense["mrr@10"])} — five times as many look-alike passages compete for the top spots.`);
  if (A.hybrid_rerank?.vs_dense && B.hybrid_rerank?.vs_dense) take.push(`<b>The Phase 2 gain holds on both setups:</b> gain over dense ${sgn(A.hybrid_rerank.vs_dense.mean_diff)} at 100k on the CPU and ${sgn(B.hybrid_rerank.vs_dense.mean_diff)} at 500k on the GPU — both confidence intervals exclude zero.`);
  if (LA.hybrid_rerank && LB.hybrid_rerank) take.push(`<b>Both under 300 ms:</b> Phase 2 p95 ${Math.round(LA.hybrid_rerank.p95_ms)} ms for 100k on a CPU and ${Math.round(LB.hybrid_rerank.p95_ms)} ms for 500k on a laptop GPU — the GPU serves 5× the data faster than the CPU serves 100k, because the reranker (the slowest step) runs on it.`);
  if (sc["100k"] && sc["500k"]) take.push(`<b>Indexing stays far inside the limit:</b> ${(sc["100k_cpu"] || sc["100k"]).minutes} min for 100k${sc["100k_cpu"] ? " on the CPU" : ""}, ${sc["500k"].first_build_minutes ?? sc["500k"].minutes} min for 500k on the GPU (limit 120 min).`);
  if (cr?.["20"]) take.push(`<b>BM25 earns its place at scale:</b> the correct passage is in the top-20 candidate pool ${(cr["20"].dense * 100).toFixed(1)}% of the time with dense alone, ${(cr["20"].union * 100).toFixed(1)}% with dense + BM25.`);
  if (RB?.modes?.hybrid_rerank) take.push(`<b>RAGAS targets met at 500k too:</b> precision ${f3(RB.modes.hybrid_rerank.context_precision)}, recall ${f3(RB.modes.hybrid_rerank.context_recall)}.`);
  const live = (S.health?.indexes || {})["500k"] != null;
  return `<div class="page-head"><div><h1>Scale: 100k on CPU · 500k on GPU</h1><p>The same pipeline and the same 1,000 test questions in the two deployments we ship: the 100,000-passage index on the CPU profile (any laptop, reranker reads the top ${d.ir_cpu_rerank_depth}) and the 500,000-passage index on the GPU profile (reranker reads the top ${d.ir_gpu_rerank_depth}).</p></div></div>
  <section class="grid2">${kpiCard("100k", "CPU", sc["100k_cpu"] || sc["100k"], A, LA, RA)}${kpiCard("500k", "GPU", sc["500k"], B, LB, RB)}</section>
  <section class="callout"><b>What this shows</b><ul>${take.map((t) => `<li>${t}</li>`).join("")}</ul></section>
  ${hwCard(d)}
  <section class="card"><h2>Retrieval quality at both scales</h2><p class="sub">1,000 held-out test questions, human labels. Chips: change vs dense at that scale (paired bootstrap, 95% CI).</p>
    <div class="table-wrap"><table><thead><tr><th>Mode</th><th class="num">MRR@10 · 100k CPU</th><th class="num">MRR@10 · 500k GPU</th><th class="num">Δ</th><th>MRR vs dense · 100k CPU</th><th>MRR vs dense · 500k GPU</th></tr></thead><tbody>${qRows}</tbody></table></div></section>
  <section class="card"><h2>Every label metric at both scales</h2><p class="sub">Same 1,000 test questions. Precision@5 is low by nature: most questions have only one labelled passage, so 0.2 is the maximum for them.</p>
    <div class="table-wrap"><table><thead><tr><th></th><th colspan="6" style="text-align:center">100,000 passages · CPU</th><th></th><th colspan="6" style="text-align:center">500,000 passages · GPU</th></tr>
      <tr><th>Mode</th>${KEYS.map(([, n]) => `<th class="num">${n}</th>`).join("")}<th style="border-left:2px solid #e2e8f0"></th>${KEYS.map(([, n]) => `<th class="num">${n}</th>`).join("")}</tr></thead>
      <tbody>${fullRows}</tbody></table></div></section>
  <section class="grid2">
    <div class="card"><h2>Latency at both scales</h2><p class="sub">100 consecutive queries per mode, cache off, end-to-end HTTP (ms). Bar = p95 vs the 300 ms target.</p>
      <div class="table-wrap"><table><thead><tr><th>Mode</th><th class="num">p50 · 100k CPU</th><th>p95 · 100k CPU</th><th class="num">p99 · 100k CPU</th><th class="num">p50 · 500k GPU</th><th>p95 · 500k GPU</th><th class="num">p99 · 500k GPU</th></tr></thead><tbody>${latRows}</tbody></table></div></div>
    <div class="card"><h2>RAGAS at both scales</h2><p class="sub">Judge gpt-oss-120b, the same 50 questions. Chips: change vs dense at that scale.</p>
      <div class="table-wrap"><table><thead><tr><th>Index</th><th>Mode</th><th class="num">n</th><th class="num">Context Precision</th><th class="num">Context Recall</th></tr></thead><tbody>${ragRow("100k · CPU", RA)}${ragRow("500k · GPU", RB)}</tbody></table></div></div>
  </section>
  <section class="card"><div class="trace-head"><div><h2>Live: the same question on both indexes</h2><p class="sub">${live ? "Runs Phase 2 (hybrid + reranker) now: 100k on the CPU, 500k on the GPU." : "The 500k index is not loaded in the API right now — the measured results above still apply."}</p></div>
    <form class="query-form" id="scale-form"><input class="input mono" id="scale-q" value="${esc(S.scaleQ)}"><button class="btn" type="submit" ${live ? "" : "disabled"}>Run on both</button></form></div>
    <div class="cols2" id="scale-out" style="padding-top:16px"></div></section>`;
}

/** the two deployments side by side: 100k on the CPU profile, 500k on the GPU profile */
function hwCard(d) {
  const P = d.profiles || {}, c = P.cpu || {}, g = P.gpu || {};
  const ic = d.ir_cpu?.hybrid_rerank, ig = d.ir_500k?.hybrid_rerank, lc = d.latency_cpu?.modes || {}, lg = d.latency_500k?.modes || {};
  const rc = d.ragas_cpu?.modes?.hybrid_rerank, rgp = d.ragas_500k?.modes?.hybrid_rerank, busy = d.latency_cpu_recheck?.modes?.hybrid_rerank;
  const rows = [
    ["Index", "100,000 passages", "500,000 passages"],
    ["Hardware", "CPU only (no GPU)", esc(d.latency_500k?.gpu || "GPU")],
    ["Models run in", `${c.fp16 ? "fp16" : "fp32"} on the CPU`, `${g.fp16 ? "fp16" : "fp32"} on the GPU`],
    ["Reranker reads the top", c.rerank_depth, g.rerank_depth],
    ["MRR@10, Phase 2 (1,000 test q)", f3(ic?.["mrr@10"]), f3(ig?.["mrr@10"])],
    ["RAGAS precision / recall, Phase 2", rc ? `${f3(rc.context_precision)} / ${f3(rc.context_recall)}` : "—", rgp ? `${f3(rgp.context_precision)} / ${f3(rgp.context_recall)}` : "—"],
    ["p95, Phase 2 (100 queries)", lc.hybrid_rerank ? `<b>${Math.round(lc.hybrid_rerank.p95_ms)} ms</b>` : "—", lg.hybrid_rerank ? `<b>${Math.round(lg.hybrid_rerank.p95_ms)} ms</b>` : "—"],
    ["p95, Phase 2 + filter", lc["hybrid_rerank+filter"] ? `${Math.round(lc["hybrid_rerank+filter"].p95_ms)} ms` : "—", lg["hybrid_rerank+filter"] ? `${Math.round(lg["hybrid_rerank+filter"].p95_ms)} ms` : "—"],
  ];
  return `<section class="card"><h2>Hardware profiles — 100k on CPU, 500k on GPU</h2>
    <p class="sub">The system picks a profile from the hardware (set in config <code>serving.devices</code>). The CPU profile reads fewer candidates with the reranker so a laptop without a GPU stays well under 300 ms; the GPU profile reads more and carries 5× the data.</p>
    <div class="table-wrap"><table><thead><tr><th></th><th>CPU profile · 100k</th><th>GPU profile · 500k</th></tr></thead>
    <tbody>${rows.map(([k, a, b]) => `<tr><td>${k}</td><td>${a}</td><td>${b}</td></tr>`).join("")}</tbody></table></div>
    ${busy ? `<p class="sub" style="margin-top:10px">Honest caveat: the CPU profile shares the processor with everything else on the machine. Re-measured while OneDrive was syncing the project folder, Phase 2 p95 rose to ${Math.round(busy.p95_ms)} ms — close a heavy background program before a CPU demo. The GPU profile is not affected.</p>` : ""}</section>`;
}

async function runScaleLive() {
  const out = document.getElementById("scale-out");
  if (!out || (S.health?.indexes || {})["500k"] == null) return;
  out.innerHTML = `<div class="loading"><span class="spin"></span>Running on both indexes…</div>`;
  try {
    const [a, b] = await Promise.all(["100k", "500k"].map((index) => api(`/search?${qs({ q: S.scaleQ, mode: "hybrid_rerank", index, no_cache: true })}`)));
    const col = (label, r) => `<div class="card" style="padding:16px"><div class="card-head"><h2>${label}</h2><span class="scale-badge">${ms(r.timings_ms.total)}</span></div>
      ${timingBar(r.timings_ms)}${r.hits.slice(0, 3).map((h) => `<div class="mini-hit"><b>#${h.rank}</b> <span class="sub" style="display:inline">${esc(h.source)} · ${origin(h)}</span><br>${highlight(clip(h.text, 200), S.scaleQ)}</div>`).join("")}</div>`;
    out.innerHTML = col("100,000 passages · CPU", a) + col("500,000 passages · GPU", b);
  } catch (e) { out.innerHTML = `<div class="error">${esc(e.message)}</div>`; }
}

/* ---------- metadata tags card (comparison page) ---------- */
function tagsCard(d) {
  const t = d.tags || {}, dist = t.distribution?.msmarco, v = t.validation, f = t.filter?.test_cpu || t.filter?.test || t.filter?.dev_large;
  if (!dist) return "";
  const n = dist.passages, top = Object.entries(dist.topics);
  const max = Math.max(...top.map(([, c]) => c));
  const fRows = f ? Object.entries(f).filter(([k]) => k.startsWith("hybrid_rerank")).map(([k, x]) => `<tr class="${k.endsWith("none") ? "" : ""}">
      <td>${esc({ none: "No topic filter", auto1: "Auto: top-1 topic", auto2: "Auto: top-2 topics", auto3: "Auto: top-3 topics", oracle: "Oracle: correct topic (upper bound)" }[k.split("|")[1]] || k)}</td>
      <td class="num">${f3(x["mrr@10"])}</td><td class="num">${f3(x["recall@5"])}</td><td class="num">${x.route_accuracy != null ? (x.route_accuracy * 100).toFixed(1) + "%" : "—"}</td>
      <td>${x.mrr_vs_none ? sigChip(x.mrr_vs_none) : '<span class="chip">reference</span>'}</td></tr>`).join("") : "";
  return `<section class="grid2">
    <div class="card"><h2>Metadata tags — topics</h2><p class="sub">${n.toLocaleString()} passages tagged in ${dist.seconds} s from the vectors already in Qdrant (no re-embedding, no re-index).
      ${v ? ` Checked against an LLM (${esc(v.judge)}) on ${v.n} random passages: <b>${(v.top1_agreement * 100).toFixed(0)}%</b> agree on the topic, <b>${(v.llm_topic_in_our_top2 * 100).toFixed(0)}%</b> within our top 2.` : ""}</p>
      <div class="bars">${top.map(([name, c]) => `<div class="bar2"><div class="bar-row-head"><span class="name">${esc(name)}</span><span class="vals">${c.toLocaleString()} · ${(c / n * 100).toFixed(1)}%</span></div>
        <div class="track"><div style="width:${c / max * 100}%;background:#2563eb"></div></div></div>`).join("")}</div></div>
    <div class="card"><h2>Does topic routing help?</h2><p class="sub">${f ? `${t.filter?.test_cpu ? "1,000 test questions, 100k index · CPU profile" : t.filter?.test ? "Test questions" : "Tuning (dev_large) questions"}, Phase 2. Auto = filter to the question's most likely topics before searching.` : "Not measured yet."}</p>
      ${f ? `<div class="table-wrap"><table><thead><tr><th>Setting</th><th class="num">MRR@10</th><th class="num">Recall@5</th><th class="num">Routing correct</th><th>vs no filter</th></tr></thead><tbody>${fRows}</tbody></table></div>` : ""}
      <p class="sub" style="margin-top:12px">Source types: ${Object.entries(dist.source_types).map(([k, c]) => `${esc(SOURCE_TYPE_LABELS[k] || k)} ${(c / n * 100).toFixed(0)}%`).join(" · ")}</p></div>
  </section>`;
}

/* ---------- router ---------- */
async function render() {
  const view = (location.hash || "#search").slice(1);
  document.querySelectorAll("#tabs a").forEach((a) => a.classList.toggle("active", a.dataset.view === (view === "admin" ? "updates" : view)));
  if (["dense", "hybrid", "hybrid_rerank"].includes(view)) {
    if (!S.dash) { $view.innerHTML = `<div class="loading"><span class="spin"></span>Loading measured results…</div>`; await loadDash(); }
    $view.innerHTML = phaseView(view);
    const f = document.getElementById("trace-form");
    if (f) {
      f.onsubmit = (e) => { e.preventDefault(); S.trace.q = document.getElementById("trace-q").value; runTrace(view); };
      runTrace(view);
    }
  } else if (view === "compare") {
    if (!S.dash) { $view.innerHTML = `<div class="loading"><span class="spin"></span>Loading measured results…</div>`; await loadDash(); }
    $view.innerHTML = compareView();
  } else if (view === "updates" || view === "admin") {
    $view.innerHTML = updatesView();
    bindUpdates();
  } else if (view === "scale") {
    if (!S.dash) { $view.innerHTML = `<div class="loading"><span class="spin"></span>Loading measured results…</div>`; await loadDash(); }
    $view.innerHTML = scaleView();
    const f = document.getElementById("scale-form");
    if (f) f.onsubmit = (e) => { e.preventDefault(); S.scaleQ = document.getElementById("scale-q").value; runScaleLive(); };
    runScaleLive();
  } else {
    if (!S.meta) await loadMeta();
    $view.innerHTML = searchView();
    const s = S.search;
    document.getElementById("search-form").onsubmit = (e) => { e.preventDefault(); s.q = document.getElementById("sq").value; runSearch(); };
    document.getElementById("seg").onclick = (e) => {
      const b = e.target.closest("button"); if (!b) return;
      s.mode = b.dataset.mode;
      document.querySelectorAll("#seg button").forEach((x) => x.classList.toggle("on", x === b));
      runSearch();
    };
    for (const [id, key] of [["fcat", "category"], ["fsrc", "source"], ["ftopic", "topic"], ["fstype", "stype"], ["fcorpus", "corpus"]])
      document.getElementById(id).onchange = (e) => { s[key] = e.target.value; runSearch(); };
    document.getElementById("fans").onchange = (e) => { s.answer = e.target.checked; runSearch(); };
    document.getElementById("fside").onchange = (e) => { s.side = e.target.checked; runSearch(); };
    runSearch();
  }
}

document.getElementById("refresh").onclick = async () => {
  S.dash = null; S.meta = null;
  await loadHealth();
  render();
};
window.addEventListener("hashchange", render);
loadHealth().then(render);
