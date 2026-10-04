# PrecisionRAG — finding the *right* passage, not just a similar one

**Team OutLiers · ADROSONIC BUILD · Problem Statement 1: Vector Database Design for Large-Scale Precision Retrieval in
RAG Systems**

A retrieval system over **MS MARCO** passages in **Qdrant**: a dense-vector baseline (Phase 1), then hybrid
dense + BM25 search with a documented, configurable fusion and a cross-encoder reranker (Phase 2), pre-retrieval
metadata filters, live upsert/delete, a query UI, RAGAS evaluation and a 100-query latency benchmark.

> **Our rule: we didn't assume a component helps — we measured it.** Every number below comes from a file in
> `results/`; `python -m scripts.make_report` turns them into a full benchmark report.

## Results

Two deployments from one codebase: the **100k index on a CPU** and the **500k index on a laptop GPU** (RTX 3050 6 GB).

| Metric (Phase 2 = hybrid + reranker) | Target | 100k · CPU | 500k · GPU |
|---|---|---|---|
| RAGAS Context Precision (50 questions) | > 0.75 | **0.809** | **0.820** |
| RAGAS Context Recall | > 0.70 | **0.900** | **0.857** |
| MRR@10 gain over dense (1,000 test questions, 95% CI) | — | +0.048 (+0.027 … +0.069) | +0.062 (+0.042 … +0.083) |
| p95 latency, 100 consecutive queries | < 300 ms | **140 ms** | **76 ms** |
| Index build | < 2 h | 21.6 min | 13.3 min |
| Correct topic filter (MRR@10) | — | +0.074 | — |
| Live update (re-embed + upsert, no re-index) | — | 84 ms | same API |

Phase 1 (dense) baseline: RAGAS 0.788 / 0.860 at 100k, 0.754 / 0.850 at 500k.

## Architecture

```
INDEXING (offline, resumable)
MS MARCO v1.1 → clean + dedup → metadata {passage_id, doc_id, source, category, version, updated_at, content_hash}
             → bge-small-en-v1.5 (dense, 384-d)  +  BM25 (sparse)  →  Qdrant (one collection, HNSW + IDF)
             → tags {topic (15 subjects), source_type, corpus} from the stored vectors — no re-embedding

QUERY (FastAPI; the web UI and benchmark scripts call the API)
question → encode (dense + BM25) [+ optional topic routing] → ONE batched Qdrant call: dense top-50 ∥ BM25 top-50,
         same pre-filter inside both → weighted RRF fusion → cross-encoder rerank of top-20 (MiniLM-L6)
         → top-5 + scores + per-stage timings [→ AI answer with citations]

LIVE UPDATES (API)
POST/DELETE /passages → point ID = UUID5(doc_id) overwrites the old version → tags → change log (every version)
                      → index_version++ (result cache can never serve stale results)
```

**Filters** (all pre-retrieval, on indexed payload fields, combinable): question type, website, **topic**, **source
type** (government / education / reference / health / Q&A / news / organization / commercial), **corpus** (web or
internal company documents). **Indexes:** the official 100k collection and a 500k scale collection, switchable live —
the 100k index is served on the **CPU profile** (rerank top-10; MRR 0.581, RAGAS 0.809 / 0.900, p95 140 ms — 532 ms when the laptop is busy) and
the 500k index on the **GPU profile** (rerank top-20), set by `serving.devices` in `config.yaml`.

| Mode | What it does |
|---|---|
| `dense` | Phase 1 baseline — meaning search only |
| `bm25` | keyword search only (ablation) |
| `hybrid` | dense + BM25, weighted RRF (dense 1.0, BM25 0.1) |
| `hybrid_rerank` | **Phase 2** — hybrid (RRF 1:1) → cross-encoder reranks the top 20 (GPU) or top 10 (CPU) |
| `dense_rerank` | ablation — reranker without BM25 |

## Quick start (Windows, no Docker needed)

**Requirements:** Python 3.11, ~5 GB free disk, 16 GB RAM recommended. An NVIDIA GPU is optional — the code picks the
**GPU profile** (500k passages, rerank top-20) or the **CPU profile** (100k passages, rerank top-10) automatically.

```bash
# 1. environment (uv: https://docs.astral.sh/uv/)
uv venv .venv --python 3.11
uv pip install --python .venv\Scripts\python.exe -r requirements.lock.txt --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match

# 2. settings
copy .env.example .env          # put your free Groq key in GROQ_API_KEY (only needed for RAGAS)

# 3. Qdrant (download qdrant-x86_64-pc-windows-msvc.zip v1.19.1 from github.com/qdrant/qdrant/releases)
set QDRANT_BIN=C:\path\to\qdrant.exe
start_qdrant.bat                # or, with Docker:  docker compose up -d

# 4. data + index (first run downloads MS MARCO and the models)
.venv\Scripts\python -m scripts.make_eval_sets
.venv\Scripts\python -m scripts.build_index --size 100000

# 5. run — the API also serves the web UI
.venv\Scripts\python -m uvicorn api.main:app --host 127.0.0.1 --port 8765     # UI: http://127.0.0.1:8765/   API docs: /docs
```

The web UI has seven views: **Search** (dense / BM25 / hybrid / hybrid + reranker toggle; filters for topic — with
✨ automatic topic routing — source type, corpus, question type and website; AI answer with citations; side-by-side),
**Phase 1**, **Phase 2: Hybrid** and **Phase 2: Hybrid + Reranker** dashboards (RAGAS, MRR, p95 latency, tuning curve,
results per question type, a live query trace), **Comparison** (every challenge target, ablation with confidence
intervals, metadata tags and topic routing), **Scale: 100k vs 500k** (quality, latency, memory, RAGAS at both sizes,
the same question run live on both) and **Live Updates** (load demo company documents, edit one, see the answer before
and after, a word diff and the version history). A 100k / 500k switch in the header picks the index for live
searches. Every number is read from `results/` through `GET /dashboard`; nothing is typed by hand.

Linux / macOS: same steps with `.venv/bin/python` and `docker compose up -d` for Qdrant.

> **Qdrant setup used for our results:** all our measurements used the native Qdrant binary (v1.19.1) on Windows,
> started with `start_qdrant.bat`. `docker-compose.yml` runs the same version for Linux / macOS users; it was not tested
> on our laptop (no Docker installed).

> **Windows tip:** always use `http://127.0.0.1`, never `localhost` — on Windows `localhost` tries IPv6 first and added
> ~2 seconds to every database call in our tests.

## Reproduce every number

| What | Command | Output |
|---|---|---|
| Label-based ablation (1,000 test questions) | `python -m scripts.eval_ir --modes dense bm25 hybrid dense_rerank hybrid_rerank --split test` | `results/ir/` |
| RAGAS (Phase 1 vs Phase 2) | `python -m scripts.eval_ragas --modes dense hybrid_rerank --judge official` | `results/ragas/` |
| Latency, 100 consecutive queries | start the API, then `python -m scripts.bench_latency --filtered` | `results/latency/` |
| Live upsert / delete demo | start the API, then `python -m scripts.demo_updates` | `results/updates_demo.json` |
| Fusion / rerank tuning (tuning questions only) | `python -m scripts.tune_fusion --split dev_large [--rerank]` | `results/tuning/` |
| Failure analysis | `python -m scripts.analyze_failures` | `results/analysis/` |
| Metadata tags (no re-index) | `python -m scripts.add_tags [--collection msmarco_500k]` | `results/tags/` |
| Topic-tag accuracy vs an LLM | `python -m scripts.validate_topics --n 150` | `results/tags/topic_validation*` |
| Topic routing effect | `python -m scripts.eval_topics --split dev_large --modes hybrid_rerank --configs none auto1 auto2 oracle` | `results/tags/topic_filter_*` |
| 500k index + RAGAS at 500k | `python -m scripts.build_index --size 500000 --collection msmarco_500k`, then `QDRANT_COLLECTION=msmarco_500k python -m scripts.eval_ragas --modes dense hybrid_rerank --tag 500k` | `results/ragas/*_500k.csv` |
| CPU deployment (100k) | `DEVICE=cpu python -m scripts.eval_ir --split test --tag cpu`, `DEVICE=cpu python -m scripts.eval_ragas --modes hybrid_rerank --tag cpu`, `python -m scripts.check_cpu_contexts` | `results/ir/*_cpu*`, `results/ragas/*_cpu*` |
| Reports (generated locally) | `python -m scripts.make_report` · `python -m scripts.make_report_book` | benchmark report (Markdown) · report book (HTML) with SVG diagrams |
| Unit tests | `python -m pytest -q` | — |

Every result file stores a stamp: time, git commit, config fingerprint, profile and hardware.

**Groq free tier:** each judge model allows 200,000 tokens/day; one 50-question RAGAS run on one mode uses about that.
The script saves after every 5 questions and resumes where it stopped. If `.env` has more than one key
(`GROQ_API_KEY`, `GROQ_API_KEY_2`, …) it moves to the next key when one reaches its limit (`--key 2` pins one key),
and stops safely when every key is used up. AI answers in the UI switch keys the same way.

## Project layout

```
precisionrag/   config · data (clean, dedup, metadata) · encoders · store (Qdrant) · fusion · rerank
                retriever (all modes, timed) · updates (upsert/delete/versioning) · changelog · tags · cache
                · metrics · dashboard · results
api/main.py     FastAPI: /search · /answer · POST /passages · GET|DELETE /passages/{doc_id} · /passages/{id}/history
                · /changes · /demo/company-docs · /health · /meta · /dashboard · serves web/ at /ui/
web/            web UI (plain HTML/CSS/JS, no build step): Search · Phase dashboards · Comparison · Scale · Live Updates
scripts/        build_index · make_eval_sets · eval_ir · eval_ragas · bench_latency · demo_updates · tune_fusion
                add_tags · validate_topics · eval_topics · analyze_failures · check_cpu_contexts
                exp_bm25_avglen · exp_union_rerank · make_report · make_report_book
tests/          48 offline unit tests (API, updates, tags, fusion, metrics, key switching)
eval/           frozen question sets (dev, dev_large, test, ragas, latency)
results/        every measurement (CSV / JSON, stamped)
```

## Tech stack (all free and open-source)
Python 3.11 · MS MARCO v1.1 (HuggingFace `datasets`) · Qdrant 1.19 · `BAAI/bge-small-en-v1.5` (sentence-transformers)
· BM25 (FastEmbed `Qdrant/bm25`) · `cross-encoder/ms-marco-MiniLM-L-6-v2` · FastAPI · plain HTML/CSS/JS web UI
· RAGAS 0.4 with Groq free-tier judges · pandas · matplotlib · pytest.

## Team OutLiers
Prakhar Agnihotri · Ayush Soni · Anantam Singh · Mohit Sharma
