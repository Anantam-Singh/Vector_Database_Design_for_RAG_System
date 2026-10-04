# The 30 hardest questions a judge can ask — and our answers

Every number below is read from a file in `results/` (named in brackets). Short versions are on the last slides of the
deck; these are the full answers to study. Say "we measured it" only when a file backs it.

> **Live demo pairing:** the web UI serves 100k on the **CPU** profile and 500k on the **GPU** profile. Where an answer
> below quotes 100k numbers from the GPU profile (MRR 0.592, p95 76 ms, RAGAS 0.814 / 0.900), the CPU numbers shown in
> the demo are: MRR 0.581 (+0.048 over dense), p95 140 ms, RAGAS Phase 2 0.809 / 0.900, correct topic filter +0.074.
> The report book (`docs/ppt_report.html`) uses the demo pairing throughout.

---

## A · Technology choices

**1. Why Qdrant and not pgvector, Weaviate or ChromaDB?**
Qdrant stores dense vectors and BM25 sparse vectors in **one collection** and computes BM25's IDF on the server
(`Modifier.IDF`), so hybrid search needs no second engine. Its filtered HNSW applies metadata filters *inside* the graph
search (true pre-filtering), payload indexes make those filters cheap, and `query_batch_points` sends the dense and the
BM25 search in one network round-trip. It also ships as a single native Windows binary, which mattered on a laptop
without Docker. pgvector would need a separate full-text path for BM25; ChromaDB has no native sparse vectors.

**2. Why bge-small-en-v1.5 and not a bigger embedding model?**
384 dimensions, strong retrieval quality for its size, ~1,000 passages/s on the RTX 3050 (100k indexed in 3.2 min), and
still usable on CPU. A bigger embedder would cost index time, RAM and query latency; we spend that budget on the
cross-encoder reranker instead, which is where the measured gain is (+0.060 MRR).

**3. Why not OpenAI embeddings or a hosted vector database?**
The rules allow free tiers only and only Qdrant / Weaviate / ChromaDB / pgvector. Everything runs locally and open-source;
data never leaves the machine — also what a company with confidential documents wants.

**4. Why build your own API and web UI?**
Models load once at start-up; the latency benchmark measures the real end-to-end HTTP path; live updates are shown
"through an API call" as the brief asks; and every number on the dashboard is read from `results/` by `GET /dashboard`
(a unit test fails if the page and the files disagree).

**5. Why no chunking?**
MS MARCO passages are already ~70 words — chunk-sized. The design supports chunks: the point ID is
`UUID5(doc_id#chunk)`, so a long company document can be split and still updated or deleted as one document.

## B · Phase 2: fusion and reranking

**6. Your dense baseline already beats 0.75 precision. What did Phase 2 add?**
On 1,000 held-out questions: MRR@10 0.531 → 0.592, **+0.060 (95% CI +0.039 … +0.081)**, Recall@5 0.799 → 0.857. RAGAS:
precision 0.788 → 0.814, recall 0.860 → 0.900. The target is a floor; Phase 2 must beat *our* baseline. [`results/ir/`,
`results/ragas/`]

**7. Hybrid alone is about equal to dense. Why keep BM25?**
(a) The brief requires hybrid fusion. (b) BM25 measurably widens the candidate pool the reranker reads: the correct
passage is in the top-20 pool 95.8% → 97.8% of the time on the tuning questions, and 89.9% → 93.1% at 500k. (c) On
company data full of product codes, ticket IDs and names, exact matching matters more — and the BM25 weight is one config
value, no re-index. We do **not** claim a ranking gain we could not measure.

**8. Why weighted RRF and not linear (score) fusion?**
RRF combines ranks, so BM25's unbounded scores need no normalisation and the method is stable when the score
distributions shift (new data, filters). Linear min-max fusion looked best on the tuning questions (MRR 0.541) but on the
test set it was not better (0.533 vs our 0.534, Recall@5 0.794 vs 0.804), so we kept RRF.

**9. How did you choose a BM25 weight of 0.1?**
A grid search on 1,000 separate tuning questions (`dev_large`): dense 0.530 · w 0.1 0.530 · w 0.2 0.524 · w 1.0 (plain RRF)
0.494. More BM25 weight measurably hurts on MS MARCO. A weight of 0.2 that won on a 200-question set did not hold on
1,000 — a lesson about small samples. [`results/tuning/dev_large_fusion_grid.csv`]

**10. Why rerank only the top 20?**
Tuned on dev_large: RRF 1 : 1 with depth 10 / 20 / 30 → MRR 0.594 / 0.598 / 0.598. Depth 30 adds cost for no gain. The CPU
profile reranks the top 10: MRR 0.581 (vs 0.592 on GPU) on the same 1,000 test questions; p95 140 ms over 100 consecutive
API queries (same-session run), but 532 ms when background programs used the CPU — CPU latency depends on the machine, so
the GPU profile is the one we stand behind for the 300 ms target.

## C · Evaluation rigour

**11. Did you tune on the test set?**
No. Frozen, disjoint question sets generated with seed 42: dev (200), dev_large (1,000) for tuning, test (1,000) for the
final claim, RAGAS (50), latency (105). When a later check on the test set disagreed with the tuning result (linear
fusion), we kept the existing setting rather than switching.

**12. How reliable is an LLM judge?**
Same judge (`gpt-oss-120b`), temperature 0, the same 50 questions and the same index for every mode — only the retrieval
mode changes. The judge tracks the human labels: Context Recall is 1.000 when the labelled passage is in the top 5 and
0.455 when it is not. A second judge (20b) on 15 questions was inconclusive, so we don't lean on it.

**13. The RAGAS confidence intervals include zero. Is the improvement real?**
50 questions is a small sample (precision +0.026, CI −0.042 … +0.095). Our claim rests on the 1,000-question label metrics,
where the interval (+0.039 … +0.081) is far from zero; RAGAS agrees in direction, and every mode passes the RAGAS targets.

**14. Why not RAGAS's non-LLM context metrics? They're free.**
They compare strings. In our test, a passage "Water boils at 100 °C" scored as correct for a question whose answer is
"Ethanol boils at 78 °C" — exactly the look-alike error we are trying to remove. The LLM judge reads the passage.

**15. The reranker was trained on MS MARCO. Isn't that an unfair advantage?**
It has a home advantage, and we state it. On a company's documents we would evaluate on that company's own questions
first, and fine-tune the reranker (and possibly the embedder) if it underperforms.

## D · Latency and scale

**16. How exactly did you measure p95?**
100 consecutive questions per mode through the HTTP API, result cache off, 5 warm-up queries first, idle machine.
Phase 2: p50 59 ms, p95 76 ms, p99 81 ms; with a metadata filter p95 87 ms. Every response also logs per-stage timings
(median: embed 9 ms, Qdrant 18 ms, rerank 30 ms). [`results/latency/summary.json`]

**17. Why is the first search after a pause slower?**
The laptop GPU drops to its deepest power state after ~10 s idle (P8: 210 MHz core, 405 MHz memory, instead of
1,732 MHz). The reranker's GPU step then takes ~157 ms instead of 12 ms, once. Consecutive use — how the challenge
measures — is unaffected; a server GPU does not idle like this.

**18. What happens at 1 million or 10 million passages?**
Memory grows with the vectors: Qdrant used 0.6 GB at 100k and 2.1 GB at 500k (~4 KB per passage). For 10M: int8 scalar
quantization (~4× less vector RAM, originals on disk for rescoring), sharding across nodes, possibly HNSW m tuning. The
reranker cost does not grow — it always reads 20 candidates.

**19. Why is 100k the official index and not 500k?**
100k fits every machine profile (including the CPU fallback) and is the brief's minimum. We ran the full pipeline at
500k as the bonus: Phase 2 MRR 0.513 (dense 0.450) — the reranker's gain holds (+0.062, CI +0.042 … +0.083) — p95 76 ms (same-session run; 132 ms in a first run under memory pressure),
build well inside the 120-minute limit. RAGAS at 500k (same judge, same 50 questions): Phase 2 precision **0.820**, recall **0.857** — both targets met — while
dense alone falls to 0.754 / 0.850 (hybrid without the reranker: 0.770 / 0.870); Phase 2 vs dense at 500k: precision +0.066 (CI −0.016 … +0.156).

**20. How much memory and time does indexing take?**
100k: 3.2 min, Qdrant 0.6 GB. 500k: 13.3 min on an idle machine (16.8 min on a rebuild that shared the GPU with other
jobs), Qdrant 2.1 GB. Limit: 120 min. [`results/index_build_*.json`, idle 500k run in `results/build_index_500k.log`]

## E · Filters and updates

**21. Pre-filter or post-filter?**
Pre-filter: the filter is part of Qdrant's search request, applied inside both the dense and the BM25 search, on indexed
payload fields. A post-filter (search, then drop non-matching results) could return fewer than 5 passages.

**22. What if the user chooses the wrong filter?**
Measured on 1,000 test questions (hybrid): no filter MRR 0.534; the correct question-type filter 0.544; a wrong one
0.001 — the answer is filtered out. Filters are a precision tool for users who know what they want, so "Any" is the
default. [`results/ir/test_summary_filter-*.json`]

**23. How do you guarantee a search never returns an outdated version?**
The point ID is `UUID5(doc_id)`, so a new version overwrites the old one — dense vector, BM25 vector and metadata in one
write; two versions can never coexist. Every write increments `index_version`, which is part of the result-cache key,
so cached results expire immediately.

**24. What does an update cost? How do deletes work in HNSW?**
70–200 ms per write (embedding + upsert with `wait=true`). Qdrant updates the HNSW graph incrementally, marks deleted
points and cleans segments in the background; BM25 IDF is recomputed from the live collection. No rebuild at any point.

**25. Can I see what changed?**
Yes. A change log (SQLite) records every insert, update, metadata change and delete with the full text of that version.
The Live Updates page asks a question, saves the edit, asks again and shows both answers side by side, a word-level
diff (e.g. 30 → 14 days) and the version history.

## F · Metadata tags, honesty, operations

**26. How do the new topic tags work, and how accurate are they?**
Each of 15 everyday topics (e.g. Animals & Nature, Food & Nutrition, Health & Medicine) is described by 2–4 short
sentences; a passage gets the topic whose description its existing dense vector is closest to. No new model, no
re-embedding: 100,000 passages tagged in 83 s straight from the vectors in Qdrant. Checked against an LLM
(gpt-oss-20b) on 150 random passages: **70% agree on the topic, 82% within our top 2** — most disagreements are between
neighbouring topics (Language vs History, Animals vs Science). Good enough for a user-chosen filter: +0.071 MRR. Source type (government, education, health site,
encyclopedia, Q&A, news…) comes from domain rules. [`results/tags/`]

**27. Does automatic topic routing make search better or faster?**
Measured on the 1,000 held-out test questions (Phase 2): a **correct** topic filter, chosen by a user who knows what
they want, raises MRR from 0.592 to **0.664 (+0.071, CI +0.060 … +0.083)** — about as much as the whole reranker. But
**guessing** the topic automatically *lowers* MRR: −0.069 with the top-2 guesses (−0.154 with the top-1 guess on the
tuning set), because the guess misses for 16–30% of questions and then the answer is filtered out. It is also not faster at 100k. So topic
filters are a user choice with "Any" as default, auto-routing is labelled experimental, and instead the Search page
**suggests** the 3 likely topics as one-click chips — the system proposes, the user decides. [`results/tags/topic_filter_*`]

**28. You used several Groq keys. Isn't that gaming the free tier?**
The plan assumed one key; the real limit turned out to be 200,000 tokens per day per model — about one 50-question RAGAS
run. The team added further keys to finish the runs sooner. Every run uses the same judge model at temperature 0, so the
number of keys does not change any result; with one key the same runs would simply take more days.

**29. Where does your system fail?**
(a) Look-alike passages: for "boiling point of ethanol" the reranker ranks a passage about denatured alcohol's
*flashpoint* first. (b) Label gaps: for "what determines social security disability" the reranker's first passage is
arguably better than the labelled one, but counts as a miss. (c) Questions the corpus can't answer: the reranker scores
are strongly negative (e.g. −8), a signal we could use to say "no relevant passage". 27 of 1,000 test questions fail in
every mode.

**30. Can someone else reproduce your results — and what would you do next?**
Yes: one command per number in the README, 154 packages pinned, frozen question sets regenerated byte-for-byte in a
fresh copy, and the dense baseline reproduced exactly (MRR 0.531). 48 offline unit tests. Next: evaluate and fine-tune
on a company's own questions, quantization for 10M+ passages, per-user access control through the same pre-filter, and
monitoring of per-stage latency and "no relevant passage" rates.
