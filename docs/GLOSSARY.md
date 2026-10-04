# Glossary — every technical word in one line

| Word | Meaning |
|---|---|
| **RAG** | Retrieval-Augmented Generation: find relevant passages first, then let an LLM answer using them. |
| **Retrieval** | Finding the passages most likely to answer a question. |
| **Passage** | A short piece of text (MS MARCO passages are ~70 words). |
| **MS MARCO** | Microsoft's public dataset of real Bing questions, candidate passages, human labels and answers. |
| **Corpus / index** | All passages we search over, stored so they can be searched fast. |
| **Embedding / dense vector** | A list of numbers (384 here) that represents the *meaning* of a text. |
| **bge-small-en-v1.5** | The open-source model that turns text into those 384 numbers. |
| **Cosine similarity** | How closely two vectors point in the same direction; 1 = identical meaning. |
| **BM25** | Classic keyword scoring: rewards passages containing the question's words, especially rare ones. |
| **Sparse vector** | A vector that only stores the words that appear (most entries are zero) — how BM25 is stored. |
| **IDF** | Inverse document frequency: rare words count more than common words. |
| **Hybrid search** | Combining meaning search (dense) with keyword search (BM25). |
| **Fusion** | Merging two ranked lists into one. |
| **RRF** | Reciprocal Rank Fusion: score = Σ weight ÷ (60 + position). Uses positions, not raw scores. |
| **Weighted RRF** | RRF where one list counts more than the other. |
| **Linear fusion** | Normalise each list's scores to 0–1, then add them with weights. |
| **Reranker / cross-encoder** | A model that reads question and passage *together* and scores the match — accurate but slow, so used only on a few candidates. |
| **Bi-encoder** | A model that encodes question and passage *separately* (our dense model) — fast, less precise. |
| **Candidate pool** | The passages handed from one stage to the next (e.g. the top-20 the reranker sees). |
| **Candidate recall** | How often the correct passage is anywhere in that pool. |
| **Top-k** | The first k results (we return the top 5). |
| **Vector database** | A database that finds items by vector similarity. |
| **Qdrant** | The open-source vector database we use. |
| **Collection** | Qdrant's name for one table of vectors + metadata. |
| **HNSW** | A graph index that finds nearest vectors quickly without checking every passage. |
| **ANN** | Approximate nearest neighbour search — fast, very slightly less exact. |
| **Payload / metadata** | Extra fields stored with each vector: source, category, version… |
| **Pre-retrieval filter** | The database only searches passages that match the filter (what the rules require). |
| **Post-retrieval filter** | Filtering after searching — can leave fewer results; not allowed for FR-4. |
| **Upsert** | Insert if new, replace if it already exists. |
| **Stale data** | An outdated version that should no longer appear. |
| **Content hash** | A fingerprint of the text; same text → same fingerprint. |
| **Deduplication** | Storing identical passages only once. |
| **Ground truth / labels** | Human-marked correct passages we score against. |
| **Dev / test split** | Separate question sets: dev for tuning, test only for final reporting. |
| **Ablation** | Turning components on one at a time to see what each one contributes. |
| **MRR@10** | 1 ÷ position of the first correct passage (0 if not in the top 10), averaged. MS MARCO's official metric. |
| **nDCG@10** | Ranking quality that rewards correct passages higher up. |
| **Hit@5** | Share of questions with at least one correct passage in the top 5. |
| **Recall@5** | Share of all correct passages that appear in the top 5. |
| **Precision@5** | Share of the top 5 that is correct (max ≈ 0.2 here, since most questions have one labelled passage). |
| **RAGAS** | A framework that uses an LLM as a judge to score retrieval and answers. |
| **Context Precision (RAGAS)** | Are the relevant retrieved passages ranked at the top? |
| **Context Recall (RAGAS)** | Do the retrieved passages contain the facts needed for the reference answer? |
| **LLM judge** | A language model that grades results (here: free Groq models at temperature 0). |
| **Confidence interval (95%)** | The range where the true effect most likely lies; if it includes 0, the effect may be luck. |
| **Paired bootstrap** | Re-sampling the same questions thousands of times to estimate that range. |
| **Latency** | Time from sending a question to getting results. |
| **p50 / p95 / p99** | The time that 50% / 95% / 99% of queries finish within. p95 = the "slow tail" users notice. |
| **Warm-up** | First queries after start-up are slower (models load into memory); we exclude and report them separately. |
| **Cache** | Remembering recent answers; ours is invalidated on every index change. |
| **API** | A program other programs call over HTTP (FastAPI here). |
| **fp16** | Half-precision numbers on the GPU — faster, same results for our purposes. |
| **Rate limit** | A cap on how much of a free service you may use per minute or per day. |
| **Profile (GPU / CPU)** | A named set of settings picked automatically for the hardware. |
| **Mock** | A fake object used in tests instead of the real database or model, so tests run fast and offline. |
| **Key pool / key switching** | Using the next API key from `.env` when the current one reaches its free-tier limit. |
| **Static files** | Files sent to the browser as they are (HTML, CSS, JS); our web UI is served this way by the API. |
| **Cache hit** | An answer served from memory instead of being computed again. |
| **GPU power state (P0 … P8)** | P0 = full speed, P8 = deepest power saving; a laptop GPU drops to P8 after a few idle seconds. |
| **Double rounding** | Rounding twice (e.g. to 4 then 3 decimals) can differ from rounding once: 0.81447 → 0.8145 → 0.815. |
| **n.s. (not significant)** | The 95% confidence interval includes zero, so the difference could be luck. |
| **Zero-shot classification** | Sorting items into categories without training examples — here by similarity to a short description of each topic. |
| **Topic routing** | Guessing a question's likely topics and searching only passages tagged with them. |
| **Change log / audit trail** | A record of every change (what, when, which version), kept separately from the live index. |
| **Word diff** | Highlighting exactly which words were removed and added between two versions of a text. |
