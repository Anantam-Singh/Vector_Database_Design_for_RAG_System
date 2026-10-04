"""Freeze the evaluation question sets ONCE, with a fixed seed, before any tuning.

    dev.jsonl      ~200 questions  → used ONLY to tune settings (fusion weights, rerank depth)
    dev_large.jsonl 1000 questions → also tuning only (added: 200 proved too noisy)
    test.jsonl     ~1000 questions → used ONLY to report label-based metrics
    ragas.jsonl    50 questions    → subset of test that has a real human answer (for RAGAS)
    latency.jsonl  105 questions   → 5 warm-up + 100 timed, disjoint from dev/test

All come from MS MARCO v1.1 validation questions that have at least one human-labelled relevant passage.

    python -m scripts.make_eval_sets
"""
from __future__ import annotations

import json
import random

from precisionrag.config import get_settings
from precisionrag.data import build_corpus, load_msmarco
from precisionrag.results import save_json

NO_ANSWER = "No Answer Present."


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")


def main():
    s = get_settings()
    e = s["eval"]
    ds = load_msmarco()
    corpus = build_corpus(None, ds)          # validation part only: gives stable passage IDs + labels
    val = ds["validation"]

    def row(qi):
        r = val[qi]
        answers = [a for a in r["answers"] if a and a != NO_ANSWER]
        return {"qid": qi, "query": r["query"], "query_type": r["query_type"].lower(),
                "relevant": sorted(corpus.qrels[qi]), "answer": answers[0] if answers else None}

    labelled = sorted(corpus.qrels)
    rng = random.Random(e["seed"])
    rng.shuffle(labelled)
    dev = labelled[:e["dev_size"]]
    test = labelled[e["dev_size"]:e["dev_size"] + e["test_size"]]
    rest = labelled[e["dev_size"] + e["test_size"]:]
    latency = rest[:e["latency_queries"] + e["latency_warmup"]]
    # added after Phase 3 showed 200 dev questions are too noisy for tuning; taken from unused questions so the
    # dev / test / latency sets above stay exactly the same
    n_lat = e["latency_queries"] + e["latency_warmup"]
    dev_large = rest[n_lat:n_lat + e["dev_large_size"]]

    test_rows = [row(qi) for qi in test]
    ragas_rows = [r for r in test_rows if r["answer"]][:e["ragas_size"]]

    out = s.eval_dir
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "dev.jsonl", [row(qi) for qi in dev])
    write_jsonl(out / "dev_large.jsonl", [row(qi) for qi in dev_large])
    write_jsonl(out / "test.jsonl", test_rows)
    write_jsonl(out / "ragas.jsonl", ragas_rows)
    write_jsonl(out / "latency.jsonl", [{"qid": qi, "query": val[qi]["query"],
                                        "query_type": val[qi]["query_type"].lower()} for qi in latency])

    def types(rows):
        c = {}
        for r in rows:
            c[r["query_type"]] = c.get(r["query_type"], 0) + 1
        return dict(sorted(c.items()))

    summary = {"labelled_validation_queries": len(corpus.qrels), "seed": e["seed"],
               "dev": len(dev), "dev_large": len(dev_large), "test": len(test_rows), "ragas": len(ragas_rows),
               "latency": len(latency), "overlap_dev_test": len(set(dev) & set(test)),
               "overlap_devlarge_test": len(set(dev_large) & set(test)),
               "test_query_types": types(test_rows),
               "avg_relevant_per_test_query": round(sum(len(r["relevant"]) for r in test_rows) / len(test_rows), 2)}
    save_json("eval_sets.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
