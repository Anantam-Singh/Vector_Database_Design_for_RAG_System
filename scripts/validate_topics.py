"""How accurate are the automatic topic tags? An LLM (Groq, free tier) labels a random sample of passages with one
topic from the same list; we report how often our zero-shot tag agrees (top-1) and how often the LLM's topic is in
our top-2.

    python -m scripts.validate_topics --n 150
"""
from __future__ import annotations

import argparse
import random
import re
import time

import numpy as np

from precisionrag.config import get_settings
from precisionrag.groq_quota import KeyPool
from precisionrag.results import save_csv, save_json
from precisionrag.store import DENSE, Store
from precisionrag.tags import TOPIC_NAMES, topic_scores

PROMPT = ("Choose the ONE topic that best describes this passage. Answer with the topic name only, exactly as "
          "written.\nTopics: {topics}\n\nPassage: {text}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--model", default=None, help="default: config groq.dev_model")
    args = ap.parse_args()
    s = get_settings()
    model = args.model or s["groq"]["dev_model"]
    store = Store()
    total = store.count()
    rng = random.Random(42)
    want = sorted(rng.sample(range(total), args.n))
    # passage_id → point: msmarco passages have doc_id "msmarco-<passage_id>"
    from qdrant_client import models
    pts = store.client.scroll(store.collection, limit=args.n, with_vectors=[DENSE], with_payload=["text", "topic"],
                              scroll_filter=models.Filter(must=[models.FieldCondition(
                                  key="doc_id", match=models.MatchAny(any=[f"msmarco-{i}" for i in want]))]))[0]
    sc = topic_scores(np.array([p.vector[DENSE] for p in pts], dtype=np.float32))
    top2 = [[TOPIC_NAMES[i] for i in np.argsort(-row)[:2]] for row in sc]
    from groq import AuthenticationError, Groq, RateLimitError
    pool = KeyPool()
    rows = []
    for p, t2 in zip(pts, top2):
        text = p.payload["text"][:1200]
        for attempt in range(6):
            try:
                r = Groq(api_key=pool.key, timeout=60).chat.completions.create(
                    model=model, temperature=0, max_tokens=400, reasoning_effort="low",
                    messages=[{"role": "user", "content": PROMPT.format(topics="; ".join(TOPIC_NAMES), text=text)}])
                out = (r.choices[0].message.content or "").strip()
                break
            except RateLimitError as e:
                if "per day" in str(e) or attempt == 5:          # daily limit → next key; per-minute → wait
                    if not pool.rotate(model, "daily limit"):
                        raise
                else:
                    time.sleep(15)
            except AuthenticationError:
                if not pool.rotate(model, "invalid key"):
                    raise
        llm = next((t for t in TOPIC_NAMES if t.lower() in out.lower()), None) or re.sub(r"\s+", " ", out)[:40]
        rows.append({"doc_id": f"msmarco-{p.payload.get('passage_id', '')}", "ours": p.payload.get("topic"),
                     "ours_top2": "|".join(t2), "llm": llm, "agree": llm == p.payload.get("topic"),
                     "llm_in_top2": llm in t2, "text": text[:160]})
        time.sleep(2.5)   # stay under 8,000 tokens/minute
    acc = float(np.mean([r["agree"] for r in rows]))
    acc2 = float(np.mean([r["llm_in_top2"] for r in rows]))
    import pandas as pd
    save_csv("tags/topic_validation.csv", pd.DataFrame(rows))
    save_json("tags/topic_validation_summary.json", {"n": len(rows), "judge": model, "top1_agreement": round(acc, 4),
                                                      "llm_topic_in_our_top2": round(acc2, 4)})
    print(f"n={len(rows)} | top-1 agreement {acc:.1%} | LLM topic in our top-2 {acc2:.1%} | judge {model}")


if __name__ == "__main__":
    main()
