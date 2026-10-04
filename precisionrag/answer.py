"""Grounded answer generation (bonus): an LLM answers ONLY from the retrieved passages and cites them as [1]..[5].

Retrieval quality is what this project measures; this module just shows the end of a RAG pipeline. It uses a
different Groq model than the RAGAS judge, so it never eats into the judge's daily quota.
"""
from __future__ import annotations

import time

from .config import get_settings
from .groq_quota import KeyPool, available_keys

SYSTEM = ("You answer questions using ONLY the numbered passages provided. Cite the passages you use like [1] or [2]. "
          "If the passages do not contain the answer, say exactly: \"The retrieved passages do not contain the answer.\" "
          "Answer in at most 3 sentences.")


_pool: KeyPool | None = None


def grounded_answer(question: str, passages: list[str]) -> dict:
    global _pool
    s = get_settings()
    if _pool is None:
        if not available_keys():
            return {"answer": None, "error": "GROQ_API_KEY not set"}
        _pool = KeyPool()
    from groq import AuthenticationError, Groq, RateLimitError
    context = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(passages, 1))
    t = time.perf_counter()
    while True:
        try:
            r = Groq(api_key=_pool.key, timeout=30).chat.completions.create(
                model=s["groq"]["answer_model"], temperature=0, max_tokens=600,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": f"Passages:\n{context}\n\nQuestion: {question}"}])
            text = r.choices[0].message.content or ""
            if "</think>" in text:                 # some models prepend their reasoning
                text = text.split("</think>", 1)[1]
            return {"answer": text.strip(), "model": s["groq"]["answer_model"], "key": _pool.label,
                    "ms": round((time.perf_counter() - t) * 1000, 1)}
        except (RateLimitError, AuthenticationError) as e:   # key used up (or revoked) -> try the next one
            if not _pool.rotate(s["groq"]["answer_model"], type(e).__name__):
                return {"answer": None, "error": f"{type(e).__name__} on every Groq key: {str(e)[:200]}"}
        except Exception as e:  # network etc. — the search result is still shown
            return {"answer": None, "error": f"{type(e).__name__}: {str(e)[:200]}"}
