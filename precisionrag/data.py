"""MS MARCO v1.1 → de-duplicated passages with metadata, plus human relevance labels (qrels).

Corpus order is deterministic: all validation passages first (so every evaluation question's correct passage is
indexed), then train passages until the target size. Passage IDs are therefore stable across runs and sizes.
"""
from __future__ import annotations

import hashlib
import html
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib.parse import urlparse

from .config import get_settings

INGEST_TIME = datetime.now(timezone.utc).isoformat(timespec="seconds")


_SPACES = re.compile(r"\s+")
_BARE_ENTITY = re.compile(r"&(deg|amp|quot|nbsp|lt|gt|#\d+)(?![;\w])")   # e.g. "175&deg F" (missing ';')


def clean(text: str) -> str:
    """Decode HTML entities (also ones missing their ';') and collapse whitespace. ~28% of raw passages need it."""
    text = _BARE_ENTITY.sub(lambda m: f"&{m.group(1)};", text)
    return _SPACES.sub(" ", html.unescape(text)).strip()


def content_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def domain(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.") or "unknown"


@dataclass
class Corpus:
    texts: list[str] = field(default_factory=list)
    meta: list[dict] = field(default_factory=list)          # payload per passage (without text)
    qrels: dict[int, set[int]] = field(default_factory=dict)  # validation query index -> relevant passage ids
    n_validation: int = 0
    _by_hash: dict[str, int] = field(default_factory=dict, repr=False)

    def add(self, text: str, url: str, category: str) -> int:
        text = clean(text)
        h = content_hash(text)
        pid = self._by_hash.get(h)
        if pid is None:                      # exact duplicates are stored once
            pid = len(self.texts)
            self._by_hash[h] = pid
            self.texts.append(text)
            self.meta.append({
                "passage_id": pid,
                "doc_id": f"msmarco-{pid}",
                "source": domain(url),
                "category": category.lower(),  # query type of the question this passage came with
                "version": 1,
                "updated_at": INGEST_TIME,
                "content_hash": h,
            })
        return pid


def load_msmarco():
    get_settings()  # sets HF_HOME before the datasets library is imported
    from datasets import load_dataset
    return load_dataset("microsoft/ms_marco", "v1.1")


def build_corpus(size: int | None, ds=None) -> Corpus:
    """size=None builds only the validation part (enough for labels and evaluation question sets)."""
    ds = ds or load_msmarco()
    corpus = Corpus()
    for qi, row in enumerate(ds["validation"]):
        p = row["passages"]
        pids = [corpus.add(t, u, row["query_type"]) for t, u in zip(p["passage_text"], p["url"])]
        relevant = {pid for pid, sel in zip(pids, p["is_selected"]) if sel}
        if relevant:
            corpus.qrels[qi] = relevant
    corpus.n_validation = len(corpus.texts)
    if size is None:
        corpus._by_hash.clear()
        return corpus
    if size < corpus.n_validation:
        raise ValueError(f"size {size:,} is smaller than the {corpus.n_validation:,} validation passages "
                         "needed for evaluation labels")
    for row in ds["train"]:
        if len(corpus.texts) >= size:
            break
        p = row["passages"]
        for t, u in zip(p["passage_text"], p["url"]):
            if len(corpus.texts) >= size:
                break
            corpus.add(t, u, row["query_type"])
    corpus._by_hash.clear()
    return corpus
