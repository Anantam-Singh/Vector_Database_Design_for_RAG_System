"""Text → vectors. Dense = bge-small (meaning). Sparse = BM25 (exact words). Models load once per process."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import get_settings


class DenseEncoder:
    def __init__(self, device: str | None = None):
        s = get_settings()
        from sentence_transformers import SentenceTransformer
        self.device = device or s.device
        profile = s["profiles"]["gpu" if self.device == "cuda" else "cpu"]
        self.model = SentenceTransformer(s["models"]["dense"], device=self.device)
        if self.device == "cuda" and profile["fp16"]:
            self.model.half()
        self.prefix = s["models"]["query_prefix"]
        self.batch = profile["embed_batch"]

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        # passages get NO prefix; vectors are L2-normalised so cosine == dot product
        return self.model.encode(texts, batch_size=self.batch, normalize_embeddings=True,
                                 convert_to_numpy=True).astype(np.float32)

    def encode_query(self, query: str) -> list[float]:
        return self.model.encode(self.prefix + query, normalize_embeddings=True).astype(np.float32).tolist()


class SparseEncoder:
    def __init__(self, **bm25_overrides):
        s = get_settings()
        from fastembed import SparseTextEmbedding
        name = s["models"]["sparse"]
        params = {**s["models"]["bm25"], **bm25_overrides}   # k, b, avg_len — only passage weights depend on them
        snapshots = Path(s.models_dir) / ("models--" + name.replace("/", "--")) / "snapshots"
        if snapshots.is_dir() and any(snapshots.iterdir()):
            # FastEmbed's offline cache check expects files the BM25 repo doesn't ship; point at the snapshot directly
            self.model = SparseTextEmbedding(name, cache_dir=str(s.models_dir),
                                             specific_model_path=str(next(snapshots.iterdir())), **params)
        else:
            self.model = SparseTextEmbedding(name, cache_dir=str(s.models_dir), **params)  # first run: downloads
        self.params = params

    def encode_passages(self, texts: list[str]) -> list:
        return list(self.model.embed(texts, batch_size=256))

    def encode_query(self, query: str):
        return next(self.model.query_embed(query))


@lru_cache(maxsize=2)
def get_dense(device: str | None = None) -> DenseEncoder:
    """One model per device (None = the machine's default device), loaded once per process."""
    return DenseEncoder(device)


@lru_cache(maxsize=1)
def get_sparse() -> SparseEncoder:
    return SparseEncoder()
