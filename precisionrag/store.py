"""Everything that talks to Qdrant: one collection holding dense vectors, BM25 sparse vectors and metadata."""
from __future__ import annotations

import os
import time
import uuid

from qdrant_client import QdrantClient, models

from .config import get_settings

ID_NAMESPACE = uuid.UUID("6f0e4c1e-6a59-4e4b-9b0e-2b1c6d1a7a10")
DENSE, SPARSE = "dense", "bm25"
PAYLOAD_INDEXES = {"category": models.PayloadSchemaType.KEYWORD,
                   "source": models.PayloadSchemaType.KEYWORD,
                   "doc_id": models.PayloadSchemaType.KEYWORD,
                   "topic": models.PayloadSchemaType.KEYWORD,          # extra tags (precisionrag/tags.py)
                   "source_type": models.PayloadSchemaType.KEYWORD,
                   "corpus": models.PayloadSchemaType.KEYWORD}
FILTER_FIELDS = ("category", "source", "topic", "source_type", "corpus")


def point_id(doc_id: str, chunk: int = 0) -> str:
    """Same document (+chunk) always maps to the same ID, so re-writing it overwrites instead of duplicating."""
    return str(uuid.uuid5(ID_NAMESPACE, f"{doc_id}#{chunk}"))


def to_sparse(emb) -> models.SparseVector:
    return models.SparseVector(indices=emb.indices.tolist(), values=emb.values.tolist())


def make_filter(category=None, source=None, topic=None, source_type=None, corpus=None) -> models.Filter | None:
    """Pre-retrieval filter (applied INSIDE Qdrant's search). Each field: one value, or a list = any of them."""
    must = []
    for key, v in (("category", category), ("source", source), ("topic", topic), ("source_type", source_type),
                   ("corpus", corpus)):
        if v is None:
            continue
        match = models.MatchAny(any=list(v)) if isinstance(v, (list, tuple)) else models.MatchValue(value=v)
        must.append(models.FieldCondition(key=key, match=match))
    return models.Filter(must=must) if must else None


def ensure_payload_indexes(client: QdrantClient, collection: str) -> None:
    """Adds any missing payload index to an existing collection (no re-index of vectors)."""
    have = client.get_collection(collection).payload_schema or {}
    for field, schema in PAYLOAD_INDEXES.items():
        if field not in have:
            client.create_payload_index(collection, field, schema, wait=True)


class Store:
    def __init__(self, collection: str | None = None):
        s = get_settings()
        self.client = QdrantClient(url=s.qdrant_url, timeout=300)
        self.collection = collection or os.getenv("QDRANT_COLLECTION") or s["qdrant"]["collection"]
        self.ef = s["qdrant"]["hnsw_ef_search"]
        self._q = s["qdrant"]

    # ---------------------------------------------------------------- collection lifecycle
    def recreate(self) -> None:
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config={DENSE: models.VectorParams(
                size=get_settings()["models"]["dense_dim"], distance=models.Distance.COSINE,
                hnsw_config=models.HnswConfigDiff(m=self._q["hnsw_m"], ef_construct=self._q["hnsw_ef_construct"]))},
            # IDF is computed by Qdrant from the live collection, so BM25 stays correct after upserts/deletes
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        for field, schema in PAYLOAD_INDEXES.items():
            self.client.create_payload_index(self.collection, field, schema)

    def exists(self) -> bool:
        return self.client.collection_exists(self.collection)

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count

    def wait_ready(self, poll: float = 2.0) -> float:
        """Waits until Qdrant has finished building the HNSW graph; returns seconds waited."""
        t = time.time()
        while self.client.get_collection(self.collection).status != models.CollectionStatus.GREEN:
            time.sleep(poll)
        return time.time() - t

    # ---------------------------------------------------------------- writes
    def upsert(self, payloads: list[dict], dense, sparse, wait: bool = True, batch_size: int = 512) -> None:
        points = [models.PointStruct(id=point_id(p["doc_id"], p.get("chunk", 0)),
                                     vector={DENSE: d.tolist() if hasattr(d, "tolist") else d, SPARSE: to_sparse(sp)},
                                     payload=p)
                  for p, d, sp in zip(payloads, dense, sparse)]
        self.client.upload_points(self.collection, points, batch_size=batch_size, wait=wait)

    def delete_doc(self, doc_id: str) -> None:
        self.client.delete(self.collection, wait=True, points_selector=models.FilterSelector(filter=models.Filter(
            must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))])))

    def get_doc(self, doc_id: str) -> list:
        points, _ = self.client.scroll(self.collection, limit=100, with_payload=True, scroll_filter=models.Filter(
            must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]))
        return points

    # ---------------------------------------------------------------- reads
    def search_dense(self, qvec: list[float], limit: int, flt=None, exact: bool = False):
        return self.client.query_points(
            self.collection, query=qvec, using=DENSE, limit=limit, query_filter=flt, with_payload=True,
            search_params=models.SearchParams(hnsw_ef=self.ef, exact=exact)).points

    def search_sparse(self, qsparse, limit: int, flt=None):
        return self.client.query_points(self.collection, query=to_sparse(qsparse), using=SPARSE, limit=limit,
                                        query_filter=flt, with_payload=True).points

    def search_both(self, qvec: list[float], qsparse, limit: int, flt=None):
        """Dense and BM25 candidates in ONE network round-trip; the same filter is applied inside both searches."""
        d, b = self.client.query_batch_points(self.collection, [
            models.QueryRequest(query=qvec, using=DENSE, limit=limit, filter=flt, with_payload=True,
                                params=models.SearchParams(hnsw_ef=self.ef)),
            models.QueryRequest(query=to_sparse(qsparse), using=SPARSE, limit=limit, filter=flt, with_payload=True),
        ])
        return d.points, b.points
