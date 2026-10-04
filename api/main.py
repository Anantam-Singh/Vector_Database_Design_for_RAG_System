"""PrecisionRAG HTTP API. Models load once at start-up; every response includes per-stage timings.

    .venv\\Scripts\\python -m uvicorn api.main:app --host 127.0.0.1 --port 8765
    web UI: http://127.0.0.1:8765/   ·   docs: http://127.0.0.1:8765/docs
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from precisionrag import changelog
from precisionrag.cache import ResultCache
from precisionrag.config import get_settings
from precisionrag.retriever import MODES, Retriever
from precisionrag.store import Store
from precisionrag.updates import Updater, index_version

state: dict = {}
INDEXES = {"100k": None, "500k": "msmarco_500k"}   # None = the official collection from config.yaml

# Fictional company documents for the live-update demo ("Acme Retail" is made up). Clearly tagged corpus=internal.
DEMO_DOCS = [
    ("acme-refund-policy", "Acme Retail refund policy: customers can return any item within 30 days of delivery for "
     "a full refund to the original payment method. Refunds are processed within 5 business days after the "
     "returned item arrives at our warehouse."),
    ("acme-shipping-policy", "Acme Retail shipping policy: standard shipping takes 3 to 5 business days and is free "
     "on orders over $50. Express shipping takes 1 to 2 business days and costs $15."),
    ("acme-warranty-policy", "Acme Retail warranty: all electronics come with a 1-year limited warranty that covers "
     "manufacturing defects. Accidental damage and water damage are not covered."),
    ("acme-remote-work-policy", "Acme Retail HR policy: employees may work remotely 2 days per week with manager "
     "approval. Remote days must be logged in the HR portal one week in advance."),
    ("acme-leave-policy", "Acme Retail HR policy: full-time employees receive 20 days of paid annual leave per year. "
     "Unused leave of up to 5 days can be carried over to the next year."),
]
DEMO_SOURCE = "acme-policies.internal"


class PassageIn(BaseModel):
    text: str = Field(..., min_length=1)
    doc_id: str | None = Field(None, description="same doc_id = new version of that document")
    source: str = "manual"
    category: str = "description"
    corpus: str = Field("internal", description="'internal' for company documents, 'web' for public pages")


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    devices = (s.raw.get("serving") or {}).get("devices", {})
    dev = lambda k: devices.get(k) if devices.get(k) in ("cpu", "cuda") and (devices.get(k) == "cpu" or s.device == "cuda") else None
    main = Retriever(device=dev("100k"))          # config serving.devices: 100k → CPU profile by default
    main.warmup()
    state["retrievers"] = {"100k": main}
    big = Store(INDEXES["500k"])
    if big.exists() and big.count() > 0:          # the 500k scale index, if it has been built
        state["retrievers"]["500k"] = Retriever(big, device=dev("500k"))
        state["retrievers"]["500k"].warmup()
    state["updater"] = Updater(main.store)
    state["cache"] = ResultCache()
    from precisionrag.dashboard import dashboard
    dashboard()   # pre-compute the web dashboard BEFORE serving, so it never competes with a search for the CPU
    yield
    state.clear()


app = FastAPI(title="PrecisionRAG", version="0.2.0", lifespan=lifespan)


def retriever(index: str = "100k") -> Retriever:
    r = state["retrievers"].get(index)
    if r is None:
        raise HTTPException(400, f"index '{index}' is not available (built: {sorted(state['retrievers'])})")
    return r


WEB = Path(__file__).resolve().parents[1] / "web"
app.mount("/ui", StaticFiles(directory=WEB, html=True), name="ui")


@app.middleware("http")
async def no_stale_ui(request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/ui"):
        response.headers["Cache-Control"] = "no-cache"   # browsers re-check, so an edited page is never stale
    return response


@app.get("/", include_in_schema=False)
def home():
    return RedirectResponse("/ui/")


@app.get("/dashboard")
def dashboard_data():
    """Every measured number for the web dashboard, read from results/."""
    from precisionrag.dashboard import dashboard
    return dashboard()


@app.get("/health")
def health():
    s = get_settings()
    r = retriever()
    return {"status": "ok", "points": r.store.count(), "collection": r.store.collection,
            "indexes": {k: v.store.count() for k, v in state["retrievers"].items()},
            "serving": {k: {"device": v.device, "profile": v.profile_name, "rerank_depth": v.profile["rerank_depth"]}
                        for k, v in state["retrievers"].items()},
            "profile": s.profile_name, "device": s.device, "config_fingerprint": s.fingerprint,
            "index_version": index_version(), "cache": state["cache"].stats()}


@app.get("/meta")
def meta(index: str = "100k"):
    """Filter values for the UI (from Qdrant's facet counts on indexed metadata)."""
    store = retriever(index).store
    out = {}
    for key, limit in (("category", 20), ("source", 30), ("topic", 30), ("source_type", 20), ("corpus", 5)):
        try:
            f = store.client.facet(store.collection, key=key, limit=limit)
            out[key] = [{"value": h.value, "count": h.count} for h in f.hits]
        except Exception:     # a collection without that tag yet
            out[key] = []
    return out


def _topics(topic: str | None):
    if not topic:
        return None
    parts = [t.strip() for t in topic.split(",") if t.strip()]
    return parts if len(parts) > 1 else parts[0]


@app.get("/search")
def search(q: str = Query(..., min_length=1), mode: str = "hybrid_rerank", k: int = Query(5, ge=1, le=50),
           category: str | None = None, source: str | None = None,
           topic: str | None = Query(None, description="one topic, or several separated by commas (= any of them)"),
           source_type: str | None = None, corpus: str | None = None,
           auto_topic: int = Query(0, ge=0, le=3, description="search only the question's N most likely topics"),
           index: str = "100k", no_cache: bool = False):
    if mode not in MODES:
        raise HTTPException(400, f"mode must be one of {MODES}")
    r = retriever(index)
    cache: ResultCache = state["cache"]
    key = cache.key(q, mode, k, category or None, source or None, index_version(), topic or None, source_type or None,
                    corpus or None, auto_topic, index)
    if not no_cache and (hit := cache.get(key)) is not None:
        return {**hit, "cached": True}
    res = r.search(q, mode=mode, k=k, category=category or None, source=source or None, topic=_topics(topic),
                   source_type=source_type or None, corpus=corpus or None, auto_topic=auto_topic).to_dict()
    res["index"] = index
    if not no_cache:
        cache.put(key, res)
    return {**res, "cached": False}


@app.get("/answer")
def answer(q: str = Query(..., min_length=1), mode: str = "hybrid_rerank", category: str | None = None,
           source: str | None = None, topic: str | None = None, source_type: str | None = None,
           corpus: str | None = None, auto_topic: int = 0, index: str = "100k"):
    """Search, then let an LLM answer ONLY from the top-5 passages, with [n] citations (bonus feature)."""
    from precisionrag.answer import grounded_answer
    res = search(q=q, mode=mode, k=5, category=category, source=source, topic=topic, source_type=source_type,
                 corpus=corpus, auto_topic=auto_topic, index=index, no_cache=True)
    return {**grounded_answer(q, [h["text"] for h in res["hits"]]), "search": res}


@app.post("/passages")
def upsert_passage(p: PassageIn):
    """Insert a new document, or a new version of an existing doc_id. No index rebuild."""
    return state["updater"].upsert(p.text, doc_id=p.doc_id, source=p.source, category=p.category, corpus=p.corpus)


@app.get("/passages/{doc_id}")
def get_passage(doc_id: str):
    doc = state["updater"].get(doc_id)
    if not doc:
        raise HTTPException(404, f"{doc_id} not found")
    return doc


@app.get("/passages/{doc_id}/history")
def passage_history(doc_id: str):
    """Every version of a document (Qdrant itself only keeps the latest)."""
    return {"doc_id": doc_id, "versions": changelog.history(retriever().store.collection, doc_id)}


@app.get("/changes")
def changes(limit: int = Query(30, ge=1, le=200)):
    """The most recent inserts / updates / deletes."""
    return {"changes": changelog.recent(retriever().store.collection, limit)}


@app.delete("/passages/{doc_id}")
def delete_passage(doc_id: str):
    res = state["updater"].delete(doc_id)
    if res["status"] == "not_found":
        raise HTTPException(404, f"{doc_id} not found")
    return res


@app.post("/demo/company-docs")
def load_demo_docs():
    """Loads 5 fictional 'Acme Retail' policy documents (corpus=internal) for the live-update demo."""
    up = state["updater"]
    return {"loaded": [up.upsert(text, doc_id=d, source=DEMO_SOURCE, category="description", corpus="internal")
                       for d, text in DEMO_DOCS]}


@app.delete("/demo/company-docs")
def remove_demo_docs():
    """Removes the demo documents again (run this before re-running any evaluation)."""
    up = state["updater"]
    return {"removed": [up.delete(d) for d, _ in DEMO_DOCS]}
