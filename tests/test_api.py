"""API tests with Qdrant and the models replaced by mocks — fast, offline, no Groq calls.
TestClient is used without `with`, so the real start-up (model loading) never runs."""
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.main import app, state
from precisionrag import updates
from precisionrag.cache import ResultCache

client = TestClient(app)


def R():
    return state["retrievers"]["100k"]


@pytest.fixture(autouse=True)
def fake_state():
    r = MagicMock()                  # no spec: Retriever.store is set in __init__, so spec=Retriever would hide it
    r.store.count.return_value = 100
    r.store.collection = "test_collection"
    r.search.return_value.to_dict.return_value = {"hits": [{"text": "passage one"}, {"text": "passage two"}]}
    state.update(retrievers={"100k": r}, updater=MagicMock(), cache=ResultCache())   # real cache: tests the caching logic
    yield state
    state.clear()


def test_health():
    data = client.get("/health").json()
    assert data["status"] == "ok" and data["points"] == 100 and data["collection"] == "test_collection"
    assert {"profile", "config_fingerprint", "index_version", "cache"} <= data.keys()


def test_web_ui_is_served():
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307) and r.headers["location"] == "/ui/"
    page = client.get("/ui/")
    assert page.status_code == 200 and "PrecisionRAG" in page.text and page.headers["cache-control"] == "no-cache"


def test_dashboard_numbers_match_saved_results():
    """The web dashboard must show exactly what results/ holds (official RAGAS summary written by eval_ragas)."""
    import json
    from precisionrag.config import get_settings
    d = client.get("/dashboard").json()
    saved = json.loads((get_settings().results_dir / "ragas" / "summary_gpt-oss-120b.json").read_text())["data"]
    for mode in ("dense", "hybrid_rerank"):
        assert abs(d["ragas"]["modes"][mode]["context_precision"] - saved[mode]["context_precision"]) < 1e-3
        assert d["ragas"]["modes"][mode]["n"] == saved[mode]["n_questions"]
    assert d["ir"]["dense"]["n"] == 1000 and "vs_dense" in d["ir"]["hybrid_rerank"]
    assert d["latency"]["modes"]["hybrid_rerank"]["pass_p95_lt_300"] is True


def test_meta_lists_facets():
    f = MagicMock()
    f.hits = [MagicMock(value="description", count=10)]
    R().store.client.facet.return_value = f
    data = client.get("/meta").json()
    assert data["category"] == [{"value": "description", "count": 10}] and "source" in data


def test_search_passes_arguments_and_filters():
    r = client.get("/search", params={"q": "what is rag", "mode": "dense", "k": 3, "category": "numeric"})
    assert r.status_code == 200 and r.json()["cached"] is False and len(r.json()["hits"]) == 2
    kw = R().search.call_args.kwargs
    assert R().search.call_args.args == ("what is rag",)
    got = (kw["mode"], kw["k"], kw["category"], kw["source"], kw["topic"], kw["auto_topic"])
    assert got == ("dense", 3, "numeric", None, None, 0)


def test_new_tag_filters_reach_the_retriever():
    client.get("/search", params={"q": "x", "topic": "Food & Nutrition,Animals & Nature", "source_type": "government",
                                  "corpus": "internal", "auto_topic": 0})
    kw = R().search.call_args.kwargs
    assert kw["topic"] == ["Food & Nutrition", "Animals & Nature"]      # comma list = any of these topics
    assert (kw["source_type"], kw["corpus"]) == ("government", "internal")


def test_unknown_index_is_400():
    assert client.get("/search", params={"q": "x", "index": "500k"}).status_code == 400


def test_empty_filter_means_no_filter():
    client.get("/search", params={"q": "x", "category": "", "source": ""})
    assert R().search.call_args.kwargs["category"] is None
    assert R().search.call_args.kwargs["source"] is None


def test_search_second_call_is_cached():
    first = client.get("/search", params={"q": "what is rag"}).json()
    second = client.get("/search", params={"q": "  What is RAG "}).json()   # same question, other spacing/case
    assert first["cached"] is False and second["cached"] is True
    assert R().search.call_count == 1


def test_no_cache_always_searches():
    for _ in range(2):
        assert client.get("/search", params={"q": "x", "no_cache": True}).json()["cached"] is False
    assert R().search.call_count == 2


def test_write_invalidates_cache():
    state["updater"].upsert.side_effect = lambda *a, **kw: {"status": "inserted", "index_version": updates._bump()}
    client.get("/search", params={"q": "x"})
    client.post("/passages", json={"text": "new policy"})
    assert client.get("/search", params={"q": "x"}).json()["cached"] is False   # old result not served
    assert R().search.call_count == 2


def test_search_unknown_mode_is_400():
    r = client.get("/search", params={"q": "x", "mode": "nope"})
    assert r.status_code == 400 and "mode must be one of" in r.json()["detail"]


@pytest.mark.parametrize("params", [{"q": ""}, {"q": "x", "k": 0}, {"q": "x", "k": 51}, {}])
def test_search_bad_input_is_422(params):
    assert client.get("/search", params=params).status_code == 422
    R().search.assert_not_called()


def test_answer_uses_retrieved_passages():
    with patch("precisionrag.answer.grounded_answer", return_value={"answer": "It is X [1]."}) as ga:
        data = client.get("/answer", params={"q": "what is x"}).json()
    ga.assert_called_once_with("what is x", ["passage one", "passage two"])
    assert data["answer"] == "It is X [1]." and len(data["search"]["hits"]) == 2


def test_upsert_passes_fields_with_defaults():
    state["updater"].upsert.return_value = {"status": "inserted", "doc_id": "d1"}
    r = client.post("/passages", json={"text": "New passage", "source": "hr"})
    assert r.status_code == 200 and r.json() == {"status": "inserted", "doc_id": "d1"}
    state["updater"].upsert.assert_called_once_with("New passage", doc_id=None, source="hr", category="description",
                                                    corpus="internal")


def test_upsert_rejects_empty_text():
    assert client.post("/passages", json={"text": ""}).status_code == 422
    state["updater"].upsert.assert_not_called()


def test_get_passage():
    state["updater"].get.return_value = {"text": "found", "doc_id": "d1"}
    assert client.get("/passages/d1").json()["text"] == "found"
    state["updater"].get.return_value = None
    assert client.get("/passages/missing").status_code == 404


def test_delete_passage():
    state["updater"].delete.return_value = {"status": "deleted", "doc_id": "d1"}
    assert client.delete("/passages/d1").json()["status"] == "deleted"
    state["updater"].delete.return_value = {"status": "not_found", "doc_id": "missing"}
    assert client.delete("/passages/missing").status_code == 404
