"""Live-update logic (insert / new version / unchanged / metadata-only / delete) with Qdrant and the models mocked."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from precisionrag import updates
from precisionrag.data import clean, content_hash
from precisionrag.store import point_id


@pytest.fixture
def updater(monkeypatch):
    with patch("precisionrag.updates.get_dense"), patch("precisionrag.updates.get_sparse"):
        u = updates.Updater(store=MagicMock())
    u.store.get_doc.return_value = []          # document not in the index yet
    monkeypatch.setattr(updates, "tags_for", lambda v, src, corpus: {"topic": "Science", "topic_conf": 0.1,
                                                                     "source_type": "commercial", "corpus": corpus})
    monkeypatch.setattr(updates, "changelog", MagicMock())     # never write to the real history file
    return u


def test_every_write_is_logged_with_its_text(updater):
    updater.upsert("Refunds within 30 days.", doc_id="d1")
    stored(updater, "Refunds within 30 days.")
    res = updater.upsert("Refunds within 14 days.", doc_id="d1")
    assert res["previous_text"] == "Refunds within 30 days." and res["previous_version"] == 1   # for the UI diff
    actions = [c.args[2] for c in updates.changelog.record.call_args_list]
    assert actions == ["inserted", "updated"]
    assert updates.changelog.record.call_args.args[4] == "Refunds within 14 days."


def test_new_documents_are_tagged(updater):
    updater.upsert("some text", doc_id="d9", source="irs.gov", corpus="internal")
    payload = updater.store.upsert.call_args.args[0][0]
    assert payload["topic"] == "Science" and payload["corpus"] == "internal"


def stored(u, text, source="manual", category="description", version=1):
    u.store.get_doc.return_value = [SimpleNamespace(payload={
        "doc_id": "d1", "passage_id": None, "content_hash": content_hash(clean(text)),
        "source": source, "category": category, "version": version, "text": clean(text)})]


def test_insert_new_document(updater):
    v = updates.index_version()
    res = updater.upsert("Remote work is allowed 2 days a week.", doc_id="d1")
    assert res["status"] == "inserted" and res["version"] == 1 and res["index_version"] == v + 1
    payload = updater.store.upsert.call_args.args[0][0]
    assert payload["doc_id"] == "d1" and payload["text"] == "Remote work is allowed 2 days a week."


def test_insert_without_doc_id_generates_one(updater):
    assert updater.upsert("some text")["doc_id"].startswith("user-")


def test_same_text_is_unchanged_and_not_reembedded(updater):
    stored(updater, "Duplicate   text")
    v = updates.index_version()
    res = updater.upsert("Duplicate text", doc_id="d1")       # same after cleaning
    assert res["status"] == "unchanged" and updates.index_version() == v
    updater.store.upsert.assert_not_called()
    updater.dense.encode_passages.assert_not_called()


def test_new_text_makes_new_version(updater):
    stored(updater, "Remote work: 2 days.", version=1)
    res = updater.upsert("Remote work: 3 days.", doc_id="d1")
    assert res["status"] == "updated" and res["version"] == 2
    assert updater.store.upsert.call_args.args[0][0]["doc_id"] == "d1"   # same doc_id -> same point ID, overwritten


def test_metadata_only_change_skips_embedding(updater):
    stored(updater, "Remote work: 2 days.", source="hr")
    res = updater.upsert("Remote work: 2 days.", doc_id="d1", source="legal")
    assert res["status"] == "metadata_updated"
    kw = updater.store.client.set_payload.call_args.kwargs
    assert kw["payload"]["source"] == "legal" and kw["points"] == [point_id("d1")]
    updater.dense.encode_passages.assert_not_called()
    updater.store.upsert.assert_not_called()


def test_empty_text_is_rejected(updater):
    with pytest.raises(ValueError):
        updater.upsert("   ")


def test_delete(updater):
    assert updater.delete("missing")["status"] == "not_found"
    updater.store.delete_doc.assert_not_called()
    stored(updater, "x")
    v = updates.index_version()
    res = updater.delete("d1")
    assert res["status"] == "deleted" and res["index_version"] == v + 1
    updater.store.delete_doc.assert_called_once_with("d1")
