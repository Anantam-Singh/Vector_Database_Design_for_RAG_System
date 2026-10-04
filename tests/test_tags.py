"""Metadata tags and filters (no models, no Qdrant)."""
from qdrant_client import models

from precisionrag.store import make_filter
from precisionrag.tags import SOURCE_TYPES, TOPIC_NAMES, source_type


def test_source_type_rules():
    cases = {"cdc.gov": "government", "gov.uk": "government", "nhs.uk": "government", "mit.edu": "education",
             "ox.ac.uk": "education", "en.wikipedia.org": "reference", "merriam-webster.com": "reference",
             "webmd.com": "health", "mayoclinic.org": "health", "answers.yahoo.com": "community",
             "bbc.co.uk": "news", "redcross.org": "organization", "costhelper.com": "commercial",
             "acme-policies.internal": "commercial", "": "commercial"}
    for domain, expected in cases.items():
        assert source_type(domain) == expected, domain
    assert set(cases.values()) <= set(SOURCE_TYPES)


def test_topic_list_has_everyday_subjects():
    assert len(TOPIC_NAMES) == 15
    assert {"Animals & Nature", "Food & Nutrition", "Health & Medicine"} <= set(TOPIC_NAMES)


def test_filter_combines_fields_and_lists():
    f = make_filter(category="numeric", topic=["Food & Nutrition", "Animals & Nature"], corpus="internal")
    conds = {c.key: c.match for c in f.must}
    assert isinstance(conds["topic"], models.MatchAny) and conds["topic"].any == ["Food & Nutrition", "Animals & Nature"]
    assert isinstance(conds["category"], models.MatchValue) and conds["corpus"].value == "internal"
    assert make_filter() is None
