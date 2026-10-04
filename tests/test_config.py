from precisionrag.config import get_settings


def test_settings_load_and_profile():
    s = get_settings()
    assert s.device in ("cuda", "cpu")
    assert s.profile_name == ("gpu" if s.device == "cuda" else "cpu")
    assert s.profile["corpus_size"] >= 100_000          # rule C-04
    assert s["retrieval"]["top_k"] == 5                 # rule FR-2 / FR-6
    assert "localhost" not in s.qdrant_url              # Windows IPv6 fallback adds ~2 s per call
    assert len(s.fingerprint) == 10


def test_fusion_is_configurable():
    r = get_settings()["retrieval"]
    assert r["fusion_hybrid"] in ("rrf", "weighted_rrf", "linear")
    assert {"dense", "bm25"} <= set(r["weights_hybrid"])
