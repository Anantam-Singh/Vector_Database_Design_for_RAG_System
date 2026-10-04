from types import SimpleNamespace

from precisionrag.cache import ResultCache
from precisionrag.data import clean
from precisionrag.fusion import fuse
from precisionrag.metrics import paired_bootstrap, per_query
from precisionrag.store import point_id


def P(i, score=1.0):
    return SimpleNamespace(id=i, score=score)


def test_rrf_rewards_agreement_and_weights_matter():
    dense, bm25 = [P("a"), P("b"), P("c")], [P("c"), P("a"), P("d")]
    plain = [p.id for p, _ in fuse("rrf", {"dense": dense, "bm25": bm25}, {})]
    assert plain[0] == "a"                       # ranked high by both lists
    dense_only = [p.id for p, _ in fuse("weighted_rrf", {"dense": dense, "bm25": bm25}, {"dense": 1, "bm25": 0})]
    assert dense_only == ["a", "b", "c"]         # weight 0 switches BM25 off completely


def test_linear_fusion_normalises_scores():
    dense = [P("a", 0.9), P("b", 0.5)]
    bm25 = [P("b", 30.0), P("a", 10.0)]          # very different scale
    ids = [p.id for p, _ in fuse("linear", {"dense": dense, "bm25": bm25}, {"dense": 1, "bm25": 1})]
    assert set(ids) == {"a", "b"}


def test_metrics():
    m = per_query([7, 3, 9, 1, 2, 5], {3})
    assert m["mrr@10"] == 0.5 and m["hit@5"] == 1.0 and m["recall@5"] == 1.0 and abs(m["precision@5"] - 0.2) < 1e-9
    assert per_query([1, 2, 3], {42})["mrr@10"] == 0.0


def test_bootstrap_detects_clear_gain_and_no_gain():
    gain = paired_bootstrap([0.0] * 100, [1.0] * 100, n=200)
    assert gain["ci95_low"] > 0 and gain["wins"] == 100
    same = paired_bootstrap([0.5] * 100, [0.5] * 100, n=200)
    assert same["mean_diff"] == 0 and same["ties"] == 100


def test_cache_is_invalidated_by_index_version():
    c = ResultCache(max_items=2)
    k1 = c.key("  What IS  x ", "dense", 5, None, None, index_version=1)
    c.put(k1, {"hits": [1]})
    assert c.get(c.key("what is x", "dense", 5, None, None, 1)) == {"hits": [1]}   # normalised query
    assert c.get(c.key("what is x", "dense", 5, None, None, 2)) is None            # after any write: miss


def test_cache_evicts_least_recent():
    c = ResultCache(max_items=2)
    for i in range(3):
        c.put(("k", i), i)
    assert c.get(("k", 0)) is None and c.get(("k", 2)) == 2


def test_clean_text():
    assert clean("175&deg F  and   AT&amp;T ") == "175° F and AT&T"


def test_point_id_is_stable_per_document():
    assert point_id("policy-1") == point_id("policy-1") != point_id("policy-2")
