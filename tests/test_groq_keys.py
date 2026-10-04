"""Groq key rotation: fake keys only — the real .env is never read here."""
import pytest

from precisionrag.groq_quota import LATEST, KeyPool, available_keys

ENV = {"GROQ_API_KEY": "gsk_one", "GROQ_API_KEY_2": "", "GROQ_API_KEY_3": "gsk_three", "GROQ_API_KEY_4": "gsk_four",
       "GROQ_API_KEY_5": "not-a-key"}


def test_keys_are_found_in_order_and_blank_ones_skipped():
    assert [label for label, _ in available_keys(ENV)] == ["1", "3", "4"]


def test_pool_moves_to_next_key_until_all_are_used():
    pool = KeyPool(env=ENV)
    LATEST["m"] = {"remaining_requests": 0}
    assert pool.key == "gsk_one"
    assert pool.rotate("m", "limit") and pool.key == "gsk_three" and "m" not in LATEST   # old counters dropped
    assert pool.rotate("m", "limit") and pool.label == "4"
    assert pool.rotate("m", "limit") is False and pool.label == "4"                      # nothing left


def test_pool_can_be_pinned_to_one_key():
    pool = KeyPool("3", env=ENV)
    assert pool.key == "gsk_three" and pool.rotate() is False


def test_missing_key_stops_clearly():
    with pytest.raises(SystemExit):
        KeyPool("2", env=ENV)


def test_rotation_message_never_shows_the_key(capsys):
    KeyPool(env=ENV).rotate("m", "limit")
    out = capsys.readouterr().out
    assert "#1" in out and "#3" in out and "gsk_" not in out
