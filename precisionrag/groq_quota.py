"""Reads Groq's rate-limit headers from every response, so scripts can stop safely before the free quota runs out.
KeyPool moves to the next key in .env (GROQ_API_KEY, GROQ_API_KEY_2, ...) when one key's limit is reached."""
from __future__ import annotations

import os
import threading

import httpx

_lock = threading.Lock()
LATEST: dict[str, dict] = {}   # model -> {"remaining_requests": int, "limit_requests": int, ...}


class QuotaExhausted(RuntimeError):
    pass


def _record(response: httpx.Response) -> None:
    h = response.headers
    if "x-ratelimit-remaining-requests" not in h:
        return
    try:
        model = response.request.read().decode("utf-8", "ignore").split('"model":"', 1)[1].split('"', 1)[0]
    except Exception:
        model = "unknown"
    with _lock:
        LATEST[model] = {"remaining_requests": int(h.get("x-ratelimit-remaining-requests", -1)),
                         "limit_requests": int(h.get("x-ratelimit-limit-requests", -1)),
                         "remaining_tokens": h.get("x-ratelimit-remaining-tokens"),
                         "reset_requests": h.get("x-ratelimit-reset-requests")}


async def _arecord(response: httpx.Response) -> None:
    _record(response)


def http_clients() -> tuple[httpx.Client, httpx.AsyncClient]:
    return (httpx.Client(timeout=120, event_hooks={"response": [_record]}),
            httpx.AsyncClient(timeout=120, event_hooks={"response": [_arecord]}))


def available_keys(env=None) -> list[tuple[str, str]]:
    """Groq keys from .env in order: ("1", GROQ_API_KEY), ("2", GROQ_API_KEY_2), ... Only the label is ever printed."""
    if env is None:
        from .config import get_settings
        get_settings()                      # loads .env
        env = os.environ
    names = [("1", "GROQ_API_KEY")] + [(str(n), f"GROQ_API_KEY_{n}") for n in range(2, 10)]
    return [(label, env[name]) for label, name in names if env.get(name, "").startswith("gsk_")]


class KeyPool:
    """Uses one Groq key at a time and moves to the next one when the current key reaches its limit."""

    def __init__(self, only: str | None = None, env=None):
        keys = available_keys(env)
        self.keys = keys if only in (None, "auto") else [k for k in keys if k[0] == only]
        if not self.keys:
            raise SystemExit(f"Groq key {only or ''} missing in .env")
        self.i = 0

    @property
    def label(self) -> str:
        return self.keys[self.i][0]

    @property
    def key(self) -> str:
        return self.keys[self.i][1]

    def rotate(self, model: str | None = None, reason: str = "") -> bool:
        """Switch to the next key. Returns False when every key has been used."""
        if self.i + 1 >= len(self.keys):
            return False
        old = self.label
        self.i += 1
        with _lock:
            LATEST.pop(model, None)         # the old key's counters say nothing about the new key
        print(f"Groq key #{old} reached its limit ({reason}) -> switching to key #{self.label}", flush=True)
        return True


def check(model: str, minimum: int) -> dict:
    info = LATEST.get(model)
    if info and 0 <= info["remaining_requests"] < minimum:
        raise QuotaExhausted(f"Groq quota low for {model}: {info['remaining_requests']} requests left "
                             f"(resets in {info['reset_requests']}). Stopped safely; results so far are saved.")
    return info or {}
