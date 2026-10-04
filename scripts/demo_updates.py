"""Live-update demo through the REAL API (rule FR-5): insert → dedup → update → metadata change → delete.

Story: a company policy changes from "2 days remote" to "3 days remote". The search must show only the newest version,
never the stale one, and the deleted document must disappear — all without rebuilding the 100k-passage index.

Needs the API running:  python -m uvicorn api.main:app --host 127.0.0.1 --port 8765
    python -m scripts.demo_updates
"""
from __future__ import annotations

import os

import httpx

from precisionrag.results import save_json

API = os.getenv("PRECISIONRAG_API", "http://127.0.0.1:8765")
DOC = "policy-remote-work"
SOURCE = "hr-policy.internal"
QUERY = "how many days per week can employees work remotely"
V1 = "Company policy (HR-07): Employees may work remotely 2 days per week, with manager approval."
V2 = "Company policy (HR-07): Employees may work remotely 3 days per week, with manager approval."

steps = []


def call(method, path, **kw):
    r = httpx.request(method, f"{API}{path}", timeout=60, **kw)
    return r.status_code, (r.json() if r.content else {})


def search(**params):
    _, res = call("GET", "/search", params={"q": QUERY, "mode": "hybrid_rerank", "k": 10, **params})
    return [h for h in res["hits"] if h["doc_id"] == DOC], res


def check(name, ok, detail):
    steps.append({"step": name, "pass": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name} — {detail}", flush=True)


def main():
    call("DELETE", f"/passages/{DOC}")                      # clean start (ignore 404)
    hits, _ = search()
    check("0. before insert: not in index", not hits, "document absent from top-10")

    _, r = call("POST", "/passages", json={"doc_id": DOC, "text": V1, "source": SOURCE, "category": "numeric"})
    hits, res = search()
    check("1. insert v1", r["status"] == "inserted" and hits and hits[0]["rank"] == 1 and "2 days" in hits[0]["text"],
          f"status={r['status']} in {r['ms']} ms; rank={hits[0]['rank'] if hits else None}, version={hits[0]['version'] if hits else None}")

    _, r = call("POST", "/passages", json={"doc_id": DOC, "text": V1, "source": SOURCE, "category": "numeric"})
    check("2. same text again (dedup)", r["status"] == "unchanged" and r["version"] == 1,
          f"status={r['status']} — no re-embedding, no duplicate")

    _, r = call("POST", "/passages", json={"doc_id": DOC, "text": V2, "source": SOURCE, "category": "numeric"})
    hits, res = search()
    texts = " ".join(h["text"] for h in res["hits"])
    check("3. update to v2 (stale prevention)",
          r["status"] == "updated" and hits and hits[0]["version"] == 2 and "3 days" in hits[0]["text"]
          and "2 days per week" not in texts and len(hits) == 1,
          f"status={r['status']} in {r['ms']} ms; only v2 returned, v1 gone, exactly 1 copy")

    hits, _ = search(source=SOURCE)
    check("4. metadata filter on the new doc", len(hits) == 1 and hits[0]["source"] == SOURCE,
          f"source={SOURCE} filter returns it")

    _, r = call("POST", "/passages", json={"doc_id": DOC, "text": V2, "source": SOURCE, "category": "description"})
    _, doc = call("GET", f"/passages/{DOC}")
    check("5. metadata-only update", r["status"] == "metadata_updated" and doc["category"] == "description"
          and doc["version"] == 2, f"status={r['status']} in {r['ms']} ms; vectors untouched, category changed")

    _, r = call("DELETE", f"/passages/{DOC}")
    hits, _ = search()
    code, _ = call("GET", f"/passages/{DOC}")
    check("6. delete", r["status"] == "deleted" and not hits and code == 404,
          f"status={r['status']} in {r['ms']} ms; gone from search and GET returns 404")

    _, health = call("GET", "/health")
    passed = sum(s["pass"] for s in steps)
    save_json("updates_demo.json", {"steps": steps, "passed": passed, "total": len(steps),
                                    "points_after": health["points"], "index_version": health["index_version"]})
    print(f"\n{passed}/{len(steps)} steps passed | index still has {health['points']:,} points (no rebuild)")


if __name__ == "__main__":
    main()
