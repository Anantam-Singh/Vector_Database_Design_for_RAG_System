"""Do the 50 RAGAS questions get the same top-5 passages on the CPU profile as on the GPU profile (100k index)?

Where the lists are identical, the judge sees exactly the same contexts, so the RAGAS scores measured once apply to both
profiles. Where they differ (the CPU reranker reads the top 10 instead of the top 20), RAGAS is run again on the CPU:
    DEVICE=cpu python -m scripts.eval_ragas --modes hybrid_rerank --tag cpu

    python -m scripts.check_cpu_contexts
"""
from __future__ import annotations

import json

from precisionrag.config import get_settings
from precisionrag.results import save_json
from precisionrag.retriever import Retriever


def main():
    s = get_settings()
    rows = [json.loads(l) for l in (s.eval_dir / "ragas.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    cpu, gpu = Retriever(device="cpu"), Retriever(device="cuda")
    out = {}
    for mode in ("dense", "hybrid", "dense_rerank", "hybrid_rerank"):
        same = sum([h.doc_id for h in cpu.search(r["query"], mode=mode).hits] ==
                   [h.doc_id for h in gpu.search(r["query"], mode=mode).hits] for r in rows)
        out[mode] = {"identical_top5": same, "n": len(rows)}
        print(f"{mode:14s} identical top-5 lists: {same}/{len(rows)}", flush=True)
    save_json("ragas/cpu_vs_gpu_contexts_100k.json", out, cpu_rerank_depth=cpu.profile["rerank_depth"],
              gpu_rerank_depth=gpu.profile["rerank_depth"])


if __name__ == "__main__":
    main()
