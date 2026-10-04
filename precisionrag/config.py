"""Loads config.yaml + .env and picks the hardware profile (GPU or CPU) automatically."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


@dataclass(frozen=True)
class Settings:
    raw: dict
    device: str                    # "cuda" or "cpu"
    profile_name: str              # "gpu" or "cpu"
    profile: dict
    hf_home: Path
    models_dir: Path
    eval_dir: Path
    results_dir: Path
    qdrant_url: str
    groq_api_key: str = field(repr=False, default="")

    def __getitem__(self, key):
        return self.raw[key]

    @property
    def fingerprint(self) -> str:
        """Short hash of the effective config, stored with every result so numbers stay traceable."""
        blob = json.dumps({"raw": self.raw, "profile": self.profile_name}, sort_keys=True)
        return hashlib.sha1(blob.encode()).hexdigest()[:10]


def _pick_device(requested: str) -> str:
    if requested in ("cuda", "cpu"):
        return requested
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv(ROOT / ".env")
    raw = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    device = _pick_device(os.getenv("DEVICE", raw.get("device", "auto")))
    profile_name = "gpu" if device == "cuda" else "cpu"

    hf_home = Path(os.getenv("HF_HOME") or _resolve(raw["paths"]["hf_home"]))
    models_dir = Path(os.getenv("MODELS_DIR") or _resolve(raw["paths"]["models_dir"]))
    os.environ["HF_HOME"] = str(hf_home)  # must be set before datasets / sentence-transformers are imported

    return Settings(
        raw=raw,
        device=device,
        profile_name=profile_name,
        profile=raw["profiles"][profile_name],
        hf_home=hf_home,
        models_dir=models_dir,
        eval_dir=_resolve(raw["paths"]["eval_dir"]),
        results_dir=_resolve(raw["paths"]["results_dir"]),
        qdrant_url=os.getenv("QDRANT_URL", raw["qdrant"]["url"]),
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
    )
