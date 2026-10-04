"""Saves measurements with a provenance stamp: when, which config, which hardware, which code version.

Rule we follow: no number goes into a document unless it was written here by a script.
"""
from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import ROOT, get_settings


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "uncommitted"
    except Exception:
        return "uncommitted"


def hardware() -> dict:
    info = {"platform": platform.platform(), "python": platform.python_version()}
    try:
        import psutil
        info["cpu_threads"] = psutil.cpu_count()
        info["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except ImportError:
        pass
    try:
        import torch
        info["torch_threads"] = torch.get_num_threads()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return info


def stamp(**extra) -> dict:
    s = get_settings()
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "config_fingerprint": s.fingerprint,
        "profile": s.profile_name,
        "device": s.device,
        "hardware": hardware(),
        **extra,
    }


def results_path(name: str) -> Path:
    path = get_settings().results_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_json(name: str, data: dict, **extra) -> Path:
    path = results_path(name)
    path.write_text(json.dumps({"meta": stamp(**extra), "data": data}, indent=2, default=str), encoding="utf-8")
    return path


def save_csv(name: str, df: pd.DataFrame, **extra) -> Path:
    """Writes the table plus a sidecar <name>.meta.json with the provenance stamp."""
    path = results_path(name)
    df.to_csv(path, index=False)
    path.with_suffix(".meta.json").write_text(json.dumps(stamp(**extra), indent=2, default=str), encoding="utf-8")
    return path
