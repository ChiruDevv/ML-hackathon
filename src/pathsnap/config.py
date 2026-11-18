"""Project paths and configuration loading."""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "config.json"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
CACHE = ROOT / "artifacts" / "cache"


def load_config(path: str | os.PathLike | None = None) -> dict:
    with open(path or CONFIG_PATH) as f:
        cfg = json.load(f)
    for k, v in cfg["data"].items():
        p = Path(os.environ.get(f"PATHSNAP_{k.upper()}", v))
        cfg["data"][k] = str(p if p.is_absolute() else ROOT / p)
    return cfg
