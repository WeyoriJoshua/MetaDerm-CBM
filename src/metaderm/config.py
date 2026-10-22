from __future__ import annotations
from copy import deepcopy
from pathlib import Path
from typing import Any
import yaml


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def load_config(*paths: str | Path) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            part = yaml.safe_load(f) or {}
        cfg = deep_merge(cfg, part)
    return cfg


def save_config(cfg: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
