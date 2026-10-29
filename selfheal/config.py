"""Configuration loading: configs/base.yaml deep-merged with configs/<SUBSET>.yaml."""
from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml

ROOT = Path(os.environ.get("SELFHEAL_HOME", Path(__file__).resolve().parents[1]))
SUBSETS = ("FD001", "FD002", "FD003", "FD004")


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(subset: str) -> dict:
    subset = subset.upper()
    if subset not in SUBSETS:
        raise ValueError(f"unknown subset {subset!r}; expected one of {SUBSETS}")
    cfg_dir = ROOT / "configs"
    cfg = yaml.safe_load((cfg_dir / "base.yaml").read_text(encoding="utf-8"))
    over = cfg_dir / f"{subset}.yaml"
    if over.exists():
        cfg = _deep_merge(cfg, yaml.safe_load(over.read_text(encoding="utf-8")) or {})
    cfg["subset"] = subset
    return cfg


def resolve(cfg: dict, key: str) -> Path:
    """Resolve cfg['paths'][key] against the project root (absolute paths win)."""
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else ROOT / p


def artifacts_dir(cfg: dict) -> Path:
    d = resolve(cfg, "artifacts_dir") / cfg["subset"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def outputs_dir(cfg: dict) -> Path:
    d = resolve(cfg, "outputs_dir") / cfg["subset"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def tickets_db_path(cfg: dict) -> Path:
    d = resolve(cfg, "outputs_dir")
    d.mkdir(parents=True, exist_ok=True)
    return d / "tickets.db"
