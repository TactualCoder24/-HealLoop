"""Monitoring: data-quality counters and drift (PSI) against the train early-life reference.

Unlabelled cycles can only support DRIFT monitoring - never supervised retraining.  To keep ageing from
looking like drift, incoming early-life cycles (<= reference_cycles) are compared with the train early-life
reference of the same sensors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-4


def bin_counts(values: np.ndarray, edges) -> np.ndarray:
    """Counts per bin for open-ended edges [-inf, e1, ..., inf] (inner edges define the bins)."""
    inner = np.asarray(edges, dtype=float)[1:-1]
    return np.bincount(np.searchsorted(inner, values, side="right"), minlength=len(inner) + 1).astype(float)


def psi(ref_p: np.ndarray, edges: np.ndarray, values: np.ndarray) -> float:
    counts = bin_counts(values, edges)
    cur = counts / max(counts.sum(), 1.0)
    ref = np.clip(np.asarray(ref_p, dtype=float), EPS, None)
    cur = np.clip(cur, EPS, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def status_for(value: float, warn: float, alert: float) -> str:
    return "alert" if value >= alert else "warn" if value >= warn else "ok"


def drift_report(reference: dict, cleaned: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """PSI per kept sensor: incoming early-life cycles vs train early-life reference."""
    mon = cfg["monitor"]
    early = cleaned[cleaned["cycle"] <= reference["cycles"]]
    rows = []
    for s in reference["sensors"]:
        b = reference["bins"][s]
        v = early[s].to_numpy()
        val = psi(b["p"], b["edges"], v) if len(v) >= mon["min_rows"] else float("nan")
        rows.append({"sensor": s, "psi": val, "n_incoming": int(len(v)),
                     "status": "insufficient_data" if val != val else status_for(val, mon["psi_warn"], mon["psi_alert"])})
    return pd.DataFrame(rows)


def quality_report(cleaned: pd.DataFrame, rejected: int, cfg: dict) -> dict:
    n = max(len(cleaned) + rejected, 1)
    rates = {
        "rows": int(len(cleaned)), "rejected_rows": int(rejected),
        "bad_data_rate": float(cleaned["data_bad"].mean()) if len(cleaned) else 0.0,
        "long_gap_rate": float(cleaned["long_gap"].mean()) if len(cleaned) else 0.0,
        "stale_rate": float(cleaned["stale"].mean()) if len(cleaned) else 0.0,
        "out_of_range_rate": float(cleaned["range_violation"].mean()) if len(cleaned) else 0.0,
        "unseen_regime_rate": float(cleaned["ood"].mean()) if len(cleaned) else 0.0,
        "imputed_rate": float((cleaned["imputed_n"] > 0).mean()) if len(cleaned) else 0.0,
        "reject_rate": rejected / n,
    }
    worst = max(rates["bad_data_rate"], rates["unseen_regime_rate"], rates["reject_rate"])
    rates["status"] = status_for(worst, cfg["monitor"]["quality_warn_frac"], cfg["monitor"]["quality_alert_frac"])
    return rates


def monitor_summary(drift: pd.DataFrame, quality: dict) -> dict:
    d = drift[drift["psi"].notna()]
    order = {"ok": 0, "warn": 1, "alert": 2}
    worst = max((order[s] for s in d["status"]), default=0)
    return {
        "drift_status": ["ok", "warn", "alert"][worst],
        "max_psi": float(d["psi"].max()) if len(d) else float("nan"),
        "max_psi_sensor": d.loc[d["psi"].idxmax(), "sensor"] if len(d) else None,
        "sensors_alert": d.loc[d["status"] == "alert", "sensor"].tolist(),
        "sensors_warn": d.loc[d["status"] == "warn", "sensor"].tolist(),
        "quality_status": quality["status"],
    }
