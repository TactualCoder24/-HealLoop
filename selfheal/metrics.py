"""Evaluation metrics: RMSE, NASA asymmetric score, interval coverage/width, risk-window metrics."""
from __future__ import annotations

import numpy as np


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.sqrt(np.mean((p - y) ** 2)))


def mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(p, float) - np.asarray(y, float))))


def nasa_score(y: np.ndarray, p: np.ndarray) -> float:
    """NASA C-MAPSS asymmetric score: late predictions (p > y) are penalised more than early ones."""
    d = np.asarray(p, float) - np.asarray(y, float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)))


def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    y = np.asarray(y, float)
    return float(np.mean((y >= lo) & (y <= hi)))


def width(lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean(np.asarray(hi, float) - np.asarray(lo, float)))


def pinball(y: np.ndarray, q: np.ndarray, alpha: float) -> float:
    d = np.asarray(y, float) - np.asarray(q, float)
    return float(np.mean(np.maximum(alpha * d, (alpha - 1) * d)))


def point_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {"rmse": rmse(y, p), "mae": mae(y, p), "nasa_score": nasa_score(y, p), "n": int(len(y))}


def endpoint_like(units: np.ndarray, cycles: np.ndarray, rul: np.ndarray, min_cycle: int, horizon: float) -> tuple[np.ndarray, np.ndarray]:
    """Mask + weights emulating 'one random cut per engine': rows with >= min_cycle cycles of history and
    true RUL <= horizon, each engine weighted equally (C-MAPSS test engines are truncated histories)."""
    m = (np.asarray(cycles) >= min_cycle) & (np.asarray(rul, float) <= horizon)
    w = np.zeros(len(units))
    if m.any():
        u = np.asarray(units)[m]
        _, inv, cnt = np.unique(u, return_inverse=True, return_counts=True)
        w[m] = 1.0 / cnt[inv]
    return m, w


def wrmse(y, p, w) -> float:
    y, p, w = (np.asarray(a, float) for a in (y, p, w))
    return float(np.sqrt(np.sum(w * (p - y) ** 2) / np.sum(w)))


def wnasa_mean(y, p, w) -> float:
    """Weighted mean of the per-row NASA asymmetric score (comparable across sample sizes)."""
    y, p, w = (np.asarray(a, float) for a in (y, p, w))
    d = p - y
    s = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(w * s) / np.sum(w))


def wcoverage(y, lo, hi, w) -> float:
    y, lo, hi, w = (np.asarray(a, float) for a in (y, lo, hi, w))
    return float(np.sum(w * ((y >= lo) & (y <= hi))) / np.sum(w))


def wwidth(lo, hi, w) -> float:
    lo, hi, w = (np.asarray(a, float) for a in (lo, hi, w))
    return float(np.sum(w * (hi - lo)) / np.sum(w))


def risk_window_metrics(y_true: np.ndarray, p50: np.ndarray, horizon: float = 30.0) -> dict[str, float]:
    """Metrics on the rare-risk window (true RUL <= horizon): rmse and late-overestimate share/bias."""
    y_true, p50 = np.asarray(y_true, float), np.asarray(p50, float)
    m = y_true <= horizon
    if not m.any():
        return {"rmse_risk": float("nan"), "late_share_risk": float("nan"), "late_bias_risk": float("nan")}
    d = p50[m] - y_true[m]
    return {
        "rmse_risk": float(np.sqrt(np.mean(d ** 2))),
        "late_share_risk": float(np.mean(d > 0)),
        "late_bias_risk": float(np.mean(np.maximum(d, 0.0))),
    }
