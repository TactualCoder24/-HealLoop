"""Stage 3 - FEATURES: past-only rolling windows (mean, SD, slope, extrema, EWMA, delta) + RUL labels.

All features are a pure function of the last `max(windows)` cleaned rows (left-padded with NaN at
engine start -> shorter-window fallback), so batch features equal streaming features exactly and
no future cycle can leak into row t.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

STATS_PER_WINDOW = ("mean", "std", "slope", "min", "max", "delta")


@dataclass
class FeatureSpec:
    sensors: list[str]
    windows: tuple[int, ...] = (5, 10, 20, 30)
    ewma_span: int = 10

    @property
    def max_window(self) -> int:
        return max(self.windows)

    def stat_names(self) -> list[str]:
        names = ["cur", "ewma"]
        for w in self.windows:
            names += [f"{s}{w}" for s in STATS_PER_WINDOW]
        return names

    def names(self) -> list[str]:
        """Feature column names: 'cycle' first, then stat-major / sensor-minor."""
        return ["cycle"] + [f"{s}__{st}" for st in self.stat_names() for s in self.sensors]

    def sensor_of(self, name: str) -> str | None:
        return None if name == "cycle" else name.split("__", 1)[0]


def window_tensor(Z: np.ndarray, W: int) -> np.ndarray:
    """(T, S) -> (T, W, S) trailing windows, NaN left-padded; row t holds Z[t-W+1 .. t]."""
    T, S = Z.shape
    pad = np.full((W - 1, S), np.nan)
    padded = np.vstack([pad, Z])
    win = np.lib.stride_tricks.sliding_window_view(padded, W, axis=0)  # (T, S, W)
    return np.ascontiguousarray(win.transpose(0, 2, 1))


def _stats(x: np.ndarray) -> dict[str, np.ndarray]:
    """NaN-aware window statistics over axis 1 of x (N, w, S); the last row is always valid."""
    N, w, S = x.shape
    m = ~np.isnan(x)
    n = m.sum(1)
    x0 = np.where(m, x, 0.0)
    mean = x0.sum(1) / n
    var = np.where(m, (x - mean[:, None, :]) ** 2, 0.0).sum(1) / n
    mn = np.where(m, x, np.inf).min(1)
    mx = np.where(m, x, -np.inf).max(1)
    t = np.arange(w, dtype=float)[None, :, None]
    St, Stt = (t * m).sum(1), ((t ** 2) * m).sum(1)
    Sx, Stx = x0.sum(1), (t * x0).sum(1)
    den = n * Stt - St ** 2
    slope = np.where(den > 0, (n * Stx - St * Sx) / np.where(den > 0, den, 1.0), 0.0)
    first = np.take_along_axis(x0, m.argmax(1)[:, None, :], axis=1)[:, 0, :]
    return {"mean": mean, "std": np.sqrt(var), "slope": slope, "min": mn, "max": mx, "delta": x0[:, -1, :] - first}


def window_features(win: np.ndarray, spec: FeatureSpec) -> np.ndarray:
    """(N, W, S) window tensor -> (N, 2 + 6*len(windows), S) -> flattened (N, F) float32 (without 'cycle')."""
    N, W, S = win.shape
    blocks = [win[:, -1, :]]
    # EWMA over the (up to) last W valid rows, newest sample weighted highest
    alpha = 2.0 / (spec.ewma_span + 1.0)
    wts = alpha * (1 - alpha) ** np.arange(W - 1, -1, -1)
    valid = ~np.isnan(win)
    blocks.append((np.where(valid, win, 0.0) * wts[None, :, None]).sum(1) / (valid * wts[None, :, None]).sum(1))
    for w in spec.windows:
        st = _stats(win[:, W - w:, :])
        blocks += [st[k] for k in STATS_PER_WINDOW]
    return np.stack(blocks, axis=1).reshape(N, -1).astype(np.float32)


def engine_features(Z: np.ndarray, cycles: np.ndarray, spec: FeatureSpec) -> np.ndarray:
    """All rows of one engine (batch path): (T, 1 + F) with the cycle counter first."""
    win = window_tensor(Z, spec.max_window)
    F = window_features(win, spec)
    return np.hstack([cycles.astype(np.float32)[:, None], F])


def build_features(cleaned: pd.DataFrame, spec: FeatureSpec) -> np.ndarray:
    """Features for a frame produced by `Cleaner.transform_frame` (engines contiguous, chronological)."""
    units = cleaned["unit"].to_numpy()
    cyc = cleaned["cycle"].to_numpy()
    Z = cleaned[spec.sensors].to_numpy(dtype=float)
    bounds = np.flatnonzero(np.r_[True, units[1:] != units[:-1], True])
    out = [engine_features(Z[a:b], cyc[a:b], spec) for a, b in zip(bounds[:-1], bounds[1:])]
    return np.vstack(out)


def add_rul_labels(df: pd.DataFrame, cap: int | None = None) -> pd.DataFrame:
    """RUL = failure cycle - current cycle (train engines run to failure). Adds `rul` and capped `rul_c`."""
    life = df.groupby("unit")["cycle"].transform("max")
    out = df[["unit", "cycle"]].copy()
    out["life"] = life
    out["rul"] = (life - df["cycle"]).astype(float)
    out["rul_c"] = out["rul"].clip(upper=cap) if cap else out["rul"]
    return out
