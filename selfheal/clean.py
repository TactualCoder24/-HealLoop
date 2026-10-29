"""Stage 2 - CLEAN & NORMALIZE.

Everything here is *causal*: the cleaned value at cycle t depends only on cycles <= t, and every
statistic (regime centres, scalers, medians, bounds) is fitted on TRAIN engines only.  The same
`Cleaner.clean_arrays` runs in batch (training / evaluation) and one-engine-at-a-time (live replay),
so training and inference transforms are identical by construction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .data import SENSORS, SETTINGS


class RegimeModel:
    """Operating-regime model: k-means on standardized settings (k=1 for single-condition subsets)."""

    def __init__(self, n_regimes: int, seed: int, ood_quantile: float, ood_margin: float):
        self.n_regimes = n_regimes
        self.seed = seed
        self.ood_quantile = ood_quantile
        self.ood_margin = ood_margin

    def fit(self, settings: np.ndarray) -> "RegimeModel":
        s = np.asarray(settings, dtype=float)
        self.mu = s.mean(0)
        self.sd = np.maximum(s.std(0), 1e-3 * np.maximum(np.abs(self.mu), 1.0))
        z = (s - self.mu) / self.sd
        if self.n_regimes == 1:
            self.centers = z.mean(0, keepdims=True)
        else:
            km = KMeans(n_clusters=self.n_regimes, n_init=10, random_state=self.seed).fit(z)
            # deterministic ordering so regime ids are stable across refits
            order = np.lexsort(km.cluster_centers_.T[::-1])
            self.centers = km.cluster_centers_[order]
        _, dist = self._assign(z)
        self.ood_threshold = float(np.quantile(dist, self.ood_quantile) * self.ood_margin)
        return self

    def _assign(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        d2 = ((z[:, None, :] - self.centers[None, :, :]) ** 2).sum(-1)
        lab = d2.argmin(1)
        return lab, np.sqrt(d2[np.arange(len(z)), lab])

    def predict(self, settings: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        z = (np.asarray(settings, dtype=float) - self.mu) / self.sd
        lab, dist = self._assign(z)
        return lab, dist, dist > self.ood_threshold


@dataclass
class Cleaned:
    """Cleaned engine history; arrays are aligned to the gap-free cycle axis `cycles`."""

    cycles: np.ndarray
    present: np.ndarray       # False for cycles that never arrived (inserted as missing rows)
    Z: np.ndarray             # (T, S_kept) regime-scaled, causally smoothed sensors
    regime: np.ndarray
    regime_dist: np.ndarray
    ood: np.ndarray           # unseen operating regime
    imputed_n: np.ndarray     # sensors forward-filled / median-filled at this row
    long_gap: np.ndarray      # (held) gap longer than max_ffill_gap
    stale: np.ndarray         # frozen feed
    range_violation: np.ndarray
    data_bad: np.ndarray      # long_gap | stale | range_violation


class Cleaner:
    def __init__(self, cfg: dict):
        c = cfg["cleaning"]
        self.max_gap = int(c["max_ffill_gap"])
        self.hold = int(c["long_gap_hold"])
        self.stale_repeat = int(c["stale_repeat"])
        self.median_window = int(c["median_window"])
        self.range_margin = float(c["range_margin"])
        self.range_frac = float(c["range_violation_frac"])
        self.min_std = float(c["min_within_regime_std"])
        r = cfg["regime"]
        self.regime = RegimeModel(int(r["n_regimes"]), int(cfg["seed"]), float(r["ood_quantile"]), float(r["ood_margin"]))

    # ------------------------------------------------------------------ fit (TRAIN engines only)
    def fit(self, train: pd.DataFrame) -> "Cleaner":
        settings = train[SETTINGS].to_numpy(dtype=float)
        X = train[SENSORS].to_numpy(dtype=float)
        self.regime.fit(settings)
        lab, _, _ = self.regime.predict(settings)
        R, S = self.regime.n_regimes, X.shape[1]
        med = np.tile(np.median(X, 0), (R, 1))
        sd = np.tile(X.std(0), (R, 1))
        pooled = np.zeros(S)
        for r in range(R):
            m = lab == r
            if m.sum() >= 2:
                med[r] = np.median(X[m], 0)
                sd[r] = X[m].std(0)
                pooled += sd[r] ** 2 * m.sum()
        pooled = np.sqrt(pooled / len(X))  # pooled WITHIN-regime std (global variance would be dominated by regimes)
        keep = pooled > self.min_std
        self.keep_idx = np.flatnonzero(keep)
        self.kept = [SENSORS[i] for i in self.keep_idx]
        self.dropped = [SENSORS[i] for i in np.flatnonzero(~keep)]
        self.sensor_std_pooled = dict(zip(SENSORS, pooled.round(8)))
        self.med_k = med[:, self.keep_idx]
        self.sd_k = np.maximum(sd[:, self.keep_idx], 1e-9)
        self.setting_median = np.median(settings, 0)
        z = (X[:, self.keep_idx] - self.med_k[lab]) / self.sd_k[lab]
        self.z_lo, self.z_hi = z.min(0), z.max(0)
        return self

    # ------------------------------------------------------------------ one engine, causal
    def clean_arrays(self, cycles: np.ndarray, settings: np.ndarray, sensors: np.ndarray) -> Cleaned:
        cycles = np.asarray(cycles, dtype=np.int64)
        full = np.arange(cycles[0], cycles[-1] + 1)
        T, k = len(full), len(self.keep_idx)
        pos = cycles - cycles[0]
        F = np.full((T, 3 + k), np.nan)
        F[pos, :3] = settings
        F[pos, 3:] = np.asarray(sensors, dtype=float)[:, self.keep_idx]
        present = np.zeros(T, dtype=bool)
        present[pos] = True

        # --- causal forward fill, limited run lengths measured explicitly --------------------------
        isnan = np.isnan(F)
        t_idx = np.arange(T)[:, None]
        last_valid = np.maximum.accumulate(np.where(isnan, -1, t_idx), axis=0)
        run = t_idx - last_valid                                   # consecutive missing steps ending at t
        filled = np.where(last_valid >= 0, F[np.maximum(last_valid, 0), np.arange(3 + k)[None, :]], np.nan)
        lead = last_valid < 0                                      # no valid observation yet -> train median
        set_f = np.where(np.isnan(filled[:, :3]), self.setting_median, filled[:, :3])
        long_raw = (run > self.max_gap).any(1)
        imputed_n = isnan[:, 3:].sum(1)

        # --- regime & regime-aware scaling (train-only medians / std) ------------------------------
        lab, dist, ood = self.regime.predict(set_f)
        z = (filled[:, 3:] - self.med_k[lab]) / self.sd_k[lab]
        z = np.where(lead[:, 3:], 0.0, z)                          # train-median fill == 0 in z space

        # --- data-quality flags ---------------------------------------------------------------------
        raw_s = F[:, 3:]
        same = np.zeros(T, dtype=bool)
        if T > 1:
            same[1:] = (raw_s[1:] == raw_s[:-1]).all(1)
        last_break = np.maximum.accumulate(np.where(~same, np.arange(T), 0))
        stale = (np.arange(T) - last_break) >= self.stale_repeat
        span = self.z_hi - self.z_lo
        lo, hi = self.z_lo - self.range_margin * span, self.z_hi + self.range_margin * span
        range_violation = ((z < lo) | (z > hi)).mean(1) > self.range_frac
        c = np.cumsum(long_raw.astype(int))
        prev = np.zeros(T, dtype=int)
        if T > self.hold:
            prev[self.hold:] = c[: T - self.hold]
        long_gap = (c - prev) > 0

        # --- causal rolling-median noise filter ------------------------------------------------------
        w = self.median_window
        if w > 1:
            stack = np.full((w, T, k), np.nan)
            for j in range(w):
                stack[j, j:] = z[: T - j]
            s = np.sort(stack, axis=0)                                  # NaN sorts last
            n = (~np.isnan(stack)).sum(0)                               # valid samples per position (start-up: < w)
            lo = np.take_along_axis(s, ((n - 1) // 2)[None], axis=0)[0]
            hi = np.take_along_axis(s, (n // 2)[None], axis=0)[0]
            zs = 0.5 * (lo + hi)                                        # == nanmedian, without its slow masked path
        else:
            zs = z
        return Cleaned(full, present, zs, lab, dist, ood, imputed_n, long_gap, stale, range_violation,
                       long_gap | stale | range_violation)

    # ------------------------------------------------------------------ whole frame
    def transform_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean every engine of `df` (rows must be chronological per engine); returns present rows only."""
        units = df["unit"].to_numpy()
        cyc = df["cycle"].to_numpy()
        settings = df[SETTINGS].to_numpy(dtype=float)
        sens = df[SENSORS].to_numpy(dtype=float)
        bounds = np.flatnonzero(np.r_[True, units[1:] != units[:-1], True])
        parts = []
        for a, b in zip(bounds[:-1], bounds[1:]):
            cl = self.clean_arrays(cyc[a:b], settings[a:b], sens[a:b])
            m = cl.present
            part = pd.DataFrame(cl.Z[m], columns=self.kept)
            part.insert(0, "unit", units[a])
            part.insert(1, "cycle", cl.cycles[m])
            part["regime"] = cl.regime[m]
            part["regime_dist"] = cl.regime_dist[m]
            part["ood"] = cl.ood[m]
            part["imputed_n"] = cl.imputed_n[m]
            part["long_gap"] = cl.long_gap[m]
            part["stale"] = cl.stale[m]
            part["range_violation"] = cl.range_violation[m]
            part["data_bad"] = cl.data_bad[m]
            parts.append(part)
        return pd.concat(parts, ignore_index=True)
