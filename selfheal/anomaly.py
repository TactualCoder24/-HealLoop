"""Healthy-baseline anomaly detection / early warning: Isolation Forest + PCA-SPE fused into a percentile A.

Fitted ONLY on the first N cycles of historical TRAIN engines (healthy proxy, no live lifetime needed).
The fused score is mapped to a percentile `A` of a held-out healthy reference (first N cycles of
calibration engines), so A=0.99 means "more abnormal than 99% of healthy cycles".
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest


class HealthyBaseline:
    def __init__(self, n_trees: int = 200, pca_variance: float = 0.95, weights=(0.5, 0.5), seed: int = 42):
        self.n_trees, self.pca_variance, self.weights, self.seed = n_trees, pca_variance, tuple(weights), seed

    def fit(self, X_healthy: np.ndarray) -> "HealthyBaseline":
        X = np.asarray(X_healthy, dtype=float)
        self.mu = X.mean(0)
        self.sd = np.maximum(X.std(0), 1e-9)
        Xs = (X - self.mu) / self.sd
        self.iforest = IsolationForest(n_estimators=self.n_trees, random_state=self.seed, n_jobs=-1).fit(Xs)
        self.iforest.n_jobs = 1  # scoring is called per fleet tick on a few hundred rows: threads only add overhead
        self.pca = PCA(n_components=self.pca_variance, svd_solver="full", random_state=self.seed).fit(Xs)
        s_if, s_spe = self._raw(Xs)
        self._m = np.array([s_if.mean(), s_spe.mean()])
        self._s = np.maximum(np.array([s_if.std(), s_spe.std()]), 1e-9)
        return self

    def _raw(self, Xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s_if = -self.iforest.score_samples(Xs)                       # higher = more abnormal
        rec = self.pca.inverse_transform(self.pca.transform(Xs))
        spe = np.log1p(((Xs - rec) ** 2).sum(1))                      # squared prediction error (log scale)
        return s_if, spe

    def score(self, X: np.ndarray) -> np.ndarray:
        """Fused standardized abnormality score (healthy-train mean 0, SD 1 per component)."""
        Xs = (np.asarray(X, dtype=float) - self.mu) / self.sd
        s_if, spe = self._raw(Xs)
        z = (np.stack([s_if, spe], 1) - self._m) / self._s
        return z @ np.asarray(self.weights)

    def set_reference(self, healthy_scores: np.ndarray) -> "HealthyBaseline":
        self.ref = np.sort(np.asarray(healthy_scores, dtype=float))
        return self

    def percentile(self, scores: np.ndarray) -> np.ndarray:
        """A in (0, 1]: share of the healthy-validation reference at or below each score."""
        n = len(self.ref)
        return np.searchsorted(self.ref, np.asarray(scores, dtype=float), side="right") / n


def sustained(flags: np.ndarray, units: np.ndarray, k: int, n: int) -> np.ndarray:
    """k-of-n persistence per engine (rows contiguous & chronological): True when >= k of the last n cycles alerted."""
    out = np.zeros(len(flags), dtype=bool)
    f = np.asarray(flags, dtype=int)
    bounds = np.flatnonzero(np.r_[True, units[1:] != units[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        c = np.cumsum(f[a:b])
        prev = np.zeros(b - a, dtype=int)
        if b - a > n:
            prev[n:] = c[: b - a - n]
        out[a:b] = (c - prev) >= k
    return out


def tune_alert(A: np.ndarray, units: np.ndarray, cycles: np.ndarray, rul: np.ndarray, grid_A, grid_persist,
               early_cycles: int, budget: float, healthy_rul: float, warm: np.ndarray | None = None) -> tuple[dict, list[dict]]:
    """Pick (A threshold, k-of-n) on validation engines.

    False alert = sustained-alert onset while the engine is still healthy (true RUL > `healthy_rul`).
    Lead time = true RUL at the earliest sustained alert inside the degradation region (RUL <= healthy_rul).
    Choose the candidate with the longest median lead among those within the false-alert `budget`
    (onsets per engine); ties -> fewer early-life alerts, then the stricter threshold.
    """
    uniq = np.unique(units)
    first_row = np.r_[True, units[1:] != units[:-1]]
    ok_row = np.ones(len(A), bool) if warm is None else ~np.asarray(warm, bool)     # no flags during warm-up
    rows = []
    for a_thr in grid_A:
        for k, n in grid_persist:
            s = sustained((A >= a_thr) & ok_row, units, k, n)
            onset = (s & ~np.r_[False, s[:-1]]) | (s & first_row)
            false_on = float(np.sum(onset & (rul > healthy_rul)) / len(uniq))
            early = float(np.sum(onset & (cycles <= early_cycles)) / len(uniq))
            leads = []
            for u in uniq:
                m = (units == u) & s & (rul <= healthy_rul)
                leads.append(float(rul[m].max()) if m.any() else 0.0)
            rows.append({"A": float(a_thr), "k": int(k), "n": int(n), "false_alerts_per_engine": false_on,
                         "early_life_alerts_per_engine": early, "median_lead": float(np.median(leads)),
                         "alerting_share": float(np.mean(np.array(leads) > 0))})
    ok = [r for r in rows if r["false_alerts_per_engine"] <= budget]
    if ok:
        best = max(ok, key=lambda r: (r["median_lead"], -r["early_life_alerts_per_engine"], r["A"]))
    else:  # nothing within budget: least-false-alert candidate
        best = min(rows, key=lambda r: (r["false_alerts_per_engine"], -r["median_lead"]))
    return {"A": best["A"], "k": best["k"], "n": best["n"]}, rows
