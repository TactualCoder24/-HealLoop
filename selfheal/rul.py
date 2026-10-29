"""Supervised RUL: LightGBM quantile models P10/P50/P90 + split-conformal (CQR) calibration of the 80% interval.

Target handling (why P10/P90 are fitted on the UNCAPPED RUL): the 125-cycle cap turns half of the labels into a
point mass at 125, so a P90 booster starts at 125 with identical gradients everywhere and never splits.
Quantiles commute with monotone transforms - P90(min(RUL, cap)) = min(P90(RUL), cap) - so the tail quantiles are
fitted on the raw RUL and clipped to the cap afterwards, which is mathematically the same quantile but learnable.
The median is fitted on the capped label (the usual piece-wise linear RUL target).
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np


def cqr_qhat(lo: np.ndarray, hi: np.ndarray, y: np.ndarray, alpha: float, weights: np.ndarray | None = None) -> float:
    """Conformalized quantile regression correction (Romano et al., 2019) on held-out calibration rows.

    With `weights` (e.g. one unit of weight per engine) it is the weighted split-conformal quantile, so the
    80% coverage holds for the weighted calibration distribution instead of being dominated by long engines.
    """
    scores = np.maximum(np.asarray(lo, float) - y, y - np.asarray(hi, float))
    n = len(scores)
    level = min(1.0, (1 - alpha) * (n + 1) / n)          # finite-sample correction
    if weights is None:
        return float(np.sort(scores)[min(int(np.ceil((n + 1) * (1 - alpha))), n) - 1])
    order = np.argsort(scores)
    cum = np.cumsum(np.asarray(weights, float)[order] / np.sum(weights))
    return float(scores[order][min(int(np.searchsorted(cum, level)), n - 1)])


class QuantileRUL:
    """Three independent LightGBM quantile regressors + conformal widening of [P10, P90]."""

    def __init__(self, quantiles=(0.1, 0.5, 0.9), params: dict | None = None, seed: int = 42, rul_cap: float = 125.0):
        self.quantiles = tuple(quantiles)
        self.params = dict(params or {})
        self.seed = seed
        self.rul_cap = float(rul_cap)
        self.models: dict[float, lgb.LGBMRegressor] = {}
        self.qhat = 0.0
        self.alpha = 0.2
        self.best_iters: dict[float, int] = {}

    def _make(self, q: float) -> lgb.LGBMRegressor:
        p = dict(self.params)
        p.pop("early_stopping_rounds", None)
        if abs(q - 0.5) < 1e-9:
            # The median is fitted on the CAPPED label. When >= 50% of rows sit on the cap (e.g. FD003's long-lived
            # engines) the default start value (the label median) equals the cap, every gradient is identical and the
            # booster never splits (RMSE ~ 65). Starting from 0 lets the first residual signs differ across rows.
            p["boost_from_average"] = False
        return lgb.LGBMRegressor(objective="quantile", alpha=q, random_state=self.seed, verbose=-1, **p)

    def fit_one(self, q: float, X, y_cap, y_raw, weight=None, X_val=None, yv_cap=None, yv_raw=None) -> lgb.LGBMRegressor:
        median = abs(q - 0.5) < 1e-9
        y, yv = (y_cap, yv_cap) if median else (y_raw, yv_raw)
        rounds = int(self.params.get("early_stopping_rounds", 0) or 0)
        m = self._make(q)
        kw = {}
        if X_val is not None and rounds:
            kw = {"eval_X": X_val, "eval_y": yv, "eval_metric": "quantile",
                  "callbacks": [lgb.early_stopping(rounds, verbose=False)]}
        m.fit(X, y, sample_weight=weight, **kw)
        self.models[q] = m
        self.best_iters[q] = int(m.best_iteration_ or m.n_estimators)
        return m

    def fit(self, X, y_cap, y_raw, weight=None, X_val=None, yv_cap=None, yv_raw=None, only: tuple[float, ...] | None = None):
        for q in (only or self.quantiles):
            self.fit_one(q, X, y_cap, y_raw, weight, X_val, yv_cap, yv_raw)
        return self

    def predict_raw(self, X) -> np.ndarray:
        """(N, 3) raw quantile predictions in [0, cap], sorted per row (rearrangement removes crossing)."""
        P = np.column_stack([self.models[q].predict(X) for q in self.quantiles])
        return np.sort(np.clip(P, 0.0, self.rul_cap), axis=1)

    def calibrate(self, X_cal, y_cal_capped, alpha: float, weights: np.ndarray | None = None) -> float:
        P = self.predict_raw(X_cal)
        self.alpha = alpha
        self.qhat = cqr_qhat(P[:, 0], P[:, -1], np.asarray(y_cal_capped, float), alpha, weights)
        return self.qhat

    def predict(self, X) -> np.ndarray:
        """(N, 3) calibrated [P10, P50, P90]: interval widened by the conformal correction, kept in [0, cap]."""
        P = self.predict_raw(X)
        lo = np.minimum(np.clip(P[:, 0] - self.qhat, 0.0, None), P[:, 1])
        hi = np.maximum(np.clip(P[:, 2] + self.qhat, None, self.rul_cap), P[:, 1])
        return np.column_stack([lo, P[:, 1], hi])

    def contributions(self, X_row: np.ndarray, q: float = 0.5) -> np.ndarray:
        """TreeSHAP per-feature contributions (last column = bias) for ONE row of the P50 model.
        Indicative of what drove the prediction - NOT proof of root cause."""
        return self.models[q].booster_.predict(np.asarray(X_row, dtype=np.float32).reshape(1, -1), pred_contrib=True)[0]
