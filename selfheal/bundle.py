"""ModelBundle: everything inference needs (cleaner, feature spec, RUL models, anomaly baseline, policy)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .anomaly import HealthyBaseline, sustained
from .clean import Cleaner
from .data import SENSOR_NAMES
from .features import FeatureSpec, build_features
from .rul import QuantileRUL
from .triage import Policy


@dataclass
class ModelBundle:
    subset: str
    cfg: dict
    cleaner: Cleaner
    spec: FeatureSpec
    rul: QuantileRUL
    anomaly: HealthyBaseline
    policy: Policy
    split: dict
    metrics: dict = field(default_factory=dict)
    reference: dict = field(default_factory=dict)   # train early-life reference for drift monitoring
    meta: dict = field(default_factory=dict)
    version: str = "unversioned"

    def __post_init__(self) -> None:
        names = self.spec.names()
        stat = self.cfg["anomaly"]["input_stat"]
        self.anom_idx = [names.index(f"{s}__{stat}") for s in self.spec.sensors]
        self._names = names

    # ------------------------------------------------------------------ batch (training / evaluation)
    def featurize(self, df_raw: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        cleaned = self.cleaner.transform_frame(df_raw)
        return cleaned, build_features(cleaned, self.spec)

    def anomaly_percentile(self, X: np.ndarray) -> np.ndarray:
        return self.anomaly.percentile(self.anomaly.score(X[:, self.anom_idx]))

    def batch_score(self, df_raw: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        cleaned, X = self.featurize(df_raw)
        P = self.rul.predict(X)
        A = self.anomaly_percentile(X)
        pol = self.policy
        out = cleaned[["unit", "cycle", "regime", "ood", "data_bad", "long_gap", "stale", "range_violation"]].copy()
        out["p10"], out["p50"], out["p90"] = P[:, 0], P[:, 1], P[:, 2]
        out["A"] = A
        out["warmup"] = out.groupby("unit").cumcount().to_numpy() + 1 < pol.warmup_rows   # rows of history seen so far
        out["alert"] = (A >= pol.alert_A) & ~out["warmup"].to_numpy()
        out["sustained"] = sustained(out["alert"].to_numpy(), out["unit"].to_numpy(), pol.alert_k, pol.alert_n)
        return out, X

    # ------------------------------------------------------------------ explanation for tickets
    def explain(self, x_row: np.ndarray, top: int = 3) -> list[dict]:
        """Top sensors by |TreeSHAP contribution| to the P50 prediction (indicative, NOT root cause)."""
        c = self.rul.contributions(x_row)[:-1]
        agg: dict[str, float] = {}
        for name, v in zip(self._names, c):
            key = self.spec.sensor_of(name) or "cycle"
            agg[key] = agg.get(key, 0.0) + float(v)
        ranked = sorted(agg.items(), key=lambda kv: -abs(kv[1]))[:top]
        return [{"sensor": k, "name": SENSOR_NAMES.get(k, "engine age (cycles)"), "contribution": round(v, 2),
                 "effect": "lowers RUL" if v < 0 else "raises RUL"} for k, v in ranked]

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)
        return path

    @staticmethod
    def load(path: str | Path) -> "ModelBundle":
        return joblib.load(path)
