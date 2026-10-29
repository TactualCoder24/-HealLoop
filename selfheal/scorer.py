"""Per-cycle scoring: ingest checks -> causal cleaning -> past-only features -> RUL bands + anomaly percentile.

`Scorer.process` consumes ONE fleet tick (at most one new reading per engine), keeping a bounded raw history per
engine.  It calls the same `Cleaner.clean_arrays` and `window_features` as the batch path, so a streamed engine
reproduces the batch scores exactly (asserted in tests/test_pipeline.py).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .bundle import ModelBundle
from .data import SENSORS, SETTINGS, CycleValidator
from .features import window_features
from .triage import CycleInputs

HISTORY = 80  # raw rows kept per engine (>> longest window + gap logic)


@dataclass
class ScoreResult:
    unit: int
    cycle: int
    status: str                      # "ok" | "rejected"
    reason: str
    inputs: CycleInputs
    x_row: np.ndarray | None = None  # feature row (for ticket evidence)
    extras: dict = field(default_factory=dict)


class _History:
    __slots__ = ("cycles", "settings", "sensors")

    def __init__(self) -> None:
        self.cycles: deque = deque(maxlen=HISTORY)
        self.settings: deque = deque(maxlen=HISTORY)
        self.sensors: deque = deque(maxlen=HISTORY)

    def add(self, cycle: int, settings: np.ndarray, sensors: np.ndarray) -> None:
        self.cycles.append(cycle)
        self.settings.append(settings)
        self.sensors.append(sensors)


class Scorer:
    def __init__(self, bundle: ModelBundle):
        self.b = bundle
        bundle.anomaly.iforest.n_jobs = 1   # per-tick scoring of a few hundred rows: threads only add overhead
        self.validator = CycleValidator()
        self.hist: dict[int, _History] = {}
        self.flags: dict[int, deque] = {}
        self.seen: dict[int, int] = {}      # accepted rows per engine (history length for the warm-up guard)
        self.W = bundle.spec.max_window

    def reset(self) -> None:
        self.validator.reset()
        self.hist.clear()
        self.flags.clear()
        self.seen.clear()

    def process(self, rows: pd.DataFrame) -> list[ScoreResult]:
        b, pol = self.b, self.b.policy
        units = rows["unit"].to_numpy()
        cycles = rows["cycle"].to_numpy()
        sets = rows[SETTINGS].to_numpy(dtype=float)
        sens = rows[SENSORS].to_numpy(dtype=float)
        results: dict[int, ScoreResult] = {}
        pending = []  # (row index, unit, cycle, window, clean flags)
        for i in range(len(rows)):
            u, c = int(units[i]), int(cycles[i])
            ok, reason, gap = self.validator.check(u, c)
            if not ok:
                results[i] = ScoreResult(u, c, "rejected", reason, CycleInputs(
                    u, c, data_bad=True, data_reasons=(f"ingest rejected row: {reason} cycle",)))
                continue
            h = self.hist.setdefault(u, _History())
            nan_s, nan_x = np.full(3, np.nan), np.full(sens.shape[1], np.nan)
            for g in range(min(gap, HISTORY)):          # cycles that never arrived become explicit missing rows
                h.add(c - min(gap, HISTORY) + g, nan_s, nan_x)
            h.add(c, sets[i], sens[i])
            cl = b.cleaner.clean_arrays(np.fromiter(h.cycles, dtype=np.int64), np.array(h.settings), np.array(h.sensors))
            z = cl.Z[cl.present][-self.W:]
            if len(z) < self.W:                          # start-up: shorter-window fallback via NaN left-padding
                z = np.vstack([np.full((self.W - len(z), z.shape[1]), np.nan), z])
            pending.append((i, u, c, z[None], cl))
        if pending:
            wins = np.concatenate([p[3] for p in pending])
            F = window_features(wins, b.spec)
            X = np.hstack([np.array([p[2] for p in pending], dtype=np.float32)[:, None], F])
            try:
                P = b.rul.predict(X)
                model_ok = np.isfinite(P).all(1)
            except Exception:                           # model failure -> agent raises a REVIEW exception ticket
                P, model_ok = np.full((len(X), 3), np.nan), np.zeros(len(X), dtype=bool)
            A = b.anomaly_percentile(X)
            for j, (i, u, c, _, cl) in enumerate(pending):
                self.seen[u] = self.seen.get(u, 0) + 1
                warm = self.seen[u] < pol.warmup_rows
                fl = self.flags.setdefault(u, deque(maxlen=pol.alert_n))
                fl.append(bool(A[j] >= pol.alert_A) and not warm)
                sustained = sum(fl) >= pol.alert_k
                reasons = []
                if cl.long_gap[-1]:
                    reasons.append("long data gap")
                if cl.stale[-1]:
                    reasons.append("stale / frozen feed")
                if cl.range_violation[-1]:
                    reasons.append("sensor values out of range")
                inp = CycleInputs(u, c, float(P[j, 0]), float(P[j, 1]), float(P[j, 2]), float(A[j]), bool(sustained),
                                  bool(cl.data_bad[-1]), tuple(reasons), bool(cl.ood[-1]), bool(model_ok[j]), warm)
                results[i] = ScoreResult(u, c, "ok", "ok", inp, X[j], {"regime": int(cl.regime[-1]), "imputed": int(cl.imputed_n[-1])})
        return [results[i] for i in range(len(rows))]
