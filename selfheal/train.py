"""Training pipeline for one C-MAPSS subset: ingest -> clean -> features -> anomaly + RUL -> calibrate -> tune -> bundle."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from . import metrics as M
from .anomaly import HealthyBaseline, sustained, tune_alert
from .bundle import ModelBundle
from .clean import Cleaner
from .config import resolve
from .data import ValidationReport, load_subset, train_bounds, validate_frame
from .features import FeatureSpec, build_features
from .monitor import bin_counts
from .rul import QuantileRUL
from .triage import Policy
from .tuning import agent_replay_metrics, policy_metrics, tune_policy

@dataclass
class TrainResult:
    bundle: ModelBundle
    report: dict            # tables / extra artifacts for MLflow
    seconds: float


def load_and_validate(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, list[ValidationReport]]:
    train, test, rul = load_subset(resolve(cfg, "raw_dir"), cfg["subset"])
    rep_tr = validate_frame(train, f"train_{cfg['subset']}")
    rep_tr.raise_if_fatal()
    rep_te = validate_frame(test, f"test_{cfg['subset']}", bounds=train_bounds(train))
    rep_te.raise_if_fatal()
    if len(rul) != test["unit"].nunique():
        raise ValueError(f"RUL file has {len(rul)} rows but test has {test['unit'].nunique()} engines")
    return train, test, rul, [rep_tr, rep_te]


def split_engines(units, cfg: dict) -> dict:
    """Grouped (by ENGINE ID) split of the TRAIN engines: calibration / validation / reserve / fit.

    `reserve` engines are held back from the first model and only 'released' (as completed, labelled run-to-failure
    histories) when the retraining flow runs; fit + reserve is what a model trained on all history would see.
    The official test set takes no part in any of this.
    """
    rng = np.random.default_rng(cfg["seed"])
    u = np.array(sorted(int(x) for x in units))
    rng.shuffle(u)
    n = len(u)
    n_cal = max(3, round(n * cfg["split"]["calib_frac"]))
    n_val = max(3, round(n * cfg["split"]["val_frac"]))
    n_res = int(round(n * cfg["retrain"]["reserve_fraction"]))
    a, b = n_cal, n_cal + n_val
    return {"calib": sorted(u[:a].tolist()), "val": sorted(u[a:b].tolist()),
            "reserve": sorted(u[b:b + n_res].tolist()), "fit": sorted(u[b + n_res:].tolist())}


# ---------------------------------------------------------------------------------------- evaluation
def endpoints(scored: pd.DataFrame) -> np.ndarray:
    u = scored["unit"].to_numpy()
    return np.r_[u[1:] != u[:-1], True]


def evaluate_test(bundle: ModelBundle, test: pd.DataFrame, rul: pd.Series) -> dict:
    """Locked official test endpoints: RMSE + NASA score (vs official RUL), interval coverage/width (vs capped RUL)."""
    sc, _ = bundle.batch_score(test)
    ep = sc[endpoints(sc)].reset_index(drop=True)
    y = rul.reindex(ep["unit"]).to_numpy()
    cap = bundle.cfg["labels"]["rul_cap"]
    yc = np.minimum(y, cap)
    out = {**M.point_metrics(y, ep["p50"]), "rmse_vs_capped_truth": M.rmse(yc, ep["p50"]),
           "coverage_80": M.coverage(yc, ep["p10"], ep["p90"]), "interval_width": M.width(ep["p10"], ep["p90"]),
           "late_share": float(np.mean(ep["p50"].to_numpy() > y)), **M.risk_window_metrics(y, ep["p50"].to_numpy(), 30)}
    return out


def baselines_on_test(cfg: dict, bundle: ModelBundle, X_fit, y_fit, life_fit_med, test: pd.DataFrame, rul: pd.Series) -> dict:
    """Simple baselines on the official test endpoints: constant, age-based, ridge on the same window features."""
    cap = cfg["labels"]["rul_cap"]
    cleaned, Xt = bundle.featurize(test)
    ep = endpoints(cleaned)
    units, cyc, Xe = cleaned["unit"].to_numpy()[ep], cleaned["cycle"].to_numpy()[ep], Xt[ep]
    y = rul.reindex(units).to_numpy()
    out = {}
    out["constant_median"] = M.point_metrics(y, np.full(len(y), float(np.median(y_fit))))
    out["age_based"] = M.point_metrics(y, np.clip(life_fit_med - cyc, 0, cap))
    sc = StandardScaler().fit(X_fit)
    ridge = Ridge(alpha=10.0).fit(sc.transform(X_fit), y_fit)
    out["ridge_window_features"] = M.point_metrics(y, np.clip(ridge.predict(sc.transform(Xe)), 0, cap))
    return out


# ---------------------------------------------------------------------------------------- training
def train_subset(cfg: dict, *, split: dict | None = None, release_reserve: bool = False, verbose: bool = True) -> TrainResult:
    """Train one subset.  `release_reserve=True` is the retraining flow: the reserved, completed run-to-failure
    histories join the fit engines (calibration and validation engines stay identical -> like-for-like gate)."""
    t0 = time.time()
    say = print if verbose else (lambda *a, **k: None)
    subset, seed, cap = cfg["subset"], cfg["seed"], cfg["labels"]["rul_cap"]

    # ---- 1. INGEST & VALIDATE -----------------------------------------------------------------------
    train, test, rul_test, reps = load_and_validate(cfg)
    for r in reps:
        say(f"  [ingest] {r.name}: {r.n_rows:,} rows, {r.n_units} engines, errors={len(r.errors)}, warnings={[(w.code, w.count) for w in r.warnings]}")
    life_map = {int(u): int(c) for u, c in train.groupby("unit")["cycle"].max().items()}
    split = split or split_engines(train["unit"].unique(), cfg)
    all_train = train
    fit_units = list(split["fit"]) + (list(split["reserve"]) if release_reserve else [])
    held = 0 if release_reserve else len(split["reserve"])
    say(f"  [split] fit={len(fit_units)}{' (incl. ' + str(len(split['reserve'])) + ' released reserve engines)' if release_reserve else ''} "
        f"calib={len(split['calib'])} val={len(split['val'])} reserve(held back)={held} engines (grouped by engine id; test untouched)")

    # ---- 2. CLEAN (train-only transforms) + 3. FEATURES ---------------------------------------------
    cleaner = Cleaner(cfg).fit(all_train[all_train["unit"].isin(fit_units)])
    say(f"  [clean] kept {len(cleaner.kept)} sensors, dropped near-constant {cleaner.dropped}; regimes={cleaner.regime.n_regimes}")
    spec = FeatureSpec(cleaner.kept, tuple(cfg["features"]["windows"]), int(cfg["features"]["ewma_span"]))
    cleaned = cleaner.transform_frame(all_train)
    X = build_features(cleaned, spec)
    unit, cyc = cleaned["unit"].to_numpy(), cleaned["cycle"].to_numpy()
    rul = (cleaned["unit"].map(life_map).to_numpy() - cyc).astype(float)       # failure cycle - current cycle
    y = np.minimum(rul, cap)
    m_fit, m_cal, m_val = (np.isin(unit, g) for g in (fit_units, split["calib"], split["val"]))
    warm_all = (pd.Series(unit).groupby(unit).cumcount().to_numpy() + 1) < cfg["policy"]["warmup_rows"]   # start-up rows
    say(f"  [features] {X.shape[1]} features x {len(X):,} rows (past-only windows {list(spec.windows)})")

    # ---- 4a. ANOMALY DETECTION (healthy proxy = first N cycles of TRAIN engines) -----------------------
    acfg = cfg["anomaly"]
    names = spec.names()
    aidx = [names.index(f"{s}__{acfg['input_stat']}") for s in spec.sensors]
    h = acfg["healthy_cycles"]
    anomaly = HealthyBaseline(acfg["if_trees"], acfg["pca_variance"], acfg["fuse_weights"], seed).fit(X[m_fit & (cyc <= h)][:, aidx])
    anomaly.set_reference(anomaly.score(X[m_cal & (cyc <= h)][:, aidx]))
    A = anomaly.percentile(anomaly.score(X[:, aidx]))
    alert, alert_rows = tune_alert(A[m_val], unit[m_val], cyc[m_val], rul[m_val], acfg["candidates"]["A"],
                                   acfg["candidates"]["persistence"], acfg["early_life_cycles"], acfg["early_alert_budget"], cap, warm_all[m_val])
    say(f"  [anomaly] IF+PCA-SPE fitted on {int((m_fit & (cyc <= h)).sum())} healthy rows; alert A>={alert['A']} in {alert['k']}/{alert['n']} cycles (validation-tuned)")

    # ---- 4b. RUL QUANTILES (LightGBM P10/P50/P90) with near-failure re-weighting -----------------------
    rp = cfg["rul"]
    thr = rp["near_failure"]["threshold"]
    Xf, yf, Xv, yv = X[m_fit], y[m_fit], X[m_val], y[m_val]
    rf, rv = rul[m_fit], rul[m_val]                       # uncapped targets for the tail quantiles
    ccfg = cfg["conformal"]
    ep_v, w_v = M.endpoint_like(unit[m_val], cyc[m_val], rv, ccfg["min_cycle"], ccfg["horizon"])    # endpoint-like validation rows
    ep_c, w_c = M.endpoint_like(unit[m_cal], cyc[m_cal], rul[m_cal], ccfg["min_cycle"], ccfg["horizon"])
    wfun = lambda w: np.where(rf <= thr, w, 1.0)
    cands, models = [], {}
    for w in rp["near_failure"]["weights"]:
        m = QuantileRUL((0.5,), rp["lgbm"], seed, cap).fit(Xf, yf, rf, wfun(w), Xv, yv, rv)
        p = m.models[0.5].predict(Xv)
        rk = M.risk_window_metrics(rv, p, thr)
        cands.append({"weight": float(w), "val_rmse": M.rmse(yv, p), "ep_rmse": M.wrmse(rv[ep_v], p[ep_v], w_v[ep_v]),
                      "ep_nasa_mean": M.wnasa_mean(rv[ep_v], p[ep_v], w_v[ep_v]),
                      "val_rmse_risk": rk["rmse_risk"], "val_late_bias_risk": rk["late_bias_risk"], "best_iter": m.best_iters[0.5]})
        models[float(w)] = m
    best_rmse = min(c["ep_rmse"] for c in cands)
    ok = [c for c in cands if c["ep_rmse"] <= 1.05 * best_rmse]              # risk-aware pick, RMSE within 5% of the best
    chosen = min(ok, key=lambda c: (c["ep_nasa_mean"], c["weight"]))
    say(f"  [rul] near-failure weight candidates {[(c['weight'], round(c['ep_rmse'], 2), round(c['ep_nasa_mean'], 3)) for c in cands]} -> chosen {chosen['weight']}")
    rul_model = QuantileRUL(tuple(rp["quantiles"]), rp["lgbm"], seed, cap)
    rul_model.fit(Xf, yf, rf, wfun(chosen["weight"]), Xv, yv, rv, only=(rp["quantiles"][0], rp["quantiles"][-1]))
    rul_model.models[0.5] = models[chosen["weight"]].models[0.5]
    rul_model.best_iters[0.5] = models[chosen["weight"]].best_iters[0.5]
    xc = X[m_cal][ep_c]
    qhat = rul_model.calibrate(xc, y[m_cal][ep_c], ccfg["alpha"], w_c[ep_c])
    say(f"  [conformal] weighted CQR correction q^={qhat:.2f} cycles on {int(ep_c.sum()):,} endpoint-like rows of {len(np.unique(unit[m_cal]))} held-out engines")

    # ---- validation metrics (all rows, and endpoint-like = one random cut per engine) -----------------------
    Pv = rul_model.predict(Xv)
    Praw = rul_model.predict_raw(Xv)
    val_metrics = {**M.point_metrics(yv, Pv[:, 1]), "nasa_score_mean": M.nasa_score(yv, Pv[:, 1]) / len(yv),
                   "coverage_80_all_rows": M.coverage(yv, Pv[:, 0], Pv[:, 2]), **M.risk_window_metrics(rv, Pv[:, 1], thr),
                   "ep_rmse": M.wrmse(rv[ep_v], Pv[ep_v, 1], w_v[ep_v]), "ep_nasa_mean": M.wnasa_mean(rv[ep_v], Pv[ep_v, 1], w_v[ep_v]),
                   "ep_coverage_80": M.wcoverage(yv[ep_v], Pv[ep_v, 0], Pv[ep_v, 2], w_v[ep_v]),
                   "ep_coverage_80_uncalibrated": M.wcoverage(yv[ep_v], Praw[ep_v, 0], Praw[ep_v, 2], w_v[ep_v]),
                   "ep_interval_width": M.wwidth(Pv[ep_v, 0], Pv[ep_v, 2], w_v[ep_v])}
    # larger held-out pool for the retrain gate: calibration + validation engines are both out-of-sample for the fit
    # (coverage is excluded: the calibration engines set q-hat), which keeps the gate from flipping on 20 noisy engines
    Pc = rul_model.predict(X[m_cal])
    h_u, h_c = np.r_[unit[m_val], unit[m_cal]], np.r_[cyc[m_val], cyc[m_cal]]
    h_r, h_p = np.r_[rv, rul[m_cal]], np.r_[Pv[:, 1], Pc[:, 1]]
    ep_h, w_h = M.endpoint_like(h_u, h_c, h_r, ccfg["min_cycle"], ccfg["horizon"])
    val_metrics.update({"hold_ep_rmse": M.wrmse(h_r[ep_h], h_p[ep_h], w_h[ep_h]),
                        "hold_ep_nasa_mean": M.wnasa_mean(h_r[ep_h], h_p[ep_h], w_h[ep_h]),
                        "hold_rmse_risk": M.risk_window_metrics(h_r, h_p, thr)["rmse_risk"],
                        "hold_engines": int(len(np.unique(h_u)))})
    say(f"  [val] RMSE(all rows, capped)={val_metrics['rmse']:.2f} | endpoint-like: RMSE={val_metrics['ep_rmse']:.2f}  NASA(mean)={val_metrics['ep_nasa_mean']:.2f}  "
        f"coverage80={val_metrics['ep_coverage_80']:.3f} (uncalibrated {val_metrics['ep_coverage_80_uncalibrated']:.3f})  width={val_metrics['ep_interval_width']:.1f}")

    # ---- 5. TRIAGE POLICY TUNING on grouped run-to-failure replay of validation engines ----------------
    pol0 = Policy.from_cfg(cfg, alert_A=alert["A"], alert_k=alert["k"], alert_n=alert["n"])
    val_df = pd.DataFrame({"unit": unit[m_val], "cycle": cyc[m_val], "rul": rul[m_val], "p10": Pv[:, 0], "p50": Pv[:, 1], "p90": Pv[:, 2],
                           "A": A[m_val], "data_bad": False, "warmup": warm_all[m_val]})
    val_df["sustained"] = sustained((A[m_val] >= pol0.alert_A) & ~warm_all[m_val], unit[m_val], pol0.alert_k, pol0.alert_n)
    policy, prep = tune_policy(val_df, cfg, pol0)
    agent_val = agent_replay_metrics(val_df, policy, subset, cfg["policy"]["tuning"])
    say(f"  [policy] tuned cut-offs {prep['chosen']}  recall(RUL<=30)={prep['tuned']['recall_risk_window']:.2f}  "
        f"false-alarm P1/P2={prep['tuned']['false_alarm_rate_P1P2']:.3f}  tickets/engine={agent_val['tickets_per_engine']:.2f}")

    # ---- 6. drift reference (early-life cycles of TRAIN engines) -----------------------------------------
    mon = cfg["monitor"]
    ref_mask = m_fit & (cyc <= mon["reference_cycles"])
    reference = {"sensors": cleaner.kept, "n": int(ref_mask.sum()), "cycles": mon["reference_cycles"], "bins": {}}
    for j, s in enumerate(cleaner.kept):
        v = cleaned[s].to_numpy()[ref_mask]
        edges = np.unique(np.quantile(v, np.linspace(0, 1, 11)))
        edges[0], edges[-1] = -np.inf, np.inf
        counts = bin_counts(v, edges)
        reference["bins"][s] = {"edges": edges.tolist(), "p": (counts / counts.sum()).tolist()}
    reference["quality_rate"] = float(cleaned["data_bad"].mean())

    bundle = ModelBundle(subset=subset, cfg=cfg, cleaner=cleaner, spec=spec, rul=rul_model, anomaly=anomaly, policy=policy,
                         split=split, reference=reference,
                         meta={"fit_units": len(fit_units), "released": bool(release_reserve), "n_features": int(X.shape[1]), "near_failure_weight": chosen["weight"],
                               "life_median_fit": float(np.median([life_map[u] for u in fit_units]))})

    # ---- 7. LOCKED OFFICIAL TEST (never used for tuning) ---------------------------------------------------
    test_metrics = evaluate_test(bundle, test, rul_test)
    base = baselines_on_test(cfg, bundle, Xf, yf, bundle.meta["life_median_fit"], test, rul_test)
    say(f"  [test] RMSE={test_metrics['rmse']:.2f}  NASA score={test_metrics['nasa_score']:.0f}  coverage80(vs capped)={test_metrics['coverage_80']:.3f}"
        f"  | baselines RMSE: " + ", ".join(f"{k}={v['rmse']:.1f}" for k, v in base.items()))

    bundle.metrics = {"val": val_metrics, "test": test_metrics, "baselines_test": base,
                      "policy_val_rule_stream": prep["tuned"], "policy_val_default": prep["default"], "agent_val": agent_val,
                      "alert": alert, "conformal_qhat": qhat}
    report = {"near_failure_candidates.csv": pd.DataFrame(cands), "alert_tuning.csv": pd.DataFrame(alert_rows),
              "policy_grid.csv": pd.DataFrame(prep["grid"]), "policy_width_search.csv": pd.DataFrame(prep["width_search"]),
              "validation_report.json": [r.summary() for r in reps],
              "feature_importance.csv": pd.DataFrame({"feature": names, "gain_p50": rul_model.models[0.5].booster_.feature_importance("gain")}).sort_values("gain_p50", ascending=False)}
    return TrainResult(bundle, report, time.time() - t0)
