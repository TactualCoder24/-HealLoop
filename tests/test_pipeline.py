"""End-to-end invariants: causality, train/inference parity, data handling, conformal calibration, split hygiene."""
import numpy as np
import pandas as pd
import pytest

from selfheal import metrics as M
from selfheal.clean import Cleaner
from selfheal.config import load_config
from selfheal.data import COLUMNS, SENSORS, SETTINGS, CycleValidator, DataValidationError, validate_frame
from selfheal.features import FeatureSpec, build_features
from selfheal.rul import cqr_qhat
from selfheal.scorer import Scorer
from selfheal.train import split_engines


# ------------------------------------------------------------------------------------------ ingest / validation
def test_clean_frame_validates(raw):
    train, _, _ = raw
    rep = validate_frame(train, "train")
    assert not rep.errors and train.shape[1] == 26 and list(train.columns) == COLUMNS


def test_validation_rejects_duplicates_and_out_of_order(raw):
    train, _, _ = raw
    dup = pd.concat([train.iloc[:10], train.iloc[[5]], train.iloc[10:]], ignore_index=True)
    with pytest.raises(DataValidationError):
        validate_frame(dup, "dup").raise_if_fatal()
    shuffled = train.iloc[:50].copy()
    shuffled.iloc[[3, 4]] = shuffled.iloc[[4, 3]].to_numpy()
    with pytest.raises(DataValidationError):
        validate_frame(shuffled, "ooo").raise_if_fatal()


def test_cycle_validator_streaming():
    v = CycleValidator()
    assert v.check(1, 1) == (True, "ok", 0)
    assert v.check(1, 2) == (True, "ok", 0)
    assert v.check(1, 2)[:2] == (False, "duplicate")
    assert v.check(1, 1)[:2] == (False, "out_of_order")
    assert v.check(1, 6) == (True, "ok", 3)          # cycles 3,4,5 never arrived


# ------------------------------------------------------------------------------------------ cleaning
@pytest.fixture(scope="module")
def cleaner(raw):
    train, _, _ = raw
    return Cleaner(load_config("FD001")).fit(train)


def test_constant_sensors_dropped_from_train_variance(cleaner):
    assert {"s1", "s5", "s10", "s16", "s18", "s19"} <= set(cleaner.dropped)
    assert "s11" in cleaner.kept


def _engine(raw, unit=1):
    d = raw[0][raw[0].unit == unit]
    return d["cycle"].to_numpy(), d[SETTINGS].to_numpy(float), d[SENSORS].to_numpy(float)


def test_cleaning_is_causal(raw, cleaner):
    c, s, x = _engine(raw)
    full = cleaner.clean_arrays(c, s, x)
    cut = 90
    part = cleaner.clean_arrays(c[:cut], s[:cut], x[:cut])
    np.testing.assert_allclose(full.Z[:cut], part.Z, atol=1e-12)          # the future never changes the past


def test_short_gap_forward_filled_long_gap_flagged(raw, cleaner):
    c, s, x = _engine(raw)
    x = x.copy()
    x[50:52] = np.nan                                  # 2-cycle gap (<= max_ffill_gap)
    x[100:108] = np.nan                                # 8-cycle gap (> max_ffill_gap)
    cl = cleaner.clean_arrays(c, s, x)
    assert cl.imputed_n[50] > 0 and not cl.long_gap[51] and not cl.data_bad[51]
    assert np.isfinite(cl.Z).all()
    assert cl.long_gap[105] and cl.data_bad[105]
    assert cl.long_gap[109]                            # stays flagged for `long_gap_hold` cycles after data resumes
    assert not cl.long_gap[130]


def test_missing_cycles_become_explicit_rows(raw, cleaner):
    c, s, x = _engine(raw)
    keep = np.r_[0:40, 46:len(c)]
    cl = cleaner.clean_arrays(c[keep], s[keep], x[keep])
    assert len(cl.cycles) == len(c) and (~cl.present).sum() == 6 and cl.long_gap[44]


def test_stale_feed_and_unseen_regime_flagged(raw, cleaner):
    c, s, x = _engine(raw)
    x, s = x.copy(), s.copy()
    x[70:76] = x[69]
    s[120:123] = [31.0, 0.55, 30.0]
    cl = cleaner.clean_arrays(c, s, x)
    assert cl.stale[74] and not cl.stale[60]
    assert cl.ood[121] and not cl.ood[10]


# ------------------------------------------------------------------------------------------ features
def test_features_are_causal_no_leakage(raw, cleaner):
    train, _, _ = raw
    spec = FeatureSpec(cleaner.kept)
    d = train[train.unit.isin([1, 2])]
    cleaned = cleaner.transform_frame(d)
    full = build_features(cleaned, spec)
    e1 = cleaned[cleaned.unit == 1]
    cut = 77
    tampered = cleaned.copy()
    m = (tampered.unit == 1) & (tampered.cycle > cut)
    tampered.loc[m, cleaner.kept] = 1e6                                        # destroy the future of engine 1
    again = build_features(tampered, spec)
    n1 = len(e1)
    np.testing.assert_array_equal(full[:cut], again[:cut])                    # features up to `cut` unchanged
    np.testing.assert_array_equal(full[n1:], again[n1:])                      # other engines untouched


def test_startup_uses_shorter_windows(raw, cleaner):
    train, _, _ = raw
    spec = FeatureSpec(cleaner.kept)
    X = build_features(cleaner.transform_frame(train[train.unit == 1]), spec)
    assert np.isfinite(X).all()
    names = spec.names()
    assert X[0, names.index(f"{cleaner.kept[0]}__std30")] == 0.0               # single sample -> zero spread, no NaN


# ------------------------------------------------------------------------------------------ streaming == batch
def test_streaming_matches_batch(raw, small_bundle):
    _, test, _ = raw
    b = small_bundle
    sub = test[test.unit.isin([1, 2, 3])]
    batch, _ = b.batch_score(sub)
    sc = Scorer(b)
    ticks = sub.assign(_p=sub.groupby("unit").cumcount()).sort_values(["_p", "unit"], kind="stable")
    rows = []
    for _, tick in ticks.groupby("_p", sort=True):
        rows += [(r.unit, r.cycle, r.inputs.p10, r.inputs.p50, r.inputs.p90, r.inputs.anomaly_A, r.inputs.sustained)
                 for r in sc.process(tick.drop(columns="_p"))]
    st = pd.DataFrame(rows, columns=["unit", "cycle", "p10", "p50", "p90", "A", "sustained"]).sort_values(["unit", "cycle"]).reset_index(drop=True)
    batch = batch.sort_values(["unit", "cycle"]).reset_index(drop=True)
    np.testing.assert_allclose(st[["p10", "p50", "p90", "A"]].to_numpy(), batch[["p10", "p50", "p90", "A"]].to_numpy(), atol=1e-5)
    assert (st["sustained"].to_numpy() == batch["sustained"].to_numpy()).all()


def test_streaming_rejects_duplicates_and_reports_gaps(raw, small_bundle):
    _, test, _ = raw
    e = test[test.unit == 4].reset_index(drop=True)
    sc = Scorer(small_bundle)
    out = [sc.process(e.iloc[[i]])[0] for i in range(40)]
    assert all(r.status == "ok" for r in out)
    dup = sc.process(e.iloc[[39]])[0]
    assert dup.status == "rejected" and dup.inputs.data_bad
    gap = sc.process(e.iloc[[49]])[0]                                      # cycles 41..49 never arrived (long gap)
    assert gap.status == "ok" and gap.inputs.data_bad and "long data gap" in gap.inputs.data_reasons


# ------------------------------------------------------------------------------------------ models / metrics
def test_conformal_restores_coverage():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 4000)
    lo, hi = np.full(4000, -0.6), np.full(4000, 0.6)                           # too narrow -> ~45% coverage
    cal, tst = slice(0, 2000), slice(2000, 4000)
    q = cqr_qhat(lo[cal], hi[cal], y[cal], 0.2)
    cov = np.mean((y[tst] >= lo[tst] - q) & (y[tst] <= hi[tst] + q))
    assert q > 0 and abs(cov - 0.8) < 0.04


def test_weighted_conformal_matches_unweighted_for_uniform_weights():
    rng = np.random.default_rng(1)
    y = rng.normal(size=1000)
    lo, hi = np.full(1000, -0.5), np.full(1000, 0.5)
    assert abs(cqr_qhat(lo, hi, y, 0.2) - cqr_qhat(lo, hi, y, 0.2, np.ones(1000))) < 0.05


def test_nasa_score_is_asymmetric():
    assert M.nasa_score([50], [60]) > M.nasa_score([50], [40])                # late costs more than early
    assert M.nasa_score([50], [50]) == 0.0


def test_endpoint_like_weights_one_per_engine():
    units = np.repeat([1, 2], [100, 50])
    cyc = np.r_[np.arange(1, 101), np.arange(1, 51)]
    rul = np.r_[100 - np.arange(1, 101), 50 - np.arange(1, 51)]
    m, w = M.endpoint_like(units, cyc, rul, 31, 150)
    assert abs(w[units == 1].sum() - 1.0) < 1e-9 and abs(w[units == 2].sum() - 1.0) < 1e-9 and not m[cyc < 31].any()


def test_engine_split_is_grouped_and_disjoint():
    cfg = load_config("FD002")
    sp = split_engines(range(1, 261), cfg)
    sets = [set(sp[k]) for k in ("fit", "calib", "val", "reserve")]
    for i in range(4):
        for j in range(i + 1, 4):
            assert not (sets[i] & sets[j])                                      # no engine appears in two groups
    assert sum(len(s) for s in sets) == 260 and len(sp["reserve"]) == round(260 * cfg["retrain"]["reserve_fraction"])
    assert split_engines(range(1, 261), cfg) == sp                              # fixed seed -> reproducible
    # calibration / validation engines do not depend on the reserve size -> champion and challenger are compared like-for-like
    cfg0 = load_config("FD002")
    cfg0["retrain"]["reserve_fraction"] = 0.0
    sp0 = split_engines(range(1, 261), cfg0)
    assert sp0["calib"] == sp["calib"] and sp0["val"] == sp["val"] and set(sp0["fit"]) == set(sp["fit"]) | set(sp["reserve"])


def test_startup_rows_are_flagged_warmup_and_never_raise_anomaly_tickets(raw, small_bundle):
    _, test, _ = raw
    e = test[test.unit == 6].reset_index(drop=True)
    sc = Scorer(small_bundle)
    res = [sc.process(e.iloc[[i]])[0] for i in range(14)]
    n = small_bundle.policy.warmup_rows
    assert all(r.inputs.warmup for r in res[: n - 1]) and not any(r.inputs.warmup for r in res[n - 1:])
    assert not any(r.inputs.sustained for r in res[: n - 1])                    # short-window anomaly percentiles are not trusted


def test_quantiles_ordered_and_bounded(raw, small_bundle):
    _, test, _ = raw
    sc, _ = small_bundle.batch_score(test[test.unit.isin([1, 2])])
    assert (sc.p10 <= sc.p50 + 1e-9).all() and (sc.p50 <= sc.p90 + 1e-9).all()
    assert sc.p10.min() >= 0 and sc.p90.max() <= small_bundle.cfg["labels"]["rul_cap"] + 1e-9
