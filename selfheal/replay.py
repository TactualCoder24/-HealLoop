"""Closed-loop replay: stream the official TEST engines one cycle at a time through
ingest -> clean -> features -> models -> triage agent -> work orders, then run the monitors.

`scenario="faults"` injects the data problems the agent must escalate to a human (long gap, frozen feed,
unseen operating regime, duplicate row) plus sensor drift for the drift monitor.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from .bundle import ModelBundle
from .config import outputs_dir, resolve, tickets_db_path
from .data import SENSORS, SETTINGS, load_subset
from .monitor import drift_report, monitor_summary, quality_report
from .scorer import Scorer
from .tickets import TicketStore
from .triage import TriageAgent
from .tuning import ticket_metrics


def inject_faults(test: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Deterministic fault injection on copies of the test stream (engine ids from configs `triage_demo.faults`)."""
    f = cfg["triage_demo"]["faults"]
    df = test.copy()
    log: dict = {}
    first = df.groupby("unit").cycle.min()
    last = df.groupby("unit").cycle.max()

    def at(u: int, frac: float) -> int:
        return int(first[u] + frac * (last[u] - first[u]))

    u = f["gap_engine"]                       # 8 consecutive cycles never arrive (> max_ffill_gap)
    c0 = at(u, 0.6)
    df = df[~((df.unit == u) & (df.cycle >= c0) & (df.cycle < c0 + 8))]
    log["long_gap"] = {"unit": u, "cycles": [c0, c0 + 7]}

    u = f["stale_engine"]                     # sensors freeze for 6 cycles
    c0 = at(u, 0.5)
    m = (df.unit == u) & (df.cycle >= c0) & (df.cycle < c0 + 6)
    ref = df[(df.unit == u) & (df.cycle == c0 - 1)][SENSORS].iloc[0]
    df.loc[m, SENSORS] = ref.to_numpy()
    log["stale_feed"] = {"unit": u, "cycles": [c0, c0 + 5]}

    u = f["regime_engine"]                    # operating settings far from any known regime
    c0 = at(u, 0.7)
    m = (df.unit == u) & (df.cycle >= c0) & (df.cycle < c0 + 4)
    df.loc[m, SETTINGS] = [31.0, 0.55, 30.0]
    log["unseen_regime"] = {"unit": u, "cycles": [c0, c0 + 3]}

    u = f["duplicate_engine"]                 # the same reading delivered twice (right behind the original)
    c0 = at(u, 0.4)
    df = df.reset_index(drop=True)
    i = int(df.index[(df.unit == u) & (df.cycle == c0)][0])
    df = pd.concat([df.iloc[: i + 1], df.iloc[[i]], df.iloc[i + 1:]], ignore_index=True)
    log["duplicate_row"] = {"unit": u, "cycle": c0}

    # sensor drift in a quarter of the fleet (moderate: below the range-violation threshold)
    sd = test[f["drift_sensor"]].std()
    drift_units = sorted(test["unit"].unique())[:: max(int(1 / f["drift_fraction"]), 1)]
    m = df.unit.isin(drift_units)
    df.loc[m, f["drift_sensor"]] = df.loc[m, f["drift_sensor"]] + f["drift_offset_sd"] * sd
    log["sensor_drift"] = {"sensor": f["drift_sensor"], "units": drift_units, "offset_sd": f["drift_offset_sd"]}
    return df.reset_index(drop=True), log


def _ticks(test: pd.DataFrame) -> list[pd.DataFrame]:
    t = test.copy()
    t["_pos"] = t.groupby("unit").cumcount()
    t = t.sort_values(["_pos", "unit"], kind="stable")
    b = np.flatnonzero(np.r_[True, t["_pos"].to_numpy()[1:] != t["_pos"].to_numpy()[:-1], True])
    cols = [c for c in t.columns if c != "_pos"]
    return [t.iloc[a:z][cols] for a, z in zip(b[:-1], b[1:])]


def stream_fleet(bundle: ModelBundle, frame: pd.DataFrame, key: str, failure_cycle: dict, store: TicketStore,
                 verbose: bool = False, max_ticks: int | None = None) -> tuple[pd.DataFrame, int, int]:
    """Core closed loop: one fleet tick at a time -> ingest checks -> clean -> features -> models -> triage agent -> tickets.

    `failure_cycle[u] - cycle` is verification metadata only (logged on ticket events); the agent never sees it.
    Returns (per-cycle log, rejected rows, ticks)."""
    scorer, agent = Scorer(bundle), TriageAgent(bundle.policy, store, key, bundle.version)
    ticks = _ticks(frame)
    if max_ticks:
        ticks = ticks[:max_ticks]
    log, rejected = [], 0
    for n, rows in enumerate(ticks):
        for res in scorer.process(rows):
            inp = res.inputs
            rejected += res.status == "rejected"
            true_rul = failure_cycle[res.unit] - res.cycle
            ex = (lambda r=res: bundle.explain(r.x_row)) if res.x_row is not None else None
            ev = agent.step(inp, explain=ex, true_rul=float(true_rul))
            log.append((res.unit, res.cycle, res.status, inp.p10, inp.p50, inp.p90, inp.anomaly_A, inp.sustained, inp.data_bad,
                        inp.unseen_regime, ev["rule_id"], ev["severity"], ev["priority"], ev["action"], ev["ticket_id"], float(true_rul)))
        if verbose and (n + 1) % 100 == 0:
            print(f"  [replay {key}] tick {n + 1}/{len(ticks)}")
    store.flush()
    cols = ["unit", "cycle", "status", "p10", "p50", "p90", "A", "sustained", "data_bad", "unseen_regime", "rule_id",
            "severity", "priority", "action", "ticket_id", "true_rul"]
    return pd.DataFrame(log, columns=cols), rejected, len(ticks)


def replay(cfg: dict, bundle: ModelBundle, scenario: str = "clean", verbose: bool = True, max_ticks: int | None = None) -> dict:
    """Replay the official TEST fleet (truncated histories, true RUL known only for verification)."""
    t0 = time.time()
    subset = cfg["subset"]
    key = subset if scenario == "clean" else f"{subset}+{scenario}"
    _, test, rul = load_subset(resolve(cfg, "raw_dir"), subset)
    last = test.groupby("unit")["cycle"].max()
    failure_cycle = (last + rul.reindex(last.index)).to_dict()
    fault_log = {}
    if scenario == "faults":
        test, fault_log = inject_faults(test, cfg)

    out_dir = outputs_dir(cfg) / scenario
    out_dir.mkdir(parents=True, exist_ok=True)
    store = TicketStore(tickets_db_path(cfg))
    store.reset(key)
    scores, rejected, n_ticks = stream_fleet(bundle, test, key, failure_cycle, store, verbose, max_ticks)
    scores.to_parquet(out_dir / "replay_scores.parquet", index=False)
    store.export(out_dir, key)

    # ---- monitors (drift vs train early-life reference, data-quality counters) ------------------------------
    cleaned = bundle.cleaner.transform_frame(test.drop_duplicates(["unit", "cycle"], keep="first"))
    drift = drift_report(bundle.reference, cleaned, cfg)
    drift.to_csv(out_dir / "drift.csv", index=False)
    quality = quality_report(cleaned, rejected, cfg)
    mon = {**monitor_summary(drift, quality), "quality": quality}

    summary = summarise(cfg, scores, store, key, last, rul, bundle)
    summary.update({"scenario": scenario, "ticks": n_ticks, "seconds": round(time.time() - t0, 1), "monitor": mon,
                    "fault_log": fault_log, "model_version": bundle.version, "policy_version": bundle.policy.version,
                    "rejected_rows": rejected})
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    store.flush()
    store.db.close()
    return summary


def summarise(cfg: dict, scores: pd.DataFrame, store: TicketStore, key: str, last: pd.Series, rul: pd.Series, bundle: ModelBundle) -> dict:
    """Agent behaviour on the official test fleet (engine states at the last observed cycle vs the official RUL)."""
    tcfg = cfg["policy"]["tuning"]
    t = store.tickets(key)
    final = scores.sort_values(["unit", "cycle"]).groupby("unit").tail(1).set_index("unit")
    end_rul = rul.reindex(final.index)
    ev = store.events(key)
    hi_ev = ev[ev["priority"].isin(["P1", "P2"]) & ev["event"].isin(["created", "escalated"])]
    ever_hi = set(hi_ev["unit"])
    risk = end_rul <= tcfg["risk_horizon"]
    healthy = end_rul > tcfg["false_alarm_rul"]
    risk_units, healthy_units = risk[risk].index, healthy[healthy].index
    sev_by_tier = pd.crosstab(pd.cut(end_rul, [-1, 15, 40, 80, 1000], labels=["<=15", "16-40", "41-80", ">80"]), final["severity"])
    tm = ticket_metrics(store, len(final), tcfg, key)
    tm.pop("late_P1P2_engines", None)   # not meaningful on truncated test engines (most are still healthy)
    out = {
        "engines": int(len(final)),
        "tickets_total": int(len(t)),
        "tickets_by_priority": t["priority"].value_counts().to_dict() if len(t) else {},
        "tickets_by_type": t["type"].value_counts().to_dict() if len(t) else {},
        "tickets_per_engine": float(len(t) / max(len(final), 1)),
        "risk_engines_rul_le_30": int(len(risk_units)),
        "risk_engines_flagged_P1P2_ever": int(sum(1 for u in risk_units if u in ever_hi)),
        "risk_engines_P1P2_at_end": int(final.loc[risk_units, "priority"].isin(["P1", "P2"]).sum()),
        "healthy_engines_rul_gt_80": int(len(healthy_units)),
        "healthy_engines_rul_gt_80_with_P1P2_at_end": int(final.loc[healthy_units, "priority"].isin(["P1", "P2"]).sum()),
        "final_severity_by_true_rul_tier": {str(k): {kk: int(vv) for kk, vv in v.items()} for k, v in sev_by_tier.to_dict(orient="index").items()},
        "review_tickets": int((t["type"] == "exception").sum()) if len(t) else 0,
        "ticket_metrics": tm,
    }
    return out
