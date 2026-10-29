"""Validation-tuning of the triage policy candidates (cut-offs, W*) on grouped run-to-failure replay.

The tuner works on the stateless rule stream (vectorised `decide_batch`); the stateful agent is evaluated
separately in `agent_replay_metrics`.  The cost is deliberately asymmetric - a late or missed alert costs far
more than a false alarm (design principle) - and every weight is in configs/base.yaml.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .tickets import TicketStore
from .triage import CycleInputs, Policy, TriageAgent, decide_batch


def _slices(units: np.ndarray):
    b = np.flatnonzero(np.r_[True, units[1:] != units[:-1], True])
    return list(zip(b[:-1], b[1:]))


def _onsets(active: np.ndarray, debounce: int) -> int:
    idx = np.flatnonzero(active)
    return 0 if len(idx) == 0 else 1 + int(np.sum(np.diff(idx) > debounce))


def policy_metrics(df: pd.DataFrame, pol: Policy, tcfg: dict) -> dict:
    """Rule-stream metrics on run-to-failure engines; `df` needs unit, rul (true, uncapped), p10/p50/p90, sustained."""
    units = df["unit"].to_numpy()
    rul = df["rul"].to_numpy(float)
    rule, pri = decide_batch(df["p10"], df["p50"], df["p90"], df["sustained"], df["data_bad"] if "data_bad" in df else np.zeros(len(df), bool), pol,
                             df["warmup"] if "warmup" in df else None)
    m30, healthy = rul <= tcfg["risk_horizon"], rul > tcfg["false_alarm_rul"]
    recall = float(np.mean(pri[m30] >= 2)) if m30.any() else float("nan")
    fa = float(np.mean(pri[healthy] >= 2)) if healthy.any() else 0.0
    ea = float(np.mean(pri[healthy] >= 1)) if healthy.any() else 0.0
    late, leads, tickets, reviews = [], [], [], []
    for a, b in _slices(units):
        p, r, ru = pri[a:b], rul[a:b], rule[a:b]
        late.append(not np.any((p >= 2) & (r >= tcfg["late_lead_cycles"])))
        hit = (p >= 2) & (r <= tcfg["false_alarm_rul"])
        if hit.any():
            leads.append(float(r[hit][0]))
        tickets.append(_onsets(p > 0, pol.deescalate_after))
        reviews.append(_onsets(ru == 3, pol.deescalate_after))
    n = len(late)
    cost_w = tcfg["cost"]
    out = {
        "recall_risk_window": recall, "false_alarm_rate_P1P2": fa, "ticket_eligible_healthy_rate": ea,
        "late_engine_frac": float(np.mean(late)), "median_lead_cycles": float(np.median(leads)) if leads else 0.0,
        "p10_lead_cycles": float(np.percentile(leads, 10)) if leads else 0.0,
        "tickets_per_engine": float(np.sum(tickets) / n), "review_onsets_per_engine": float(np.sum(reviews) / n),
        "engines": n,
    }
    out["cost"] = float(
        cost_w["late_engine"] * out["late_engine_frac"] + cost_w["miss_recall"] * (1 - (recall if recall == recall else 1.0))
        + cost_w["false_alarm"] * fa + cost_w["early_alerts"] * ea + cost_w["tickets"] * out["tickets_per_engine"])
    return out


def tune_policy(val: pd.DataFrame, cfg: dict, base: Policy) -> tuple[Policy, dict]:
    tcfg = cfg["policy"]["tuning"]
    grid = tcfg["grid"]
    widths = (val["p90"] - val["p10"]).to_numpy()

    # stage 1 - W*: the tightest width quantile whose exception (REVIEW) load fits the per-engine budget
    w_rows, chosen_w = [], None
    for q in sorted(grid["width_q"]):
        pol = base.replace(width_star=float(np.quantile(widths, q)))
        m = policy_metrics(val, pol, tcfg)
        w_rows.append({"width_q": q, "W_star": pol.width_star, "review_onsets_per_engine": m["review_onsets_per_engine"]})
        if chosen_w is None and m["review_onsets_per_engine"] <= tcfg["review_budget_per_engine"]:
            chosen_w = pol.width_star
    if chosen_w is None:
        chosen_w = w_rows[-1]["W_star"]
    base_w = base.replace(width_star=chosen_w)

    # stage 2 - cut-offs: grid search of the asymmetric cost
    rows = []
    for c, h, m_ in itertools.product(grid["critical"], grid["high"], grid["medium"]):
        if not c < h < m_:
            continue
        met = policy_metrics(val, base_w.replace(crit=float(c), high=float(h), med=float(m_)), tcfg)
        rows.append({"critical": c, "high": h, "medium": m_, **met})
    best = min(rows, key=lambda r: (r["cost"], abs(r["critical"] - base.crit) + abs(r["high"] - base.high) + abs(r["medium"] - base.med)))
    tuned = base_w.replace(crit=float(best["critical"]), high=float(best["high"]), med=float(best["medium"]))
    report = {
        "default": policy_metrics(val, base.replace(width_star=chosen_w), tcfg),
        "tuned": {k: v for k, v in best.items() if k not in ("critical", "high", "medium")},
        "chosen": {"critical": tuned.crit, "high": tuned.high, "medium": tuned.med, "width_star": tuned.width_star},
        "width_search": w_rows, "grid": rows,
    }
    return tuned, report


def agent_replay_metrics(val: pd.DataFrame, pol: Policy, subset: str, tcfg: dict, model_version: str = "val") -> dict:
    """Run the real stateful TriageAgent over run-to-failure engines (in-memory ticket store) and score the tickets."""
    store = TicketStore(":memory:")
    agent = TriageAgent(pol, store, subset, model_version)
    for u, g in val.groupby("unit", sort=False):
        for r in g.itertuples(index=False):
            agent.step(CycleInputs(int(r.unit), int(r.cycle), r.p10, r.p50, r.p90, r.A, bool(r.sustained), bool(getattr(r, "data_bad", False)),
                                   warmup=bool(getattr(r, "warmup", False))), true_rul=float(r.rul))
    out = ticket_metrics(store, val["unit"].nunique(), tcfg, subset)
    store.db.close()
    return out


def ticket_metrics(store: TicketStore, n_engines: int, tcfg: dict, subset: str | None = None) -> dict:
    """Score tickets from the audit trail: the first moment each engine reached P1/P2 (not the ticket's final state).
    Needs `true_rul` on the events, i.e. a replay on engines whose RUL is known."""
    t = store.tickets(subset)
    ev = store.events(subset)
    ev = ev[ev["event"].isin(["created", "escalated"]) & ev["priority"].isin(["P1", "P2"]) & ev["true_rul"].notna()]
    first = ev.sort_values("id").groupby("unit").first() if len(ev) else ev
    out = {
        "tickets_per_engine": float(len(t) / max(n_engines, 1)),
        "engines_with_P1P2": int(len(first)),
        "engines_total": int(n_engines),
        "median_lead_cycles_first_P1P2": float(first["true_rul"].median()) if len(first) else 0.0,
        "p10_lead_cycles_first_P1P2": float(first["true_rul"].quantile(0.1)) if len(first) else 0.0,
        "false_alarm_P1P2_engines": int((first["true_rul"] > tcfg["false_alarm_rul"]).sum()) if len(first) else 0,
        "late_P1P2_engines": int((first["true_rul"] < tcfg["late_lead_cycles"]).sum()) + int(n_engines - len(first)),
    }
    return out
