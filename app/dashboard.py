"""Streamlit dashboard: fleet health, RUL bands, severity, live work orders, data quality / drift and MLOps.

Run:  python -m selfheal dashboard      (needs `python -m selfheal run-all` or `train` + `replay` first)
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from selfheal import mlops
from selfheal.config import SUBSETS, artifacts_dir, load_config, outputs_dir, resolve, tickets_db_path
from selfheal.data import SENSOR_NAMES, load_subset
from selfheal.replay import _ticks
from selfheal.scorer import Scorer
from selfheal.tickets import TicketStore
from selfheal.triage import PRI_RANK, SEV_RANK, CycleInputs, TriageAgent

st.set_page_config(page_title="HealLoop · Self-Healing Machines", page_icon="🛠️", layout="wide")

SEV_COLOR = {"CRITICAL": "#d62728", "HIGH": "#ff7f0e", "REVIEW": "#9467bd", "MEDIUM": "#e6b800", "MONITOR": "#2ca02c"}
PRI_COLOR = {"P1": "#d62728", "P2": "#ff7f0e", "P3": "#e6b800", None: "#2ca02c"}
TIER_LABELS = ["<=15", "16-40", "41-80", ">80"]


# ----------------------------------------------------------------------------------------------- cached loaders
def _mtime(p: Path) -> float:
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_resource(show_spinner=False)
def get_bundle(subset: str, stamp: float):
    return mlops.load_champion(load_config(subset))


@st.cache_data(show_spinner=False)
def load_scores(path: str, stamp: float) -> pd.DataFrame:
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_test(subset: str):
    cfg = load_config(subset)
    return load_subset(resolve(cfg, "raw_dir"), subset)


def available_subsets() -> list[str]:
    """Subsets with a trained champion (looked up without creating directories)."""
    return [s for s in SUBSETS if (resolve(load_config(s), "artifacts_dir") / s / "champion.json").exists()]


def read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def open_store(cfg: dict) -> TicketStore:
    return TicketStore(tickets_db_path(cfg))


def rank_of(priority, severity) -> tuple[int, int]:
    """Sort key that tolerates the NaN a missing priority (MONITOR) becomes after a parquet round-trip."""
    return PRI_RANK.get(priority if isinstance(priority, str) else None, 0), SEV_RANK.get(severity, 0)


def sev_badge(sev: str) -> str:
    return f"<span style='background:{SEV_COLOR.get(sev, '#888')};color:white;padding:2px 8px;border-radius:10px;font-size:0.8rem'>{sev}</span>"


# ----------------------------------------------------------------------------------------------- sidebar
subsets = available_subsets()
st.sidebar.title("🛠️ HealLoop")
st.sidebar.caption("Self-Healing Machines · closed-loop predictive maintenance · NASA C-MAPSS")
if not subsets:
    st.error("No trained models found. Run `python -m selfheal run-all` (or `train` then `replay`) first.")
    st.stop()
subset = st.sidebar.selectbox("Subset", subsets, index=0)
cfg = load_config(subset)
stamp = _mtime(artifacts_dir(cfg) / "champion.json")
bundle = get_bundle(subset, stamp)
scenario = st.sidebar.radio("Replay scenario", ["clean", "faults"], horizontal=True,
                            help="'faults' injects a data gap, frozen feed, unseen regime, duplicate row and sensor drift.")
key = subset if scenario == "clean" else f"{subset}+{scenario}"
page = st.sidebar.radio("View", ["Fleet overview", "Engine detail", "Work orders", "Data quality & drift", "Model & MLOps", "Live simulation"])
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Champion** `{bundle.version}`  \n**Policy** `{bundle.policy.version}`")
st.sidebar.caption("RUL bands = conformal-calibrated P10–P90 · agent = plain-Python rules, no LLM")

out_dir = outputs_dir(cfg) / scenario
scores_path = out_dir / "replay_scores.parquet"
summary = read_json(out_dir / "summary.json")
has_replay = scores_path.exists()
scores = load_scores(str(scores_path), _mtime(scores_path)) if has_replay else None
if has_replay and summary and summary.get("model_version") != bundle.version:
    st.sidebar.warning(f"Replay was produced by {summary['model_version']}; champion is now {bundle.version}. "
                       f"Re-run `python -m selfheal replay --subset {subset}`.")

st.title(f"{subset} · {page}")


# ----------------------------------------------------------------------------------------------- pages
def page_overview() -> None:
    if not has_replay:
        st.info(f"No replay yet for {subset}/{scenario}. Run `python -m selfheal replay --subset {subset} --scenario {scenario}`.")
        return
    s = scores
    tmax = int(s["cycle"].max())
    clock = st.slider("Fleet clock (cycle)", 1, tmax, tmax, help="Rewind the replay: shows each engine's state at this cycle.")
    cur = s[s["cycle"] <= clock].sort_values(["unit", "cycle"]).groupby("unit").tail(1).set_index("unit")
    st_ = open_store(cfg)
    ev = st_.events(key)
    t = st_.tickets(key)
    st_.db.close()
    created = t[t["created_cycle"] <= clock]
    c = st.columns(5)
    c[0].metric("Engines", len(cur))
    c[1].metric("Tickets raised", len(created))
    c[2].metric("P1 (critical) engines now", int((cur["priority"] == "P1").sum()))
    c[3].metric("P2 (high) engines now", int((cur["priority"] == "P2").sum()))
    c[4].metric("Review / exception tickets", int((created["type"] == "exception").sum()) if len(created) else 0)

    left, right = st.columns([1, 2])
    with left:
        counts = cur["severity"].value_counts().reindex(list(SEV_COLOR), fill_value=0)
        fig = go.Figure(go.Bar(x=counts.index, y=counts.values, marker_color=[SEV_COLOR[k] for k in counts.index]))
        fig.update_layout(title="Agent decision per engine", height=320, margin=dict(l=10, r=10, t=40, b=10), yaxis_title="engines")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        d = cur.reset_index()
        fig = go.Figure()
        for sev, col in SEV_COLOR.items():
            m = d["severity"] == sev
            if m.any():
                fig.add_trace(go.Scatter(x=d.loc[m, "cycle"], y=d.loc[m, "p50"], mode="markers", name=sev, marker=dict(color=col, size=8, opacity=0.85),
                                         text=d.loc[m, "unit"].astype(str), hovertemplate="engine %{text}<br>cycle %{x}<br>P50 %{y:.0f}<extra></extra>"))
        for cut in (bundle.policy.crit, bundle.policy.high, bundle.policy.med):
            fig.add_hline(y=cut, line_dash="dot", line_color="#888", annotation_text=f"{cut:g}", annotation_position="right")
        fig.update_layout(title="Fleet map: predicted RUL (P50) vs cycles run", height=320, margin=dict(l=10, r=10, t=40, b=10),
                          xaxis_title="cycle", yaxis_title="P50 RUL")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Engines needing attention")
    show = cur.reset_index()
    show["rank"] = show.apply(lambda r: rank_of(r["priority"], r["severity"]), axis=1)
    show = show.sort_values("rank", ascending=False).drop(columns=["rank"])
    show = show[["unit", "cycle", "severity", "priority", "p10", "p50", "p90", "A", "sustained", "ticket_id", "true_rul"]].rename(
        columns={"A": "anomaly pct", "true_rul": "true RUL (ground truth, not seen by agent)"})
    st.dataframe(show[show["severity"] != "MONITOR"].round(2), use_container_width=True, hide_index=True, height=320)
    if summary:
        with st.expander("Agent behaviour on this replay (official test fleet)"):
            st.json({k: summary[k] for k in ("engines", "tickets_total", "tickets_by_priority", "tickets_by_type", "risk_engines_rul_le_30",
                                              "risk_engines_flagged_P1P2_ever", "risk_engines_P1P2_at_end", "healthy_engines_rul_gt_80",
                                              "healthy_engines_rul_gt_80_with_P1P2_at_end", "ticket_metrics") if k in summary})
            if "final_severity_by_true_rul_tier" in summary:
                st.caption("Final decision vs the official RUL of each test engine (rows = true RUL tier)")
                st.dataframe(pd.DataFrame(summary["final_severity_by_true_rul_tier"]).T.fillna(0).astype(int), use_container_width=True)


def page_engine() -> None:
    if not has_replay:
        st.info("Run a replay first.")
        return
    units = sorted(scores["unit"].unique())
    flagged = scores[scores["priority"].isin(["P1", "P2"])]["unit"].unique().tolist()
    unit = st.selectbox("Engine", units, index=units.index(flagged[0]) if flagged else 0,
                        format_func=lambda u: f"engine {u}" + (" ⚠" if u in flagged else ""))
    e = scores[scores["unit"] == unit].sort_values("cycle")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=e["cycle"], y=e["p90"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=e["cycle"], y=e["p10"], fill="tonexty", fillcolor="rgba(31,119,180,0.2)", line=dict(width=0), name="P10–P90 (80% conformal)"))
    fig.add_trace(go.Scatter(x=e["cycle"], y=e["p50"], line=dict(color="#1f77b4", width=2), name="P50 RUL"))
    fig.add_trace(go.Scatter(x=e["cycle"], y=e["true_rul"].clip(upper=bundle.cfg["labels"]["rul_cap"] + 40), line=dict(color="black", dash="dot"), name="true RUL (ground truth)"))
    for cut, col in ((bundle.policy.crit, SEV_COLOR["CRITICAL"]), (bundle.policy.high, SEV_COLOR["HIGH"]), (bundle.policy.med, SEV_COLOR["MEDIUM"])):
        fig.add_hline(y=cut, line_dash="dot", line_color=col)
    for sev in ("CRITICAL", "HIGH", "REVIEW", "MEDIUM"):
        m = e["severity"] == sev
        if m.any():
            fig.add_trace(go.Scatter(x=e.loc[m, "cycle"], y=np.zeros(int(m.sum())) - 4, mode="markers", marker=dict(color=SEV_COLOR[sev], size=7, symbol="square"), name=f"decision: {sev}"))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="cycle", yaxis_title="RUL (cycles)")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fa = go.Figure(go.Scatter(x=e["cycle"], y=e["A"], line=dict(color="#9467bd")))
        fa.add_hline(y=bundle.policy.alert_A, line_dash="dot", line_color="#d62728", annotation_text=f"A={bundle.policy.alert_A}")
        sus = e[e["sustained"]]
        fa.add_trace(go.Scatter(x=sus["cycle"], y=sus["A"], mode="markers", marker=dict(color="#d62728", size=6), name="sustained"))
        fa.update_layout(title=f"Anomaly percentile A (alert: {bundle.policy.alert_k}-of-{bundle.policy.alert_n} cycles ≥ {bundle.policy.alert_A})",
                         height=300, margin=dict(l=10, r=10, t=40, b=10), showlegend=False, yaxis_range=[0, 1.02])
        st.plotly_chart(fa, use_container_width=True)
    with c2:
        _, test, _ = load_test(subset)
        sensors = [s for s in bundle.cleaner.kept]
        pick = st.multiselect("Sensors", sensors, default=[s for s in ("s11", "s4", "s12") if s in sensors][:3], format_func=lambda s: f"{s} {SENSOR_NAMES.get(s, '')}")
        raw_e = test[test["unit"] == unit]
        fs = go.Figure()
        for s in pick:
            v = raw_e[s]
            fs.add_trace(go.Scatter(x=raw_e["cycle"], y=(v - v.iloc[:20].mean()) / (v.std() + 1e-9), name=s))
        fs.update_layout(title="Sensor trends (standardised, replayed raw stream)", height=300, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fs, use_container_width=True)

    st_ = open_store(cfg)
    ev = st_.events(key)
    t = st_.tickets(key)
    st_.db.close()
    ev, t = ev[ev["unit"] == unit], t[t["unit"] == unit]
    st.subheader("Work order(s) and audit trail")
    if t.empty:
        st.success("No ticket for this engine - dashboard only (MONITOR).")
    for _, r in t.iterrows():
        st.markdown(f"**{r['ticket_id']}** {sev_badge(r['severity'])} priority **{r['priority']}** · status `{r['status']}` · created cycle {r['created_cycle']} · "
                    f"due cycle {r['due_cycle']} · rule `{r['rule_name']}`", unsafe_allow_html=True)
        ks = json.loads(r["key_sensors"]) if r["key_sensors"] else []
        if ks:
            st.caption("Key sensor signals (TreeSHAP on the P50 model — indicative only, NOT proof of root cause): " +
                       ", ".join(f"{k['sensor']} {k['name']} ({k['contribution']:+.1f}, {k['effect']})" for k in ks))
    if not ev.empty:
        st.dataframe(ev[["cycle", "event", "from_severity", "to_severity", "priority", "rule_id", "note"]], use_container_width=True, hide_index=True)


def page_orders() -> None:
    st_ = open_store(cfg)
    t = st_.tickets(key)
    if t.empty:
        st.info("No work orders yet for this scenario.")
        st_.db.close()
        return
    c = st.columns(4)
    status = c[0].multiselect("Status", ["open", "acknowledged", "closed"], default=["open", "acknowledged"])
    pri = c[1].multiselect("Priority", ["P1", "P2", "P3"], default=["P1", "P2", "P3"])
    typ = c[2].multiselect("Type", ["predictive", "exception"], default=["predictive", "exception"])
    q = c[3].text_input("Engine id contains")
    v = t[t["status"].isin(status) & t["priority"].isin(pri) & t["type"].isin(typ)]
    if q:
        v = v[v["unit"].astype(str).str.contains(q)]
    v = v.assign(_r=[rank_of(p, sv)[0] * 10 + rank_of(p, sv)[1] for p, sv in zip(v["priority"], v["severity"])]).sort_values("_r", ascending=False)
    st.dataframe(v[["ticket_id", "unit", "severity", "priority", "type", "status", "created_cycle", "updated_cycle", "due_cycle", "p10", "p50", "p90", "anomaly_a",
                    "rule_name", "closure_outcome"]].round(2), use_container_width=True, hide_index=True, height=360)
    st.download_button("Download work orders (CSV)", v.drop(columns=["_r"]).to_csv(index=False), file_name=f"work_orders_{key}.csv")

    st.subheader("Engineer actions (human in the loop)")
    open_t = t[t["status"] != "closed"]
    if open_t.empty:
        st.caption("Nothing open.")
        st_.db.close()
        return
    tid = st.selectbox("Ticket", open_t["ticket_id"].tolist())
    r = open_t[open_t["ticket_id"] == tid].iloc[0]
    st.markdown(f"{sev_badge(r['severity'])} priority **{r['priority']}** · engine {r['unit']} · P10/P50/P90 = {r['p10']:.0f}/{r['p50']:.0f}/{r['p90']:.0f} · A={r['anomaly_a']}", unsafe_allow_html=True)
    ev = json.loads(r["evidence"]) if r["evidence"] else {}
    st.write({"why": ev.get("reasons"), "notes": ev.get("notes")})
    who = st.text_input("Your name", "engineer")
    a, b, c3 = st.columns(3)
    if a.button("Acknowledge"):
        st_.acknowledge(key, int(r["unit"]), who, int(r["updated_cycle"]))
        st.rerun()
    outcome = b.selectbox("Closure outcome", ["serviced", "confirmed_degradation", "false_alarm", "no_fault_found"])
    if b.button("Close ticket"):
        st_.close(key, int(r["unit"]), int(r["updated_cycle"]), outcome, f"closed from dashboard by {who}", who=who)
        st.rerun()
    new = c3.selectbox("Override severity", ["CRITICAL/P1", "HIGH/P2", "MEDIUM/P3"])
    if c3.button("Apply override"):
        sev, p = new.split("/")
        st_.override_severity(key, int(r["unit"]), int(r["updated_cycle"]), who, sev, p, "manual override from dashboard")
        st.rerun()
    st_.db.close()


def page_quality() -> None:
    if not summary:
        st.info("No replay summary yet.")
        return
    mon = summary["monitor"]
    c = st.columns(4)
    c[0].metric("Drift status", mon["drift_status"].upper())
    c[1].metric("Max PSI", f"{mon['max_psi']:.3f}", help=f"sensor {mon['max_psi_sensor']}")
    c[2].metric("Data-quality status", mon["quality_status"].upper())
    c[3].metric("Rejected rows", summary.get("rejected_rows", 0))
    st.caption("Unlabelled cycles feed DRIFT monitoring only (never supervised retraining). Early-life cycles of the incoming fleet are compared with the train early-life "
               "reference per sensor (PSI; warn ≥ 0.10, alert ≥ 0.25), so normal ageing is not mistaken for drift.")
    d = pd.read_csv(out_dir / "drift.csv")
    colr = d["status"].map({"ok": "#2ca02c", "warn": "#e6b800", "alert": "#d62728"}).fillna("#888")
    fig = go.Figure(go.Bar(x=d["sensor"], y=d["psi"], marker_color=colr))
    fig.add_hline(y=cfg["monitor"]["psi_warn"], line_dash="dot", line_color="#e6b800")
    fig.add_hline(y=cfg["monitor"]["psi_alert"], line_dash="dot", line_color="#d62728")
    fig.update_layout(title="Sensor drift (PSI vs train early-life reference)", height=330, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)
    q = mon["quality"]
    st.dataframe(pd.DataFrame({"metric": [k for k in q if k != "status"], "value": [q[k] for k in q if k != "status"]}), hide_index=True, use_container_width=True)
    if summary.get("fault_log"):
        st.subheader("Injected faults (scenario = faults)")
        st.json(summary["fault_log"])
        t_ = open_store(cfg)
        t = t_.tickets(key)
        t_.db.close()
        ex = t[t["type"] == "exception"][["ticket_id", "unit", "created_cycle", "severity", "priority", "action", "evidence"]]
        ex["why"] = ex["evidence"].map(lambda s: "; ".join(json.loads(s).get("reasons", [])) if s else "")
        st.markdown("**Exception tickets raised by the agent (engineer checks data):**")
        st.dataframe(ex.drop(columns=["evidence"]), hide_index=True, use_container_width=True)


def page_mlops() -> None:
    m = bundle.metrics
    t, v, base = m["test"], m["val"], m["baselines_test"]
    st.subheader("Model quality (locked official test endpoints — never used for tuning)")
    c = st.columns(5)
    c[0].metric("RMSE (vs official RUL)", f"{t['rmse']:.2f}")
    c[1].metric("NASA score", f"{t['nasa_score']:.0f}")
    c[2].metric("RMSE vs capped (125) truth", f"{t['rmse_vs_capped_truth']:.2f}", help="the model is trained on the piece-wise-linear (capped) RUL")
    c[3].metric("80% interval coverage", f"{t['coverage_80']:.2f}", help="conformal-calibrated, vs capped truth")
    c[4].metric("Interval width", f"{t['interval_width']:.1f}")
    bt = pd.DataFrame({"model": ["LightGBM P50 (ours)"] + [k.replace("_", " ") for k in base],
                       "RMSE": [t["rmse"]] + [b["rmse"] for b in base.values()], "NASA score": [t["nasa_score"]] + [b["nasa_score"] for b in base.values()]})
    st.dataframe(bt.round(1), hide_index=True, use_container_width=True)
    with st.expander("Validation (held-out engines) and tuned policy"):
        st.json({"validation": v, "alert_rule": m["alert"], "policy": bundle.policy.__dict__, "policy_val_default": m["policy_val_default"],
                 "policy_val_tuned_rule_stream": m["policy_val_rule_stream"], "agent_val_tickets": m["agent_val"]})

    st.subheader("Model registry & lineage")
    hist = mlops.read_history(cfg)
    try:
        vers = pd.DataFrame(mlops.list_versions(cfg))
        if not vers.empty:
            st.dataframe(vers, hide_index=True, use_container_width=True)
    except Exception as ex:  # noqa: BLE001
        st.caption(f"MLflow registry unavailable ({ex}); showing local history only.")
    if hist:
        st.dataframe(pd.DataFrame(hist).astype(str), hide_index=True, use_container_width=True)
    ptr = read_json(artifacts_dir(cfg) / "champion.json")
    if ptr:
        st.caption(f"champion = v{ptr['version']} · previous (rollback target) = {('v' + str(ptr['previous'])) if ptr.get('previous') else 'none'}")
        if ptr.get("previous") and st.checkbox("I want to roll the champion back to the previous version"):
            if st.button("Rollback now"):
                nv = mlops.rollback(cfg)
                st.cache_resource.clear()
                st.success(f"Rolled back to v{nv}")
                st.rerun()

    rr = read_json(outputs_dir(cfg) / "retrain" / "retrain_result.json")
    st.subheader("Retraining flow (reserved completed histories released → challenger gate)")
    if not rr:
        st.info(f"Not run yet: `python -m selfheal retrain --subset {subset}`")
    else:
        d = rr["decision"]
        cv, hv = rr["champion_val"], rr["challenger_val"]
        st.markdown(f"**{rr['released']}** reserved engines were released as completed, labelled run-to-failure histories "
                    f"(+ simulated ticket closure with verified outcomes: confirmed {rr['feedback']['confirmed_degradation']}, false alarms {rr['feedback']['false_alarm']}, "
                    f"scheduled service {rr['feedback']['scheduled_service']}). "
                    f"Challenger **v{rr['challenger']}** vs champion **v{rr['champion_before']}** on identical held-out engines (calibration + validation, {cv['hold_engines']}) → "
                    f"**{'PROMOTED' if rr['promoted'] else 'rejected, champion kept'}**.")
        gate = pd.DataFrame({"check": list(d["checks"]), "passed": list(d["checks"].values())})
        c1, c2 = st.columns(2)
        c1.dataframe(gate, hide_index=True, use_container_width=True)
        cmp = pd.DataFrame({"metric": ["NASA score (mean, endpoint-like)", "RMSE (endpoint-like)", "risk-window RMSE (RUL≤30)", "80% coverage (validation engines)"],
                            "champion": [cv["hold_ep_nasa_mean"], cv["hold_ep_rmse"], cv["hold_rmse_risk"], cv["ep_coverage_80"]],
                            "challenger": [hv["hold_ep_nasa_mean"], hv["hold_ep_rmse"], hv["hold_rmse_risk"], hv["ep_coverage_80"]]})
        c2.dataframe(cmp.round(3), hide_index=True, use_container_width=True)
        st.caption("Informational (official test set, never used for training or for the decision): champion RMSE "
                   f"{rr['champion_test']['rmse']:.2f} -> challenger {rr['challenger_test']['rmse']:.2f}; NASA score "
                   f"{rr['champion_test']['nasa_score']:.0f} -> {rr['challenger_test']['nasa_score']:.0f}")

    st.subheader("What drives the P50 model (gain, aggregated per sensor)")
    imp = bundle.rul.models[0.5].booster_.feature_importance("gain")
    agg: dict[str, float] = {}
    for n, g in zip(bundle.spec.names(), imp):
        k = bundle.spec.sensor_of(n) or "cycle"
        agg[k] = agg.get(k, 0.0) + float(g)
    s = pd.Series(agg).sort_values(ascending=False).head(12)
    fig = go.Figure(go.Bar(x=[f"{k} {SENSOR_NAMES.get(k, 'engine age')}" for k in s.index][::-1], y=(s / s.sum()).values[::-1], orientation="h"))
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="share of total gain")
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------------------------- live simulation
class LiveSim:
    """Steps the real Scorer + TriageAgent through the test fleet; faults can be injected on the fly."""

    def __init__(self, subset: str):
        self.cfg = load_config(subset)
        self.bundle = mlops.load_champion(self.cfg)
        _, test, rul = load_subset(resolve(self.cfg, "raw_dir"), subset)
        self.key = f"{subset}+live"
        self.store = TicketStore(tickets_db_path(self.cfg))
        self.store.reset(self.key)
        self.scorer = Scorer(self.bundle)
        self.agent = TriageAgent(self.bundle.policy, self.store, self.key, self.bundle.version)
        self.ticks = _ticks(test)
        last = test.groupby("unit")["cycle"].max()
        self.failure = (last + rul.reindex(last.index)).to_dict()
        self.i = 0
        self.state: dict[int, dict] = {}
        self.faults: dict[int, list] = {}
        self.last_row: dict[int, pd.Series] = {}
        self.log: list[str] = []

    def inject(self, unit: int, kind: str) -> None:
        self.faults.setdefault(unit, []).append([kind, {"stale": 6, "drop": 8, "regime": 4, "dup": 1}[kind]])

    def _apply_faults(self, rows: pd.DataFrame) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
        extra, drop_idx = [], []
        for idx, r in rows.iterrows():
            u = int(r["unit"])
            for f in list(self.faults.get(u, [])):
                kind, left = f
                if kind == "drop":
                    drop_idx.append(idx)
                elif kind == "stale" and u in self.last_row:
                    rows.loc[idx, [c for c in rows.columns if c.startswith("s")]] = self.last_row[u][[c for c in rows.columns if c.startswith("s")]].to_numpy()
                elif kind == "regime":
                    rows.loc[idx, ["op1", "op2", "op3"]] = [31.0, 0.55, 30.0]
                elif kind == "dup" and u in self.last_row:
                    extra.append(self.last_row[u].to_frame().T.astype(rows.dtypes.to_dict()))
                f[1] -= 1
                if f[1] <= 0:
                    self.faults[u].remove(f)
            self.last_row[u] = rows.loc[idx].copy()
        return rows.drop(index=drop_idx), extra

    def step(self, n: int) -> None:
        for _ in range(n):
            if self.i >= len(self.ticks):
                break
            rows, extra = self._apply_faults(self.ticks[self.i].copy())
            for batch in [rows] + extra:
                if batch.empty:
                    continue
                for res in self.scorer.process(batch):
                    inp = res.inputs
                    ex = (lambda r=res: self.bundle.explain(r.x_row)) if res.x_row is not None else None
                    ev = self.agent.step(inp, explain=ex, true_rul=float(self.failure[res.unit] - res.cycle))
                    self.state[res.unit] = {"unit": res.unit, "cycle": res.cycle, "status": res.status, "p10": inp.p10, "p50": inp.p50, "p90": inp.p90,
                                            "A": inp.anomaly_A, "decision": ev["severity"], "priority": ev["priority"], "action": ev["action"],
                                            "ticket": ev["ticket_id"], "why": "; ".join(inp.data_reasons)}
                    if ev["action"] in ("created", "escalated", "deescalated"):
                        self.log.append(f"cycle {res.cycle} · engine {res.unit} · {ev['action']} → {ev['severity']} ({ev['priority']}) {ev['ticket_id']}")
            self.i += 1
        self.store.flush()


def page_live() -> None:
    st.caption("This runs the real pipeline (ingest checks → cleaning → features → models → rule agent → work orders) one fleet cycle at a time, "
               "on the official test engines. Inject a data problem and watch the agent hand it to an engineer.")
    sk = f"sim_{subset}"
    if sk not in st.session_state or st.sidebar.button("Reset simulation"):
        with st.spinner("loading models ..."):
            st.session_state[sk] = LiveSim(subset)
    sim: LiveSim = st.session_state[sk]
    c = st.columns([1, 1, 1, 3])
    n = c[0].number_input("Cycles to advance", 1, 200, 20)
    if c[1].button("▶ Step", type="primary"):
        with st.spinner("scoring ..."):
            sim.step(int(n))
    if c[2].button("⏭ +100"):
        with st.spinner("scoring ..."):
            sim.step(100)
    c[3].progress(min(sim.i / len(sim.ticks), 1.0), text=f"fleet clock: tick {sim.i}/{len(sim.ticks)}")
    f1, f2, f3 = st.columns([1, 1, 1])
    unit = f1.number_input("Engine id", 1, int(max(sim.failure)), 7)
    kind = f2.selectbox("Inject data fault", ["stale", "drop", "regime", "dup"],
                        format_func=lambda k: {"stale": "frozen sensor feed (6 cycles)", "drop": "data gap (8 cycles)", "regime": "unseen operating regime (4 cycles)", "dup": "duplicate row"}[k])
    if f3.button("Inject on next cycles"):
        sim.inject(int(unit), kind)
        st.toast(f"{kind} fault queued for engine {unit}")
    if not sim.state:
        st.info("Press ▶ Step to start the replay.")
        return
    d = pd.DataFrame(sim.state.values())
    d["rank"] = d.apply(lambda r: rank_of(r["priority"], r["decision"]), axis=1)
    d = d.sort_values("rank", ascending=False).drop(columns="rank")
    m = st.columns(5)
    m[0].metric("Engines reporting", len(d))
    m[1].metric("P1", int((d["priority"] == "P1").sum()))
    m[2].metric("P2", int((d["priority"] == "P2").sum()))
    m[3].metric("Review / data", int((d["decision"] == "REVIEW").sum()))
    t = sim.store.tickets(sim.key)
    m[4].metric("Tickets (one open per engine)", len(t))
    st.dataframe(d[d["decision"] != "MONITOR"].round(1), hide_index=True, use_container_width=True, height=300)
    with st.expander("Agent log", expanded=True):
        st.code("\n".join(sim.log[-25:]) or "(no ticket events yet)")


PAGES = {"Fleet overview": page_overview, "Engine detail": page_engine, "Work orders": page_orders,
         "Data quality & drift": page_quality, "Model & MLOps": page_mlops, "Live simulation": page_live}
PAGES[page]()
