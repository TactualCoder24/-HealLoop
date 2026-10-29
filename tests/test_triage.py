"""Triage agent: rule order, one open incident per engine, escalate/de-escalate, guardrails, idempotent replay."""
import numpy as np
import pytest

from selfheal.tickets import TicketStore
from selfheal.triage import CycleInputs, Policy, TriageAgent, decide, decide_batch

POL = Policy(crit=15, high=40, med=80, width_star=60, conflict_healthy_p10=110)


def I(p10=120, p50=125, p90=125, sus=False, bad=False, unseen=False, ok=True, unit=1, cycle=100):
    return CycleInputs(unit, cycle, p10, p50, p90, 0.5, sus, bad, ("stale / frozen feed",) if bad else (), unseen, ok)


# ---------------------------------------------------------------------------------------------- rules (first match wins)
def test_each_rule():
    assert decide(I(), POL).rule_id == 6 and decide(I(), POL).priority is None
    assert decide(I(p10=75, p50=100, p90=110), POL).rule_id == 5
    assert decide(I(p10=40, p50=55, p90=60), POL).rule_id == 4 and decide(I(p10=40, p50=55, p90=60), POL).priority == "P2"
    d = decide(I(p10=10, p50=15, p90=25), POL)
    assert d.rule_id == 2 and d.severity == "CRITICAL" and d.priority == "P1"
    assert decide(I(p10=15, p50=30, p90=45, sus=True), POL).rule_id == 2      # P10<=15 AND sustained anomaly
    assert decide(I(p10=15, p50=30, p90=45, sus=False), POL).rule_id == 4     # ... but not without persistence
    assert decide(I(sus=True), POL).rule_id == 3                               # anomaly vs healthy RUL -> conflict review
    assert decide(I(p10=60, p50=100, p90=125, sus=True), POL).rule_id == 3     # width 65 > W*=60 and spans tiers
    assert decide(I(p10=75, p50=100, p90=110, sus=True), POL).rule_id == 5     # narrow band: sustained anomaly -> MEDIUM


def test_rule_priority_order():
    assert decide(I(p10=1, p50=5, p90=10, bad=True), POL).rule_id == 1         # bad data beats CRITICAL
    assert decide(I(p10=1, p50=5, p90=10, unseen=True), POL).rule_id == 1
    assert decide(I(ok=False), POL).rule_id == 1
    assert decide(I(p10=np.nan, p50=np.nan, p90=np.nan), POL).rule_id == 1     # model failure
    assert decide(I(p10=5, p50=10, p90=120), POL).rule_id == 2                 # CRITICAL beats uncertainty REVIEW
    wide = decide(I(p10=30, p50=60, p90=120), POL)                              # wide + spans tiers
    assert wide.rule_id == 3 and wide.priority == "P2"                          # inherits what HIGH would have been
    assert decide(I(p10=30, p50=60, p90=120), POL.replace(width_star=200)).rule_id == 4


def test_warmup_suppresses_anomaly_and_uncertainty_rules_but_never_critical():
    w = I(sus=True)
    w.warmup = True
    assert decide(w, POL).rule_id == 6              # sustained anomaly vs healthy RUL would be a conflict REVIEW - not during warm-up
    wide = I(p10=30, p50=60, p90=120)
    wide.warmup = True
    assert decide(wide, POL).rule_id == 4           # uncertainty review skipped, HIGH still applies
    crit = I(p10=5, p50=10, p90=20)
    crit.warmup = True
    assert decide(crit, POL).rule_id == 2           # safety first: CRITICAL never waits for warm-up


def test_ingest_rejected_row_is_not_reported_as_model_failure():
    d = decide(CycleInputs(1, 5, data_bad=True, data_reasons=("ingest rejected row: duplicate cycle",)), POL)
    assert d.rule_id == 1 and d.reasons == ["ingest rejected row: duplicate cycle"]
    assert decide(CycleInputs(1, 5, model_ok=False), POL).reasons == ["model failure / non-finite prediction"]


def test_batch_decide_matches_scalar():
    rng = np.random.default_rng(3)
    n = 4000
    p10 = rng.uniform(0, 125, n)
    p50 = np.clip(p10 + rng.uniform(0, 60, n), 0, 125)
    p90 = np.clip(p50 + rng.uniform(0, 60, n), 0, 125)
    sus, bad, warm = rng.random(n) < 0.2, rng.random(n) < 0.05, rng.random(n) < 0.1
    rule, pri = decide_batch(p10, p50, p90, sus, bad, POL, warm)
    pr = {None: 0, "P3": 1, "P2": 2, "P1": 3}
    for i in range(n):
        inp = I(p10[i], p50[i], p90[i], bool(sus[i]), bool(bad[i]))
        inp.warmup = bool(warm[i])
        d = decide(inp, POL)
        assert d.rule_id == rule[i] and pr[d.priority] == pri[i], (i, d, rule[i], pri[i])


# ---------------------------------------------------------------------------------------------- agent / tickets
@pytest.fixture()
def agent():
    st = TicketStore(":memory:")
    return TriageAgent(POL, st, "FDT", "m1"), st


def run(agent, seq, unit=1, start=1):
    return [agent.step(I(*s, unit=unit, cycle=start + k) if isinstance(s, tuple) else s) for k, s in enumerate(seq)]


def test_one_open_incident_never_duplicates(agent):
    ag, st = agent
    ev = run(ag, [(75, 100, 110)] * 3 + [(35, 50, 70)] * 3 + [(5, 8, 20)] * 3)
    t = st.tickets("FDT")
    assert len(t) == 1 and t.iloc[0]["priority"] == "P1" and t.iloc[0]["severity"] == "CRITICAL"
    assert [e["action"] for e in ev if e["action"] in ("created", "escalated")] == ["created", "escalated", "escalated"]
    assert st.tickets("FDT", "open").shape[0] == 1
    with pytest.raises(Exception):                                              # DB-level guarantee
        st.create({"subset": "FDT", "unit": 1, "severity": "HIGH", "priority": "P2", "status": "open"})


def test_deescalates_only_after_stable_cycles(agent):
    ag, st = agent
    run(ag, [(35, 50, 70)])                                                     # HIGH / P2
    acts = [e["action"] for e in run(ag, [(75, 100, 110)] * 6, start=2)]       # healthier, MEDIUM-level
    assert acts[:4] == ["stable_lower"] * 4 and acts[4] == "deescalated"
    assert st.tickets("FDT").iloc[0]["severity"] == "MEDIUM"


def test_bad_data_never_deescalates_a_live_alert(agent):
    ag, st = agent
    run(ag, [(5, 10, 20)])                                                      # CRITICAL / P1
    acts = [e["action"] for e in run(ag, [I(bad=True, cycle=2 + k) for k in range(10)])]
    assert set(acts) == {"annotated"}
    t = st.tickets("FDT").iloc[0]
    assert t["priority"] == "P1" and t["severity"] == "CRITICAL"
    assert "data/model review" in t["evidence"]


def test_exception_ticket_for_bad_data_on_healthy_engine(agent):
    ag, st = agent
    assert run(ag, [I(bad=True)])[0]["action"] == "created"
    t = st.tickets("FDT").iloc[0]
    assert t["type"] == "exception" and t["severity"] == "REVIEW" and t["rule_id"] == 1


def test_critical_is_immediate_while_anomaly_needs_persistence(agent):
    ag, st = agent
    assert run(ag, [(5, 10, 30)])[0]["action"] == "created"                    # P1 on the very first cycle, no persistence
    assert st.tickets("FDT").iloc[0]["priority"] == "P1"
    ag2 = TriageAgent(POL, TicketStore(":memory:"), "FDT2", "m1")
    assert ag2.step(I(sus=False, cycle=1))["action"] == "none"                  # a lone anomaly flag (not sustained) is ignored


def test_human_ack_close_and_cooldown(agent):
    ag, st = agent
    run(ag, [(35, 50, 70)])
    st.acknowledge("FDT", 1, "alice", 5)
    assert st.get_open("FDT", 1)["status"] == "acknowledged"
    st.close("FDT", 1, 6, "serviced")
    assert ag.step(I(35, 50, 70, cycle=7))["action"] == "suppressed"            # cool-down, not worse than before
    assert ag.step(I(5, 8, 20, cycle=8))["action"] == "created"                 # worse -> new incident, new ticket id
    ids = st.tickets("FDT")["ticket_id"].tolist()
    assert ids == ["WO-FDT-0001-001", "WO-FDT-0001-002"]


def test_human_override_blocks_auto_deescalation(agent):
    ag, st = agent
    run(ag, [(5, 10, 20)])
    st.override_severity("FDT", 1, 2, "bob", "HIGH", "P2", "borderline, engineer judgement")
    acts = [e["action"] for e in run(ag, [(75, 100, 110)] * 8, start=3)]
    assert set(acts) == {"held_by_override"} and st.tickets("FDT").iloc[0]["severity"] == "HIGH"
    ev = st.events("FDT")
    assert "override" in ev["event"].tolist()


def test_replay_is_idempotent(tmp_path):
    def once():
        st = TicketStore(tmp_path / "t.db")
        st.reset("FDX")
        ag = TriageAgent(POL, st, "FDX", "m1")
        for u in (1, 2, 3):
            for c, s in enumerate([(120, 125, 125)] * 5 + [(70, 90, 100)] * 5 + [(30, 45, 60)] * 5 + [(4, 7, 15)] * 5, start=1):
                ag.step(I(*s, unit=u, cycle=c))
        out = st.tickets("FDX").drop(columns=[]).to_dict("records")
        st.db.close()
        return out
    assert once() == once()


def test_ticket_export_json_csv(tmp_path):
    st = TicketStore(":memory:")
    ag = TriageAgent(POL, st, "FDE", "m1")
    ag.step(I(5, 10, 20), explain=lambda: [{"sensor": "s11", "contribution": -9.1}])
    df = st.export(tmp_path, "FDE")
    assert (tmp_path / "work_orders.csv").exists() and (tmp_path / "work_orders" / "WO-FDE-0001-001.json").exists()
    assert df.iloc[0]["due_cycle"] == 100                                       # P1: immediate (due this cycle)
