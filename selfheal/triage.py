"""Stage 5 - TRIAGE AGENT (no LLM): plain-Python, first-match-wins rules + one-open-incident ticket logic.

Rule order (design deck): 1 REVIEW(data/regime/model) -> 2 CRITICAL -> 3 REVIEW(uncertainty/conflict)
-> 4 HIGH -> 5 MEDIUM -> 6 MONITOR.  All numeric cut-offs live in `Policy` and are validation-tuned
policy candidates, not measured results.

Safety guard (design principle "late is costlier than early"): a rule-3 REVIEW ticket is an exception
ticket for an engineer, but it inherits the priority that rules 4/5 *would* have assigned, so an
uncertainty review can never silently demote an engine that is already HIGH/MEDIUM-eligible.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np

from .tickets import TicketStore

SEV_RANK = {"MONITOR": 0, "MEDIUM": 1, "REVIEW": 2, "HIGH": 3, "CRITICAL": 4}
PRI_RANK = {None: 0, "P3": 1, "P2": 2, "P1": 3}
RULE_NAMES = {1: "R1_REVIEW_DATA_OR_MODEL", 2: "R2_CRITICAL", 3: "R3_REVIEW_UNCERTAINTY", 4: "R4_HIGH",
              5: "R5_MEDIUM", 6: "R6_MONITOR"}


@dataclass(frozen=True)
class Policy:
    crit: float = 15.0            # P50 <= crit (or P10 <= crit & sustained anomaly) -> CRITICAL
    high: float = 40.0            # P10 <= high -> HIGH
    med: float = 80.0             # P10 <= med  -> MEDIUM
    width_star: float = 60.0      # W*: interval width above which (and spanning tiers) -> REVIEW
    conflict_healthy_p10: float = 110.0   # sustained anomaly while P10 >= this -> detector/RUL conflict
    deescalate_after: int = 5
    reopen_cooldown: int = 10
    due_p1: int = 0
    due_p2: int = 10
    due_review: int = 5
    service_interval: int = 50
    review_priority: str = "P3"
    alert_A: float = 0.99         # anomaly percentile threshold ...
    alert_k: int = 3              # ... k-of-n persistence
    alert_n: int = 5
    warmup_rows: int = 10         # fewer rows of history than this: windows are short -> no anomaly flags, no rule 3

    @classmethod
    def from_cfg(cls, cfg: dict, **over) -> "Policy":
        p = cfg["policy"]
        a = cfg["anomaly"]["default"]
        base = dict(
            crit=p["cutoffs"]["critical"], high=p["cutoffs"]["high"], med=p["cutoffs"]["medium"],
            width_star=p["width_star"], conflict_healthy_p10=p["conflict_healthy_p10"],
            deescalate_after=p["deescalate_after"], reopen_cooldown=p["reopen_cooldown"],
            due_p1=p["due"]["P1"], due_p2=p["due"]["P2"], due_review=p["due"]["review"],
            service_interval=p["service_interval"], review_priority=p["review_priority"], warmup_rows=p["warmup_rows"],
            alert_A=a["A"], alert_k=a["k"], alert_n=a["n"],
        )
        base.update(over)
        return cls(**base)

    def replace(self, **kw) -> "Policy":
        return Policy(**{**asdict(self), **kw})

    @property
    def version(self) -> str:
        return "pol-" + hashlib.sha1(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:8]

    def tier(self, rul: float) -> int:
        return 0 if rul <= self.crit else 1 if rul <= self.high else 2 if rul <= self.med else 3


@dataclass
class CycleInputs:
    """What the agent sees each cycle (it never sees the true RUL)."""
    unit: int
    cycle: int
    p10: float | None = None
    p50: float | None = None
    p90: float | None = None
    anomaly_A: float | None = None
    sustained: bool = False
    data_bad: bool = False
    data_reasons: tuple[str, ...] = ()
    unseen_regime: bool = False
    model_ok: bool = True
    warmup: bool = False          # start-up: too little history for stable windows / anomaly percentiles


@dataclass
class Decision:
    rule_id: int
    severity: str
    priority: str | None
    action: str
    reasons: list[str] = field(default_factory=list)
    shadow_priority: str | None = None

    @property
    def rule_name(self) -> str:
        return RULE_NAMES[self.rule_id]

    @property
    def rank(self) -> tuple[int, int]:
        return PRI_RANK[self.priority], SEV_RANK[self.severity]


def decide(inp: CycleInputs, pol: Policy) -> Decision:
    """Pure rule evaluation, first match wins."""
    # 1. REVIEW - bad / stale data, unseen regime, model failure  (checked before any model value is trusted)
    why = []
    if inp.data_bad:
        why += list(inp.data_reasons) or ["bad or stale data"]
    if inp.unseen_regime:
        why.append("unseen operating regime")
    vals = (inp.p10, inp.p50, inp.p90)
    if not why and (not inp.model_ok or any(v is None or not np.isfinite(v) for v in vals)):
        why.append("model failure / non-finite prediction")      # (rows rejected at ingest carry no model values by design)
    if why:
        return Decision(1, "REVIEW", pol.review_priority, "Exception ticket; engineer checks data", why)
    p10, p50, p90 = float(inp.p10), float(inp.p50), float(inp.p90)
    sus = bool(inp.sustained) and not inp.warmup
    # 2. CRITICAL
    if p50 <= pol.crit or (p10 <= pol.crit and sus):
        r = [f"P50={p50:.0f} <= {pol.crit:g}"] if p50 <= pol.crit else [f"P10={p10:.0f} <= {pol.crit:g} with sustained anomaly"]
        return Decision(2, "CRITICAL", "P1", "P1 ticket; supervisor alert; safe-stop decision by human", r)
    shadow = "P2" if p10 <= pol.high else ("P3" if (p10 <= pol.med or sus) else None)
    # 3. REVIEW - wide interval spanning risk tiers, or detector/RUL conflict
    wide = (p90 - p10) > pol.width_star and pol.tier(p10) != pol.tier(p90)
    conflict = sus and p10 >= pol.conflict_healthy_p10
    if (wide or conflict) and not inp.warmup:                     # not trusted while the windows are still filling
        why = []
        if wide:
            why.append(f"interval width {p90 - p10:.0f} > W*={pol.width_star:.0f} and spans risk tiers")
        if conflict:
            why.append("sustained anomaly but RUL model says healthy (model conflict)")
        return Decision(3, "REVIEW", shadow or pol.review_priority, "Review ticket + evidence", why, shadow)
    # 4. HIGH
    if p10 <= pol.high:
        return Decision(4, "HIGH", "P2", f"P2 ticket; due within {pol.due_p2} cycles", [f"P10={p10:.0f} <= {pol.high:g}"])
    # 5. MEDIUM
    if p10 <= pol.med or sus:
        r = [f"P10={p10:.0f} <= {pol.med:g}"] if p10 <= pol.med else ["sustained anomaly"]
        return Decision(5, "MEDIUM", "P3", "P3 ticket; next scheduled service", r)
    # 6. MONITOR
    return Decision(6, "MONITOR", None, "Dashboard only", [])


def decide_batch(p10, p50, p90, sustained, data_bad, pol: Policy, warmup=None) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised twin of `decide` (used by policy tuning). Returns (rule_id, priority_rank)."""
    p10, p50, p90 = (np.asarray(a, float) for a in (p10, p50, p90))
    warm = np.zeros(len(p10), bool) if warmup is None else np.asarray(warmup, bool)
    sus, bad = np.asarray(sustained, bool) & ~warm, np.asarray(data_bad, bool)
    tier = lambda r: np.where(r <= pol.crit, 0, np.where(r <= pol.high, 1, np.where(r <= pol.med, 2, 3)))
    r1 = bad | ~np.isfinite(p10) | ~np.isfinite(p50) | ~np.isfinite(p90)
    r2 = (p50 <= pol.crit) | ((p10 <= pol.crit) & sus)
    shadow = np.where(p10 <= pol.high, 2, np.where((p10 <= pol.med) | sus, 1, 0))
    r3 = ((((p90 - p10) > pol.width_star) & (tier(p10) != tier(p90))) | (sus & (p10 >= pol.conflict_healthy_p10))) & ~warm
    r4, r5 = p10 <= pol.high, (p10 <= pol.med) | sus
    rule = np.select([r1, r2, r3, r4, r5], [1, 2, 3, 4, 5], default=6)
    rev_pri = PRI_RANK[pol.review_priority]
    pri = np.select([rule == 1, rule == 2, rule == 3, rule == 4, rule == 5],
                    [rev_pri, 3, np.where(shadow > 0, shadow, rev_pri), 2, 1], default=0)
    return rule, pri


def due_cycle(dec: Decision, cycle: int, pol: Policy) -> int:
    if dec.priority == "P1":
        due = cycle + pol.due_p1
    elif dec.priority == "P2":
        due = cycle + pol.due_p2
    else:  # P3 -> next scheduled service
        due = (cycle // pol.service_interval + 1) * pol.service_interval
    if dec.severity == "REVIEW":
        due = min(due, cycle + pol.due_review)
    return int(due)


class TriageAgent:
    """Turns per-cycle decisions into at most ONE open incident per engine: create, escalate, de-escalate."""

    def __init__(self, policy: Policy, store: TicketStore, subset: str, model_version: str = "n/a"):
        self.policy, self.store, self.subset, self.model_version = policy, store, subset, model_version

    def step(self, inp: CycleInputs, explain: Callable[[], list[dict]] | None = None,
             true_rul: float | None = None) -> dict:
        pol, st = self.policy, self.store
        dec = decide(inp, pol)
        out = {"unit": inp.unit, "cycle": inp.cycle, "rule_id": dec.rule_id, "severity": dec.severity,
               "priority": dec.priority, "action": "none", "ticket_id": None}
        t = st.get_open(self.subset, inp.unit)
        live = {"updated_cycle": inp.cycle, "p10": _f(inp.p10), "p50": _f(inp.p50), "p90": _f(inp.p90),
                "anomaly_a": _f(inp.anomaly_A)}

        if t is None:
            if dec.severity == "MONITOR":
                return out
            last = st.last_closed(self.subset, inp.unit)
            if last and inp.cycle < (last["closed_cycle"] or 0) + pol.reopen_cooldown and dec.rank <= (
                    PRI_RANK[last["priority"]], SEV_RANK[last["severity"]]):
                out["action"] = "suppressed"   # inside the post-closure cool-down and not worse than before
                return out
            t = st.create(self._new_ticket(inp, dec, explain, true_rul))
            st.log_event(t, inp.cycle, "created", None, dec.severity, "; ".join(dec.reasons), true_rul)
            st.db.commit()
            out.update(action="created", ticket_id=t["ticket_id"])
            return out

        out["ticket_id"] = t["ticket_id"]
        cur = (PRI_RANK[t["priority"]], SEV_RANK[t["severity"]])
        if dec.rank > cur:                                   # ---- escalate
            old = t["severity"]
            fields = self._decision_fields(inp, dec, explain)
            fields.update(live, stable_count=0)
            st.update(t, **fields)
            st.log_event(t, inp.cycle, "escalated", old, dec.severity, "; ".join(dec.reasons), true_rul)
            st.db.commit()
            out["action"] = "escalated"
        elif dec.rank == cur:
            st.touch(t, **live, stable_count=0)              # ---- same level: refresh evidence, never duplicate
            out["action"] = "refreshed"
        else:                                                # ---- lower than the open ticket
            if dec.rule_id == 1:                             # bad data must never de-escalate a live alert
                st.touch(t, **live)
                if not any("data/model review" in str(e) for e in (t.get("evidence") or {}).get("notes", [])):
                    ev = dict(t.get("evidence") or {})
                    ev["notes"] = list(ev.get("notes", [])) + [f"cycle {inp.cycle}: data/model review requested: " + "; ".join(dec.reasons)]
                    st.update(t, evidence=ev)
                    st.log_event(t, inp.cycle, "annotated", t["severity"], t["severity"], "; ".join(dec.reasons), true_rul)
                    st.db.commit()
                out["action"] = "annotated"
            elif t.get("human_override"):
                st.touch(t, **live)                          # engineer override holds the level
                out["action"] = "held_by_override"
            else:
                n = int(t.get("stable_count") or 0) + 1
                if n >= pol.deescalate_after and dec.severity != "MONITOR":
                    old = t["severity"]
                    fields = self._decision_fields(inp, dec, explain)
                    fields.update(live, stable_count=0)
                    st.update(t, **fields)
                    st.log_event(t, inp.cycle, "deescalated", old, dec.severity, f"{n} stable cycles; " + "; ".join(dec.reasons), true_rul)
                    st.db.commit()
                    out["action"] = "deescalated"
                else:
                    st.touch(t, **live, stable_count=n)
                    out["action"] = "recommend_close" if dec.severity == "MONITOR" and n >= pol.deescalate_after else "stable_lower"
        return out

    # ------------------------------------------------------------------ helpers
    def _decision_fields(self, inp: CycleInputs, dec: Decision, explain) -> dict:
        ev = {"rule_path": dec.rule_id, "reasons": dec.reasons, "sustained_anomaly": bool(inp.sustained),
              "shadow_priority": dec.shadow_priority, "notes": []}
        f = {"type": "exception" if dec.severity == "REVIEW" else "predictive", "severity": dec.severity,
             "priority": dec.priority, "rule_id": dec.rule_id, "rule_name": dec.rule_name, "action": dec.action,
             "due_cycle": due_cycle(dec, inp.cycle, self.policy), "evidence": ev,
             "model_version": self.model_version, "policy_version": self.policy.version}
        if explain is not None and dec.rule_id != 1:
            f["key_sensors"] = explain()
        return f

    def _new_ticket(self, inp: CycleInputs, dec: Decision, explain, true_rul) -> dict:
        t = {"subset": self.subset, "unit": inp.unit, "status": "open", "created_cycle": inp.cycle,
             "updated_cycle": inp.cycle, "p10": _f(inp.p10), "p50": _f(inp.p50), "p90": _f(inp.p90),
             "anomaly_a": _f(inp.anomaly_A), "key_sensors": [], "true_rul_at_open": _f(true_rul)}
        t.update(self._decision_fields(inp, dec, explain))
        return t


def _f(x) -> float | None:
    return None if x is None or not np.isfinite(x) else float(x)
