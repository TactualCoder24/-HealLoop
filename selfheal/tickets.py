"""Stage 6 - WORK ORDERS: idempotent ticket store (SQLite) with JSON / CSV export.

One open incident per (subset, engine) is enforced by a partial unique index, so an agent can only ever
update / escalate - never duplicate.  Ticket ids are deterministic (WO-<subset>-<engine>-<seq>) and no
wall-clock values are stored, so re-running a replay reproduces the identical ticket set.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
  ticket_id TEXT PRIMARY KEY, subset TEXT NOT NULL, unit INTEGER NOT NULL, seq INTEGER NOT NULL,
  type TEXT, severity TEXT, priority TEXT, rule_id INTEGER, rule_name TEXT, action TEXT,
  status TEXT NOT NULL, created_cycle INTEGER, updated_cycle INTEGER, due_cycle INTEGER,
  p10 REAL, p50 REAL, p90 REAL, anomaly_a REAL,
  key_sensors TEXT, evidence TEXT, model_version TEXT, policy_version TEXT,
  ack_by TEXT, ack_cycle INTEGER, closed_cycle INTEGER, closure_outcome TEXT, closure_note TEXT,
  stable_count INTEGER DEFAULT 0, overrides TEXT DEFAULT '[]', human_override INTEGER DEFAULT 0,
  true_rul_at_open REAL, true_rul_at_close REAL
);
CREATE UNIQUE INDEX IF NOT EXISTS one_open_per_engine ON tickets(subset, unit) WHERE status != 'closed';
CREATE TABLE IF NOT EXISTS ticket_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ticket_id TEXT, subset TEXT, unit INTEGER, cycle INTEGER,
  event TEXT, from_severity TEXT, to_severity TEXT, priority TEXT, rule_id INTEGER, note TEXT, true_rul REAL
);
"""

JSON_COLS = ("key_sensors", "evidence", "overrides")
LIVE_COLS = ("updated_cycle", "p10", "p50", "p90", "anomaly_a", "stable_count", "evidence", "key_sensors")


def _row_to_dict(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    d = dict(row)
    for c in JSON_COLS:
        if d.get(c):
            d[c] = json.loads(d[c])
    return d


class TicketStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.db.commit()
        self._open: dict[tuple[str, int], dict] = {}   # cache of open tickets (agent hot path)
        self._dirty: set[tuple[str, int]] = set()

    # ------------------------------------------------------------------ housekeeping
    def reset(self, subset: str | None = None) -> None:
        """Wipe tickets (+events) of one subset so a replay starts from a clean, reproducible state."""
        if subset:
            self.db.execute("DELETE FROM tickets WHERE subset=?", (subset,))
            self.db.execute("DELETE FROM ticket_events WHERE subset=?", (subset,))
        else:
            self.db.execute("DELETE FROM tickets")
            self.db.execute("DELETE FROM ticket_events")
        self.db.commit()
        self._open = {k: v for k, v in self._open.items() if subset and k[0] != subset}
        self._dirty.clear()

    def close_connection(self) -> None:
        self.flush()
        self.db.close()

    # ------------------------------------------------------------------ reads
    def get_open(self, subset: str, unit: int) -> dict | None:
        key = (subset, unit)
        if key in self._open:
            return self._open[key]
        row = self.db.execute(
            "SELECT * FROM tickets WHERE subset=? AND unit=? AND status!='closed'", (subset, unit)
        ).fetchone()
        t = _row_to_dict(row)
        if t:
            self._open[key] = t
        return t

    def last_closed(self, subset: str, unit: int) -> dict | None:
        row = self.db.execute(
            "SELECT * FROM tickets WHERE subset=? AND unit=? AND status='closed' ORDER BY seq DESC LIMIT 1",
            (subset, unit),
        ).fetchone()
        return _row_to_dict(row)

    def tickets(self, subset: str | None = None, status: str | None = None) -> pd.DataFrame:
        self.flush()
        q, args = "SELECT * FROM tickets WHERE 1=1", []
        if subset:
            q += " AND subset=?"; args.append(subset)
        if status == "open":
            q += " AND status!='closed'"
        elif status:
            q += " AND status=?"; args.append(status)
        return pd.read_sql_query(q + " ORDER BY subset, unit, seq", self.db, params=args)

    def events(self, subset: str | None = None) -> pd.DataFrame:
        q, args = "SELECT * FROM ticket_events", []
        if subset:
            q += " WHERE subset=?"; args.append(subset)
        return pd.read_sql_query(q + " ORDER BY id", self.db, params=args)

    # ------------------------------------------------------------------ writes
    def create(self, t: dict) -> dict:
        """Insert a new open ticket (fails on a second open incident for the same engine)."""
        n = self.db.execute("SELECT COUNT(*) FROM tickets WHERE subset=? AND unit=?", (t["subset"], t["unit"])).fetchone()[0]
        t = dict(t)
        t["seq"] = n + 1
        t["ticket_id"] = f"WO-{t['subset']}-{int(t['unit']):04d}-{t['seq']:03d}"
        t.setdefault("status", "open")
        t.setdefault("stable_count", 0)
        t.setdefault("overrides", [])
        t.setdefault("human_override", 0)
        cols = list(t)
        vals = [json.dumps(t[c], sort_keys=True) if c in JSON_COLS else t[c] for c in cols]
        self.db.execute(f"INSERT INTO tickets ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})", vals)
        self.db.commit()
        self._open[(t["subset"], t["unit"])] = t
        return t

    def update(self, ticket: dict, **fields) -> dict:
        """Write-through update of structural changes (escalation, ack, close, annotation)."""
        ticket.update(fields)
        self._persist(ticket)
        self.db.commit()
        key = (ticket["subset"], ticket["unit"])
        if ticket["status"] == "closed":
            self._open.pop(key, None)
            self._dirty.discard(key)
        else:
            self._open[key] = ticket
            self._dirty.discard(key)
        return ticket

    def touch(self, ticket: dict, **fields) -> None:
        """Cache-only refresh of live evidence (latest quantiles, A ...); persisted on flush()."""
        ticket.update(fields)
        self._dirty.add((ticket["subset"], ticket["unit"]))

    def _persist(self, t: dict) -> None:
        cols = [c for c in t if c not in ("ticket_id",)]
        vals = [json.dumps(t[c], sort_keys=True) if c in JSON_COLS else t[c] for c in cols]
        self.db.execute(f"UPDATE tickets SET {','.join(f'{c}=?' for c in cols)} WHERE ticket_id=?", vals + [t["ticket_id"]])

    def flush(self) -> None:
        for key in list(self._dirty):
            t = self._open.get(key)
            if t:
                self._persist(t)
        self._dirty.clear()
        self.db.commit()

    def log_event(self, ticket: dict, cycle: int, event: str, from_sev: str | None, to_sev: str | None,
                  note: str = "", true_rul: float | None = None) -> None:
        """Append-only audit trail: what changed, at which cycle, under which rule (and the ticket priority then).
        `true_rul` is verification metadata that exists only in replay/evaluation; the agent never reads it."""
        self.db.execute(
            "INSERT INTO ticket_events (ticket_id, subset, unit, cycle, event, from_severity, to_severity, priority, rule_id, note, true_rul)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (ticket["ticket_id"], ticket["subset"], ticket["unit"], int(cycle), event, from_sev, to_sev,
             ticket.get("priority"), ticket.get("rule_id"), note, true_rul),
        )

    # ------------------------------------------------------------------ human actions (engineer in the loop)
    def acknowledge(self, subset: str, unit: int, who: str, cycle: int) -> dict | None:
        t = self.get_open(subset, unit)
        if not t or t["status"] == "closed":
            return None
        self.update(t, status="acknowledged", ack_by=who, ack_cycle=int(cycle))
        self.log_event(t, cycle, "acknowledged", t["severity"], t["severity"], f"by {who}")
        self.db.commit()
        return t

    def close(self, subset: str, unit: int, cycle: int, outcome: str, note: str = "", who: str = "engineer",
              true_rul: float | None = None) -> dict | None:
        t = self.get_open(subset, unit)
        if not t:
            return None
        self.update(t, status="closed", closed_cycle=int(cycle), closure_outcome=outcome, closure_note=note,
                    true_rul_at_close=true_rul)
        self.log_event(t, cycle, "closed", t["severity"], None, f"{outcome} by {who}: {note}")
        self.db.commit()
        return t

    def override_severity(self, subset: str, unit: int, cycle: int, who: str, severity: str, priority: str,
                          reason: str = "") -> dict | None:
        """Human override of the agent's decision; it is logged and blocks automatic de-escalation."""
        t = self.get_open(subset, unit)
        if not t:
            return None
        ov = list(t.get("overrides") or [])
        ov.append({"cycle": int(cycle), "by": who, "from": [t["severity"], t["priority"]],
                   "to": [severity, priority], "reason": reason})
        old = t["severity"]
        self.update(t, severity=severity, priority=priority, overrides=ov, human_override=1)
        self.log_event(t, cycle, "override", old, severity, f"{who}: {reason}")
        self.db.commit()
        return t

    # ------------------------------------------------------------------ export
    def export(self, out_dir: str | Path, subset: str) -> pd.DataFrame:
        """Write work_orders.csv + one JSON per ticket (deterministic content -> idempotent)."""
        self.flush()
        out = Path(out_dir)
        (out / "work_orders").mkdir(parents=True, exist_ok=True)
        df = self.tickets(subset)
        flat = df.copy()
        flat.to_csv(out / "work_orders.csv", index=False)
        for _, row in df.iterrows():
            d = {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}
            for c in JSON_COLS:
                if d.get(c):
                    d[c] = json.loads(d[c])
            d = {k: (v.item() if hasattr(v, "item") else v) for k, v in d.items()}
            (out / "work_orders" / f"{d['ticket_id']}.json").write_text(json.dumps(d, indent=2, sort_keys=True), encoding="utf-8")
        return df
