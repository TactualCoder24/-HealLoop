"""Orchestration of the MLOps loop: train + register, and the retraining flow with a grouped challenger gate."""
from __future__ import annotations

import json

from . import mlops
from .config import artifacts_dir, outputs_dir, tickets_db_path
from .replay import replay, stream_fleet
from .tickets import TicketStore
from .train import TrainResult, load_and_validate, train_subset
from .tuning import ticket_metrics


def _next_local_version(cfg: dict) -> int:
    d = artifacts_dir(cfg)
    nums = [int(p.name[1:]) for p in d.glob("v*") if p.is_dir() and p.name[1:].isdigit()]
    return max(nums, default=0) + 1


def register(cfg: dict, res: TrainResult, label: str, use_mlflow: bool, extra_report: dict | None = None) -> dict:
    """Track the run in MLflow (params/metrics/splits/scalers/thresholds/bundle), register a model version, mirror locally."""
    report = {**res.report, **(extra_report or {})}
    if use_mlflow:
        info = mlops.log_training_run(cfg, res.bundle, report, label, report)
        version, run_id = info["version"], info["run_id"]
    else:
        version, run_id = _next_local_version(cfg), "local"
    mlops.save_version(cfg, res.bundle, version, run_id)
    return {"version": version, "run_id": run_id}


def train_and_promote(cfg: dict, use_mlflow: bool = True, force: bool = False, verbose: bool = True) -> dict:
    """First model: trained on the historical (fit) engines only - the reserve pool is held back for the retraining demo."""
    subset = cfg["subset"]
    if not force and mlops.champion_version(cfg) is not None:
        v = mlops.champion_version(cfg)
        if verbose:
            print(f"[{subset}] champion v{v} already exists (use --force to retrain from scratch)")
        return {"version": v, "skipped": True}
    if verbose:
        print(f"[{subset}] training the initial model ...")
    res = train_subset(cfg, verbose=verbose)
    info = register(cfg, res, "initial", use_mlflow)
    mlops.promote(cfg, info["version"], "initial model", use_mlflow)
    if verbose:
        print(f"[{subset}] registered {mlops.registry_name(cfg)} v{info['version']} -> champion  ({res.seconds:.0f}s)")
    return {**info, "skipped": False, "seconds": res.seconds}


# ------------------------------------------------------------------------------------ verified feedback
def reserve_feedback(cfg: dict, champion, train, reserve: list[int]) -> dict:
    """The reserved engines 'run to failure' in the fleet under the CURRENT champion: the agent raises tickets, then a simulated
    engineer closes each one at the failure with an outcome verified against ground truth (this is the 'verified feedback' arrow)."""
    subset = cfg["subset"]
    key = f"{subset}+reserve"
    tcfg = cfg["policy"]["tuning"]
    frame = train[train["unit"].isin(reserve)]
    life = {int(u): int(c) for u, c in frame.groupby("unit")["cycle"].max().items()}
    store = TicketStore(tickets_db_path(cfg))
    store.reset(key)
    stream_fleet(champion, frame, key, life, store)
    ev = store.events(key)
    tickets = store.tickets(key)
    counts = {"confirmed_degradation": 0, "false_alarm": 0, "scheduled_service": 0}
    for _, t in tickets.iterrows():
        u = int(t["unit"])
        hi = ev[(ev["unit"] == u) & ev["priority"].isin(["P1", "P2"]) & ev["event"].isin(["created", "escalated"])]
        if len(hi):
            outcome = "confirmed_degradation" if float(hi.iloc[0]["true_rul"]) <= tcfg["false_alarm_rul"] else "false_alarm"
        else:
            outcome = "scheduled_service"
        store.close(key, u, life[u], outcome, "engine ran to failure; outcome verified against the completed history",
                    who="simulated-engineer", true_rul=0.0)
        counts[outcome] += 1
    tm = ticket_metrics(store, len(reserve), tcfg, key)
    d = outputs_dir(cfg) / "retrain"
    d.mkdir(parents=True, exist_ok=True)
    store.export(d, key)
    store.db.close()
    n_hi = counts["confirmed_degradation"] + counts["false_alarm"]
    return {**counts, "p1p2_precision": counts["confirmed_degradation"] / n_hi if n_hi else None, "agent_on_completed_histories": tm}


# ------------------------------------------------------------------------------------ retraining
def gate(cfg: dict, champ_val: dict, chall_val: dict) -> dict:
    """Promote only with a better risk/error trade-off on identical held-out validation engines."""
    g = cfg["retrain"]["gate"]
    # endpoint-like metrics (one random cut per engine) on the held-out pool = calibration + validation engines, which are
    # identical for champion and challenger; coverage comes from the validation engines only (calibration sets q-hat)
    score_gain = (champ_val["hold_ep_nasa_mean"] - chall_val["hold_ep_nasa_mean"]) / champ_val["hold_ep_nasa_mean"]
    rmse_regress = (chall_val["hold_ep_rmse"] - champ_val["hold_ep_rmse"]) / champ_val["hold_ep_rmse"]
    cov_drop = champ_val["ep_coverage_80"] - chall_val["ep_coverage_80"]
    risk_regress = (chall_val["hold_rmse_risk"] - champ_val["hold_rmse_risk"]) / max(champ_val["hold_rmse_risk"], 1e-9)
    checks = {
        "score_improves": score_gain >= g["min_score_gain"],
        "rmse_not_worse": rmse_regress <= g["max_rmse_regress"],
        "risk_window_not_worse": risk_regress <= 0.05,
        "coverage_kept": cov_drop <= g["max_coverage_drop"],
    }
    return {"promote": all(checks.values()), "checks": checks,
            "score_gain": float(score_gain), "rmse_regress": float(rmse_regress),
            "risk_window_rmse_regress": float(risk_regress), "coverage_drop": float(cov_drop)}


def retrain_demo(cfg: dict, use_mlflow: bool = True, verbose: bool = True) -> dict:
    """Completed, labelled run-to-failure histories arrive -> retrain -> grouped champion/challenger gate -> promote or keep."""
    subset = cfg["subset"]
    say = print if verbose else (lambda *a, **k: None)
    champion = mlops.load_champion(cfg)
    champ_ver = mlops.champion_version(cfg)
    reserve = champion.split.get("reserve", [])
    if champion.meta.get("released") or not reserve:
        say(f"[{subset}] champion v{champ_ver} already includes the released histories - nothing new to retrain on")
        return {"subset": subset, "skipped": True, "champion_after": champ_ver, "promoted": False, "challenger": champ_ver}
    train, _, _, _ = load_and_validate(cfg)
    say(f"[{subset}] retrain: {len(reserve)} reserved engines are released as completed run-to-failure histories "
        f"(unlabeled cycles would only feed drift monitoring; the official test set is never used)")

    # 1. the agent works the reserved engines to failure; a simulated engineer closes the tickets with verified outcomes
    feedback = reserve_feedback(cfg, champion, train, reserve)
    say(f"[{subset}] verified feedback on {len(reserve)} completed histories: confirmed={feedback['confirmed_degradation']}, "
        f"false alarms={feedback['false_alarm']}, scheduled service={feedback['scheduled_service']}, "
        f"P1/P2 precision={feedback['p1p2_precision']}")

    # 2. challenger = retrained on fit + released engines; SAME calibration/validation engines as the champion
    res = train_subset(cfg, split=champion.split, release_reserve=True, verbose=verbose)

    # 3. like-for-like gate on the held-out validation engines (+ the untouched official test set, informational only)
    champ_val, chall_val = champion.metrics["val"], res.bundle.metrics["val"]
    decision = gate(cfg, champ_val, chall_val)
    champ_test, chall_test = champion.metrics["test"], res.bundle.metrics["test"]
    say(f"[{subset}] gate (held-out calibration+validation engines, endpoint-like): NASA-score gain {decision['score_gain']:+.1%}, "
        f"RMSE {decision['rmse_regress']:+.1%}, coverage drop {decision['coverage_drop']:+.3f} -> "
        f"{'PROMOTE' if decision['promote'] else 'KEEP CHAMPION'}   (test RMSE {champ_test['rmse']:.2f} -> {chall_test['rmse']:.2f}, informational)")

    extra_report = {"retrain_decision.json": {"champion_version": champ_ver, "decision": decision, "feedback": feedback,
                                              "released_engines": reserve, "champion_val": champ_val, "challenger_val": chall_val,
                                              "champion_test": champ_test, "challenger_test": chall_test}}
    info = register(cfg, res, "retrain", use_mlflow, extra_report)
    if decision["promote"]:
        mlops.promote(cfg, info["version"], f"gate passed: NASA-score gain {decision['score_gain']:+.1%}", use_mlflow)
    else:
        mlops.log_history(cfg, {"event": "challenger_rejected", "version": info["version"], "champion": champ_ver, "decision": decision})
    result = {"subset": subset, "skipped": False, "champion_before": champ_ver, "challenger": info["version"],
              "promoted": decision["promote"], "decision": decision, "feedback": feedback, "released": len(reserve),
              "champion_val": champ_val, "challenger_val": chall_val, "champion_test": champ_test, "challenger_test": chall_test,
              "champion_after": mlops.champion_version(cfg)}
    rd = outputs_dir(cfg) / "retrain"
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "retrain_result.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    return result


__all__ = ["train_and_promote", "retrain_demo", "replay", "gate", "register"]
