"""Command line: `python -m selfheal <command>`  (run-all is the one-command pipeline)."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import warnings

from . import mlops
from .config import ROOT, SUBSETS, load_config, resolve

warnings.filterwarnings("ignore", category=UserWarning)


def _subsets(arg: str | list[str]) -> list[str]:
    if isinstance(arg, str):
        arg = [arg]
    out: list[str] = []
    for a in arg:
        out += list(SUBSETS) if a.lower() == "all" else [a.upper()]
    return out


def cmd_download(a) -> None:
    from .download import download
    print("data ->", download(resolve(load_config("FD001"), "raw_dir"), force=a.force))


def cmd_train(a) -> None:
    from .pipeline import train_and_promote
    for s in _subsets(a.subset):
        train_and_promote(load_config(s), use_mlflow=not a.no_mlflow, force=a.force)


def cmd_replay(a) -> None:
    from .pipeline import replay
    for s in _subsets(a.subset):
        cfg = load_config(s)
        b = mlops.load_champion(cfg)
        for sc in a.scenario:
            r = replay(cfg, b, sc, verbose=not a.quiet)
            print(f"[{s}] replay '{sc}' with {r['model_version']}: {r['tickets_total']} tickets, "
                  f"risk engines flagged P1/P2 {r['risk_engines_flagged_P1P2_ever']}/{r['risk_engines_rul_le_30']}, "
                  f"review tickets {r['review_tickets']}, drift={r['monitor']['drift_status']}, quality={r['monitor']['quality_status']} ({r['seconds']}s)")


def cmd_retrain(a) -> None:
    from .pipeline import retrain_demo
    for s in _subsets(a.subset):
        r = retrain_demo(load_config(s), use_mlflow=not a.no_mlflow)
        if r.get("skipped"):
            continue
        print(f"[{s}] challenger v{r['challenger']} {'PROMOTED' if r['promoted'] else 'rejected'}; champion is now v{r['champion_after']}")


def cmd_rollback(a) -> None:
    for s in _subsets(a.subset):
        v = mlops.rollback(load_config(s), use_mlflow=not a.no_mlflow)
        print(f"[{s}] rolled back; champion is now v{v}")


def cmd_report(a) -> None:
    from .report import write_report
    print(write_report())


def cmd_status(a) -> None:
    for s in _subsets(a.subset):
        cfg = load_config(s)
        v = mlops.champion_version(cfg)
        print(f"{s}: champion={'v' + str(v) if v else 'none'}")
        if not a.no_mlflow:
            for r in mlops.list_versions(cfg):
                print(f"   v{r['version']:<3} {r['label']:<8} aliases={r['aliases']}  val_rmse={r['val_rmse']}  test_rmse={r['test_rmse']}")


def cmd_run_all(a) -> None:
    from .download import download
    from .pipeline import replay, retrain_demo, train_and_promote
    from .report import write_report
    t0 = time.time()
    subsets = _subsets(a.subsets)
    download(resolve(load_config(subsets[0]), "raw_dir"))
    for s in subsets:
        cfg = load_config(s)
        train_and_promote(cfg, use_mlflow=not a.no_mlflow, force=a.force)          # first model: historical engines only
        if not a.skip_retrain:                                                      # completed histories released -> gate
            r = retrain_demo(cfg, use_mlflow=not a.no_mlflow)
            if not r.get("skipped"):
                print(f"[{s}] retrain: challenger v{r['challenger']} {'PROMOTED' if r['promoted'] else 'rejected'} -> champion v{r['champion_after']}")
        b = mlops.load_champion(cfg)                                                # replay the test fleet with the final champion
        for sc in ("clean", "faults"):
            r = replay(cfg, b, sc, verbose=False)
            print(f"[{s}] replay '{sc}' with {r['model_version']}: {r['tickets_total']} tickets ({r['seconds']}s), drift={r['monitor']['drift_status']}")
    print(write_report())
    print(f"\nall done in {time.time() - t0:.0f}s. Dashboard: python -m selfheal dashboard   MLflow UI: python -m selfheal mlflow-ui")


def cmd_dashboard(a) -> None:
    # headless: skips Streamlit's interactive first-run e-mail prompt; we open the browser ourselves (not in containers)
    cmd = [sys.executable, "-m", "streamlit", "run", str(ROOT / "app" / "dashboard.py"), "--server.port", str(a.port),
           "--server.headless", "true", "--browser.gatherUsageStats", "false", "--server.address", a.host or "localhost"]
    url = f"http://localhost:{a.port}"
    print(f"dashboard -> {url}   (Ctrl+C to stop)")
    proc = subprocess.Popen(cmd)
    if not a.host and not a.no_browser:
        import threading
        import webbrowser
        threading.Timer(4.0, lambda: webbrowser.open(url)).start()
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()


def cmd_mlflow_ui(a) -> None:
    uri = mlops.tracking_uri(load_config("FD001"))
    subprocess.run([sys.executable, "-m", "mlflow", "ui", "--backend-store-uri", uri, "--port", str(a.port)]
                   + (["--host", a.host] if a.host else []), check=False)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="selfheal", description="Self-healing machines: closed-loop predictive maintenance on NASA C-MAPSS")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help_, subset=True):
        sp = sub.add_parser(name, help=help_)
        sp.set_defaults(fn=fn)
        if subset:
            sp.add_argument("--subset", nargs="+", default=["all"], help="FD001..FD004 or all")
        sp.add_argument("--no-mlflow", action="store_true", help="skip MLflow tracking/registry (local files only)")
        return sp

    add("download", cmd_download, "download the official C-MAPSS files", subset=False).add_argument("--force", action="store_true")
    add("train", cmd_train, "train + register + promote the first champion").add_argument("--force", action="store_true")
    r = add("replay", cmd_replay, "stream the test fleet cycle-by-cycle through the agent")
    r.add_argument("--scenario", nargs="+", default=["clean"], choices=["clean", "faults"])
    r.add_argument("--quiet", action="store_true")
    add("retrain", cmd_retrain, "demo retraining: released labelled histories -> challenger gate")
    add("rollback", cmd_rollback, "roll the champion back to the previous version")
    add("report", cmd_report, "write outputs/RESULTS.md", subset=False)
    add("status", cmd_status, "show champions and registry versions")
    ra = add("run-all", cmd_run_all, "ONE COMMAND: download -> train -> retrain demo -> replay -> fault replay -> report", subset=False)
    ra.add_argument("--subsets", nargs="+", default=["all"])
    ra.add_argument("--force", action="store_true")
    ra.add_argument("--skip-retrain", action="store_true")
    d = add("dashboard", cmd_dashboard, "launch the Streamlit dashboard", subset=False)
    d.add_argument("--port", type=int, default=8501)
    d.add_argument("--host", default=None, help="bind address (default localhost; use 0.0.0.0 in containers - disables the auto-open of the browser)")
    d.add_argument("--no-browser", action="store_true")
    m = add("mlflow-ui", cmd_mlflow_ui, "launch the MLflow tracking UI", subset=False)
    m.add_argument("--port", type=int, default=5000)
    m.add_argument("--host", default=None)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
