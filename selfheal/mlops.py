"""MLOps: MLflow experiment tracking, model registry (champion / previous aliases) and rollback.

Source of truth for lineage is the MLflow registry (sqlite backend, no server needed); a small local
pointer file `artifacts/<subset>/champion.json` mirrors it so the dashboard and replay work even when
MLflow is unavailable.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import warnings
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow  # noqa: E402
from mlflow.tracking import MlflowClient

from .bundle import ModelBundle
from .config import artifacts_dir, resolve

logging.getLogger("mlflow").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", module="mlflow")


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = str(v)[:480]
    return out


def tracking_uri(cfg: dict) -> str:
    # NOTE: deliberately NOT MLFLOW_TRACKING_URI - mlflow.set_tracking_uri() exports that variable itself, so reading it here
    # pinned every later config in the process to whichever URI was set first (tests leaked into the real registry).
    env = os.environ.get("SELFHEAL_MLFLOW_URI")
    if env:
        return env
    d = resolve(cfg, "mlflow_dir")
    d.mkdir(parents=True, exist_ok=True)
    return "sqlite:///" + (d / "mlflow.db").resolve().as_posix()


def setup(cfg: dict) -> MlflowClient:
    uri = tracking_uri(cfg)
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)   # pin the registry too: MLflow keeps a separate global that goes stale if the URI changes in-process
    name = cfg["mlflow"]["experiment"]
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    if client.get_experiment_by_name(name) is None:
        art = resolve(cfg, "mlflow_dir") / "artifacts"
        art.mkdir(parents=True, exist_ok=True)
        client.create_experiment(name, artifact_location=art.resolve().as_uri())
    mlflow.set_experiment(name)
    return client


def registry_name(cfg: dict) -> str:
    return cfg["mlflow"]["registry_name"].format(subset=cfg["subset"])


def _flat_metrics(prefix: str, d: dict, out: dict) -> None:
    for k, v in d.items():
        if isinstance(v, dict):
            _flat_metrics(f"{prefix}{k}.", v, out)
        elif isinstance(v, (int, float)) and v == v and abs(v) != float("inf"):
            out[f"{prefix}{k}".replace(" ", "_")] = float(v)


def log_training_run(cfg: dict, bundle: ModelBundle, report: dict, label: str, extra_artifacts: dict[str, object]) -> dict:
    """Track one training run: params, metrics, split ids, scalers/thresholds, tuning tables, the bundle; register a version."""
    client = setup(cfg)
    reg = registry_name(cfg)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        with mlflow.start_run(run_name=f"{cfg['subset']}-{label}") as run:
            mlflow.set_tags({"subset": cfg["subset"], "label": label, "stage": "train"})
            mlflow.log_params({k: v for k, v in _flatten(cfg).items()})
            mlflow.log_param("policy_version", bundle.policy.version)
            mlflow.log_param("kept_sensors", ",".join(bundle.cleaner.kept))
            mlflow.log_param("dropped_sensors", ",".join(bundle.cleaner.dropped))
            m: dict = {}
            _flat_metrics("", bundle.metrics, m)
            mlflow.log_metrics(m)
            mlflow.log_dict(bundle.split, "split_engine_ids.json")
            mlflow.log_dict(
                {"policy": bundle.policy.__dict__, "anomaly_alert": {"A": bundle.policy.alert_A, "k": bundle.policy.alert_k, "n": bundle.policy.alert_n},
                 "conformal_qhat": bundle.rul.qhat, "ood_threshold": bundle.cleaner.regime.ood_threshold,
                 "regime_centers": bundle.cleaner.regime.centers.tolist(),
                 "scaler_median": bundle.cleaner.med_k.tolist(), "scaler_std": bundle.cleaner.sd_k.tolist()},
                "thresholds_and_scalers.json")
            for name, obj in extra_artifacts.items():
                p = tmp / name
                if name.endswith(".csv"):
                    obj.to_csv(p, index=False)
                else:
                    p.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
                mlflow.log_artifact(str(p), "reports")
            bpath = bundle.save(tmp / "bundle" / "bundle.joblib")
            mlflow.log_artifact(str(bpath), "bundle")
            run_id = run.info.run_id
    try:
        client.create_registered_model(reg)
    except Exception:
        pass
    mv = client.create_model_version(reg, source=f"{mlflow.get_run(run_id).info.artifact_uri}/bundle", run_id=run_id,
                                     tags={"label": label, "subset": cfg["subset"]})
    return {"run_id": run_id, "version": int(mv.version), "registry": reg}


def set_alias(cfg: dict, alias: str, version: int) -> None:
    client = setup(cfg)
    client.set_registered_model_alias(registry_name(cfg), alias, str(version))


def get_alias(cfg: dict, alias: str) -> int | None:
    client = setup(cfg)
    try:
        return int(client.get_model_version_by_alias(registry_name(cfg), alias).version)
    except Exception:
        return None


def list_versions(cfg: dict) -> list[dict]:
    client = setup(cfg)
    reg = registry_name(cfg)
    try:
        aliases = client.get_registered_model(reg).aliases
    except Exception:
        return []
    rows = []
    for mv in client.search_model_versions(f"name='{reg}'"):
        run = client.get_run(mv.run_id)
        rows.append({"version": int(mv.version), "label": mv.tags.get("label", ""), "run_id": mv.run_id,
                     "aliases": [a for a, v in aliases.items() if str(v) == str(mv.version)],
                     "test_rmse": run.data.metrics.get("test.rmse"), "val_rmse": run.data.metrics.get("val.ep_rmse")})
    return sorted(rows, key=lambda r: r["version"])


# ---------------------------------------------------------------------- local mirror (dashboard / offline)
def version_dir(cfg: dict, version: int) -> Path:
    d = artifacts_dir(cfg) / f"v{version}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_version(cfg: dict, bundle: ModelBundle, version: int, run_id: str) -> Path:
    bundle.version = f"{registry_name(cfg)}/v{version}"
    path = bundle.save(version_dir(cfg, version) / "bundle.joblib")
    (version_dir(cfg, version) / "meta.json").write_text(json.dumps(
        {"version": version, "run_id": run_id, "metrics": bundle.metrics, "policy": bundle.policy.__dict__}, indent=2, default=str), encoding="utf-8")
    return path


def _pointer(cfg: dict) -> Path:
    return artifacts_dir(cfg) / "champion.json"


def history_path(cfg: dict) -> Path:
    return artifacts_dir(cfg) / "history.json"


def log_history(cfg: dict, event: dict) -> None:
    p = history_path(cfg)
    hist = json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
    hist.append(event)
    p.write_text(json.dumps(hist, indent=2, default=str), encoding="utf-8")


def read_history(cfg: dict) -> list[dict]:
    p = history_path(cfg)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


def promote(cfg: dict, version: int, reason: str, use_mlflow: bool = True) -> None:
    """Make `version` the champion (previous champion keeps alias `previous` -> rollback target)."""
    ptr = _pointer(cfg)
    prev = json.loads(ptr.read_text(encoding="utf-8")) if ptr.exists() else None
    if use_mlflow:
        old = get_alias(cfg, "champion")
        if old is not None and old != version:
            set_alias(cfg, "previous", old)
        set_alias(cfg, "champion", version)
    ptr.write_text(json.dumps({"version": version, "bundle": str((version_dir(cfg, version) / "bundle.joblib").relative_to(resolve(cfg, "artifacts_dir").parent)),
                               "previous": prev["version"] if prev and prev["version"] != version else (prev or {}).get("previous")},
                              indent=2), encoding="utf-8")
    log_history(cfg, {"event": "promote", "version": version, "reason": reason})


def rollback(cfg: dict, use_mlflow: bool = True) -> int:
    ptr = _pointer(cfg)
    cur = json.loads(ptr.read_text(encoding="utf-8"))
    target = cur.get("previous")
    if target is None:
        raise RuntimeError("no previous version to roll back to")
    if use_mlflow:
        set_alias(cfg, "champion", target)
        set_alias(cfg, "previous", cur["version"])
    ptr.write_text(json.dumps({"version": target, "bundle": str((version_dir(cfg, target) / "bundle.joblib").relative_to(resolve(cfg, "artifacts_dir").parent)),
                               "previous": cur["version"]}, indent=2), encoding="utf-8")
    log_history(cfg, {"event": "rollback", "from": cur["version"], "to": target})
    return target


def champion_version(cfg: dict) -> int | None:
    ptr = _pointer(cfg)
    return json.loads(ptr.read_text(encoding="utf-8"))["version"] if ptr.exists() else None


def load_champion(cfg: dict) -> ModelBundle:
    from .config import ROOT
    ptr = _pointer(cfg)
    if not ptr.exists():
        raise FileNotFoundError(f"no champion model for {cfg['subset']} - run `python -m selfheal train --subset {cfg['subset']}`")
    info = json.loads(ptr.read_text(encoding="utf-8"))
    b = ModelBundle.load(ROOT / info["bundle"] if not Path(info["bundle"]).is_absolute() else info["bundle"])
    return b


def load_version(cfg: dict, version: int) -> ModelBundle:
    return ModelBundle.load(version_dir(cfg, version) / "bundle.joblib")
