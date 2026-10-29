"""MLOps: challenger gate, champion/previous pointers, rollback, MLflow registry round-trip."""
import copy

import pytest

from selfheal import mlops
from selfheal.config import load_config
from selfheal.pipeline import gate

CHAMP = {"hold_ep_nasa_mean": 5.0, "hold_ep_rmse": 15.0, "ep_coverage_80": 0.80, "hold_rmse_risk": 6.0}


def test_gate_promotes_only_on_better_risk_error_tradeoff():
    cfg = load_config("FD001")
    assert gate(cfg, CHAMP, {**CHAMP, "hold_ep_nasa_mean": 4.5, "hold_ep_rmse": 14.8})["promote"]
    assert not gate(cfg, CHAMP, {**CHAMP, "hold_ep_nasa_mean": 4.99})["promote"]                       # gain below min_score_gain
    assert not gate(cfg, CHAMP, {**CHAMP, "hold_ep_nasa_mean": 4.0, "hold_ep_rmse": 16.0})["promote"]  # RMSE regression > 2%
    assert not gate(cfg, CHAMP, {**CHAMP, "hold_ep_nasa_mean": 4.0, "ep_coverage_80": 0.70})["promote"]
    assert not gate(cfg, CHAMP, {**CHAMP, "hold_ep_nasa_mean": 4.0, "hold_rmse_risk": 7.0})["promote"]  # worse near failure


@pytest.fixture()
def cfg_tmp(tmp_path):
    cfg = copy.deepcopy(load_config("FD001"))
    cfg["paths"].update(artifacts_dir=str(tmp_path / "artifacts"), outputs_dir=str(tmp_path / "outputs"), mlflow_dir=str(tmp_path / "mlflow"))
    return cfg


def test_promote_and_rollback_local_pointer(cfg_tmp):
    mlops.promote(cfg_tmp, 1, "initial", use_mlflow=False)
    mlops.promote(cfg_tmp, 2, "gate passed", use_mlflow=False)
    assert mlops.champion_version(cfg_tmp) == 2
    assert mlops.rollback(cfg_tmp, use_mlflow=False) == 1 and mlops.champion_version(cfg_tmp) == 1
    events = [h["event"] for h in mlops.read_history(cfg_tmp)]
    assert events == ["promote", "promote", "rollback"]


def test_mlflow_registry_roundtrip(cfg_tmp, small_bundle, tmp_path):
    # regression: MLflow keeps the registry URI as process-global state; switching configs in-process once sent registry
    # writes (model versions, aliases) of this test into the project's REAL registry
    other = copy.deepcopy(cfg_tmp)
    other["paths"]["mlflow_dir"] = str(tmp_path / "other_mlflow")
    mlops.setup(other)
    mlops.setup(cfg_tmp)
    assert tmp_path.resolve().as_posix() in mlops.tracking_uri(cfg_tmp)
    v1 = mlops.log_training_run(cfg_tmp, small_bundle, {}, "initial", {"note.json": {"hello": "world"}})["version"]
    mlops.promote(cfg_tmp, v1, "initial")
    v2 = mlops.log_training_run(cfg_tmp, small_bundle, {}, "retrain", {})["version"]
    mlops.promote(cfg_tmp, v2, "gate passed")
    assert (v1, v2) == (1, 2)
    assert mlops.get_alias(cfg_tmp, "champion") == 2 and mlops.get_alias(cfg_tmp, "previous") == 1
    assert mlops.rollback(cfg_tmp) == 1 and mlops.get_alias(cfg_tmp, "champion") == 1
    vers = mlops.list_versions(cfg_tmp)
    assert [v["version"] for v in vers] == [1, 2] and "champion" in vers[0]["aliases"]
    assert mlops.list_versions(other) == []                                   # nothing leaked into the other registry
