import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import pytest

from selfheal.config import load_config, resolve
from selfheal.data import load_subset


@pytest.fixture(scope="session")
def raw():
    cfg = load_config("FD001")
    train, test, rul = load_subset(resolve(cfg, "raw_dir"), "FD001")
    return train, test, rul


@pytest.fixture(scope="session")
def small_bundle():
    """A genuinely trained (but small/fast) FD001 bundle used by streaming / agent tests."""
    from selfheal.train import train_subset
    cfg = load_config("FD001")
    cfg["rul"]["lgbm"].update(n_estimators=80, learning_rate=0.1, early_stopping_rounds=15)
    cfg["anomaly"]["if_trees"] = 50
    cfg["policy"]["tuning"]["grid"].update(critical=[15], high=[40], medium=[80], width_q=[0.9])
    cfg["rul"]["near_failure"]["weights"] = [1.0]
    return train_subset(cfg, verbose=False).bundle
