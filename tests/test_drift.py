import numpy as np
import pandas as pd

from src.benchmark import NUMERIC
from src.drift import check_drift, ks_statistic, psi


def _frame(seed: int, scale: float = 1.0, n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({c: rng.lognormal(8, 1, n) * scale for c in NUMERIC})


def test_same_distribution_has_low_psi():
    assert psi(_frame(1)["Fresh"], _frame(2)["Fresh"]) < 0.10


def test_shifted_distribution_is_flagged():
    report = check_drift(_frame(1), _frame(2, scale=3.0))
    assert (report["status"] == "alert").all()


def test_ks_bounds():
    s = pd.Series(np.arange(100.0))
    assert ks_statistic(s, s) == 0.0
    assert ks_statistic(s, s + 1000) == 1.0


def test_constant_feature_does_not_crash():
    const = pd.Series([5.0] * 50)
    assert psi(const, const) == 0.0
