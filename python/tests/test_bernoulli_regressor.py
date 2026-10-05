import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from regressors import BernoulliRegressor


def _assert_calibrated_bernoulli(regressor: BernoulliRegressor) -> BernoulliRegressor:
    calibrated = regressor.calibrate()
    assert calibrated.beta_0 is not None
    assert calibrated.beta_1.shape == regressor.beta_1.shape
    return calibrated


def test_bernoulli_regressor_samples_without_calibration():
    X = np.zeros((100, 1), dtype=float)
    regressor = BernoulliRegressor(
        success_prob=0.35,
        X=X,
        beta_1=np.array([0.0], dtype=float),
        beta_0=0.0,
    )

    sample = regressor.sample(2000)
    assert sample.shape == (2000,)
    assert set(np.unique(sample)).issubset({0, 1})


def test_bernoulli_regressor_calibrate_overrides_supplied_beta_0():
    rng = np.random.default_rng(4)
    X = rng.normal(loc=0.0, scale=1.0, size=(300, 3))
    regressor = BernoulliRegressor(
        success_prob=0.65,
        X=X,
        beta_1=np.array([0.1, -0.05, 0.08], dtype=float),
        beta_0=-99.0,
    )

    calibrated = _assert_calibrated_bernoulli(regressor)
    assert not np.isclose(calibrated.beta_0, -99.0)


def test_bernoulli_regressor_calibrate_produces_valid_samples():
    rng = np.random.default_rng(8)
    X = rng.normal(loc=0.0, scale=1.0, size=(250, 4))
    regressor = BernoulliRegressor(
        success_prob=0.55,
        X=X,
        beta_1=np.array([0.06, -0.04, 0.03, 0.02], dtype=float),
    )

    calibrated = _assert_calibrated_bernoulli(regressor)
    sample = calibrated.sample(5000)
    assert sample.shape == (5000,)
    assert set(np.unique(sample)).issubset({0, 1})
    assert np.isfinite(float(np.mean(sample)))
