import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from regressors import BinomialRegressor


def _assert_calibrated_binomial(regressor: BinomialRegressor) -> BinomialRegressor:
    calibrated = regressor.calibrate()
    assert calibrated.beta_0 is not None
    assert calibrated.beta_1.shape == regressor.beta_1.shape
    return calibrated


def test_binomial_regressor_samples_without_calibration():
    X = np.zeros((100, 1), dtype=float)
    regressor = BinomialRegressor(
        success_prob=0.4,
        n_trials=5,
        min=0,
        max=5,
        truncated=False,
        X=X,
        beta_1=np.array([0.0], dtype=float),
        beta_0=0.0,
    )

    sample = regressor.sample(2000)
    assert sample.shape == (2000,)
    assert np.all(sample >= 0)
    assert np.all(sample <= 5)


def test_binomial_regressor_calibrate_overrides_supplied_beta_0():
    rng = np.random.default_rng(4)
    X = rng.normal(loc=0.0, scale=1.0, size=(300, 3))
    regressor = BinomialRegressor(
        success_prob=0.4,
        n_trials=5,
        min=0,
        max=5,
        truncated=False,
        X=X,
        beta_1=np.array([0.1, -0.05, 0.08], dtype=float),
        beta_0=-99.0,
    )

    calibrated = _assert_calibrated_binomial(regressor)
    assert not np.isclose(calibrated.beta_0, -99.0)


def test_binomial_regressor_truncated_calibration_not_implemented():
    X = np.zeros((20, 1), dtype=float)
    regressor = BinomialRegressor(
        success_prob=0.4,
        n_trials=5,
        min=0,
        max=5,
        truncated=True,
        X=X,
        beta_1=np.array([0.0], dtype=float),
    )

    try:
        regressor.calibrate()
    except NotImplementedError as exc:
        assert "truncated calibration not yet implemented" in str(exc)
    else:
        raise AssertionError("Expected calibrate() to raise NotImplementedError")
