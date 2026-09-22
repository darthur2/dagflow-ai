import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from regressors import NormalRegressor


def _assert_calibrated_normal(regressor: NormalRegressor) -> NormalRegressor:
    calibrated = regressor.calibrate()
    assert calibrated.beta_0 is not None
    assert calibrated.sigma2 is not None
    assert calibrated.beta_1.shape == regressor.beta_1.shape
    return calibrated


def _assert_close(actual: float, expected: float, *, rtol: float = 0.15) -> None:
    assert np.isclose(actual, expected, rtol=rtol)


def test_normal_regressor_samples_without_calibration():
    X = np.zeros((100, 1), dtype=float)
    regressor = NormalRegressor(
        mean=1.25,
        standard_deviation=0.5,
        min=-2.0,
        max=4.0,
        truncated=False,
        X=X,
        beta_1=np.array([0.0], dtype=float),
        beta_0=1.25,
        sigma2=0.25,
    )

    sample = regressor.sample(2000)
    assert sample.shape == (2000,)
    assert np.all(sample >= -2.0)
    assert np.all(sample <= 4.0)


def test_normal_regressor_calibrate_overrides_supplied_beta_0():
    rng = np.random.default_rng(5)
    X = rng.normal(loc=0.0, scale=1.0, size=(250, 3))
    regressor = NormalRegressor(
        mean=3.0,
        standard_deviation=2.0,
        min=-10.0,
        max=10.0,
        truncated=False,
        X=X,
        beta_1=np.array([0.2, -0.1, 0.05], dtype=float),
        beta_0=-99.0,
        sigma2=1.0,
    )

    calibrated = _assert_calibrated_normal(regressor)
    assert not np.isclose(calibrated.beta_0, -99.0)


def test_normal_regressor_moments_match_calibration():
    rng = np.random.default_rng(9)
    X = rng.normal(loc=0.0, scale=1.0, size=(300, 4))
    regressor = NormalRegressor(
        mean=5.0,
        standard_deviation=1.5,
        min=-50.0,
        max=50.0,
        truncated=False,
        X=X,
        beta_1=np.array([0.1, -0.05, 0.02, 0.03], dtype=float),
    )

    calibrated = _assert_calibrated_normal(regressor)
    sample = calibrated.sample(5000)
    assert sample.shape == (5000,)
    assert np.all(sample >= -50.0)
    assert np.all(sample <= 50.0)
    _assert_close(float(np.mean(sample)), 5.0, rtol=0.15)
    _assert_close(float(np.var(sample)), 1.5**2, rtol=0.2)


def test_normal_regressor_raises_on_infeasible_sigma2():
    X = np.linspace(-1.0, 1.0, 50, dtype=float).reshape(-1, 1)
    regressor = NormalRegressor(
        mean=1.0,
        standard_deviation=0.1,
        min=-1.0,
        max=3.0,
        truncated=False,
        X=X,
        beta_1=np.array([2.0], dtype=float),
    )

    try:
        regressor.calibrate()
    except ValueError as exc:
        message = str(exc)
        assert "Unable to calibrate NormalRegressor" in message
        assert "sigma2" in message
        assert "X_shape" in message
    else:
        raise AssertionError("Expected calibrate() to raise ValueError")
