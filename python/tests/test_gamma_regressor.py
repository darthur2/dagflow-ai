import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from regressors import GammaRegressor


def _assert_calibrated_gamma(regressor: GammaRegressor) -> GammaRegressor:
    calibrated = regressor.calibrate()
    assert calibrated.shape > 0
    assert np.isfinite(calibrated.beta_0)
    assert calibrated.beta_1.shape == regressor.beta_1.shape
    return calibrated


def _assert_close(actual: float, expected: float, *, rtol: float = 1e-1) -> None:
    assert np.isclose(actual, expected, rtol=rtol)


def test_gamma_regressor_intercept_only():
    X = np.zeros((50, 1), dtype=float)
    regressor = GammaRegressor(
        shape=4.0,
        rate=2.0,
        min=0.0,
        max=100.0,
        truncated=False,
        X=X,
        beta_0=1.5,
        beta_1=np.array([0.0], dtype=float),
    )

    calibrated = _assert_calibrated_gamma(regressor)
    sample = calibrated.sample(5000)
    assert sample.shape == (5000,)
    assert np.all(sample >= 0.0)
    _assert_close(float(np.mean(sample)), 4.0 / 2.0, rtol=0.15)
    _assert_close(float(np.var(sample)), 4.0 / (2.0**2), rtol=0.15)


def test_gamma_regressor_mixed_sign_predictors():
    rng = np.random.default_rng(7)
    X = rng.normal(loc=0.0, scale=1.0, size=(200, 4))
    regressor = GammaRegressor(
        shape=10.0,
        rate=1.25,
        min=0.0,
        max=200.0,
        truncated=False,
        X=X,
        beta_0=0.25,
        beta_1=np.array([0.15, -0.08, 0.05, 0.07], dtype=float),
    )

    calibrated = _assert_calibrated_gamma(regressor)
    sample = calibrated.sample(5000)
    assert sample.shape == (5000,)
    assert np.all(sample >= 0.0)
    _assert_close(float(np.mean(sample)), 10.0 / 1.25, rtol=0.15)
    _assert_close(float(np.var(sample)), 10.0 / (1.25**2), rtol=0.15)


def test_gamma_regressor_large_target_moments():
    rng = np.random.default_rng(11)
    X=rng.normal(loc=0.0, scale=0.35, size=(400, 4))
    regressor = GammaRegressor(
        shape=4.0,
        rate=0.00001,
        min=0.0,
        max=1500000.0,
        truncated=False,
        X=X,
        beta_0=4.0,
        beta_1=np.array([0.08, -0.04, 0.02, 0.01], dtype=float),
    )

    calibrated = _assert_calibrated_gamma(regressor)
    sample = calibrated.sample(10000)
    assert sample.shape == (10000,)
    assert np.all(sample >= 0.0)
    _assert_close(float(np.mean(sample)), 4.0 / 0.00001, rtol=0.15)
    _assert_close(float(np.var(sample)), 4.0 / (0.00001**2), rtol=0.15)


def test_gamma_regressor_large_monte_carlo_sample():
    rng = np.random.default_rng(23)
    X = rng.normal(loc=0.2, scale=0.8, size=(5000, 4))
    regressor = GammaRegressor(
        shape=6.0,
        rate=3.0,
        min=0.0,
        max=500.0,
        truncated=False,
        X=X,
        beta_0=0.4,
        beta_1=np.array([0.05, -0.03, 0.02, 0.01], dtype=float),
    )

    calibrated = _assert_calibrated_gamma(regressor)
    sample = calibrated.sample(5000)
    assert sample.shape == (5000,)
    assert np.all(sample >= 0.0)
    _assert_close(float(np.mean(sample)), 6.0 / 3.0, rtol=0.15)
    _assert_close(float(np.var(sample)), 6.0 / (3.0**2), rtol=0.15)


def test_gamma_regressor_samples_without_calibration_are_truncated():
    X = np.zeros((25, 1), dtype=float)
    regressor = GammaRegressor(
        shape=2.5,
        rate=1.5,
        min=0.25,
        max=4.0,
        truncated=False,
        X=X,
        beta_0=0.2,
        beta_1=np.array([0.0], dtype=float),
    )

    sample = regressor.sample(2000)
    assert sample.shape == (2000,)
    assert np.all(sample >= 0.25)
    assert np.all(sample <= 4.0)


def test_gamma_regressor_calibrate_overrides_supplied_beta_0():
    X = np.zeros((40, 1), dtype=float)
    regressor = GammaRegressor(
        shape=3.0,
        rate=2.0,
        min=0.0,
        max=100.0,
        truncated=False,
        X=X,
        beta_0=-5.0,
        beta_1=np.array([0.0], dtype=float),
    )

    calibrated = regressor.calibrate()
    assert not np.isclose(calibrated.beta_0, -5.0)
