import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from distributions import LogNormal
from regressors import LogNormalRegressor


def _assert_calibrated_lognormal(regressor: LogNormalRegressor) -> LogNormalRegressor:
    calibrated = regressor.calibrate()
    assert calibrated.beta_0 is not None
    assert calibrated.log_standard_deviation > 0
    assert calibrated.beta_1.shape == regressor.beta_1.shape
    return calibrated


def _assert_close(actual: float, expected: float, *, rtol: float = 0.15) -> None:
    assert np.isclose(actual, expected, rtol=rtol)


def test_lognormal_regressor_intercept_only():
    rng = np.random.default_rng(3)
    X = np.zeros((100, 1), dtype=float)
    regressor = LogNormalRegressor(
        log_mean=0.2,
        log_standard_deviation=0.4,
        min=0.01,
        max=25.0,
        X=X,
        beta_1=np.array([0.0], dtype=float),
    )

    calibrated = _assert_calibrated_lognormal(regressor)
    sample = calibrated.sample(5000)
    assert sample.shape == (5000,)
    assert np.all(sample >= 0.01)
    target = LogNormal(log_mean=0.2, log_standard_deviation=0.4, min=0.01, max=25.0, truncated=False)
    _assert_close(float(np.mean(sample)), target._get_untruncated_mean())
    _assert_close(float(np.var(sample)), target._get_untruncated_variance())


def test_lognormal_regressor_mixed_sign_predictors():
    rng = np.random.default_rng(7)
    X = rng.normal(loc=0.0, scale=1.0, size=(200, 4))
    regressor = LogNormalRegressor(
        log_mean=0.35,
        log_standard_deviation=0.55,
        min=0.01,
        max=40.0,
        X=X,
        beta_1=np.array([0.12, -0.08, 0.05, 0.07], dtype=float),
    )

    calibrated = _assert_calibrated_lognormal(regressor)
    sample = calibrated.sample(5000)
    assert sample.shape == (5000,)
    assert np.all(sample >= 0.01)
    target = LogNormal(log_mean=0.35, log_standard_deviation=0.55, min=0.01, max=40.0, truncated=False)
    _assert_close(float(np.mean(sample)), target._get_untruncated_mean())
    _assert_close(float(np.var(sample)), target._get_untruncated_variance())


def test_lognormal_regressor_large_moments():
    rng = np.random.default_rng(11)
    X = rng.normal(loc=0.0, scale=0.35, size=(400, 4))
    regressor = LogNormalRegressor(
        log_mean=13.0,
        log_standard_deviation=0.5,
        min=0.01,
        max=5000000.0,
        X=X,
        beta_1=np.array([0.08, -0.04, 0.02, 0.01], dtype=float),
    )

    calibrated = _assert_calibrated_lognormal(regressor)
    sample = calibrated.sample(10000)
    assert sample.shape == (10000,)
    assert np.all(sample >= 0.01)
    target = LogNormal(log_mean=13.0, log_standard_deviation=0.5, min=0.01, max=5000000.0, truncated=False)
    _assert_close(float(np.mean(sample)), target._get_untruncated_mean())
    _assert_close(float(np.var(sample)), target._get_untruncated_variance())


def test_lognormal_truncation_sanity():
    dist = LogNormal(log_mean=0.0, log_standard_deviation=0.5, min=0.01, max=100.0, truncated=True)
    _assert_close(dist._get_truncated_mean(), dist._get_untruncated_mean(), rtol=1e-3)
    _assert_close(dist._get_truncated_variance(), dist._get_untruncated_variance(), rtol=1e-3)


def test_lognormal_regressor_samples_without_calibration_are_truncated():
    X = np.zeros((25, 1), dtype=float)
    regressor = LogNormalRegressor(
        log_mean=0.2,
        log_standard_deviation=0.4,
        min=0.01,
        max=6.0,
        X=X,
        beta_1=np.array([0.0], dtype=float),
        beta_0=0.2,
    )

    sample = regressor.sample(2000)
    assert sample.shape == (2000,)
    assert np.all(sample >= 0.01)
    assert np.all(sample <= 6.0)


def test_lognormal_regressor_calibrate_overrides_supplied_beta_0():
    X = np.zeros((40, 1), dtype=float)
    regressor = LogNormalRegressor(
        log_mean=0.2,
        log_standard_deviation=0.4,
        min=0.01,
        max=25.0,
        X=X,
        beta_1=np.array([0.0], dtype=float),
        beta_0=-10.0,
    )

    calibrated = regressor.calibrate()
    assert not np.isclose(calibrated.beta_0, -10.0)
