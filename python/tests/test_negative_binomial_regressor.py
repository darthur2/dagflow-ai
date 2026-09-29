import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from regressors import NegativeBinomialRegressor


def _assert_calibrated_negative_binomial(regressor: NegativeBinomialRegressor) -> NegativeBinomialRegressor:
    calibrated = regressor.calibrate()
    assert calibrated.beta_0 is not None
    assert np.isfinite(calibrated.shape)
    assert calibrated.beta_1.shape == regressor.beta_1.shape
    return calibrated


def test_negative_binomial_regressor_samples_without_calibration():
    X = np.zeros((100, 1), dtype=float)
    regressor = NegativeBinomialRegressor(
        shape=4.0,
        mean=2.0,
        min=0,
        max=10,
        truncated=False,
        X=X,
        beta_1=np.array([0.0], dtype=float),
        beta_0=0.0,
    )

    sample = regressor.sample(2000)
    assert sample.shape == (2000,)
    assert np.all(sample >= 0)
    assert np.all(sample <= 10)


def test_negative_binomial_regressor_calibrate_overrides_supplied_beta_0():
    rng = np.random.default_rng(4)
    X = rng.normal(loc=0.0, scale=1.0, size=(300, 3))
    regressor = NegativeBinomialRegressor(
        shape=4.0,
        mean=2.0,
        min=0,
        max=10,
        truncated=False,
        X=X,
        beta_1=np.array([0.1, -0.05, 0.08], dtype=float),
        beta_0=-99.0,
    )

    calibrated = _assert_calibrated_negative_binomial(regressor)
    assert not np.isclose(calibrated.beta_0, -99.0)


def test_negative_binomial_regressor_truncated_calibration_not_implemented():
    X = np.zeros((20, 1), dtype=float)
    regressor = NegativeBinomialRegressor(
        shape=4.0,
        mean=2.0,
        min=0,
        max=10,
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


def test_negative_binomial_regressor_calibration_rejects_bad_fit():
    rng = np.random.default_rng(7)
    X = np.column_stack(
        [
            rng.integers(0, 2, size=300).astype(float),
            rng.integers(0, 2, size=300).astype(float),
            rng.uniform(10.0, 32.0, size=300),
            rng.uniform(20.0, 90.0, size=300),
        ]
    )

    regressor = NegativeBinomialRegressor(
        shape=7.0,
        mean=12.0,
        min=0,
        max=80,
        truncated=False,
        X=X,
        beta_1=np.array([0.2, 0.4, 0.065, -0.01], dtype=float),
    )

    try:
        regressor.calibrate()
    except ValueError as exc:
        message = str(exc)
        assert "fitted moments are too far from targets" in message
        assert "relative_mean_error" in message
        assert "relative_variance_error" in message
    else:
        raise AssertionError("Expected calibration to fail for the pathological extreme_heat_days fit")
