import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from regressors import BetaRegressor


def _assert_calibrated_beta(regressor: BetaRegressor) -> BetaRegressor:
    calibrated = regressor.calibrate()
    assert calibrated.beta_0 is not None
    assert calibrated.phi is not None
    assert calibrated.beta_1.shape == regressor.beta_1.shape
    return calibrated


def test_beta_regressor_samples_without_calibration():
    X = np.zeros((100, 1), dtype=float)
    regressor = BetaRegressor(
        shape_1=2.0,
        shape_2=3.0,
        min=0.0,
        max=1.0,
        X=X,
        beta_1=np.array([0.0], dtype=float),
        beta_0=0.0,
        phi=5.0,
    )

    sample = regressor.sample(2000)
    assert sample.shape == (2000,)
    assert np.all(sample >= 0.0)
    assert np.all(sample <= 1.0)


def test_beta_regressor_calibrate_overrides_supplied_beta_0():
    rng = np.random.default_rng(4)
    X = rng.normal(loc=0.0, scale=1.0, size=(300, 3))
    regressor = BetaRegressor(
        shape_1=2.0,
        shape_2=3.0,
        min=0.0,
        max=1.0,
        X=X,
        beta_1=np.array([0.1, -0.05, 0.08], dtype=float),
        beta_0=-99.0,
        phi=4.0,
    )

    calibrated = _assert_calibrated_beta(regressor)
    assert not np.isclose(calibrated.beta_0, -99.0)


def test_beta_regressor_calibrate_produces_valid_samples():
    rng = np.random.default_rng(8)
    X = rng.normal(loc=0.0, scale=1.0, size=(250, 4))
    regressor = BetaRegressor(
        shape_1=2.0,
        shape_2=3.0,
        min=0.0,
        max=1.0,
        X=X,
        beta_1=np.array([0.06, -0.04, 0.03, 0.02], dtype=float),
    )

    calibrated = _assert_calibrated_beta(regressor)
    sample = calibrated.sample(5000)
    assert sample.shape == (5000,)
    assert np.all(sample >= 0.0)
    assert np.all(sample <= 1.0)
    assert np.isfinite(float(np.mean(sample)))
    assert np.isfinite(float(np.var(sample)))
