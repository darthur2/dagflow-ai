import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from regressors import PoissonRegressor


def _assert_calibrated_poisson(regressor: PoissonRegressor) -> PoissonRegressor:
    calibrated = regressor.calibrate()
    assert calibrated.beta_0 is not None
    assert calibrated.beta_1.shape == regressor.beta_1.shape
    return calibrated


def test_poisson_regressor_samples_without_calibration():
    X = np.zeros((100, 1), dtype=float)
    regressor = PoissonRegressor(
        rate=2.0,
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


def test_poisson_regressor_calibrate_overrides_supplied_beta_0():
    rng = np.random.default_rng(4)
    X = rng.normal(loc=0.0, scale=1.0, size=(300, 3))
    regressor = PoissonRegressor(
        rate=2.0,
        min=0,
        max=10,
        truncated=False,
        X=X,
        beta_1=np.array([0.1, -0.05, 0.08], dtype=float),
        beta_0=-99.0,
    )

    calibrated = _assert_calibrated_poisson(regressor)
    assert not np.isclose(calibrated.beta_0, -99.0)


def test_poisson_regressor_truncated_sampling_bounds():
    X = np.zeros((50, 1), dtype=float)
    regressor = PoissonRegressor(
        rate=2.0,
        min=1,
        max=4,
        truncated=False,
        X=X,
        beta_1=np.array([0.0], dtype=float),
        beta_0=0.5,
    )

    sample = regressor.sample(2000)
    assert np.all(sample >= 1)
    assert np.all(sample <= 4)
