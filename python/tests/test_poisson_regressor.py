import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from regressors import PoissonRegressor

from tests.regressor_test_helpers import build_x_matrix


def test_poisson_regressor_calibration_and_sampling() -> None:
    X = build_x_matrix(n=180, p=4, seed=17)
    beta_1_init = np.array([0.35, -0.2, 0.15, 0.1], dtype=float)

    regressor = PoissonRegressor(
        target_mean=4.0,
        target_snr=0.55,
        X=X,
        beta_1_init=beta_1_init,
        min=0,
        max=11,
    ).calibrate()

    model_mean = regressor.target_mean_value(regressor.beta_0, regressor.c)
    model_snr = regressor.target_snr_value(regressor.beta_0, regressor.c)

    assert np.isclose(model_mean, 4.0, atol=0.15)
    assert np.isclose(model_snr, 0.55, atol=0.15)

    samples = regressor.sample(3000)
    assert np.issubdtype(samples.dtype, np.floating)
    assert np.all(samples >= 0)
    assert np.all(samples <= 11)
    assert np.allclose(samples, np.round(samples))
    assert np.isclose(samples.mean(), 4.0, atol=0.35)


def test_poisson_regressor_low_rate_scenario() -> None:
    X = build_x_matrix(n=160, p=3, seed=23)
    beta_1_init = np.array([0.2, -0.15, 0.25], dtype=float)

    regressor = PoissonRegressor(
        target_mean=1.8,
        target_snr=0.25,
        X=X,
        beta_1_init=beta_1_init,
        min=0,
        max=8,
    ).calibrate()

    model_mean = regressor.target_mean_value(regressor.beta_0, regressor.c)
    model_snr = regressor.target_snr_value(regressor.beta_0, regressor.c)

    assert np.isclose(model_mean, 1.8, atol=0.2)
    assert np.isclose(model_snr, 0.25, atol=0.2)

    samples = regressor.sample(2500)
    assert np.all(samples >= 0)
    assert np.all(samples <= 8)
    assert np.allclose(samples, np.round(samples))
    assert np.isclose(samples.mean(), 1.8, atol=0.3)


def test_poisson_regressor_higher_rate_scenario() -> None:
    X = build_x_matrix(n=220, p=5, seed=31)
    beta_1_init = np.array([-0.3, 0.18, 0.22, -0.12, 0.14], dtype=float)

    regressor = PoissonRegressor(
        target_mean=6.2,
        target_snr=0.7,
        X=X,
        beta_1_init=beta_1_init,
        min=1,
        max=14,
    ).calibrate()

    model_mean = regressor.target_mean_value(regressor.beta_0, regressor.c)
    model_snr = regressor.target_snr_value(regressor.beta_0, regressor.c)

    assert np.isclose(model_mean, 6.2, atol=0.2)
    assert np.isclose(model_snr, 0.7, atol=0.2)

    samples = regressor.sample(3500)
    assert np.all(samples >= 1)
    assert np.all(samples <= 14)
    assert np.allclose(samples, np.round(samples))
    assert np.isclose(samples.mean(), 6.2, atol=0.4)
