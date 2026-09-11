import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from regressors import NegativeBinomialRegressor

from tests.regressor_test_helpers import build_x_matrix


def test_negative_binomial_regressor_calibration_and_sampling() -> None:
    X = build_x_matrix(n=240, p=4, seed=61)
    beta_1_init = np.array([0.25, -0.2, 0.15, 0.1], dtype=float)

    regressor = NegativeBinomialRegressor(
        target_mean=5.0,
        target_variance=12.0,
        target_snr=0.8,
        X=X,
        beta_1_init=beta_1_init,
        min=0,
        max=18,
    ).calibrate()

    model_mean = regressor.target_mean_value(regressor.beta_0, regressor.c, regressor.shape)
    model_variance = regressor.target_variance_value(regressor.beta_0, regressor.c, regressor.shape)
    model_snr = regressor.target_snr_value(regressor.beta_0, regressor.c, regressor.shape)

    assert np.isclose(model_mean, 5.0, atol=0.25)
    assert np.isclose(model_variance, 12.0, atol=0.6)
    assert np.isclose(model_snr, 0.8, atol=0.2)

    samples = regressor.sample(3500)
    assert np.all(samples >= 0)
    assert np.all(samples <= 18)
    assert np.allclose(samples, np.round(samples))
    assert np.isclose(samples.mean(), 5.0, atol=0.4)


def test_negative_binomial_regressor_lower_snr_scenario() -> None:
    X = build_x_matrix(n=220, p=3, seed=67)
    beta_1_init = np.array([-0.18, 0.22, -0.14], dtype=float)

    regressor = NegativeBinomialRegressor(
        target_mean=3.2,
        target_variance=8.4,
        target_snr=0.5,
        X=X,
        beta_1_init=beta_1_init,
        min=0,
        max=14,
    ).calibrate()

    model_mean = regressor.target_mean_value(regressor.beta_0, regressor.c, regressor.shape)
    model_variance = regressor.target_variance_value(regressor.beta_0, regressor.c, regressor.shape)
    model_snr = regressor.target_snr_value(regressor.beta_0, regressor.c, regressor.shape)

    assert np.isclose(model_mean, 3.2, atol=0.25)
    assert np.isclose(model_variance, 8.4, atol=0.6)
    assert np.isclose(model_snr, 0.5, atol=0.2)

    samples = regressor.sample(3000)
    assert np.all(samples >= 0)
    assert np.all(samples <= 14)
    assert np.allclose(samples, np.round(samples))


def test_negative_binomial_regressor_higher_shape_scenario() -> None:
    X = build_x_matrix(n=260, p=5, seed=71)
    beta_1_init = np.array([0.3, -0.24, 0.18, 0.12, -0.16], dtype=float)

    regressor = NegativeBinomialRegressor(
        target_mean=7.0,
        target_variance=16.0,
        target_snr=1.1,
        X=X,
        beta_1_init=beta_1_init,
        min=0,
        max=22,
    ).calibrate()

    model_mean = regressor.target_mean_value(regressor.beta_0, regressor.c, regressor.shape)
    model_variance = regressor.target_variance_value(regressor.beta_0, regressor.c, regressor.shape)
    model_snr = regressor.target_snr_value(regressor.beta_0, regressor.c, regressor.shape)

    assert np.isclose(model_mean, 7.0, atol=0.35)
    assert np.isclose(model_variance, 16.0, atol=0.8)
    assert np.isclose(model_snr, 1.1, atol=0.25)

    samples = regressor.sample(4000)
    assert np.all(samples >= 0)
    assert np.all(samples <= 22)
    assert np.allclose(samples, np.round(samples))
