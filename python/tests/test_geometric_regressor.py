import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from regressors import GeometricRegressor

from tests.regressor_test_helpers import build_x_matrix


def test_geometric_regressor_calibration_and_sampling() -> None:
    X = build_x_matrix(n=200, p=4, seed=41)
    beta_1_init = np.array([0.3, -0.25, 0.2, 0.15], dtype=float)

    regressor = GeometricRegressor(
        target_mean=2.2,
        target_snr=0.45,
        X=X,
        beta_1_init=beta_1_init,
        min=0,
        max=12,
    ).calibrate()

    model_mean = regressor.target_mean_value(regressor.beta_0, regressor.c)
    model_snr = regressor.target_snr_value(regressor.beta_0, regressor.c)

    assert np.isclose(model_mean, 2.2, atol=0.2)
    assert np.isclose(model_snr, 0.45, atol=0.2)

    samples = regressor.sample(3000)
    assert np.all(samples >= 0)
    assert np.all(samples <= 12)
    assert np.allclose(samples, np.round(samples))
    assert np.isclose(samples.mean(), 2.2, atol=0.5)


def test_geometric_regressor_lower_mean_scenario() -> None:
    X = build_x_matrix(n=180, p=3, seed=43)
    beta_1_init = np.array([-0.2, 0.18, -0.12], dtype=float)

    regressor = GeometricRegressor(
        target_mean=0.9,
        target_snr=0.2,
        X=X,
        beta_1_init=beta_1_init,
        min=0,
        max=8,
    ).calibrate()

    model_mean = regressor.target_mean_value(regressor.beta_0, regressor.c)
    model_snr = regressor.target_snr_value(regressor.beta_0, regressor.c)

    assert np.isclose(model_mean, 0.9, atol=0.15)
    assert np.isclose(model_snr, 0.2, atol=0.15)

    samples = regressor.sample(2500)
    assert np.all(samples >= 0)
    assert np.all(samples <= 8)
    assert np.allclose(samples, np.round(samples))


def test_geometric_regressor_higher_mean_scenario() -> None:
    X = build_x_matrix(n=220, p=5, seed=47)
    beta_1_init = np.array([0.4, -0.35, 0.22, 0.14, -0.18], dtype=float)

    regressor = GeometricRegressor(
        target_mean=4.5,
        target_snr=0.75,
        X=X,
        beta_1_init=beta_1_init,
        min=0,
        max=16,
    ).calibrate()

    model_mean = regressor.target_mean_value(regressor.beta_0, regressor.c)
    model_snr = regressor.target_snr_value(regressor.beta_0, regressor.c)

    assert np.isclose(model_mean, 4.5, atol=0.25)
    assert np.isclose(model_snr, 0.75, atol=0.25)

    samples = regressor.sample(3500)
    assert np.all(samples >= 0)
    assert np.all(samples <= 16)
    assert np.allclose(samples, np.round(samples))
