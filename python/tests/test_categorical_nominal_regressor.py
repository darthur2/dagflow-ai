import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from regressors import CategoricalNominalRegressor

from tests.regressor_test_helpers import build_x_matrix


def test_categorical_nominal_regressor_calibration_and_sampling() -> None:
    X = build_x_matrix(n=240, p=3, seed=91)
    beta_1 = np.array(
        [
            [0.8, -0.4],
            [0.3, 0.2],
            [-0.25, 0.15],
        ],
        dtype=float,
    )
    target_probabilities = np.array([0.22, 0.31, 0.47], dtype=float)

    regressor = CategoricalNominalRegressor(
        target_probabilities=target_probabilities,
        X=X,
        beta_1=beta_1,
    ).calibrate()

    model_probabilities = regressor.target_probabilities_value(regressor.beta_0)

    assert np.allclose(model_probabilities, target_probabilities, atol=0.04)

    samples = regressor.sample(5000)
    assert np.issubdtype(samples.dtype, np.integer)
    assert np.all(samples >= 0)
    assert np.all(samples <= 2)

    sample_probabilities = np.bincount(samples, minlength=3) / len(samples)
    assert np.allclose(sample_probabilities, target_probabilities, atol=0.05)


def test_categorical_nominal_regressor_different_target_mix() -> None:
    X = build_x_matrix(n=180, p=4, seed=93)
    beta_1 = np.array(
        [
            [0.5, -0.3, 0.2],
            [0.15, 0.25, -0.1],
            [-0.2, 0.18, 0.12],
            [0.1, -0.05, 0.08],
        ],
        dtype=float,
    )
    target_probabilities = np.array([0.14, 0.27, 0.19, 0.40], dtype=float)

    regressor = CategoricalNominalRegressor(
        target_probabilities=target_probabilities,
        X=X,
        beta_1=beta_1,
    ).calibrate()

    model_probabilities = regressor.target_probabilities_value(regressor.beta_0)

    assert np.allclose(model_probabilities, target_probabilities, atol=0.05)

    samples = regressor.sample(4000)
    assert np.issubdtype(samples.dtype, np.integer)
    assert np.all(samples >= 0)
    assert np.all(samples <= 3)
