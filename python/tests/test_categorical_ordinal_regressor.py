import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from regressors import CategoricalOrdinalRegressor

from tests.regressor_test_helpers import build_x_matrix


def test_categorical_ordinal_regressor_calibration_and_sampling() -> None:
    X = build_x_matrix(n=260, p=4, seed=101)
    beta_1 = np.array([0.45, -0.3, 0.2, 0.1], dtype=float)
    target_probabilities = np.array([0.18, 0.27, 0.31, 0.24], dtype=float)

    regressor = CategoricalOrdinalRegressor(
        target_probabilities=target_probabilities,
        X=X,
        beta_1=beta_1,
    ).calibrate()

    model_probabilities = regressor.target_probabilities_value(regressor.beta_0)
    assert np.allclose(model_probabilities, target_probabilities, atol=0.06)

    samples = regressor.sample(5000)
    assert np.issubdtype(samples.dtype, np.integer)
    assert np.all(samples >= 0)
    assert np.all(samples <= 3)

    sample_probabilities = np.bincount(samples, minlength=4) / len(samples)
    assert np.allclose(sample_probabilities, target_probabilities, atol=0.06)


def test_categorical_ordinal_regressor_more_skewed_mix() -> None:
    X = build_x_matrix(n=200, p=3, seed=103)
    beta_1 = np.array([-0.25, 0.35, -0.15], dtype=float)
    target_probabilities = np.array([0.08, 0.22, 0.3, 0.4], dtype=float)

    regressor = CategoricalOrdinalRegressor(
        target_probabilities=target_probabilities,
        X=X,
        beta_1=beta_1,
    ).calibrate()

    model_probabilities = regressor.target_probabilities_value(regressor.beta_0)
    assert np.allclose(model_probabilities, target_probabilities, atol=0.07)

    samples = regressor.sample(4500)
    assert np.issubdtype(samples.dtype, np.integer)
    assert np.all(samples >= 0)
    assert np.all(samples <= 3)
