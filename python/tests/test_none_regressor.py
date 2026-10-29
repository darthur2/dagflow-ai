import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from regressors import NoneRegressor


def test_none_regressor_applies_predictor_transformations():
    X = np.array([[1.0], [4.0], [9.0]])
    regressor = NoneRegressor(
        X=X,
        beta_0=1.0,
        beta_1=np.array([2.0]),
        predictor_names=["x"],
        predictor_transformations={"x": "sqrt"},
    )

    sample = regressor.calibrate().sample(3)

    assert np.allclose(sample, 1.0 + 2.0 * np.sqrt(X.reshape(-1)))


def test_none_regressor_quantitative_returns_exact_linear_predictor():
    X = np.array([[0.0], [1.0], [2.0]])
    regressor = NoneRegressor(
        X=X,
        beta_0=3.0,
        beta_1=np.array([-0.5]),
        predictor_names=["x"],
    )

    sample = regressor.sample(3)

    assert np.allclose(sample, 3.0 - 0.5 * X.reshape(-1))


def test_none_regressor_nominal_uses_reference_baseline_and_argmax():
    X = np.array([[0.0], [1.0], [2.0]])
    regressor = NoneRegressor(
        X=X,
        beta_0=np.array([1.0, 0.0]),
        beta_1=np.array([[4.0, 5.0]]),
        predictor_names=["x"],
        response_type="categorical_nominal",
        categories=["reference", "middle", "highest"],
    )

    sample = regressor.sample(3)

    assert sample.tolist() == ["middle", "middle", "highest"]


def test_none_regressor_ordinal_uses_thresholds():
    X = np.array([[-1.0], [0.5], [2.0]])
    regressor = NoneRegressor(
        X=X,
        beta_0=np.array([0.0, 1.0]),
        beta_1=np.array([1.0]),
        predictor_names=["x"],
        response_type="categorical_ordinal",
        categories=["low", "middle", "high"],
    )

    sample = regressor.sample(3)

    assert sample.tolist() == ["low", "middle", "high"]


def test_none_regressor_rejects_non_increasing_ordinal_thresholds():
    regressor = NoneRegressor(
        X=np.array([[0.0], [1.0]]),
        beta_0=np.array([1.0, 0.0]),
        beta_1=np.array([1.0]),
        predictor_names=["x"],
        response_type="categorical_ordinal",
        categories=["low", "middle", "high"],
    )

    try:
        regressor.sample(2)
    except ValueError as exc:
        assert "strictly increasing" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-increasing thresholds")


def test_none_regressor_log1p_predictor_accepts_zero():
    X = np.array([[0.0], [1.0], [3.0]])
    regressor = NoneRegressor(
        X=X,
        beta_0=2.0,
        beta_1=np.array([0.5]),
        predictor_names=["x"],
        predictor_transformations={"x": "log1p"},
    )

    sample = regressor.sample(3)

    assert np.allclose(sample, 2.0 + 0.5 * np.log1p(X.reshape(-1)))


def test_none_regressor_log_predictor_rejects_zero():
    with pytest.raises(ValueError, match="log transformation requires positive values"):
        NoneRegressor(
            X=np.array([[0.0], [1.0]]),
            beta_0=0.0,
            beta_1=np.array([1.0]),
            predictor_names=["x"],
            predictor_transformations={"x": "log"},
        )


def test_none_regressor_log1p_predictor_rejects_values_at_or_below_minus_one():
    with pytest.raises(ValueError, match="log1p transformation requires values greater than -1"):
        NoneRegressor(
            X=np.array([[-1.0], [-0.5]]),
            beta_0=0.0,
            beta_1=np.array([1.0]),
            predictor_names=["x"],
            predictor_transformations={"x": "log1p"},
        )


def test_none_regressor_applies_log_response_transformation():
    X = np.array([[0.0], [1.0], [2.0]])
    regressor = NoneRegressor(
        X=X,
        beta_0=0.5,
        beta_1=np.array([1.0]),
        response_transformation="log",
    )

    sample = regressor.sample(3)

    assert np.allclose(sample, np.exp(0.5 + X.reshape(-1)))


def test_none_regressor_applies_log1p_response_transformation_with_zero_output():
    X = np.array([[0.0], [1.0], [2.0]])
    regressor = NoneRegressor(
        X=X,
        beta_0=0.0,
        beta_1=np.array([1.0]),
        response_transformation="log1p",
    )

    sample = regressor.sample(3)

    assert np.allclose(sample, np.expm1(X.reshape(-1)))
    assert sample[0] == 0.0


def test_none_regressor_combines_log_predictor_and_log1p_response():
    X = np.array([[1.0], [2.0], [4.0]])
    regressor = NoneRegressor(
        X=X,
        beta_0=0.0,
        beta_1=np.array([1.0]),
        predictor_names=["x"],
        predictor_transformations={"x": "log"},
        response_transformation="log1p",
    )

    sample = regressor.sample(3)

    assert np.allclose(sample, X.reshape(-1) - 1.0)


def test_none_regressor_applies_cubic_response_transformation_to_signed_values():
    X = np.array([[-8.0], [0.0], [8.0]])
    regressor = NoneRegressor(
        X=X,
        beta_0=0.0,
        beta_1=np.array([1.0]),
        response_transformation="cubic",
    )

    sample = regressor.sample(3)

    assert np.allclose(sample, [-2.0, 0.0, 2.0])


@pytest.mark.parametrize(
    ("transformation", "values", "message"),
    [
        ("exp", np.array([[0.0]]), "requires the linear predictor to be positive"),
        ("sqrt", np.array([[-1.0]]), "requires a non-negative linear predictor"),
        ("inverse", np.array([[0.0]]), "requires a non-zero linear predictor"),
    ],
)
def test_none_regressor_rejects_invalid_response_inverse_domains(transformation, values, message):
    regressor = NoneRegressor(
        X=values,
        beta_0=0.0,
        beta_1=np.array([1.0]),
        response_transformation=transformation,
    )

    with pytest.raises(ValueError, match=message):
        regressor.sample(1)


def test_none_regressor_rejects_response_transformation_for_categorical_response():
    with pytest.raises(ValueError, match="response transformations are only supported for quantitative responses"):
        NoneRegressor(
            X=np.array([[0.0]]),
            beta_0=np.array([1.0]),
            beta_1=np.array([[1.0]]),
            response_type="categorical_nominal",
            categories=["a", "b"],
            response_transformation="log",
        )


def test_none_regressor_rejects_unknown_response_transformation():
    with pytest.raises(ValueError, match="Unsupported response_transformation"):
        NoneRegressor(
            X=np.array([[0.0]]),
            beta_0=0.0,
            beta_1=np.array([1.0]),
            response_transformation="square",
        )
