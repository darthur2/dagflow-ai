import numpy as np

from utils import get_beta_0, get_categories, make_beta_1


def test_quantitative_formula_parsing() -> None:
    formulas = {
        "y": {
            "type": "quantitative",
            "intercept": 1.5,
            "snr": 2.0,
            "predictors": {
                "x1": {"coefficient": 0.3, "transformation": "none"},
                "x2": {"coefficient": -0.7, "transformation": "sqrt"},
            },
        }
    }

    beta_1 = make_beta_1(formulas, "y")
    beta_0 = get_beta_0(formulas, "y")

    assert beta_1.shape == (2,)
    assert np.allclose(beta_1, np.array([0.3, -0.7], dtype=float))
    assert isinstance(beta_0, float)
    assert beta_0 == 1.5


def test_categorical_nominal_formula_parsing() -> None:
    formulas = {
        "y": {
            "type": "categorical_nominal",
            "reference_category": "A",
            "category_models": {
                "B": {
                    "intercept": 0.1,
                    "predictors": {
                        "x1": {"coefficient": 0.5, "transformation": "none"},
                        "x2": {"coefficient": -0.2, "transformation": "log"},
                    },
                },
                "C": {
                    "intercept": -0.4,
                    "predictors": {
                        "x1": {"coefficient": 0.2, "transformation": "none"},
                        "x2": {"coefficient": 0.8, "transformation": "log"},
                    },
                },
            },
        }
    }

    beta_1 = make_beta_1(formulas, "y")
    beta_0 = get_beta_0(formulas, "y")
    categories = get_categories(formulas, "y")

    assert beta_1.shape == (2, 2)
    assert np.allclose(beta_1, np.array([[0.5, 0.2], [-0.2, 0.8]], dtype=float))
    assert beta_0.shape == (2,)
    assert np.allclose(beta_0, np.array([0.1, -0.4], dtype=float))
    assert categories == ["A", "B", "C"]


def test_categorical_ordinal_formula_parsing() -> None:
    formulas = {
        "y": {
            "type": "categorical_ordinal",
            "reference_category": "low",
            "predictors": {
                "x1": {"coefficient": 0.4, "transformation": "none"},
                "x2": {"coefficient": -0.1, "transformation": "sqrt"},
            },
            "thresholds": {
                "medium": {"intercept": -0.2},
                "high": {"intercept": 0.9},
            },
        }
    }

    beta_1 = make_beta_1(formulas, "y")
    beta_0 = get_beta_0(formulas, "y")
    categories = get_categories(formulas, "y")

    assert beta_1.shape == (2,)
    assert np.allclose(beta_1, np.array([0.4, -0.1], dtype=float))
    assert beta_0.shape == (2,)
    assert np.allclose(beta_0, np.array([-0.2, 0.9], dtype=float))
    assert categories == ["low", "medium", "high"]
