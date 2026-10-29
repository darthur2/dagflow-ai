import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import validators


def _quantitative_variable() -> dict:
    return {
        "data_type": "Quantitative",
        "description": "description",
        "justification": "justification",
        "measurement_level": "Ratio",
        "classification": "Continuous",
        "skew": "None",
    }


def _nominal_variable() -> dict:
    return {
        "data_type": "Categorical",
        "description": "description",
        "justification": "justification",
        "measurement_level": "Nominal",
    }


def _normal_distribution() -> dict:
    return {
        "distribution": "Normal",
        "mean": 0.0,
        "standard_deviation": 1.0,
        "min": -5.0,
        "max": 5.0,
    }


def _validate(variables, dag, distributions, formulas):
    return validators.validate_all_available_files(
        variables_data=variables,
        dag_data=dag,
        distributions_data=distributions,
        formulas_data=formulas,
    )


def test_deterministic_quantitative_node_passes_validation():
    variables = {"parent": _quantitative_variable(), "deterministic_var": _quantitative_variable()}
    dag = {
        "nodes": {
            "parent": {"type": "Stochastic"},
            "deterministic_var": {"type": "Deterministic"},
        },
        "edges": {"e1": {"parent": "parent", "child": "deterministic_var"}},
    }
    distributions = {"parent": _normal_distribution(), "deterministic_var": {"distribution": "None"}}
    formulas = {
        "deterministic_var": {
            "type": "quantitative",
            "intercept": 1.0,
            "predictors": {"parent": {"coefficient": 2.0, "transformation": "none"}},
        }
    }

    report = _validate(variables, dag, distributions, formulas)

    assert report.ok, report.errors


def test_deterministic_nominal_node_passes_validation():
    variables = {"group": _nominal_variable(), "deterministic_var": _nominal_variable()}
    dag = {
        "nodes": {
            "group": {"type": "Stochastic"},
            "deterministic_var": {"type": "Deterministic"},
        },
        "edges": {"e1": {"parent": "group", "child": "deterministic_var"}},
    }
    distributions = {
        "group": {
            "distribution": "Categorical Nominal",
            "categories": ["A", "B"],
            "probabilities": [0.5, 0.5],
        },
        "deterministic_var": {"distribution": "None"},
    }
    formulas = {
        "deterministic_var": {
            "type": "categorical_nominal",
            "reference_category": "A",
            "category_models": {
                "B": {
                    "intercept": 1.0,
                    "predictors": {
                        "group": {
                            "reference_category": "A",
                            "other_categories": {"B": {"coefficient": 1.0}},
                        }
                    },
                }
            },
        }
    }

    report = _validate(variables, dag, distributions, formulas)

    assert report.ok, report.errors


def test_deterministic_node_with_non_none_distribution_fails_validation():
    variables = {"parent": _quantitative_variable(), "deterministic_var": _quantitative_variable()}
    dag = {
        "nodes": {
            "parent": {"type": "Stochastic"},
            "deterministic_var": {"type": "Deterministic"},
        },
        "edges": {"e1": {"parent": "parent", "child": "deterministic_var"}},
    }
    distributions = {"parent": _normal_distribution(), "deterministic_var": _normal_distribution()}
    formulas = {
        "deterministic_var": {
            "type": "quantitative",
            "intercept": 1.0,
            "predictors": {"parent": {"coefficient": 2.0, "transformation": "none"}},
        }
    }

    report = _validate(variables, dag, distributions, formulas)

    assert not report.ok
    details = report.errors[0].details
    assert details["deterministic_nodes_with_non_none_distribution"] == ["deterministic_var"]


def test_stochastic_node_with_none_distribution_fails_validation():
    variables = {"stochastic_var": _quantitative_variable()}
    dag = {"nodes": {"stochastic_var": {"type": "Stochastic"}}, "edges": {}}
    distributions = {"stochastic_var": {"distribution": "None"}}

    report = _validate(variables, dag, distributions, {})

    assert not report.ok
    details = report.errors[0].details
    assert details["stochastic_nodes_with_none_distribution"] == ["stochastic_var"]


def test_deterministic_node_without_parents_fails_validation():
    variables = {"deterministic_var": _quantitative_variable()}
    dag = {"nodes": {"deterministic_var": {"type": "Deterministic"}}, "edges": {}}
    distributions = {"deterministic_var": {"distribution": "None"}}

    report = _validate(variables, dag, distributions, {})

    assert not report.ok
    details = report.errors[0].details
    assert details["deterministic_nodes_without_parents"] == ["deterministic_var"]


def test_deterministic_quantitative_node_accepts_response_transformation():
    variables = {"parent": _quantitative_variable(), "deterministic_var": _quantitative_variable()}
    dag = {
        "nodes": {
            "parent": {"type": "Stochastic"},
            "deterministic_var": {"type": "Deterministic"},
        },
        "edges": {"e1": {"parent": "parent", "child": "deterministic_var"}},
    }
    distributions = {"parent": _normal_distribution(), "deterministic_var": {"distribution": "None"}}
    formulas = {
        "deterministic_var": {
            "type": "quantitative",
            "intercept": 1.0,
            "response_transformation": "log1p",
            "predictors": {"parent": {"coefficient": 2.0, "transformation": "log1p"}},
        }
    }

    report = _validate(variables, dag, distributions, formulas)

    assert report.ok, report.errors


def test_stochastic_node_rejects_response_transformation():
    variables = {"parent": _quantitative_variable(), "stochastic_var": _quantitative_variable()}
    dag = {
        "nodes": {
            "parent": {"type": "Stochastic"},
            "stochastic_var": {"type": "Stochastic"},
        },
        "edges": {"e1": {"parent": "parent", "child": "stochastic_var"}},
    }
    distributions = {"parent": _normal_distribution(), "stochastic_var": _normal_distribution()}
    formulas = {
        "stochastic_var": {
            "type": "quantitative",
            "intercept": 1.0,
            "response_transformation": "log",
            "predictors": {"parent": {"coefficient": 2.0, "transformation": "none"}},
        }
    }

    report = _validate(variables, dag, distributions, formulas)

    assert not report.ok
    error = next(error for error in report.errors if error.code == "invalid_response_transformation_context")
    assert error.details["offending_formulas"]["stochastic_var"]["node_type"] == "Stochastic"


def test_stochastic_node_accepts_explicit_none_response_transformation():
    variables = {"parent": _quantitative_variable(), "stochastic_var": _quantitative_variable()}
    dag = {
        "nodes": {
            "parent": {"type": "Stochastic"},
            "stochastic_var": {"type": "Stochastic"},
        },
        "edges": {"e1": {"parent": "parent", "child": "stochastic_var"}},
    }
    distributions = {"parent": _normal_distribution(), "stochastic_var": _normal_distribution()}
    formulas = {
        "stochastic_var": {
            "type": "quantitative",
            "intercept": 1.0,
            "response_transformation": "none",
            "predictors": {"parent": {"coefficient": 2.0, "transformation": "none"}},
        }
    }

    report = _validate(variables, dag, distributions, formulas)

    assert report.ok, report.errors


def test_categorical_formula_rejects_response_transformation_field():
    variables = {"group": _nominal_variable(), "deterministic_var": _nominal_variable()}
    dag = {
        "nodes": {
            "group": {"type": "Stochastic"},
            "deterministic_var": {"type": "Deterministic"},
        },
        "edges": {"e1": {"parent": "group", "child": "deterministic_var"}},
    }
    distributions = {
        "group": {
            "distribution": "Categorical Nominal",
            "categories": ["A", "B"],
            "probabilities": [0.5, 0.5],
        },
        "deterministic_var": {"distribution": "None"},
    }
    formulas = {
        "deterministic_var": {
            "type": "categorical_nominal",
            "reference_category": "A",
            "response_transformation": "log",
            "category_models": {
                "B": {
                    "intercept": 1.0,
                    "predictors": {
                        "group": {
                            "reference_category": "A",
                            "other_categories": {"B": {"coefficient": 1.0}},
                        }
                    },
                }
            },
        }
    }

    report = _validate(variables, dag, distributions, formulas)

    assert not report.ok
    assert any(
        error.code == "extra_forbidden" and "response_transformation" in error.message
        for error in report.errors
    )


def test_unsupported_response_transformation_fails_validation():
    variables = {"parent": _quantitative_variable(), "deterministic_var": _quantitative_variable()}
    dag = {
        "nodes": {
            "parent": {"type": "Stochastic"},
            "deterministic_var": {"type": "Deterministic"},
        },
        "edges": {"e1": {"parent": "parent", "child": "deterministic_var"}},
    }
    distributions = {"parent": _normal_distribution(), "deterministic_var": {"distribution": "None"}}
    formulas = {
        "deterministic_var": {
            "type": "quantitative",
            "intercept": 1.0,
            "response_transformation": "square",
            "predictors": {"parent": {"coefficient": 2.0, "transformation": "none"}},
        }
    }

    report = _validate(variables, dag, distributions, formulas)

    assert not report.ok
    assert any(error.code == "literal_error" for error in report.errors)
