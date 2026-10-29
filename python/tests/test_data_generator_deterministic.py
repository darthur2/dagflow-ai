import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import data_generator


def _write_project(path: Path, *, variables, dag, distributions, formulas) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for name, payload in {
        "variables.json": variables,
        "dag.json": dag,
        "distributions.json": distributions,
        "formulas.json": formulas,
    }.items():
        (path / name).write_text(json.dumps(payload), encoding="utf-8")


def test_generate_data_evaluates_deterministic_quantitative_formula(tmp_path, monkeypatch):
    monkeypatch.setattr(data_generator, "SYNTHDATA", tmp_path)
    _write_project(
        tmp_path,
        variables={
            "x": {
                "data_type": "Quantitative",
                "description": "x",
                "justification": "x",
                "measurement_level": "Ratio",
                "classification": "Continuous",
                "skew": "None",
            },
            "y": {
                "data_type": "Quantitative",
                "description": "y",
                "justification": "y",
                "measurement_level": "Ratio",
                "classification": "Continuous",
                "skew": "None",
            },
        },
        dag={
            "nodes": {"x": {"type": "Stochastic"}, "y": {"type": "Deterministic"}},
            "edges": {"e1": {"parent": "x", "child": "y"}},
        },
        distributions={
            "x": {
                "distribution": "Normal",
                "mean": 0.0,
                "standard_deviation": 1.0,
                "min": -5.0,
                "max": 5.0,
            },
            "y": {"distribution": "None"},
        },
        formulas={
            "y": {
                "type": "quantitative",
                "intercept": 1.0,
                "predictors": {"x": {"coefficient": 2.0, "transformation": "none"}},
            }
        },
    )

    dataframe = data_generator.generate_data(50)

    assert np.allclose(dataframe["y"], 1.0 + 2.0 * dataframe["x"])


def test_generate_data_uses_deterministic_categorical_parent_in_design_matrix(tmp_path, monkeypatch):
    monkeypatch.setattr(data_generator, "SYNTHDATA", tmp_path)
    _write_project(
        tmp_path,
        variables={
            "base": {
                "data_type": "Quantitative",
                "description": "base",
                "justification": "base",
                "measurement_level": "Ratio",
                "classification": "Continuous",
                "skew": "None",
            },
            "tier": {
                "data_type": "Categorical",
                "description": "tier",
                "justification": "tier",
                "measurement_level": "Nominal",
            },
            "score": {
                "data_type": "Quantitative",
                "description": "score",
                "justification": "score",
                "measurement_level": "Ratio",
                "classification": "Continuous",
                "skew": "None",
            },
        },
        dag={
            "nodes": {
                "base": {"type": "Stochastic"},
                "tier": {"type": "Deterministic"},
                "score": {"type": "Deterministic"},
            },
            "edges": {
                "e1": {"parent": "base", "child": "tier"},
                "e2": {"parent": "tier", "child": "score"},
            },
        },
        distributions={
            "base": {
                "distribution": "Normal",
                "mean": 0.0,
                "standard_deviation": 1.0,
                "min": -5.0,
                "max": 5.0,
            },
            "tier": {"distribution": "None"},
            "score": {"distribution": "None"},
        },
        formulas={
            "tier": {
                "type": "categorical_nominal",
                "reference_category": "low",
                "category_models": {
                    "high": {
                        "intercept": 0.0,
                        "predictors": {"base": {"coefficient": 1.0, "transformation": "none"}},
                    }
                },
            },
            "score": {
                "type": "quantitative",
                "intercept": 0.0,
                "predictors": {
                    "tier": {
                        "reference_category": "low",
                        "other_categories": {"high": {"coefficient": 10.0}},
                    }
                },
            },
        },
    )

    dataframe = data_generator.generate_data(50)

    expected_tier = np.where(dataframe["base"] > 0.0, "high", "low")
    assert dataframe["tier"].tolist() == expected_tier.tolist()
    assert np.allclose(dataframe["score"], np.where(expected_tier == "high", 10.0, 0.0))


def test_generate_data_evaluates_bmi_log_response_transformation(tmp_path, monkeypatch):
    monkeypatch.setattr(data_generator, "SYNTHDATA", tmp_path)
    _write_project(
        tmp_path,
        variables={
            "height": {
                "data_type": "Quantitative",
                "description": "height",
                "justification": "height",
                "measurement_level": "Ratio",
                "classification": "Continuous",
                "skew": "None",
            },
            "weight": {
                "data_type": "Quantitative",
                "description": "weight",
                "justification": "weight",
                "measurement_level": "Ratio",
                "classification": "Continuous",
                "skew": "Right",
            },
            "bmi": {
                "data_type": "Quantitative",
                "description": "bmi",
                "justification": "bmi",
                "measurement_level": "Ratio",
                "classification": "Continuous",
                "skew": "Right",
            },
        },
        dag={
            "nodes": {
                "height": {"type": "Stochastic"},
                "weight": {"type": "Stochastic"},
                "bmi": {"type": "Deterministic"},
            },
            "edges": {
                "height_to_bmi": {"parent": "height", "child": "bmi"},
                "weight_to_bmi": {"parent": "weight", "child": "bmi"},
            },
        },
        distributions={
            "height": {
                "distribution": "Normal",
                "mean": 69.0,
                "standard_deviation": 4.0,
                "min": 56.0,
                "max": 82.0,
            },
            "weight": {
                "distribution": "Normal",
                "mean": 180.0,
                "standard_deviation": 30.0,
                "min": 90.0,
                "max": 300.0,
            },
            "bmi": {"distribution": "None"},
        },
        formulas={
            "bmi": {
                "type": "quantitative",
                "intercept": float(np.log(703.0)),
                "response_transformation": "log",
                "predictors": {
                    "height": {"coefficient": -2.0, "transformation": "log"},
                    "weight": {"coefficient": 1.0, "transformation": "log"},
                },
            }
        },
    )

    dataframe = data_generator.generate_data(200)
    expected = 703.0 * dataframe["weight"] / dataframe["height"] ** 2

    assert np.all(np.isfinite(dataframe["bmi"]))
    assert np.allclose(dataframe["bmi"], expected)


def test_generate_data_rejects_stochastic_response_transformation(tmp_path, monkeypatch):
    monkeypatch.setattr(data_generator, "SYNTHDATA", tmp_path)
    _write_project(
        tmp_path,
        variables={
            "parent": {
                "data_type": "Quantitative",
                "description": "parent",
                "justification": "parent",
                "measurement_level": "Ratio",
                "classification": "Continuous",
                "skew": "None",
            },
            "child": {
                "data_type": "Quantitative",
                "description": "child",
                "justification": "child",
                "measurement_level": "Ratio",
                "classification": "Continuous",
                "skew": "None",
            },
        },
        dag={
            "nodes": {
                "parent": {"type": "Stochastic"},
                "child": {"type": "Stochastic"},
            },
            "edges": {"parent_to_child": {"parent": "parent", "child": "child"}},
        },
        distributions={
            "parent": {
                "distribution": "Normal",
                "mean": 0.0,
                "standard_deviation": 1.0,
                "min": -5.0,
                "max": 5.0,
            },
            "child": {
                "distribution": "Normal",
                "mean": 0.0,
                "standard_deviation": 1.0,
                "min": -5.0,
                "max": 5.0,
            },
        },
        formulas={
            "child": {
                "type": "quantitative",
                "intercept": 0.0,
                "response_transformation": "log",
                "predictors": {"parent": {"coefficient": 1.0, "transformation": "none"}},
            }
        },
    )

    with pytest.raises(ValueError, match="Stochastic variable 'child' cannot use response transformation"):
        data_generator.generate_data(20)


def test_generate_data_applies_log1p_response_transformation(tmp_path, monkeypatch):
    monkeypatch.setattr(data_generator, "SYNTHDATA", tmp_path)
    _write_project(
        tmp_path,
        variables={
            "x": {
                "data_type": "Quantitative",
                "description": "x",
                "justification": "x",
                "measurement_level": "Ratio",
                "classification": "Continuous",
                "skew": "None",
            },
            "y": {
                "data_type": "Quantitative",
                "description": "y",
                "justification": "y",
                "measurement_level": "Ratio",
                "classification": "Continuous",
                "skew": "None",
            },
        },
        dag={
            "nodes": {"x": {"type": "Stochastic"}, "y": {"type": "Deterministic"}},
            "edges": {"e1": {"parent": "x", "child": "y"}},
        },
        distributions={
            "x": {
                "distribution": "Normal",
                "mean": 3.0,
                "standard_deviation": 1.0,
                "min": 0.0,
                "max": 6.0,
            },
            "y": {"distribution": "None"},
        },
        formulas={
            "y": {
                "type": "quantitative",
                "intercept": 1.0,
                "response_transformation": "log1p",
                "predictors": {"x": {"coefficient": 2.0, "transformation": "log1p"}},
            }
        },
    )

    dataframe = data_generator.generate_data(100)

    assert np.allclose(dataframe["y"], np.expm1(1.0 + 2.0 * np.log1p(dataframe["x"])))
