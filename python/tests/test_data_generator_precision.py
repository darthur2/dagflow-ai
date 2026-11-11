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


def _continuous_variable(decimal_places: int) -> dict:
    return {
        "data_type": "Quantitative",
        "description": "description",
        "justification": "justification",
        "measurement_level": "Ratio",
        "classification": "Continuous",
        "skew": "None",
        "decimal_places": decimal_places,
    }


def _discrete_variable() -> dict:
    return {
        "data_type": "Quantitative",
        "description": "description",
        "justification": "justification",
        "measurement_level": "Ratio",
        "classification": "Discrete",
        "skew": "Right",
    }


def test_continuous_variable_is_rounded_to_decimal_places(tmp_path, monkeypatch):
    monkeypatch.setattr(data_generator, "SYNTHDATA", tmp_path)
    _write_project(
        tmp_path,
        variables={"x": _continuous_variable(2)},
        dag={"nodes": {"x": {"type": "Stochastic"}}, "edges": {}},
        distributions={
            "x": {
                "distribution": "Normal",
                "mean": 0.0,
                "standard_deviation": 1.0,
                "min": -5.0,
                "max": 5.0,
            }
        },
        formulas={},
    )

    dataframe = data_generator.generate_data(200)
    values = dataframe["x"].to_numpy()

    assert np.array_equal(values, np.round(values, 2))


def test_deterministic_continuous_variable_is_rounded(tmp_path, monkeypatch):
    monkeypatch.setattr(data_generator, "SYNTHDATA", tmp_path)
    _write_project(
        tmp_path,
        variables={"x": _continuous_variable(0), "y": _continuous_variable(6)},
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

    dataframe = data_generator.generate_data(100)

    assert np.array_equal(dataframe["x"].to_numpy(), np.round(dataframe["x"].to_numpy(), 0))
    assert np.array_equal(dataframe["y"].to_numpy(), np.round(dataframe["y"].to_numpy(), 6))
    assert np.array_equal(dataframe["y"].to_numpy(), 1.0 + 2.0 * dataframe["x"].to_numpy())


def test_discrete_variable_is_not_rounded_and_needs_no_decimal_places(tmp_path, monkeypatch):
    monkeypatch.setattr(data_generator, "SYNTHDATA", tmp_path)
    _write_project(
        tmp_path,
        variables={"count": _discrete_variable()},
        dag={"nodes": {"count": {"type": "Stochastic"}}, "edges": {}},
        distributions={
            "count": {
                "distribution": "Poisson",
                "rate": 3.0,
                "min": 0,
                "max": 30,
                "truncated": False,
            }
        },
        formulas={},
    )

    dataframe = data_generator.generate_data(100)
    values = dataframe["count"].to_numpy()

    assert np.array_equal(values, np.round(values))


def test_continuous_variable_without_decimal_places_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(data_generator, "SYNTHDATA", tmp_path)
    variables = {"x": _continuous_variable(2)}
    del variables["x"]["decimal_places"]
    _write_project(
        tmp_path,
        variables=variables,
        dag={"nodes": {"x": {"type": "Stochastic"}}, "edges": {}},
        distributions={
            "x": {
                "distribution": "Normal",
                "mean": 0.0,
                "standard_deviation": 1.0,
                "min": -5.0,
                "max": 5.0,
            }
        },
        formulas={},
    )

    with pytest.raises(ValueError, match="decimal_places"):
        data_generator.generate_data(20)
