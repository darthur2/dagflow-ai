import json
from pathlib import Path
from typing import Any

import numpy as np

from distributions import (
    Beta,
    Bernoulli,
    Binomial,
    CategoricalNominal,
    CategoricalOrdinal,
    DiscreteUniform,
    Exponential,
    Gamma,
    Geometric,
    LogNormal,
    NegativeBinomial,
    NoneDistribution,
    Normal,
    Poisson,
    Uniform,
)


def load_json(path: Path):
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def make_distribution(distributions_data: dict, variable_name: str):
    if variable_name not in distributions_data:
        raise ValueError(f"Unknown variable: {variable_name}")

    item = distributions_data[variable_name]
    distribution_name = item.get("distribution")
    parameters = {key: value for key, value in item.items() if key != "distribution"}

    if distribution_name == "Normal":
        return Normal(**parameters)
    if distribution_name == "Exponential":
        return Exponential(**parameters)
    if distribution_name == "Gamma":
        return Gamma(**parameters)
    if distribution_name == "Log Normal":
        return LogNormal(**parameters)
    if distribution_name == "Beta":
        return Beta(**parameters)
    if distribution_name == "Uniform":
        return Uniform(**parameters)
    if distribution_name == "Discrete Uniform":
        return DiscreteUniform(**parameters)
    if distribution_name == "Bernoulli":
        return Bernoulli(**parameters)
    if distribution_name == "Binomial":
        return Binomial(**parameters)
    if distribution_name == "Poisson":
        return Poisson(**parameters)
    if distribution_name == "Geometric":
        return Geometric(**parameters)
    if distribution_name == "Negative Binomial":
        return NegativeBinomial(**parameters)
    if distribution_name == "Categorical Nominal":
        return CategoricalNominal(**parameters)
    if distribution_name == "Categorical Ordinal":
        return CategoricalOrdinal(**parameters)
    if distribution_name == "None":
        return NoneDistribution()

    raise ValueError(f"Unsupported distribution: {distribution_name}")


def get_snr(formulas_data: dict, variable_name: str):
    if variable_name not in formulas_data:
        raise ValueError(f"Unknown variable: {variable_name}")

    return formulas_data[variable_name].get("snr")


def _predictor_coefficients(predictors: dict[str, Any]) -> list[float]:
    coefficients: list[float] = []
    for predictor in predictors.values():
        if "coefficient" in predictor:
            coefficients.append(float(predictor["coefficient"]))
            continue

        if "reference_category" in predictor and "other_categories" in predictor:
            for category in predictor["other_categories"].values():
                coefficients.append(float(category["coefficient"]))
            continue

        raise ValueError("Unsupported predictor schema")

    return coefficients


def _predictor_names(predictors: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for predictor_name, predictor in predictors.items():
        if "coefficient" in predictor:
            names.append(predictor_name)
            continue

        if "reference_category" in predictor and "other_categories" in predictor:
            for _ in predictor["other_categories"].values():
                names.append(predictor_name)
            continue

        raise ValueError("Unsupported predictor schema")

    return names


def _predictor_transformations(predictors: dict[str, Any]) -> dict[str, str]:
    transformations: dict[str, str] = {}
    for predictor_name, predictor in predictors.items():
        if "coefficient" in predictor:
            transformations[predictor_name] = str(predictor.get("transformation", "none"))
            continue

        if "reference_category" in predictor and "other_categories" in predictor:
            transformations[predictor_name] = str(predictor.get("transformation", "none"))
            continue

        raise ValueError("Unsupported predictor schema")

    return transformations


def _flatten_predictor_coefficients_and_names(
    predictors: dict[str, Any],
) -> tuple[list[float], list[str], dict[str, str]]:
    coefficients: list[float] = []
    names: list[str] = []
    transformations: dict[str, str] = {}

    for predictor_name, predictor in predictors.items():
        if "coefficient" in predictor:
            coefficients.append(float(predictor["coefficient"]))
            names.append(predictor_name)
            transformations[predictor_name] = str(predictor.get("transformation", "none"))
            continue

        if "reference_category" in predictor and "other_categories" in predictor:
            category_names = list(predictor["other_categories"].keys())
            for category_name in category_names:
                category_data = predictor["other_categories"][category_name]
                coefficients.append(float(category_data["coefficient"]))
                names.append(predictor_name)
            transformations[predictor_name] = str(predictor.get("transformation", "none"))
            continue

        raise ValueError("Unsupported predictor schema")

    return coefficients, names, transformations


def make_beta_1(formulas_data: dict, variable_name: str):
    if variable_name not in formulas_data:
        raise ValueError(f"Unknown variable: {variable_name}")

    formula = formulas_data[variable_name]

    if "intercept" in formula and "snr" in formula and "predictors" in formula:
        return np.asarray(_predictor_coefficients(formula["predictors"]), dtype=float)

    if "reference_category" in formula and "other_categories" in formula and "predictors" in formula:
        coefficients, _, _ = _flatten_predictor_coefficients_and_names(formula["predictors"])
        return np.asarray(coefficients, dtype=float)

    if "reference_category" in formula and "other_categories" in formula:
        category_vectors = []
        for category_block in formula["other_categories"].values():
            if not isinstance(category_block, dict):
                raise ValueError(f"Unsupported formula schema for variable: {variable_name}")
            category_predictors = category_block.get("predictors", {})
            if not isinstance(category_predictors, dict):
                raise ValueError(f"Unsupported formula schema for variable: {variable_name}")
            coefficients, _, _ = _flatten_predictor_coefficients_and_names(category_predictors)
            category_vectors.append(coefficients)

        if not category_vectors:
            return np.asarray([], dtype=float)

        return np.asarray(category_vectors, dtype=float)

    raise ValueError(f"Unsupported formula schema for variable: {variable_name}")


def get_predictor_names(formulas_data: dict, variable_name: str) -> list[str]:
    if variable_name not in formulas_data:
        raise ValueError(f"Unknown variable: {variable_name}")

    formula = formulas_data[variable_name]

    if "intercept" in formula and "snr" in formula and "predictors" in formula:
        return _predictor_names(formula["predictors"])

    if "reference_category" in formula and "other_categories" in formula and "predictors" in formula:
        _, names, _ = _flatten_predictor_coefficients_and_names(formula["predictors"])
        return names

    if "reference_category" in formula and "other_categories" in formula:
        names: list[str] = []
        for category_block in formula["other_categories"].values():
            names.extend(_predictor_names(category_block.get("predictors", {})))
        return names

    raise ValueError(f"Unsupported formula schema for variable: {variable_name}")


def get_predictor_transformations(formulas_data: dict, variable_name: str) -> dict[str, str]:
    if variable_name not in formulas_data:
        raise ValueError(f"Unknown variable: {variable_name}")

    formula = formulas_data[variable_name]

    if "intercept" in formula and "snr" in formula and "predictors" in formula:
        return _predictor_transformations(formula["predictors"])

    if "reference_category" in formula and "other_categories" in formula and "predictors" in formula:
        _, _, transformations = _flatten_predictor_coefficients_and_names(formula["predictors"])
        return transformations

    if "reference_category" in formula and "other_categories" in formula:
        transformations: dict[str, str] = {}
        for category_block in formula["other_categories"].values():
            transformations.update(_predictor_transformations(category_block.get("predictors", {})))
        return transformations

    raise ValueError(f"Unsupported formula schema for variable: {variable_name}")


def get_beta_0(formulas_data: dict, variable_name: str):
    if variable_name not in formulas_data:
        raise ValueError(f"Unknown variable: {variable_name}")

    formula = formulas_data[variable_name]

    if "intercept" in formula and "snr" in formula and "predictors" in formula:
        return float(formula["intercept"])

    if "reference_category" in formula and "other_categories" in formula:
        intercepts = [float(category_block["intercept"]) for category_block in formula["other_categories"].values()]
        return np.asarray(intercepts, dtype=float)

    raise ValueError(f"Unsupported formula schema for variable: {variable_name}")


def get_categories(formulas_data: dict, variable_name: str) -> list[str]:
    if variable_name not in formulas_data:
        raise ValueError(f"Unknown variable: {variable_name}")

    formula = formulas_data[variable_name]

    if "reference_category" in formula and "other_categories" in formula:
        reference_category = formula["reference_category"]
        if not isinstance(reference_category, str):
            raise ValueError(f"Invalid reference category for variable: {variable_name}")
        return [reference_category, *formula["other_categories"].keys()]

    raise ValueError(f"Unsupported formula schema for variable: {variable_name}")


def get_dag_order(dag_data: dict) -> list[str]:
    nodes = dag_data.get("nodes", {})
    edges = dag_data.get("edges", {})

    if not isinstance(nodes, dict) or not isinstance(edges, dict):
        raise ValueError("Invalid DAG schema")

    node_order = list(nodes.keys())
    parents: dict[str, set[str]] = {node_name: set() for node_name in node_order}
    children: dict[str, set[str]] = {node_name: set() for node_name in node_order}

    for edge_data in edges.values():
        parent = edge_data.get("parent")
        child = edge_data.get("child")
        if parent not in parents or child not in parents:
            raise ValueError("DAG edge references an unknown node")
        parents[child].add(parent)
        children[parent].add(child)

    available = [node_name for node_name in node_order if not parents[node_name]]
    visited: set[str] = set()
    ordered: list[str] = []

    while available:
        current = available.pop(0)
        if current in visited:
            continue

        visited.add(current)
        ordered.append(current)

        for child in node_order:
            if child in visited:
                continue
            if child in children[current] and parents[child].issubset(visited) and child not in available:
                available.append(child)

    if len(ordered) != len(node_order):
        raise ValueError("DAG contains a cycle or disconnected invalid structure")

    return ordered
