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
    Gamma,
    LogNormal,
    NegativeBinomial,
    NoneDistribution,
    Normal,
    Poisson,
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
    if distribution_name == "Gamma":
        return Gamma(**parameters)
    if distribution_name == "Log Normal":
        return LogNormal(**parameters)
    if distribution_name == "Beta":
        return Beta(**parameters)
    if distribution_name == "Bernoulli":
        return Bernoulli(**parameters)
    if distribution_name == "Binomial":
        return Binomial(**parameters)
    if distribution_name == "Poisson":
        return Poisson(**parameters)
    if distribution_name == "Negative Binomial":
        return NegativeBinomial(**parameters)
    if distribution_name == "Categorical Nominal":
        return CategoricalNominal(**parameters)
    if distribution_name == "Categorical Ordinal":
        return CategoricalOrdinal(**parameters)
    if distribution_name == "None":
        return NoneDistribution()

    raise ValueError(f"Unsupported distribution: {distribution_name}")


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


def resolve_predictor_transformations(predictors: dict[str, Any]) -> dict[str, str]:
    """Resolve the declared transformation for each predictor.

    A malformed predictor block raises rather than silently defaulting to "none",
    so a typo in a formula surfaces instead of quietly producing unscaled data.
    """
    transformations: dict[str, str] = {}
    for predictor_name, predictor in predictors.items():
        if not isinstance(predictor, dict):
            raise ValueError(f"Unsupported predictor schema for predictor: {predictor_name}")

        if "coefficient" in predictor or ("reference_category" in predictor and "other_categories" in predictor):
            transformations[predictor_name] = str(predictor.get("transformation", "none"))
            continue

        raise ValueError(f"Unsupported predictor schema for predictor: {predictor_name}")

    return transformations


def _predictor_dimension(predictor: dict[str, Any]) -> int:
    if "coefficient" in predictor:
        return 1

    if "reference_category" in predictor and "other_categories" in predictor:
        other_categories = predictor.get("other_categories", {})
        if not isinstance(other_categories, dict):
            raise ValueError("Unsupported predictor schema")
        return len(other_categories)

    raise ValueError("Unsupported predictor schema")


def _align_coefficients_to_predictor_names(
    coefficients: list[float],
    coefficient_names: list[str],
    target_names: list[str],
    context: str,
) -> list[float]:
    """Reorder declared coefficients into design-matrix column order.

    The design matrix is built by walking a node's DAG parents, while coefficients
    are declared in formulas.json key order. Matching on predictor name rather
    than on position keeps every coefficient attached to the variable it was
    written for. Duplicates (one entry per non-reference category of a categorical
    predictor) are consumed in declaration order.
    """
    if len(coefficients) != len(coefficient_names):
        raise ValueError(
            f"{context}: internal error, {len(coefficients)} coefficients declared for "
            f"{len(coefficient_names)} predictor names"
        )

    available: dict[str, list[int]] = {}
    for index, coefficient_name in enumerate(coefficient_names):
        available.setdefault(coefficient_name, []).append(index)

    aligned: list[float] = []
    for target_name in target_names:
        candidates = available.get(target_name)
        if not candidates:
            raise ValueError(
                f"{context}: design matrix column '{target_name}' has no matching formula predictor; "
                f"declared predictors were {coefficient_names}"
            )
        aligned.append(coefficients[candidates.pop(0)])

    unused = sorted(name for name, indexes in available.items() if indexes)
    if unused:
        raise ValueError(
            f"{context}: formula predictors {unused} are not columns of this node's design matrix, "
            f"which was built from {target_names}"
        )

    return aligned


def collect_formula_predictors(formulas_data: dict, variable_name: str) -> dict[str, Any]:
    """Return the predictor mapping that describes a variable's design matrix.

    For categorical_nominal formulas the predictors live inside each category
    model rather than at the top level. Every model must declare the same
    predictors, so the first one is representative.
    """
    formula = formulas_data.get(variable_name)
    if not isinstance(formula, dict):
        return {}

    if formula.get("type") == "categorical_nominal":
        category_models = formula.get("category_models")
        if isinstance(category_models, dict):
            for category_block in category_models.values():
                if isinstance(category_block, dict):
                    predictors = category_block.get("predictors")
                    if isinstance(predictors, dict):
                        return predictors
        return {}

    predictors = formula.get("predictors")
    return predictors if isinstance(predictors, dict) else {}


def make_beta_1(formulas_data: dict, variable_name: str, predictor_names: list[str]) -> np.ndarray:
    """Build beta_1 with one coefficient per design-matrix column.

    `predictor_names` is the per-column provenance returned by the design-matrix
    builder, so coefficients are attached to columns by predictor name instead of
    by declaration position.
    """
    if variable_name not in formulas_data:
        raise ValueError(f"Unknown variable: {variable_name}")

    formula = formulas_data[variable_name]
    formula_type = formula.get("type")
    context = f"variable '{variable_name}'"

    if formula_type in {"quantitative", "categorical_ordinal"}:
        predictors = formula.get("predictors")
        if not isinstance(predictors, dict):
            raise ValueError(f"Unsupported formula schema for {context}")
        aligned = _align_coefficients_to_predictor_names(
            _predictor_coefficients(predictors),
            _predictor_names(predictors),
            list(predictor_names),
            context,
        )
        return np.asarray(aligned, dtype=float)

    if formula_type == "categorical_nominal":
        category_models = formula.get("category_models", {})
        if not isinstance(category_models, dict):
            raise ValueError(f"Unsupported formula schema for {context}")

        rows: list[list[float]] = []
        for category_name, category_block in category_models.items():
            if not isinstance(category_block, dict):
                raise ValueError(f"Unsupported formula schema for {context}")
            predictors = category_block.get("predictors")
            if not isinstance(predictors, dict):
                raise ValueError(f"Unsupported formula schema for {context}")

            declared_dimensions = sum(_predictor_dimension(predictor) for predictor in predictors.values())
            if declared_dimensions != len(predictor_names):
                raise ValueError(
                    f"{context} category '{category_name}': predictors declare {declared_dimensions} "
                    f"columns but the design matrix has {len(predictor_names)}"
                )

            rows.append(
                _align_coefficients_to_predictor_names(
                    _predictor_coefficients(predictors),
                    _predictor_names(predictors),
                    list(predictor_names),
                    f"{context} category '{category_name}'",
                )
            )

        if not rows:
            return np.asarray([], dtype=float)

        return np.column_stack(rows)

    raise ValueError(f"Unsupported formula schema for {context}")


def get_beta_0(formulas_data: dict, variable_name: str):
    if variable_name not in formulas_data:
        raise ValueError(f"Unknown variable: {variable_name}")

    formula = formulas_data[variable_name]

    if formula.get("type") == "quantitative":
        return float(formula["intercept"])

    if formula.get("type") == "categorical_nominal":
        category_models = formula.get("category_models", {})
        if not isinstance(category_models, dict):
            raise ValueError(f"Unsupported formula schema for variable: {variable_name}")
        intercepts = [float(category_block["intercept"]) for category_block in category_models.values() if isinstance(category_block, dict)]
        return np.asarray(intercepts, dtype=float)

    if formula.get("type") == "categorical_ordinal":
        thresholds = formula.get("thresholds", {})
        if not isinstance(thresholds, dict):
            raise ValueError(f"Unsupported formula schema for variable: {variable_name}")
        return np.asarray([float(threshold["intercept"]) for threshold in thresholds.values() if isinstance(threshold, dict)], dtype=float)

    raise ValueError(f"Unsupported formula schema for variable: {variable_name}")


def get_formula_categories(formulas_data: dict, variable_name: str) -> list[str]:
    if variable_name not in formulas_data:
        raise ValueError(f"Unknown variable: {variable_name}")

    formula = formulas_data[variable_name]

    if formula.get("type") == "categorical_nominal":
        reference_category = formula.get("reference_category")
        category_models = formula.get("category_models", {})
        if not isinstance(reference_category, str) or not isinstance(category_models, dict):
            raise ValueError(f"Unsupported formula schema for variable: {variable_name}")
        return [reference_category, *category_models.keys()]

    if formula.get("type") == "categorical_ordinal":
        reference_category = formula.get("reference_category")
        thresholds = formula.get("thresholds", {})
        if not isinstance(reference_category, str) or not isinstance(thresholds, dict):
            raise ValueError(f"Unsupported formula schema for variable: {variable_name}")
        return [reference_category, *thresholds.keys()]

    raise ValueError(f"Unsupported formula schema for variable: {variable_name}")


def get_distribution_categories(distributions_data: dict, variable_name: str) -> list[str]:
    if variable_name not in distributions_data:
        raise ValueError(f"Unknown variable: {variable_name}")

    distribution = distributions_data[variable_name]
    categories = distribution.get("categories")
    if not isinstance(categories, list) or len(categories) < 2:
        raise ValueError(f"Unsupported distribution schema for variable: {variable_name}")

    return [str(category) for category in categories]


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
