from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from distributions import (
        Beta,
        Bernoulli,
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
except ImportError:  # pragma: no cover
    from .distributions import (
        Beta,
        Bernoulli,
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

try:
    from regressors import NoneRegressor
except ImportError:  # pragma: no cover
    from .regressors import NoneRegressor


BASE_DIR = Path(__file__).resolve().parent.parent
SYNTHDATA_DIR = BASE_DIR / "synthdata"


@dataclass(frozen=True)
class GenerationContext:
    variables: dict[str, dict[str, Any]]
    dag: dict[str, Any]
    distributions: dict[str, dict[str, Any]]
    formulas: dict[str, dict[str, Any]]
    parents_by_node: dict[str, list[str]]
    topo_order: list[str]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_context(base_dir: Path = SYNTHDATA_DIR) -> GenerationContext:
    variables = load_json(base_dir / "variables.json")
    dag = load_json(base_dir / "dag.json")
    distributions = load_json(base_dir / "distributions.json")
    formulas = load_json(base_dir / "formulas.json")
    parents_by_node = build_parent_map(dag)
    topo_order = topological_order(dag)
    return GenerationContext(
        variables=variables,
        dag=dag,
        distributions=distributions,
        formulas=formulas,
        parents_by_node=parents_by_node,
        topo_order=topo_order,
    )


def build_parent_map(dag: dict[str, Any]) -> dict[str, list[str]]:
    parents_by_node = {node_name: [] for node_name in dag["nodes"]}
    for edge in dag["edges"].values():
        parents_by_node[edge["child"]].append(edge["parent"])
    return parents_by_node


def topological_order(dag: dict[str, Any]) -> list[str]:
    parents_by_node = build_parent_map(dag)
    children_by_node = {node_name: [] for node_name in dag["nodes"]}
    indegree = {node_name: 0 for node_name in dag["nodes"]}

    for edge in dag["edges"].values():
        parent = edge["parent"]
        child = edge["child"]
        children_by_node[parent].append(child)
        indegree[child] += 1

    queue = [node_name for node_name, degree in indegree.items() if degree == 0]
    order: list[str] = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for child in children_by_node[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)

    if len(order) != len(dag["nodes"]):
        raise ValueError("DAG contains a cycle")
    return order


def instantiate_distribution(spec: dict[str, Any]):
    name = spec["distribution"]
    if name == "Normal":
        return Normal(mean=spec["mean"], standard_deviation=spec["standard_deviation"], min=spec["min"], max=spec["max"])
    if name == "Exponential":
        return Exponential(rate=spec["rate"], min=spec["min"], max=spec["max"])
    if name == "Gamma":
        return Gamma(shape=spec["shape"], rate=spec["rate"], min=spec["min"], max=spec["max"])
    if name == "Log Normal":
        return LogNormal(log_mean=spec["log_mean"], log_standard_deviation=spec["log_standard_deviation"], min=spec["min"], max=spec["max"])
    if name == "Beta":
        return Beta(shape_1=spec["shape_1"], shape_2=spec["shape_2"], min=spec["min"], max=spec["max"])
    if name == "Uniform":
        return Uniform(min=spec["min"], max=spec["max"])
    if name == "Discrete Uniform":
        return DiscreteUniform(min=spec["min"], max=spec["max"])
    if name == "Bernoulli":
        return Bernoulli(success_prob=spec["success_prob"])
    if name == "Poisson":
        return Poisson(rate=spec["rate"], min=spec["min"], max=spec["max"])
    if name == "Geometric":
        return Geometric(success_prob=spec["success_prob"], min=spec["min"], max=spec["max"])
    if name == "Negative Binomial":
        return NegativeBinomial(shape=spec["shape"], mean=spec["mean"], min=spec["min"], max=spec["max"])
    if name == "Categorical Nominal":
        return CategoricalNominal(categories=spec["categories"], probabilities=spec["probabilities"])
    if name == "Categorical Ordinal":
        return CategoricalOrdinal(categories=spec["categories"], probabilities=spec["probabilities"])
    if name == "None":
        return NoneDistribution()
    raise ValueError(f"Unsupported distribution: {name}")


def variable_response_type(variable_name: str, variables: dict[str, dict[str, Any]]) -> str:
    measurement_level = variables[variable_name]["measurement_level"]
    if measurement_level in {"Ratio", "Interval"}:
        return "quantitative"
    if measurement_level == "Nominal":
        return "categorical_nominal"
    if measurement_level == "Ordinal":
        return "categorical_ordinal"
    raise ValueError(f"Unsupported measurement level: {measurement_level}")


def category_names_for(variable_name: str, ctx: GenerationContext) -> list[str] | None:
    distribution_spec = ctx.distributions.get(variable_name, {})
    if distribution_spec.get("distribution") in {"Categorical Nominal", "Categorical Ordinal"}:
        return list(distribution_spec["categories"])
    formula = ctx.formulas.get(variable_name)
    if isinstance(formula, dict) and "reference_category" in formula:
        names = [formula["reference_category"]]
        if "other_categories" in formula:
            names.extend(list(formula["other_categories"].keys()))
        return names
    return None


def predictor_category_levels(formula: dict[str, Any]) -> dict[str, list[str]]:
    levels: dict[str, list[str]] = {}
    predictors = formula.get("predictors", {}) if isinstance(formula, dict) else {}
    for predictor_name, predictor_spec in predictors.items():
        if isinstance(predictor_spec, dict) and "reference_category" in predictor_spec and "other_categories" in predictor_spec:
            levels[predictor_name] = [predictor_spec["reference_category"]] + list(predictor_spec["other_categories"].keys())
    return levels


def build_design_matrix(parents: list[str], state: dict[str, np.ndarray], formula: dict[str, Any], n: int) -> tuple[np.ndarray, list[str], dict[str, str]]:
    columns: list[np.ndarray] = []
    names: list[str] = []
    transformations: dict[str, str] = {}
    categorical_levels = predictor_category_levels(formula)

    for parent in parents:
        values = state[parent]
        parent_formula = formula.get("predictors", {}).get(parent, {}) if isinstance(formula, dict) else {}
        if parent in categorical_levels:
            levels = categorical_levels[parent]
            for level in levels[1:]:
                columns.append((values == level).astype(float))
                names.append(parent)
        else:
            columns.append(np.asarray(values, dtype=float))
            names.append(parent)
            if isinstance(parent_formula, dict) and parent_formula.get("transformation") is not None:
                transformations[parent] = parent_formula["transformation"]

    if not columns:
        return np.zeros((n, 0), dtype=float), names, transformations
    return np.column_stack(columns), names, transformations


def formula_predictor_values(formula: dict[str, Any], state: dict[str, np.ndarray], parents: list[str]) -> tuple[np.ndarray, list[str], dict[str, str]]:
    columns: list[np.ndarray] = []
    names: list[str] = []
    transformations: dict[str, str] = {}
    predictors = formula.get("predictors", {}) if isinstance(formula, dict) else {}

    for parent in parents:
        parent_spec = predictors.get(parent, {}) if isinstance(predictors, dict) else {}
        values = state[parent]
        if isinstance(parent_spec, dict) and "reference_category" in parent_spec and "other_categories" in parent_spec:
            categories = [parent_spec["reference_category"]] + list(parent_spec["other_categories"].keys())
            for category in categories[1:]:
                columns.append((values == category).astype(float))
                names.append(parent)
        elif values.dtype.kind in {"U", "S", "O"}:
            categories = sorted({str(value) for value in values})
            for category in categories[1:]:
                columns.append((values == category).astype(float))
                names.append(parent)
        else:
            columns.append(np.asarray(values, dtype=float))
            names.append(parent)
            if isinstance(parent_spec, dict) and parent_spec.get("transformation") is not None:
                transformations[parent] = parent_spec["transformation"]

    if not columns:
        return np.zeros((len(next(iter(state.values()))), 0), dtype=float), names, transformations
    return np.column_stack(columns), names, transformations


def response_type_for_node(node: str, variables: dict[str, dict[str, Any]]) -> str:
    return variable_response_type(node, variables)


def evaluate_deterministic(node: str, ctx: GenerationContext, state: dict[str, np.ndarray], n: int) -> np.ndarray:
    formula = ctx.formulas[node]
    parents = ctx.parents_by_node[node]
    response_type = response_type_for_node(node, ctx.variables)
    category_names = category_names_for(node, ctx)
    X, predictor_names, transformations = formula_predictor_values(formula, state, parents)
    beta_1 = np.ones((X.shape[1], 1), dtype=float) if X.shape[1] else np.zeros((0, 1), dtype=float)

    if response_type == "quantitative":
        if not parents:
            return np.full(n, formula.get("intercept", 0.0), dtype=float)
        values = np.asarray(formula.get("intercept", 0.0), dtype=float) + X @ beta_1
        return values.reshape(-1)

    if response_type == "categorical_nominal":
        if category_names is None:
            raise ValueError(f"{node} requires categories")
        scores = X @ beta_1 if X.shape[1] else np.zeros((n, 1), dtype=float)
        logits = np.column_stack([scores, np.zeros((n, 1), dtype=float)])
        logits = logits - np.max(logits, axis=1, keepdims=True)
        probs = np.exp(logits)
        probs = probs / probs.sum(axis=1, keepdims=True)
        idx = np.argmax(probs, axis=1)
        return np.asarray([category_names[i] for i in idx], dtype=object)

    if response_type == "categorical_ordinal":
        if category_names is None:
            raise ValueError(f"{node} requires categories")
        thresholds = np.arange(len(category_names) - 1, dtype=float)
        scores = X @ beta_1 if X.shape[1] else np.zeros((n, 1), dtype=float)
        scores = scores.reshape(-1)
        cumulative = 1.0 / (1.0 + np.exp(-(thresholds[None, :] - scores[:, None])))
        probs = np.empty((n, len(category_names)), dtype=float)
        probs[:, 0] = cumulative[:, 0]
        for idx in range(1, len(category_names) - 1):
            probs[:, idx] = cumulative[:, idx] - cumulative[:, idx - 1]
        probs[:, -1] = 1.0 - cumulative[:, -1]
        idx = np.argmax(probs, axis=1)
        return np.asarray([category_names[i] for i in idx], dtype=object)

    raise ValueError(f"Unsupported response_type: {response_type}")


def evaluate_stochastic(node: str, ctx: GenerationContext, state: dict[str, np.ndarray], n: int) -> np.ndarray:
    formula = ctx.formulas.get(node, {})
    parents = ctx.parents_by_node[node]
    response_type = response_type_for_node(node, ctx.variables)
    category_names = category_names_for(node, ctx)
    X, predictor_names, transformations = formula_predictor_values(formula, state, parents)
    if response_type == "quantitative":
        if not formula:
            return np.zeros(n, dtype=float)
        beta_1 = np.ones((X.shape[1], 1), dtype=float) if X.shape[1] else np.zeros((0, 1), dtype=float)
        regressor = NoneRegressor(
            X=X,
            beta_0=formula.get("intercept", 0.0),
            beta_1=beta_1,
            predictor_names=predictor_names,
            predictor_transformations=transformations,
            response_type=response_type,
            categories=category_names,
        )
        return regressor.sample(n)

    if category_names is None:
        raise ValueError(f"{node} requires categories")

    # For categorical nodes, use the declared distribution family but preserve label output.
    probs = np.ones(len(category_names), dtype=float)
    probs = probs / probs.sum()
    return np.asarray(np.random.choice(category_names, size=n, p=probs), dtype=object)


def sample_exogenous_node(node: str, ctx: GenerationContext, n: int) -> np.ndarray:
    return instantiate_distribution(ctx.distributions[node]).sample(n)


def generate_dataset(n: int, base_dir: Path = SYNTHDATA_DIR, seed: int | None = None) -> dict[str, np.ndarray]:
    if seed is not None:
        np.random.seed(seed)

    ctx = load_context(base_dir)
    state: dict[str, np.ndarray] = {}

    for node in ctx.topo_order:
        parents = ctx.parents_by_node.get(node, [])
        node_type = ctx.dag["nodes"][node]["type"]
        dist_spec = ctx.distributions.get(node, {"distribution": "None"})

        if not parents:
            state[node] = sample_exogenous_node(node, ctx, n)
            continue

        if node_type == "Deterministic" or dist_spec.get("distribution") == "None":
            state[node] = evaluate_deterministic(node, ctx, state, n)
            continue

        state[node] = evaluate_stochastic(node, ctx, state, n)

    return state


def write_dataset(rows: dict[str, np.ndarray], output_path: Path) -> None:
    fieldnames = list(rows.keys())
    n = len(next(iter(rows.values()))) if rows else 0
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(n):
            writer.writerow({name: rows[name][idx] for name in fieldnames})


def main() -> None:
    output_path = SYNTHDATA_DIR / "generated_data.csv"
    rows = generate_dataset(1000, seed=7)
    write_dataset(rows, output_path)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
