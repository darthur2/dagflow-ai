from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PYTHON_ROOT = Path(__file__).resolve().parents[1]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from distributions import CategoricalNominal, CategoricalOrdinal
from regressors import (
    BetaRegressor,
    BernoulliRegressor,
    BinomialRegressor,
    CategoricalNominalRegressor,
    CategoricalOrdinalRegressor,
    GammaRegressor,
    LogNormalRegressor,
    NegativeBinomialRegressor,
    NoneRegressor,
    NormalRegressor,
    PoissonRegressor,
)
from utils import (
    get_beta_0,
    get_categories,
    get_dag_order,
    get_snr,
    load_json,
    make_beta_1,
    make_distribution,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SYNTHDATA = REPO_ROOT / "synthdata"


def _one_hot(values: np.ndarray, categories: list[str]) -> np.ndarray:
    return np.column_stack([(values == category).astype(float) for category in categories])


def _build_parent_matrix(
    parent_names: list[str],
    samples: dict[str, np.ndarray],
    distributions: dict[str, object],
) -> tuple[np.ndarray, list[str], dict[str, str]]:
    columns: list[np.ndarray] = []
    predictor_names: list[str] = []
    predictor_transformations: dict[str, str] = {}

    for parent_name in parent_names:
        parent_values = np.asarray(samples[parent_name])
        parent_distribution = distributions[parent_name]
        if parent_distribution.__class__.__name__ in {"CategoricalNominal", "CategoricalOrdinal"}:
            categories = list(parent_distribution.categories)
            if len(categories) < 2:
                raise ValueError(f"Categorical parent {parent_name} must have at least 2 categories")
            encoded = _one_hot(parent_values, categories)[:, 1:]
            columns.append(encoded)
            predictor_names.extend([parent_name] * encoded.shape[1])
            predictor_transformations[parent_name] = "none"
            continue

        columns.append(parent_values.reshape(-1, 1).astype(float))
        predictor_names.append(parent_name)
        predictor_transformations.setdefault(parent_name, "none")

    if not columns:
        return np.empty((len(next(iter(samples.values()))), 0), dtype=float), predictor_names, predictor_transformations

    matrix = np.column_stack(columns)
    return matrix, predictor_names, predictor_transformations


def _sample_exogenous(distribution, n: int) -> np.ndarray:
    samples = distribution.sample(n)
    return np.asarray(samples)


def _safe_sample_regressor(regressor, n: int, fallback_distribution=None) -> np.ndarray:
    calibrated = regressor.calibrate()
    return np.asarray(calibrated.sample(n))


def _raise_generation_error(
    variable_name: str,
    distribution_name: str,
    parent_names: list[str],
    X: np.ndarray,
    predictor_names: list[str],
    predictor_transformations: dict[str, str],
    beta_0,
    beta_1,
    stage: str,
    error: Exception,
) -> None:
    raise RuntimeError(
        "Failed to generate variable "
        f"'{variable_name}' during {stage}. "
        f"distribution={distribution_name}; parents={parent_names}; "
        f"X_shape={X.shape}; predictor_names={predictor_names}; "
        f"predictor_transformations={predictor_transformations}; "
        f"beta_0_type={type(beta_0).__name__}; beta_0_shape={getattr(beta_0, 'shape', None)}; "
        f"beta_1_type={type(beta_1).__name__}; beta_1_shape={getattr(beta_1, 'shape', None)}; "
        f"original_error={type(error).__name__}: {error}"
    ) from error


def _regressor_for_distribution_name(distribution_name: str):
    mapping = {
        "Normal": NormalRegressor,
        "Gamma": GammaRegressor,
        "Log Normal": LogNormalRegressor,
        "Beta": BetaRegressor,
        "Bernoulli": BernoulliRegressor,
        "Binomial": BinomialRegressor,
        "Poisson": PoissonRegressor,
        "Negative Binomial": NegativeBinomialRegressor,
        "Categorical Nominal": CategoricalNominalRegressor,
        "Categorical Ordinal": CategoricalOrdinalRegressor,
        "None": NoneRegressor,
    }
    if distribution_name not in mapping:
        raise ValueError(f"Unsupported distribution for regression: {distribution_name}")
    return mapping[distribution_name]


def _distribution_name(distribution) -> str:
    name = distribution.__class__.__name__
    if name in {"CategoricalNominal", "CategoricalOrdinal"}:
        return "Categorical Nominal" if name == "CategoricalNominal" else "Categorical Ordinal"
    if name == "NoneDistribution":
        return "None"
    if name == "LogNormal":
        return "Log Normal"
    if name == "NegativeBinomial":
        return "Negative Binomial"
    return name


def generate_data(n: int = 1000) -> pd.DataFrame:
    dag_data = load_json(SYNTHDATA / "dag.json") or {}
    distributions_data = load_json(SYNTHDATA / "distributions.json") or {}
    formulas_data = load_json(SYNTHDATA / "formulas.json") or {}

    distributions = {name: make_distribution(distributions_data, name) for name in distributions_data}
    dag_order = get_dag_order(dag_data)

    samples: dict[str, np.ndarray] = {}
    dataframe = pd.DataFrame(index=range(n))
    parents_by_child: dict[str, list[str]] = {}
    for edge_data in dag_data.get("edges", {}).values():
        parents_by_child.setdefault(edge_data["child"], []).append(edge_data["parent"])

    for variable_name in dag_order:
        distribution = distributions.get(variable_name)
        if distribution is None:
            raise ValueError(f"Missing distribution for variable: {variable_name}")

        parent_names = parents_by_child.get(variable_name, [])
        if not parent_names:
            sampled = _sample_exogenous(distribution, n)
            samples[variable_name] = sampled
            dataframe[variable_name] = sampled
            continue

        formula = formulas_data.get(variable_name)
        if formula is None:
            raise ValueError(f"Missing formula for endogenous variable: {variable_name}")

        X, predictor_names, predictor_transformations = _build_parent_matrix(parent_names, samples, distributions)
        distribution_name = _distribution_name(distribution)
        regressor_cls = _regressor_for_distribution_name(distribution_name)

        beta_0 = get_beta_0(formulas_data, variable_name)
        beta_1 = make_beta_1(formulas_data, variable_name)
        categories = get_categories(formulas_data, variable_name) if "reference_category" in formula else None

        try:
            if distribution_name == "Normal":
                regressor = regressor_cls(
                    mean=distribution.mean,
                    standard_deviation=distribution.standard_deviation,
                    target_snr=get_snr(formulas_data, variable_name),
                    min=distribution.min,
                    max=distribution.max,
                    X=X,
                    beta_1=beta_1,
                    predictor_names=predictor_names,
                    predictor_transformations=predictor_transformations,
                )
            elif distribution_name == "Gamma":
                regressor = regressor_cls(
                    shape=distribution.shape,
                    rate=distribution.rate,
                    min=distribution.min,
                    max=distribution.max,
                    truncated=distribution.truncated,
                    target_snr=get_snr(formulas_data, variable_name),
                    X=X,
                    beta_1=beta_1,
                    predictor_names=predictor_names,
                    predictor_transformations=predictor_transformations,
                )
            elif distribution_name == "Log Normal":
                regressor = regressor_cls(
                    log_mean=distribution.log_mean,
                    log_standard_deviation=distribution.log_standard_deviation,
                    min=distribution.min,
                    max=distribution.max,
                    truncated=distribution.truncated,
                    target_snr=get_snr(formulas_data, variable_name),
                    X=X,
                    beta_1=beta_1,
                    predictor_names=predictor_names,
                    predictor_transformations=predictor_transformations,
                )
            elif distribution_name == "Beta":
                regressor = regressor_cls(
                    shape_1=distribution.shape_1,
                    shape_2=distribution.shape_2,
                    min=distribution.min,
                    max=distribution.max,
                    target_snr=get_snr(formulas_data, variable_name),
                    X=X,
                    beta_1=beta_1,
                    predictor_names=predictor_names,
                    predictor_transformations=predictor_transformations,
                )
            elif distribution_name == "Bernoulli":
                regressor = regressor_cls(
                    success_prob=distribution.success_prob,
                    target_snr=get_snr(formulas_data, variable_name),
                    X=X,
                    beta_1=beta_1,
                    predictor_names=predictor_names,
                    predictor_transformations=predictor_transformations,
                )
            elif distribution_name == "Binomial":
                regressor = regressor_cls(
                    success_prob=distribution.success_prob,
                    n_trials=distribution.n_trials,
                    min=distribution.min,
                    max=distribution.max,
                    truncated=distribution.truncated,
                    target_snr=get_snr(formulas_data, variable_name),
                    X=X,
                    beta_1=beta_1,
                    predictor_names=predictor_names,
                    predictor_transformations=predictor_transformations,
                )
            elif distribution_name == "Poisson":
                regressor = regressor_cls(
                    rate=distribution.rate,
                    min=distribution.min,
                    max=distribution.max,
                    truncated=distribution.truncated,
                    target_snr=get_snr(formulas_data, variable_name),
                    X=X,
                    beta_1=beta_1,
                    predictor_names=predictor_names,
                    predictor_transformations=predictor_transformations,
                )
            elif distribution_name == "Negative Binomial":
                regressor = regressor_cls(
                    shape=distribution.shape,
                    mean=distribution.mean,
                    min=distribution.min,
                    max=distribution.max,
                    truncated=distribution.truncated,
                    target_snr=get_snr(formulas_data, variable_name),
                    X=X,
                    beta_1=beta_1,
                    predictor_names=predictor_names,
                    predictor_transformations=predictor_transformations,
                )
            elif distribution_name == "Categorical Nominal":
                regressor = regressor_cls(
                    target_probabilities=np.asarray(distribution.probabilities, dtype=float),
                    X=X,
                    beta_1=beta_1,
                    predictor_names=predictor_names,
                    predictor_transformations=predictor_transformations,
                    categories=categories,
                )
            elif distribution_name == "Categorical Ordinal":
                regressor = regressor_cls(
                    target_probabilities=np.asarray(distribution.probabilities, dtype=float),
                    X=X,
                    beta_1=beta_1,
                    predictor_names=predictor_names,
                    predictor_transformations=predictor_transformations,
                    categories=categories,
                )
            elif distribution_name == "None":
                regressor = regressor_cls(X=X, beta_0=beta_0, beta_1=beta_1, predictor_names=predictor_names, predictor_transformations=predictor_transformations, response_type="quantitative")
            else:
                raise ValueError(f"Unsupported distribution for variable: {variable_name}")
        except Exception as exc:
            _raise_generation_error(
                variable_name=variable_name,
                distribution_name=distribution_name,
                parent_names=parent_names,
                X=X,
                predictor_names=predictor_names,
                predictor_transformations=predictor_transformations,
                beta_0=beta_0,
                beta_1=beta_1,
                stage="regressor construction",
                error=exc,
            )

        try:
            sampled = _safe_sample_regressor(regressor, n)
        except Exception as exc:
            _raise_generation_error(
                variable_name=variable_name,
                distribution_name=distribution_name,
                parent_names=parent_names,
                X=X,
                predictor_names=predictor_names,
                predictor_transformations=predictor_transformations,
                beta_0=beta_0,
                beta_1=beta_1,
                stage="sampling",
                error=exc,
            )
        samples[variable_name] = np.asarray(sampled)
        dataframe[variable_name] = sampled

    ordered_columns = [name for name in dag_order if name in dataframe.columns]
    dataframe = dataframe[ordered_columns]
    return dataframe


def main() -> None:
    dataframe = generate_data()
    output_path = SYNTHDATA / "generated_data.csv"
    dataframe.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
