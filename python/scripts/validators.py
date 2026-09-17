from dataclasses import dataclass
from pathlib import Path
from math import isfinite
from typing import Any, Literal
import sys

from pydantic import BaseModel, ConfigDict, RootModel, ValidationError as PydanticValidationError, field_validator

PYTHON_ROOT = Path(__file__).resolve().parents[1]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))


@dataclass(frozen=True)
class ValidationError:
    code: str
    message: str
    path: str
    details: dict[str, Any] | None = None


@dataclass(frozen=True)
class ValidationReport:
    ok: bool
    errors: list[ValidationError]


def _is_snake_case(value: str) -> bool:
    return value.islower() and value.replace("_", "").isalnum() and value[0].islower() and " " not in value and "-" not in value


def _name_diff_report(
    variables_names: set[str] | None,
    dag_names: set[str] | None,
    distributions_names: set[str] | None,
) -> ValidationReport:
    present_sets = [name_set for name_set in [variables_names, dag_names, distributions_names] if name_set is not None]
    if len(present_sets) < 2:
        return ValidationReport(ok=True, errors=[])

    reference = present_sets[0]
    all_equal = all(name_set == reference for name_set in present_sets[1:])
    if all_equal:
        return ValidationReport(ok=True, errors=[])

    def union_of_others(current_name: str) -> set[str]:
        if current_name == "variables":
            other_sets = [dag_names, distributions_names]
        elif current_name == "dag":
            other_sets = [variables_names, distributions_names]
        else:
            other_sets = [variables_names, dag_names]
        return set().union(*(name_set for name_set in other_sets if name_set is not None))

    variables_only = sorted((variables_names or set()) - union_of_others("variables"))
    dag_only = sorted((dag_names or set()) - union_of_others("dag"))
    distributions_only = sorted((distributions_names or set()) - union_of_others("distributions"))

    missing_from_variables = sorted(union_of_others("variables") - (variables_names or set())) if variables_names is not None else []
    missing_from_dag = sorted(union_of_others("dag") - (dag_names or set())) if dag_names is not None else []
    missing_from_distributions = (
        sorted(union_of_others("distributions") - (distributions_names or set())) if distributions_names is not None else []
    )

    return ValidationReport(
        ok=False,
        errors=[
            ValidationError(
                code="inconsistent_variable_sets",
                message="Variable names differ across the present core files",
                path="",
                details={
                    "present_files": {
                        "variables": variables_names is not None,
                        "dag": dag_names is not None,
                        "distributions": distributions_names is not None,
                    },
                    "variables_only": variables_only,
                    "dag_only": dag_only,
                    "distributions_only": distributions_only,
                    "missing_from_variables": missing_from_variables,
                    "missing_from_dag": missing_from_dag,
                    "missing_from_distributions": missing_from_distributions,
                },
            )
        ],
    )


def _dag_node_parents(dag_data: dict[str, Any]) -> dict[str, set[str]]:
    nodes = dag_data.get("nodes", {})
    edges = dag_data.get("edges", {})
    if not isinstance(nodes, dict):
        return {}

    parents: dict[str, set[str]] = {node_name: set() for node_name in nodes.keys()}

    if not isinstance(nodes, dict) or not isinstance(edges, dict):
        return parents

    for edge_data in edges.values():
        if not isinstance(edge_data, dict):
            continue
        parent = edge_data.get("parent")
        child = edge_data.get("child")
        if parent in parents and child in parents:
            parents[child].add(parent)

    return parents


def _formula_dag_consistency_report(formulas_data: dict[str, Any], dag_data: dict[str, Any]) -> ValidationReport:
    if not isinstance(formulas_data, dict) or not isinstance(dag_data, dict):
        return ValidationReport(ok=True, errors=[])

    dag_nodes = dag_data.get("nodes")
    dag_edges = dag_data.get("edges")
    if not isinstance(dag_nodes, dict) or not isinstance(dag_edges, dict):
        return ValidationReport(ok=True, errors=[])

    parents = _dag_node_parents(dag_data)
    endogenous_nodes = {node_name for node_name, parent_names in parents.items() if parent_names}
    formula_names = set(formulas_data.keys())
    dag_node_names = set(dag_nodes.keys())

    formulas_for_non_endogenous = sorted(formula_names - endogenous_nodes)
    missing_formulas_for_endogenous = sorted(endogenous_nodes - formula_names)
    extra_formulas_for_non_dag_nodes = sorted(formula_names - dag_node_names)
    formulas_for_exogenous_dag_nodes = sorted((formula_names & dag_node_names) - endogenous_nodes)

    if not formulas_for_non_endogenous and not missing_formulas_for_endogenous:
        return ValidationReport(ok=True, errors=[])

    return ValidationReport(
        ok=False,
        errors=[
            ValidationError(
                code="inconsistent_formula_dag_sets",
                message="Formula names must match the endogenous DAG nodes",
                path="",
                details={
                    "endogenous_dag_nodes": sorted(endogenous_nodes),
                    "formula_names": sorted(formula_names),
                    "missing_formulas_for_endogenous": missing_formulas_for_endogenous,
                    "extra_formulas_for_non_endogenous": formulas_for_non_endogenous,
                    "extra_formulas_for_non_dag_nodes": extra_formulas_for_non_dag_nodes,
                    "formulas_for_exogenous_dag_nodes": formulas_for_exogenous_dag_nodes,
                },
            )
        ],
    )


def _dag_distribution_consistency_report(dag_data: dict[str, Any], distributions_data: dict[str, Any]) -> ValidationReport:
    if not isinstance(dag_data, dict) or not isinstance(distributions_data, dict):
        return ValidationReport(ok=True, errors=[])

    dag_nodes = dag_data.get("nodes")
    if not isinstance(dag_nodes, dict):
        return ValidationReport(ok=True, errors=[])

    deterministic_nodes = sorted(
        node_name
        for node_name, node_data in dag_nodes.items()
        if isinstance(node_data, dict) and node_data.get("type") == "Deterministic"
    )

    distributions_nodes = set(distributions_data.keys())
    deterministic_with_non_none: list[str] = []
    missing_from_distributions: list[str] = []
    distributions_missing_from_dag: list[str] = []

    for node_name in deterministic_nodes:
        distribution_item = distributions_data.get(node_name)
        if distribution_item is None:
            missing_from_distributions.append(node_name)
            continue
        if not isinstance(distribution_item, dict) or distribution_item.get("distribution") != "None":
            deterministic_with_non_none.append(node_name)

    distributions_missing_from_dag = sorted(distributions_nodes - set(dag_nodes.keys()))

    if not deterministic_with_non_none and not missing_from_distributions and not distributions_missing_from_dag:
        return ValidationReport(ok=True, errors=[])

    return ValidationReport(
        ok=False,
        errors=[
            ValidationError(
                code="inconsistent_dag_distribution_sets",
                message="Deterministic DAG nodes must use None distributions",
                path="",
                details={
                    "deterministic_nodes": deterministic_nodes,
                    "deterministic_nodes_with_non_none_distribution": deterministic_with_non_none,
                    "dag_nodes_missing_from_distributions": missing_from_distributions,
                    "distributions_missing_from_dag": distributions_missing_from_dag,
                },
            )
        ],
    )


def _dag_formula_predictor_consistency_report(dag_data: dict[str, Any], formulas_data: dict[str, Any]) -> ValidationReport:
    if not isinstance(dag_data, dict) or not isinstance(formulas_data, dict):
        return ValidationReport(ok=True, errors=[])

    dag_nodes = dag_data.get("nodes")
    dag_edges = dag_data.get("edges")
    if not isinstance(dag_nodes, dict) or not isinstance(dag_edges, dict):
        return ValidationReport(ok=True, errors=[])

    parents = _dag_node_parents(dag_data)
    formula_names = set(formulas_data.keys())
    dag_node_names = set(dag_nodes.keys())
    endogenous_nodes = {node_name for node_name, parent_names in parents.items() if parent_names}

    missing_formulas_for_dag_nodes = sorted(endogenous_nodes - formula_names)
    extra_formulas_for_non_dag_nodes = sorted(formula_names - endogenous_nodes)

    missing_predictors_from_formula: dict[str, list[str]] = {}
    extra_predictors_not_in_dag: dict[str, list[str]] = {}

    for formula_name in sorted(formula_names & endogenous_nodes):
        dag_parents = parents.get(formula_name, set())
        formula_item = formulas_data.get(formula_name)
        if not isinstance(formula_item, dict):
            continue

        predictor_names = set()
        if formula_item.get("type") == "categorical_nominal":
            category_models = formula_item.get("category_models", {})
            if isinstance(category_models, dict):
                for category_item in category_models.values():
                    if isinstance(category_item, dict):
                        predictors = category_item.get("predictors", {})
                        if isinstance(predictors, dict):
                            predictor_names.update(predictors.keys())
        else:
            predictors = formula_item.get("predictors", {})
            if isinstance(predictors, dict):
                predictor_names.update(predictors.keys())

        missing_predictors = sorted(dag_parents - predictor_names)
        extra_predictors = sorted(predictor_names - dag_parents)
        if missing_predictors:
            missing_predictors_from_formula[formula_name] = missing_predictors
        if extra_predictors:
            extra_predictors_not_in_dag[formula_name] = extra_predictors

    if (
        not missing_formulas_for_dag_nodes
        and not extra_formulas_for_non_dag_nodes
        and not missing_predictors_from_formula
        and not extra_predictors_not_in_dag
    ):
        return ValidationReport(ok=True, errors=[])

    return ValidationReport(
        ok=False,
        errors=[
            ValidationError(
                code="inconsistent_dag_formula_predictors",
                message="Formula predictors must match DAG parent sets",
                path="",
                details={
                    "missing_formulas_for_dag_nodes": missing_formulas_for_dag_nodes,
                    "extra_formulas_for_non_dag_nodes": extra_formulas_for_non_dag_nodes,
                    "missing_predictors_from_formula": missing_predictors_from_formula,
                    "extra_predictors_not_in_dag": extra_predictors_not_in_dag,
                },
            )
        ],
    )


def _variables_distribution_consistency_report(variables_data: dict[str, Any], distributions_data: dict[str, Any]) -> ValidationReport:
    if not isinstance(variables_data, dict) or not isinstance(distributions_data, dict):
        return ValidationReport(ok=True, errors=[])

    shared_names = set(variables_data.keys()) & set(distributions_data.keys())
    quantitative_continuous_mismatches: dict[str, str] = {}
    quantitative_discrete_mismatches: dict[str, str] = {}
    categorical_nominal_mismatches: dict[str, str] = {}
    categorical_ordinal_mismatches: dict[str, str] = {}

    continuous_distributions = {"Normal", "Gamma", "Log Normal", "Beta"}
    discrete_distributions = {"Bernoulli", "Binomial", "Poisson", "Negative Binomial"}

    for variable_name in sorted(shared_names):
        variable_item = variables_data.get(variable_name)
        distribution_item = distributions_data.get(variable_name)
        if not isinstance(variable_item, dict) or not isinstance(distribution_item, dict):
            continue

        variable_type = variable_item.get("data_type")
        variable_level = variable_item.get("classification") if variable_type == "Quantitative" else variable_item.get("measurement_level")
        distribution_name = distribution_item.get("distribution")

        if variable_type == "Quantitative" and variable_level == "Continuous" and distribution_name not in continuous_distributions:
            quantitative_continuous_mismatches[variable_name] = str(distribution_name)
        elif variable_type == "Quantitative" and variable_level == "Discrete" and distribution_name not in discrete_distributions:
            quantitative_discrete_mismatches[variable_name] = str(distribution_name)
        elif variable_type == "Categorical" and variable_level == "Nominal" and distribution_name != "Categorical Nominal":
            categorical_nominal_mismatches[variable_name] = str(distribution_name)
        elif variable_type == "Categorical" and variable_level == "Ordinal" and distribution_name != "Categorical Ordinal":
            categorical_ordinal_mismatches[variable_name] = str(distribution_name)

    if (
        not quantitative_continuous_mismatches
        and not quantitative_discrete_mismatches
        and not categorical_nominal_mismatches
        and not categorical_ordinal_mismatches
    ):
        return ValidationReport(ok=True, errors=[])

    return ValidationReport(
        ok=False,
        errors=[
            ValidationError(
                code="inconsistent_variable_distribution_types",
                message="Variable types must match their distributions for shared names",
                path="",
                details={
                    "quantitative_continuous_mismatches": quantitative_continuous_mismatches,
                    "quantitative_discrete_mismatches": quantitative_discrete_mismatches,
                    "categorical_nominal_mismatches": categorical_nominal_mismatches,
                    "categorical_ordinal_mismatches": categorical_ordinal_mismatches,
                },
            )
        ],
    )


def _variables_formula_consistency_report(variables_data: dict[str, Any], formulas_data: dict[str, Any]) -> ValidationReport:
    if not isinstance(variables_data, dict) or not isinstance(formulas_data, dict):
        return ValidationReport(ok=True, errors=[])

    shared_names = set(variables_data.keys()) & set(formulas_data.keys())
    quantitative_type_mismatches: dict[str, str] = {}
    categorical_nominal_type_mismatches: dict[str, str] = {}
    categorical_ordinal_type_mismatches: dict[str, str] = {}
    quantitative_predictor_type_mismatches: dict[str, list[str]] = {}
    categorical_predictor_type_mismatches: dict[str, list[str]] = {}

    for variable_name in sorted(shared_names):
        variable_item = variables_data.get(variable_name)
        formula_item = formulas_data.get(variable_name)
        if not isinstance(variable_item, dict) or not isinstance(formula_item, dict):
            continue

        variable_type = variable_item.get("data_type")
        measurement_level = variable_item.get("measurement_level")
        formula_type = formula_item.get("type")

        if variable_type == "Quantitative" and formula_type != "quantitative":
            quantitative_type_mismatches[variable_name] = str(formula_type)
        elif variable_type == "Categorical" and measurement_level == "Nominal" and formula_type != "categorical_nominal":
            categorical_nominal_type_mismatches[variable_name] = str(formula_type)
        elif variable_type == "Categorical" and measurement_level == "Ordinal" and formula_type != "categorical_ordinal":
            categorical_ordinal_type_mismatches[variable_name] = str(formula_type)

        if formula_type == "quantitative":
            predictors = formula_item.get("predictors", {})
            if isinstance(predictors, dict):
                mismatches = [
                    predictor_name
                    for predictor_name, predictor_item in predictors.items()
                    if isinstance(predictor_item, dict) and "reference_category" in predictor_item and "other_categories" in predictor_item
                ]
                if mismatches:
                    quantitative_predictor_type_mismatches[variable_name] = sorted(mismatches)
        elif formula_type == "categorical_nominal":
            category_models = formula_item.get("category_models", {})
            if isinstance(category_models, dict):
                mismatches: set[str] = set()
                for category_item in category_models.values():
                    if not isinstance(category_item, dict):
                        continue
                    predictors = category_item.get("predictors", {})
                    if not isinstance(predictors, dict):
                        continue
                    for predictor_name, predictor_item in predictors.items():
                        if isinstance(predictor_item, dict) and "reference_category" in predictor_item and "other_categories" in predictor_item:
                            mismatches.add(predictor_name)
                if mismatches:
                    categorical_predictor_type_mismatches[variable_name] = sorted(mismatches)
        elif formula_type == "categorical_ordinal":
            predictors = formula_item.get("predictors", {})
            if isinstance(predictors, dict):
                mismatches = [
                    predictor_name
                    for predictor_name, predictor_item in predictors.items()
                    if isinstance(predictor_item, dict) and "reference_category" not in predictor_item and "coefficient" not in predictor_item
                ]
                if mismatches:
                    categorical_predictor_type_mismatches[variable_name] = sorted(mismatches)

    if (
        not quantitative_type_mismatches
        and not categorical_nominal_type_mismatches
        and not categorical_ordinal_type_mismatches
        and not quantitative_predictor_type_mismatches
        and not categorical_predictor_type_mismatches
    ):
        return ValidationReport(ok=True, errors=[])

    return ValidationReport(
        ok=False,
        errors=[
            ValidationError(
                code="inconsistent_variable_formula_types",
                message="Variable types must match formula types for shared names",
                path="",
                details={
                    "quantitative_type_mismatches": quantitative_type_mismatches,
                    "categorical_nominal_type_mismatches": categorical_nominal_type_mismatches,
                    "categorical_ordinal_type_mismatches": categorical_ordinal_type_mismatches,
                    "quantitative_predictor_type_mismatches": quantitative_predictor_type_mismatches,
                    "categorical_predictor_type_mismatches": categorical_predictor_type_mismatches,
                },
            )
        ],
    )


def _formulas_distribution_consistency_report(formulas_data: dict[str, Any], distributions_data: dict[str, Any]) -> ValidationReport:
    if not isinstance(formulas_data, dict) or not isinstance(distributions_data, dict):
        return ValidationReport(ok=True, errors=[])

    shared_names = set(formulas_data.keys()) & set(distributions_data.keys())
    formula_distribution_type_mismatches: dict[str, str] = {}
    categorical_response_category_mismatches: dict[str, dict[str, Any]] = {}
    categorical_predictor_category_mismatches: dict[str, dict[str, Any]] = {}

    quantitative_distributions = {
        "Normal",
        "Gamma",
        "Log Normal",
        "Beta",
        "Bernoulli",
        "Binomial",
        "Poisson",
        "Negative Binomial",
    }

    for variable_name in sorted(shared_names):
        formula_item = formulas_data.get(variable_name)
        distribution_item = distributions_data.get(variable_name)
        if not isinstance(formula_item, dict) or not isinstance(distribution_item, dict):
            continue

        formula_type = formula_item.get("type")
        distribution_name = distribution_item.get("distribution")

        if formula_type == "quantitative" and distribution_name not in quantitative_distributions:
            formula_distribution_type_mismatches[variable_name] = f"expected a quantitative distribution, found {distribution_name!r}"
        elif formula_type == "categorical_nominal" and distribution_name != "Categorical Nominal":
            formula_distribution_type_mismatches[variable_name] = f"expected 'Categorical Nominal', found {distribution_name!r}"
        elif formula_type == "categorical_ordinal" and distribution_name != "Categorical Ordinal":
            formula_distribution_type_mismatches[variable_name] = f"expected 'Categorical Ordinal', found {distribution_name!r}"

        if distribution_name == "Categorical Nominal" and formula_type == "categorical_nominal":
            categories = distribution_item.get("categories", [])
            if isinstance(categories, list):
                reference_category = formula_item.get("reference_category")
                category_models = formula_item.get("category_models", {})
                category_model_names = list(category_models.keys()) if isinstance(category_models, dict) else []
                missing_names = [name for name in [reference_category, *category_model_names] if name is not None and name not in categories]
                count_mismatch = len(category_model_names) != max(len(categories) - 1, 0)
                if missing_names or count_mismatch:
                    categorical_response_category_mismatches[variable_name] = {
                        "kind": "nominal",
                        "reference_category": reference_category,
                        "category_models": sorted(category_model_names),
                        "categories": categories,
                        "missing_names": sorted(set(missing_names)),
                        "expected_category_models": max(len(categories) - 1, 0),
                        "actual_category_models": len(category_model_names),
                    }
        elif distribution_name == "Categorical Ordinal" and formula_type == "categorical_ordinal":
            categories = distribution_item.get("categories", [])
            if isinstance(categories, list):
                reference_category = formula_item.get("reference_category")
                thresholds = formula_item.get("thresholds", {})
                threshold_names = list(thresholds.keys()) if isinstance(thresholds, dict) else []
                missing_names = [name for name in [reference_category, *threshold_names] if name is not None and name not in categories]
                count_mismatch = len(threshold_names) != max(len(categories) - 1, 0)
                if missing_names or count_mismatch:
                    categorical_response_category_mismatches[variable_name] = {
                        "kind": "ordinal",
                        "reference_category": reference_category,
                        "thresholds": sorted(threshold_names),
                        "categories": categories,
                        "missing_names": sorted(set(missing_names)),
                        "expected_thresholds": max(len(categories) - 1, 0),
                        "actual_thresholds": len(threshold_names),
                    }

        predictor_blocks: dict[str, dict[str, Any]] = {}
        if formula_type == "categorical_nominal":
            category_models = formula_item.get("category_models", {})
            if isinstance(category_models, dict):
                for category_item in category_models.values():
                    if not isinstance(category_item, dict):
                        continue
                    predictors = category_item.get("predictors", {})
                    if isinstance(predictors, dict):
                        for predictor_name, predictor_item in predictors.items():
                            if isinstance(predictor_item, dict) and "reference_category" in predictor_item and "other_categories" in predictor_item:
                                predictor_blocks[predictor_name] = predictor_item
        elif formula_type == "categorical_ordinal":
            predictors = formula_item.get("predictors", {})
            if isinstance(predictors, dict):
                for predictor_name, predictor_item in predictors.items():
                    if isinstance(predictor_item, dict) and "reference_category" in predictor_item and "other_categories" in predictor_item:
                        predictor_blocks[predictor_name] = predictor_item

        for predictor_name, predictor_item in predictor_blocks.items():
            predictor_distribution = distributions_data.get(predictor_name)
            if not isinstance(predictor_distribution, dict) or predictor_distribution.get("distribution") != "Categorical Nominal":
                continue
            categories = predictor_distribution.get("categories", [])
            if not isinstance(categories, list):
                continue
            reference_category = predictor_item.get("reference_category")
            other_categories = predictor_item.get("other_categories", {})
            other_category_names = list(other_categories.keys()) if isinstance(other_categories, dict) else []
            missing_names = [name for name in [reference_category, *other_category_names] if name is not None and name not in categories]
            count_mismatch = len(other_category_names) != max(len(categories) - 1, 0)
            if missing_names or count_mismatch:
                categorical_predictor_category_mismatches[f"{variable_name}.{predictor_name}"] = {
                    "reference_category": reference_category,
                    "other_categories": sorted(other_category_names),
                    "categories": categories,
                    "missing_names": sorted(set(missing_names)),
                    "expected_other_categories": max(len(categories) - 1, 0),
                    "actual_other_categories": len(other_category_names),
                }

    if (
        not formula_distribution_type_mismatches
        and not categorical_response_category_mismatches
        and not categorical_predictor_category_mismatches
    ):
        return ValidationReport(ok=True, errors=[])

    return ValidationReport(
        ok=False,
        errors=[
            ValidationError(
                code="inconsistent_formula_distribution_types",
                message="Formula distributions and categorical category definitions must match for shared names",
                path="",
                details={
                    "formula_distribution_type_mismatches": formula_distribution_type_mismatches,
                    "categorical_response_category_mismatches": categorical_response_category_mismatches,
                    "categorical_predictor_category_mismatches": categorical_predictor_category_mismatches,
                },
            )
        ],
    )

    distributions_nodes = set(distributions_data.keys())
    deterministic_with_non_none: list[str] = []
    missing_from_distributions: list[str] = []
    distributions_missing_from_dag: list[str] = []

    for node_name in deterministic_nodes:
        distribution_item = distributions_data.get(node_name)
        if distribution_item is None:
            missing_from_distributions.append(node_name)
            continue
        if not isinstance(distribution_item, dict) or distribution_item.get("distribution") != "None":
            deterministic_with_non_none.append(node_name)

    distributions_missing_from_dag = sorted(distributions_nodes - set(dag_nodes.keys()))

    if not deterministic_with_non_none and not missing_from_distributions and not distributions_missing_from_dag:
        return ValidationReport(ok=True, errors=[])

    return ValidationReport(
        ok=False,
        errors=[
            ValidationError(
                code="inconsistent_dag_distribution_sets",
                message="Deterministic DAG nodes must use None distributions",
                path="",
                details={
                    "deterministic_nodes": deterministic_nodes,
                    "deterministic_nodes_with_non_none_distribution": deterministic_with_non_none,
                    "dag_nodes_missing_from_distributions": missing_from_distributions,
                    "distributions_missing_from_dag": distributions_missing_from_dag,
                },
            )
        ],
    )

    return ValidationReport(
        ok=False,
        errors=[
            ValidationError(
                code="inconsistent_variable_sets",
                message="Variable names differ across the present core files",
                path="",
                details={
                    "present_files": {
                        "variables": variables_names is not None,
                        "dag": dag_names is not None,
                        "distributions": distributions_names is not None,
                    },
                    "variables_only": variables_only,
                    "dag_only": dag_only,
                    "distributions_only": distributions_only,
                    "missing_from_variables": missing_from_variables,
                    "missing_from_dag": missing_from_dag,
                    "missing_from_distributions": missing_from_distributions,
                },
            )
        ],
    )


class QuantitativeVariable(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_type: Literal["Quantitative"]
    description: str
    justification: str
    measurement_level: Literal["Interval", "Ratio"]
    classification: Literal["Discrete", "Continuous"]
    skew: Literal["Left", "Right", "None"]


def _format_field_errors(variable_name: str, errors: list[dict[str, Any]]) -> list[ValidationError]:
    formatted: list[ValidationError] = []
    for error in errors:
        error_type = str(error.get("type", "validation_error"))
        path_parts = [variable_name, *[str(part) for part in error.get("loc", [])]]
        path = ".".join(path_parts)
        message = error.get("msg", "Invalid value")
        context = error.get("ctx") if isinstance(error.get("ctx"), dict) else None
        wrapped_error = context.get("error") if context else None
        if wrapped_error:
            message = str(wrapped_error)
        if error_type == "missing":
            message = f"Missing required field at {path}"
        elif error_type == "extra_forbidden":
            message = f"Unexpected field at {path}"
        elif error_type == "literal_error":
            expected = context.get("expected") if context else None
            if expected:
                message = f"Expected {expected} at {path}"
        elif error_type == "finite_number":
            message = f"{path} must be finite"
        formatted.append(
            ValidationError(
                code=error_type,
                message=message,
                path=path,
                details={"type": error_type, **({"context": context} if context is not None else {})},
            )
        )
    return formatted


def _format_model_errors(errors: list[dict[str, Any]]) -> list[ValidationError]:
    formatted: list[ValidationError] = []
    for error in errors:
        error_type = str(error.get("type", "validation_error"))
        path = ".".join(str(part) for part in error.get("loc", []))
        message = error.get("msg", "Invalid value")
        context = error.get("ctx") if isinstance(error.get("ctx"), dict) else None
        wrapped_error = context.get("error") if context else None
        if wrapped_error:
            message = str(wrapped_error)
        if error_type == "missing":
            message = f"Missing required field at {path}"
        elif error_type == "extra_forbidden":
            message = f"Unexpected field at {path}"
        elif error_type == "literal_error":
            expected = context.get("expected") if context else None
            if expected:
                message = f"Expected {expected} at {path}"
        elif error_type == "finite_number":
            message = f"{path} must be finite"
        formatted.append(
            ValidationError(
                code=error_type,
                message=message,
                path=path,
                details={"type": error_type, **({"context": context} if context is not None else {})},
            )
        )
    return formatted


def _validate_finite_min(value: float | int) -> float | int:
    if not isfinite(value):
        raise ValueError("min must be finite")
    return value


def _validate_finite_max(value: float | int, info):
    if not isfinite(value):
        raise ValueError("max must be finite")
    min_value = info.data.get("min")
    if min_value is not None and value < min_value:
        raise ValueError("max must be greater than or equal to min")
    return value


class BoundedDistribution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min: float | int
    max: float | int

    @field_validator("min")
    @classmethod
    def _validate_min(cls, value: float | int) -> float | int:
        return _validate_finite_min(value)

    @field_validator("max")
    @classmethod
    def _validate_max(cls, value: float | int, info):
        return _validate_finite_max(value, info)


class CategoricalVariable(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_type: Literal["Categorical"]
    description: str
    justification: str
    measurement_level: Literal["Nominal", "Ordinal"]


class DAGNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["Stochastic", "Deterministic"]


class DAGEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parent: str
    child: str


class DAGFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nodes: dict[str, DAGNode]
    edges: dict[str, DAGEdge]

    @field_validator("nodes")
    @classmethod
    def _validate_node_names(cls, value: dict[str, DAGNode]) -> dict[str, DAGNode]:
        for node_name in value:
            if not _is_snake_case(node_name):
                raise ValueError(f"Node name '{node_name}' must be snake_case")
        return value


class NormalDistribution(BoundedDistribution):
    model_config = ConfigDict(extra="forbid")

    distribution: Literal["Normal"]
    mean: float
    standard_deviation: float

    @field_validator("standard_deviation")
    @classmethod
    def _validate_standard_deviation(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("standard_deviation must be positive")
        return value

class GammaDistribution(BoundedDistribution):
    model_config = ConfigDict(extra="forbid")

    distribution: Literal["Gamma"]
    shape: float
    rate: float

    @field_validator("shape", "rate")
    @classmethod
    def _validate_positive(cls, value: float, info):
        if value <= 0:
            raise ValueError(f"{info.field_name} must be positive")
        return value

class LogNormalDistribution(BoundedDistribution):
    model_config = ConfigDict(extra="forbid")

    distribution: Literal["Log Normal"]
    log_mean: float
    log_standard_deviation: float

    @field_validator("log_standard_deviation")
    @classmethod
    def _validate_log_standard_deviation(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("log_standard_deviation must be positive")
        return value

class BetaDistribution(BoundedDistribution):
    model_config = ConfigDict(extra="forbid")

    distribution: Literal["Beta"]
    shape_1: float
    shape_2: float

    @field_validator("shape_1", "shape_2")
    @classmethod
    def _validate_shapes(cls, value: float, info):
        if value <= 0:
            raise ValueError(f"{info.field_name} must be positive")
        return value

class BernoulliDistribution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    distribution: Literal["Bernoulli"]
    success_prob: float

    @field_validator("success_prob")
    @classmethod
    def _validate_success_prob(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("success_prob must be in [0, 1]")
        return value


class BinomialDistribution(BoundedDistribution):
    model_config = ConfigDict(extra="forbid")

    distribution: Literal["Binomial"]
    n_trials: int
    success_prob: float

    @field_validator("n_trials")
    @classmethod
    def _validate_trials(cls, value: int) -> int:
        if value < 1:
            raise ValueError("n_trials must be >= 1")
        return value

    @field_validator("success_prob")
    @classmethod
    def _validate_success_prob(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("success_prob must be in [0, 1]")
        return value

class PoissonDistribution(BoundedDistribution):
    model_config = ConfigDict(extra="forbid")

    distribution: Literal["Poisson"]
    rate: float

    @field_validator("rate")
    @classmethod
    def _validate_rate(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("rate must be positive")
        return value

class NegativeBinomialDistribution(BoundedDistribution):
    model_config = ConfigDict(extra="forbid")

    distribution: Literal["Negative Binomial"]
    shape: float
    mean: float

    @field_validator("shape", "mean")
    @classmethod
    def _validate_positive(cls, value: float, info):
        if value <= 0:
            raise ValueError(f"{info.field_name} must be positive")
        return value

class CategoricalNominalDistribution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    distribution: Literal["Categorical Nominal"]
    categories: list[str]
    probabilities: list[float]

    @field_validator("probabilities")
    @classmethod
    def _validate_probabilities(cls, value: list[float], info):
        categories = info.data.get("categories", [])
        if len(categories) != len(value):
            raise ValueError("categories and probabilities must have the same length")
        if not value:
            raise ValueError("probabilities must not be empty")
        total = sum(value)
        if abs(total - 1.0) > 1e-6:
            raise ValueError("probabilities must sum to 1 within tolerance")
        return value


class CategoricalOrdinalDistribution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    distribution: Literal["Categorical Ordinal"]
    categories: list[str]
    probabilities: list[float]

    @field_validator("probabilities")
    @classmethod
    def _validate_probabilities(cls, value: list[float], info):
        categories = info.data.get("categories", [])
        if len(categories) != len(value):
            raise ValueError("categories and probabilities must have the same length")
        if not value:
            raise ValueError("probabilities must not be empty")
        total = sum(value)
        if abs(total - 1.0) > 1e-6:
            raise ValueError("probabilities must sum to 1 within tolerance")
        return value


class NoneDistribution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    distribution: Literal["None"]


class QuantitativePredictor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    coefficient: float
    transformation: Literal["none", "exp", "log", "sqrt", "inverse", "square", "cubic", "quartic", "sin", "cos"] = "none"


class CategoricalPredictorCategory(BaseModel):
    model_config = ConfigDict(extra="forbid")

    coefficient: float


class CategoricalPredictor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reference_category: str
    other_categories: dict[str, CategoricalPredictorCategory]


class QuantitativePredictors(RootModel[dict[str, QuantitativePredictor]]):
    pass


class CategoricalPredictors(RootModel[dict[str, CategoricalPredictor]]):
    pass


class QuantitativeFormula(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["quantitative"]
    intercept: float
    snr: float
    predictors: dict[str, QuantitativePredictor | CategoricalPredictor]


class NominalCategoryFormula(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intercept: float
    predictors: dict[str, QuantitativePredictor | CategoricalPredictor]


class CategoricalNominalFormula(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["categorical_nominal"]
    reference_category: str
    category_models: dict[str, NominalCategoryFormula]


class OrdinalThreshold(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intercept: float


class CategoricalOrdinalFormula(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["categorical_ordinal"]
    reference_category: str
    predictors: dict[str, QuantitativePredictor | CategoricalPredictor]
    thresholds: dict[str, OrdinalThreshold]


FormulaValue = QuantitativeFormula | CategoricalNominalFormula | CategoricalOrdinalFormula


_DISTRIBUTION_MODELS: dict[str, type[BaseModel]] = {
    "Normal": NormalDistribution,
    "Gamma": GammaDistribution,
    "Log Normal": LogNormalDistribution,
    "Beta": BetaDistribution,
    "Bernoulli": BernoulliDistribution,
    "Binomial": BinomialDistribution,
    "Poisson": PoissonDistribution,
    "Negative Binomial": NegativeBinomialDistribution,
    "Categorical Nominal": CategoricalNominalDistribution,
    "Categorical Ordinal": CategoricalOrdinalDistribution,
    "None": NoneDistribution,
}


def _validate_distribution_item(variable_name: str, item: Any) -> list[ValidationError]:
    if not isinstance(item, dict):
        return [
            ValidationError(
                code="invalid_type",
                message=f"Distribution '{variable_name}' must be an object",
                path=variable_name,
                details={"received_type": type(item).__name__},
            )
        ]

    distribution_name = item.get("distribution")
    if not isinstance(distribution_name, str):
        return [
            ValidationError(
                code="missing_distribution",
                message=f"Distribution '{variable_name}' must define a distribution name",
                path=f"{variable_name}.distribution",
                details=None,
            )
        ]

    model = _DISTRIBUTION_MODELS.get(distribution_name)
    if model is None:
        return [
            ValidationError(
                code="unsupported_distribution",
                message=f"Unsupported distribution '{distribution_name}' for '{variable_name}'",
                path=f"{variable_name}.distribution",
                details={"distribution": distribution_name},
            )
        ]

    try:
        model.model_validate(item)
    except PydanticValidationError as exc:
        return _format_field_errors(variable_name, exc.errors())

    return []


def validate_variables_file(data: dict[str, Any]) -> ValidationReport:
    errors: list[ValidationError] = []

    if not isinstance(data, dict):
        return ValidationReport(
            ok=False,
            errors=[
                ValidationError(
                    code="invalid_type",
                    message="Variables file must be an object",
                    path="",
                    details={"received_type": type(data).__name__},
                )
            ],
        )

    for variable_name, item in data.items():
        if not _is_snake_case(variable_name):
            errors.append(
                ValidationError(
                    code="invalid_name",
                    message=f"Variable name '{variable_name}' must be snake_case",
                    path=variable_name,
                    details=None,
                )
            )
            continue

        if not isinstance(item, dict):
            errors.append(
                ValidationError(
                    code="invalid_type",
                    message=f"Variable '{variable_name}' must be an object",
                    path=variable_name,
                    details={"received_type": type(item).__name__},
                )
            )
            continue

        data_type = item.get("data_type")
        if data_type == "Quantitative":
            try:
                QuantitativeVariable.model_validate(item)
            except PydanticValidationError as exc:
                errors.extend(_format_field_errors(variable_name, exc.errors()))
            continue

        if data_type == "Categorical":
            try:
                CategoricalVariable.model_validate(item)
            except PydanticValidationError as exc:
                errors.extend(_format_field_errors(variable_name, exc.errors()))
            continue

        errors.append(
            ValidationError(
                code="unsupported_data_type",
                message=f"Variable '{variable_name}' must declare data_type as Quantitative or Categorical",
                path=f"{variable_name}.data_type",
                details={"data_type": data_type},
            )
        )

    if errors:
        return ValidationReport(ok=False, errors=errors)

    return ValidationReport(ok=True, errors=[])


def validate_dag_file(data: dict[str, Any]) -> ValidationReport:
    if not isinstance(data, dict):
        return ValidationReport(
            ok=False,
            errors=[
                ValidationError(
                    code="invalid_type",
                    message="DAG file must be an object",
                    path="",
                    details={"received_type": type(data).__name__},
                )
            ],
        )

    errors: list[ValidationError] = []
    nodes = data.get("nodes")
    edges = data.get("edges")

    if not isinstance(nodes, dict):
        errors.append(
            ValidationError(
                code="invalid_type",
                message="DAG file must define nodes as an object",
                path="nodes",
                details={"received_type": type(nodes).__name__},
            )
        )
    if not isinstance(edges, dict):
        errors.append(
            ValidationError(
                code="invalid_type",
                message="DAG file must define edges as an object",
                path="edges",
                details={"received_type": type(edges).__name__},
            )
        )
    if errors:
        return ValidationReport(ok=False, errors=errors)

    try:
        validated_nodes = {name: DAGNode.model_validate(node) for name, node in nodes.items()}
        validated_edges = {name: DAGEdge.model_validate(edge) for name, edge in edges.items()}
    except PydanticValidationError as exc:
        return ValidationReport(ok=False, errors=_format_model_errors(exc.errors()))

    for node_name in validated_nodes:
        if not _is_snake_case(node_name):
            errors.append(
                ValidationError(
                    code="invalid_name",
                    message=f"Node name '{node_name}' must be snake_case",
                    path=f"nodes.{node_name}",
                    details=None,
                )
            )
    for edge_name in validated_edges:
        if not _is_snake_case(edge_name):
            errors.append(
                ValidationError(
                    code="invalid_name",
                    message=f"Edge name '{edge_name}' must be snake_case",
                    path=f"edges.{edge_name}",
                    details=None,
                )
            )

    node_names = set(validated_nodes.keys())
    graph = {
        "nodes": validated_nodes,
        "edges": {edge_name: edge.model_dump() for edge_name, edge in validated_edges.items()},
    }
    has_unknown_node_references = False
    for edge_name, edge in validated_edges.items():
        if edge.parent not in node_names:
            has_unknown_node_references = True
            errors.append(
                ValidationError(
                    code="unknown_parent_node",
                    message=f"Edge '{edge_name}' references unknown parent node '{edge.parent}'",
                    path=f"edges.{edge_name}.parent",
                    details={"edge": edge_name, "parent": edge.parent},
                )
            )
        if edge.child not in node_names:
            has_unknown_node_references = True
            errors.append(
                ValidationError(
                    code="unknown_child_node",
                    message=f"Edge '{edge_name}' references unknown child node '{edge.child}'",
                    path=f"edges.{edge_name}.child",
                    details={"edge": edge_name, "child": edge.child},
                )
            )

    if not has_unknown_node_references:
        from utils import get_dag_order

        try:
            get_dag_order(graph)
        except ValueError as exc:
            errors.append(
                ValidationError(
                    code="invalid_dag_structure",
                    message=str(exc),
                    path="",
                    details=None,
                )
            )

    if errors:
        return ValidationReport(ok=False, errors=errors)

    return ValidationReport(ok=True, errors=[])


def validate_distributions_file(data: dict[str, Any]) -> ValidationReport:
    errors: list[ValidationError] = []

    if not isinstance(data, dict):
        return ValidationReport(
            ok=False,
            errors=[
                ValidationError(
                    code="invalid_type",
                    message="Distributions file must be an object",
                    path="",
                    details={"received_type": type(data).__name__},
                )
            ],
        )

    for variable_name, item in data.items():
        if not _is_snake_case(variable_name):
            errors.append(
                ValidationError(
                    code="invalid_name",
                    message=f"Variable name '{variable_name}' must be snake_case",
                    path=variable_name,
                    details=None,
                )
            )
            continue
        errors.extend(_validate_distribution_item(variable_name, item))

    if errors:
        return ValidationReport(ok=False, errors=errors)

    return ValidationReport(ok=True, errors=[])


def _validate_predictor_item(formula_name: str, predictor_name: str, item: Any) -> list[ValidationError]:
    path_prefix = f"{formula_name}.predictors.{predictor_name}"
    if not isinstance(item, dict):
        return [
            ValidationError(
                code="invalid_type",
                message=f"Predictor '{predictor_name}' in '{formula_name}' must be an object",
                path=path_prefix,
                details={"received_type": type(item).__name__},
            )
        ]

    if set(item.keys()).issubset({"coefficient", "transformation"}):
        try:
            QuantitativePredictor.model_validate(item)
        except PydanticValidationError as exc:
            return _format_field_errors(path_prefix, exc.errors())
        return []

    if set(item.keys()).issubset({"reference_category", "other_categories"}):
        try:
            CategoricalPredictor.model_validate(item)
        except PydanticValidationError as exc:
            return _format_field_errors(path_prefix, exc.errors())
        return []

    return [
        ValidationError(
            code="unsupported_predictor_schema",
            message=(
                f"Predictor '{predictor_name}' in '{formula_name}' must be either "
                "a quantitative predictor with coefficient and optional transformation, "
                "or a categorical predictor with reference_category and other_categories"
            ),
            path=path_prefix,
            details=None,
        )
    ]


def _validate_formula_item(formula_name: str, item: Any) -> list[ValidationError]:
    if not isinstance(item, dict):
        return [
            ValidationError(
                code="invalid_type",
                message=f"Formula '{formula_name}' must be an object",
                path=formula_name,
                details={"received_type": type(item).__name__},
            )
        ]

    formula_type = item.get("type")
    errors: list[ValidationError] = []

    if formula_type == "quantitative":
        try:
            QuantitativeFormula.model_validate(item)
        except PydanticValidationError as exc:
            return _format_field_errors(formula_name, exc.errors())
        for predictor_name, predictor_item in item.get("predictors", {}).items():
            errors.extend(_validate_predictor_item(formula_name, predictor_name, predictor_item))
        return errors

    if formula_type == "categorical_nominal":
        try:
            CategoricalNominalFormula.model_validate(item)
        except PydanticValidationError as exc:
            return _format_field_errors(formula_name, exc.errors())
        for category_name, category_item in item.get("category_models", {}).items():
            if not isinstance(category_item, dict):
                errors.append(
                    ValidationError(
                        code="invalid_type",
                        message=f"Nominal category '{category_name}' in '{formula_name}' must be an object",
                        path=f"{formula_name}.category_models.{category_name}",
                        details={"received_type": type(category_item).__name__},
                    )
                )
                continue
            for predictor_name, predictor_item in category_item.get("predictors", {}).items():
                errors.extend(_validate_predictor_item(formula_name, predictor_name, predictor_item))
        return errors

    if formula_type == "categorical_ordinal":
        try:
            CategoricalOrdinalFormula.model_validate(item)
        except PydanticValidationError as exc:
            return _format_field_errors(formula_name, exc.errors())
        for predictor_name, predictor_item in item.get("predictors", {}).items():
            errors.extend(_validate_predictor_item(formula_name, predictor_name, predictor_item))
        return errors

    if "type" not in item:
        return [
            ValidationError(
                code="missing_type",
                message=f"Formula '{formula_name}' must declare a type",
                path=f"{formula_name}.type",
                details=None,
            )
        ]

    return [
        ValidationError(
            code="unsupported_formula_type",
            message=f"Formula '{formula_name}' has unsupported type '{formula_type}'",
            path=f"{formula_name}.type",
            details={"type": formula_type},
        )
    ]


def validate_formulas_file(data: dict[str, Any]) -> ValidationReport:
    errors: list[ValidationError] = []

    if not isinstance(data, dict):
        return ValidationReport(
            ok=False,
            errors=[
                ValidationError(
                    code="invalid_type",
                    message="Formulas file must be an object",
                    path="",
                    details={"received_type": type(data).__name__},
                )
            ],
        )

    for formula_name, item in data.items():
        if not _is_snake_case(formula_name):
            errors.append(
                ValidationError(
                    code="invalid_name",
                    message=f"Formula name '{formula_name}' must be snake_case",
                    path=formula_name,
                    details=None,
                )
            )
            continue
        errors.extend(_validate_formula_item(formula_name, item))

    if errors:
        return ValidationReport(ok=False, errors=errors)

    return ValidationReport(ok=True, errors=[])


def load_and_validate_dag_file(path: str | Path) -> ValidationReport:
    file_path = Path(path)
    import json

    with file_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    return validate_dag_file(data)


def load_and_validate_distributions_file(path: str | Path) -> ValidationReport:
    file_path = Path(path)
    import json

    with file_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    return validate_distributions_file(data)


def load_and_validate_variables_file(path: str | Path) -> ValidationReport:
    file_path = Path(path)
    import json

    with file_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    return validate_variables_file(data)


def load_and_validate_formulas_file(path: str | Path) -> ValidationReport:
    file_path = Path(path)
    import json

    with file_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    return validate_formulas_file(data)


def validate_core_files(
    variables_data: dict[str, Any] | None,
    dag_data: dict[str, Any] | None,
    distributions_data: dict[str, Any] | None,
) -> ValidationReport:
    errors: list[ValidationError] = []

    if variables_data is not None:
        variables_report = validate_variables_file(variables_data)
        errors.extend(variables_report.errors)
    if dag_data is not None:
        dag_report = validate_dag_file(dag_data)
        errors.extend(dag_report.errors)
    if distributions_data is not None:
        distributions_report = validate_distributions_file(distributions_data)
        errors.extend(distributions_report.errors)

    if errors:
        return ValidationReport(ok=False, errors=errors)

    variables_names = set(variables_data.keys()) if isinstance(variables_data, dict) else None
    dag_names = set(dag_data.get("nodes", {}).keys()) if isinstance(dag_data, dict) else None
    distributions_names = set(distributions_data.keys()) if isinstance(distributions_data, dict) else None

    return _name_diff_report(variables_names, dag_names, distributions_names)


def validate_formulas_and_dag_consistency(
    formulas_data: dict[str, Any] | None,
    dag_data: dict[str, Any] | None,
) -> ValidationReport:
    if formulas_data is None or dag_data is None:
        return ValidationReport(ok=True, errors=[])

    formulas_report = validate_formulas_file(formulas_data)
    dag_report = validate_dag_file(dag_data)
    errors = [*formulas_report.errors, *dag_report.errors]
    if errors:
        return ValidationReport(ok=False, errors=errors)

    return _formula_dag_consistency_report(formulas_data, dag_data)


def validate_dag_and_distributions_consistency(
    dag_data: dict[str, Any] | None,
    distributions_data: dict[str, Any] | None,
) -> ValidationReport:
    if dag_data is None or distributions_data is None:
        return ValidationReport(ok=True, errors=[])

    dag_report = validate_dag_file(dag_data)
    distributions_report = validate_distributions_file(distributions_data)
    errors = [*dag_report.errors, *distributions_report.errors]
    if errors:
        return ValidationReport(ok=False, errors=errors)

    return _dag_distribution_consistency_report(dag_data, distributions_data)


def validate_dag_and_formulas_predictor_consistency(
    dag_data: dict[str, Any] | None,
    formulas_data: dict[str, Any] | None,
) -> ValidationReport:
    if dag_data is None or formulas_data is None:
        return ValidationReport(ok=True, errors=[])

    dag_report = validate_dag_file(dag_data)
    formulas_report = validate_formulas_file(formulas_data)
    errors = [*dag_report.errors, *formulas_report.errors]
    if errors:
        return ValidationReport(ok=False, errors=errors)

    return _dag_formula_predictor_consistency_report(dag_data, formulas_data)


def validate_variables_and_distributions_consistency(
    variables_data: dict[str, Any] | None,
    distributions_data: dict[str, Any] | None,
) -> ValidationReport:
    if variables_data is None or distributions_data is None:
        return ValidationReport(ok=True, errors=[])

    variables_report = validate_variables_file(variables_data)
    distributions_report = validate_distributions_file(distributions_data)
    errors = [*variables_report.errors, *distributions_report.errors]
    if errors:
        return ValidationReport(ok=False, errors=errors)

    return _variables_distribution_consistency_report(variables_data, distributions_data)


def validate_variables_and_formulas_consistency(
    variables_data: dict[str, Any] | None,
    formulas_data: dict[str, Any] | None,
) -> ValidationReport:
    if variables_data is None or formulas_data is None:
        return ValidationReport(ok=True, errors=[])

    variables_report = validate_variables_file(variables_data)
    formulas_report = validate_formulas_file(formulas_data)
    errors = [*variables_report.errors, *formulas_report.errors]
    if errors:
        return ValidationReport(ok=False, errors=errors)

    return _variables_formula_consistency_report(variables_data, formulas_data)


def validate_formulas_and_distributions_consistency(
    formulas_data: dict[str, Any] | None,
    distributions_data: dict[str, Any] | None,
) -> ValidationReport:
    if formulas_data is None or distributions_data is None:
        return ValidationReport(ok=True, errors=[])

    formulas_report = validate_formulas_file(formulas_data)
    distributions_report = validate_distributions_file(distributions_data)
    errors = [*formulas_report.errors, *distributions_report.errors]
    if errors:
        return ValidationReport(ok=False, errors=errors)

    return _formulas_distribution_consistency_report(formulas_data, distributions_data)


def load_and_validate_core_files(
    variables_path: str | Path | None = None,
    dag_path: str | Path | None = None,
    distributions_path: str | Path | None = None,
) -> ValidationReport:
    import json

    variables_data = None
    dag_data = None
    distributions_data = None

    if variables_path is not None:
        variables_file = Path(variables_path)
        with variables_file.open("r", encoding="utf-8") as handle:
            variables_data = json.load(handle)

    if dag_path is not None:
        dag_file = Path(dag_path)
        with dag_file.open("r", encoding="utf-8") as handle:
            dag_data = json.load(handle)

    if distributions_path is not None:
        distributions_file = Path(distributions_path)
        with distributions_file.open("r", encoding="utf-8") as handle:
            distributions_data = json.load(handle)

    return validate_core_files(variables_data, dag_data, distributions_data)


def _load_json_file(path: str | Path) -> Any:
    import json

    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_all_available_files(
    *,
    variables_data: dict[str, Any] | None = None,
    dag_data: dict[str, Any] | None = None,
    distributions_data: dict[str, Any] | None = None,
    formulas_data: dict[str, Any] | None = None,
) -> ValidationReport:
    errors: list[ValidationError] = []

    if variables_data is not None:
        errors.extend(validate_variables_file(variables_data).errors)
    if dag_data is not None:
        errors.extend(validate_dag_file(dag_data).errors)
    if distributions_data is not None:
        errors.extend(validate_distributions_file(distributions_data).errors)
    if formulas_data is not None:
        errors.extend(validate_formulas_file(formulas_data).errors)

    if errors:
        return ValidationReport(ok=False, errors=errors)

    if variables_data is not None and dag_data is not None and distributions_data is not None:
        errors.extend(validate_core_files(variables_data, dag_data, distributions_data).errors)
    if formulas_data is not None and dag_data is not None:
        errors.extend(validate_formulas_and_dag_consistency(formulas_data, dag_data).errors)
        errors.extend(validate_dag_and_formulas_predictor_consistency(dag_data, formulas_data).errors)
    if dag_data is not None and distributions_data is not None:
        errors.extend(validate_dag_and_distributions_consistency(dag_data, distributions_data).errors)
    if variables_data is not None and distributions_data is not None:
        errors.extend(validate_variables_and_distributions_consistency(variables_data, distributions_data).errors)
    if variables_data is not None and formulas_data is not None:
        errors.extend(validate_variables_and_formulas_consistency(variables_data, formulas_data).errors)
    if formulas_data is not None and distributions_data is not None:
        errors.extend(validate_formulas_and_distributions_consistency(formulas_data, distributions_data).errors)

    if errors:
        return ValidationReport(ok=False, errors=errors)

    return ValidationReport(ok=True, errors=[])


def _format_validation_report(report: ValidationReport) -> str:
    lines = ["Validation passed" if report.ok else "Validation failed"]
    if not report.errors:
        return "\n".join(lines)

    lines.append(f"Errors: {len(report.errors)}")
    for index, error in enumerate(report.errors, start=1):
        lines.append(f"{index}. {error.code}")
        lines.append(f"   message: {error.message}")
        if error.path:
            lines.append(f"   path: {error.path}")
        if error.details is not None:
            lines.append(f"   details: {error.details}")
    return "\n".join(lines)


def main() -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Validate synthdata JSON files")
    parser.add_argument("--variables", default="synthdata/variables.json")
    parser.add_argument("--dag", default="synthdata/dag.json")
    parser.add_argument("--distributions", default="synthdata/distributions.json")
    parser.add_argument("--formulas", default="synthdata/formulas.json")
    args = parser.parse_args()

    variables_data = _load_json_file(args.variables) if Path(args.variables).exists() else None
    dag_data = _load_json_file(args.dag) if Path(args.dag).exists() else None
    distributions_data = _load_json_file(args.distributions) if Path(args.distributions).exists() else None
    formulas_data = _load_json_file(args.formulas) if Path(args.formulas).exists() else None

    report = validate_all_available_files(
        variables_data=variables_data,
        dag_data=dag_data,
        distributions_data=distributions_data,
        formulas_data=formulas_data,
    )
    print(_format_validation_report(report))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
