import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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


def error_to_dict(error: ValidationError) -> dict[str, Any]:
    payload = {
        "code": error.code,
        "message": error.message,
        "path": error.path,
    }
    if error.details is not None:
        payload["details"] = error.details
    return payload


def report_to_dict(report: ValidationReport) -> dict[str, Any]:
    return {
        "ok": report.ok,
        "errors": [error_to_dict(error) for error in report.errors],
    }


QUANTITATIVE_REQUIRED_FIELDS = {
    "description",
    "justification",
    "effect_type",
    "measurement_level",
    "classification",
    "skew",
}

CATEGORICAL_REQUIRED_FIELDS = {
    "description",
    "justification",
    "effect_type",
    "measurement_level",
}

QUANTITATIVE_ALLOWED_FIELDS = QUANTITATIVE_REQUIRED_FIELDS
CATEGORICAL_ALLOWED_FIELDS = CATEGORICAL_REQUIRED_FIELDS

DAG_NODE_REQUIRED_FIELDS = {"type"}
DAG_NODE_ALLOWED_FIELDS = DAG_NODE_REQUIRED_FIELDS
DAG_EDGE_REQUIRED_FIELDS = {"parent", "child"}
DAG_EDGE_ALLOWED_FIELDS = DAG_EDGE_REQUIRED_FIELDS

ALLOWED_EFFECT_TYPES = {"Fixed", "Random"}
ALLOWED_QUANTITATIVE_MEASUREMENT_LEVELS = {"Interval", "Ratio"}
ALLOWED_CATEGORICAL_MEASUREMENT_LEVELS = {"Nominal", "Ordinal"}
ALLOWED_CLASSIFICATIONS = {"Discrete", "Continuous"}
ALLOWED_SKEWS = {"Left", "Right", "None"}
DAG_NODE_TYPES = {"Stochastic", "Deterministic"}
ALLOWED_DISTRIBUTIONS = {
    "Normal",
    "Exponential",
    "Gamma",
    "Log Normal",
    "Beta",
    "Uniform",
    "Discrete Uniform",
    "Bernoulli",
    "Binomial",
    "Poisson",
    "Geometric",
    "Negative Binomial",
    "Categorical Nominal",
    "Categorical Ordinal",
    "None",
}

ALLOWED_DISTRIBUTIONS_BY_VARIABLE_SCHEMA = {
    "quantitative": {
        "Continuous": {"Normal", "Exponential", "Gamma", "Log Normal", "Beta", "Uniform"},
        "Discrete": {
            "Discrete Uniform",
            "Bernoulli",
            "Binomial",
            "Poisson",
            "Geometric",
            "Negative Binomial",
        },
    },
    "categorical": {"Categorical Nominal", "Categorical Ordinal"},
}

ALLOWED_FORMULA_TRANSFORMATIONS = {"none", "exp", "log", "sqrt", "inverse", "square", "cubic", "quartic", "sin", "cos"}
FORMULA_VARIABLE_TYPES = {"quantitative", "categorical nominal", "categorical ordinal"}

VARIABLE_DISTRIBUTION_FIELDS = {
    "Normal": {
        "required": {"distribution", "mean", "standard_deviation", "min", "max"},
        "types": {
            "distribution": str,
            "mean": float,
            "standard_deviation": float,
            "min": float,
            "max": float,
        },
    },
    "Exponential": {
        "required": {"distribution", "rate", "min", "max"},
        "types": {"distribution": str, "rate": float, "min": float, "max": float},
    },
    "Gamma": {
        "required": {"distribution", "shape", "rate", "min", "max"},
        "types": {
            "distribution": str,
            "shape": float,
            "rate": float,
            "min": float,
            "max": float,
        },
    },
    "Log Normal": {
        "required": {"distribution", "log_mean", "log_standard_deviation", "min", "max"},
        "types": {
            "distribution": str,
            "log_mean": float,
            "log_standard_deviation": float,
            "min": float,
            "max": float,
        },
    },
    "Beta": {
        "required": {"distribution", "shape_1", "shape_2", "min", "max"},
        "types": {
            "distribution": str,
            "shape_1": float,
            "shape_2": float,
            "min": float,
            "max": float,
        },
    },
    "Uniform": {
        "required": {"distribution", "min", "max"},
        "types": {"distribution": str, "min": float, "max": float},
    },
    "Discrete Uniform": {
        "required": {"distribution", "min", "max"},
        "types": {"distribution": str, "min": int, "max": int},
    },
    "Bernoulli": {
        "required": {"distribution", "success_prob"},
        "types": {"distribution": str, "success_prob": float},
    },
    "Binomial": {
        "required": {"distribution", "n_trials", "success_prob", "min", "max"},
        "types": {
            "distribution": str,
            "n_trials": int,
            "success_prob": float,
            "min": int,
            "max": int,
        },
    },
    "Poisson": {
        "required": {"distribution", "rate", "min", "max"},
        "types": {"distribution": str, "rate": float, "min": int, "max": int},
    },
    "Geometric": {
        "required": {"distribution", "success_prob", "min", "max"},
        "types": {"distribution": str, "success_prob": float, "min": int, "max": int},
    },
    "Negative Binomial": {
        "required": {"distribution", "shape", "mean", "min", "max"},
        "types": {
            "distribution": str,
            "shape": float,
            "mean": float,
            "min": int,
            "max": int,
        },
    },
    "Categorical Nominal": {
        "required": {"distribution", "categories", "probabilities"},
        "types": {"distribution": str, "categories": list, "probabilities": list},
    },
    "Categorical Ordinal": {
        "required": {"distribution", "categories", "probabilities"},
        "types": {"distribution": str, "categories": list, "probabilities": list},
    },
    "None": {
        "required": {"distribution"},
        "types": {"distribution": str},
    },
}


def load_variables(path: str | Path) -> tuple[dict[str, dict[str, Any]] | None, list[ValidationError]]:
    file_path = Path(path)
    if not file_path.exists():
        return None, [
            ValidationError(
                code="VARIABLES_FILE_MISSING",
                message=f"Variables file not found: {file_path}",
                path=str(file_path),
            )
        ]

    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        return None, [
            ValidationError(
                code="VARIABLES_JSON_INVALID",
                message="Variables file is not valid JSON.",
                path=str(file_path),
                details={"line": exc.lineno, "column": exc.colno, "error": exc.msg},
            )
        ]

    if not isinstance(data, dict):
        return None, [
            ValidationError(
                code="VARIABLES_ROOT_INVALID",
                message="Variables file must contain a JSON object at the top level.",
                path=str(file_path),
                details={"actual_type": type(data).__name__},
            )
        ]

    return data, []


def load_dag(path: str | Path) -> tuple[dict[str, Any] | None, list[ValidationError]]:
    file_path = Path(path)
    if not file_path.exists():
        return None, []

    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        return None, [
            ValidationError(
                code="DAG_JSON_INVALID",
                message="DAG file is not valid JSON.",
                path=str(file_path),
                details={"line": exc.lineno, "column": exc.colno, "error": exc.msg},
            )
        ]

    if not isinstance(data, dict):
        return None, [
            ValidationError(
                code="DAG_ROOT_INVALID",
                message="DAG file must contain a JSON object at the top level.",
                path=str(file_path),
                details={"actual_type": type(data).__name__},
            )
        ]

    return data, []


def load_formulas(path: str | Path) -> tuple[dict[str, Any] | None, list[ValidationError]]:
    file_path = Path(path)
    if not file_path.exists():
        return None, []

    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        return None, [
            ValidationError(
                code="FORMULAS_JSON_INVALID",
                message="Formulas file is not valid JSON.",
                path=str(file_path),
                details={"line": exc.lineno, "column": exc.colno, "error": exc.msg},
            )
        ]

    if not isinstance(data, dict):
        return None, [
            ValidationError(
                code="FORMULAS_ROOT_INVALID",
                message="Formulas file must contain a JSON object at the top level.",
                path=str(file_path),
                details={"actual_type": type(data).__name__},
            )
        ]

    return data, []


def is_nan(value: Any) -> bool:
    return isinstance(value, float) and value != value


def classify_variable_schema(variable_name: str, variable_data: dict[str, Any]) -> tuple[str | None, ValidationError | None]:
    measurement_level = variable_data.get("measurement_level")
    has_classification = "classification" in variable_data
    has_skew = "skew" in variable_data

    if measurement_level in ALLOWED_CATEGORICAL_MEASUREMENT_LEVELS:
        unexpected_fields = sorted(set(variable_data.keys()) & {"classification", "skew"})
        if unexpected_fields:
            return None, ValidationError(
                code="VARIABLE_SCHEMA_MISMATCH",
                message="Categorical variables must not include quantitative-only fields.",
                path=variable_name,
                details={"unexpected_fields": unexpected_fields},
            )
        return "categorical", None

    if measurement_level in ALLOWED_QUANTITATIVE_MEASUREMENT_LEVELS:
        if not has_classification or not has_skew:
            return None, ValidationError(
                code="VARIABLE_SCHEMA_MISMATCH",
                message="Quantitative variables must include both classification and skew.",
                path=variable_name,
                details={"missing_field": "classification" if not has_classification else "skew"},
            )
        return "quantitative", None

    return None, ValidationError(
        code="VARIABLE_SCHEMA_MISMATCH",
        message="measurement_level must identify the variable schema.",
        path=variable_name,
        details={"actual_value": measurement_level},
    )


def validate_quantitative_variable_schema_fields(variable_name: str, variable_data: dict[str, Any]) -> ValidationError | None:
    missing_fields = sorted(QUANTITATIVE_REQUIRED_FIELDS - variable_data.keys())
    if missing_fields:
        return ValidationError(
            code="VARIABLE_SCHEMA_MISSING_FIELD",
            message="A required quantitative variable field is missing.",
            path=variable_name,
            details={"missing_fields": missing_fields},
        )

    unexpected_fields = sorted(set(variable_data.keys()) - QUANTITATIVE_ALLOWED_FIELDS)
    if unexpected_fields:
        return ValidationError(
            code="VARIABLE_SCHEMA_UNKNOWN_FIELD",
            message="A quantitative variable contains unknown fields.",
            path=variable_name,
            details={"unexpected_fields": unexpected_fields},
        )

    return None


def validate_categorical_variable_schema_fields(variable_name: str, variable_data: dict[str, Any]) -> ValidationError | None:
    missing_fields = sorted(CATEGORICAL_REQUIRED_FIELDS - variable_data.keys())
    if missing_fields:
        return ValidationError(
            code="VARIABLE_SCHEMA_MISSING_FIELD",
            message="A required categorical variable field is missing.",
            path=variable_name,
            details={"missing_fields": missing_fields},
        )

    unexpected_fields = sorted(set(variable_data.keys()) - CATEGORICAL_ALLOWED_FIELDS)
    if unexpected_fields:
        return ValidationError(
            code="VARIABLE_SCHEMA_UNKNOWN_FIELD",
            message="A categorical variable contains unknown fields.",
            path=variable_name,
            details={"unexpected_fields": unexpected_fields},
        )

    return None


def validate_variable_field_values(variable_name: str, variable_data: dict[str, Any], schema_type: str) -> ValidationError | None:
    required_fields = QUANTITATIVE_REQUIRED_FIELDS if schema_type == "quantitative" else CATEGORICAL_REQUIRED_FIELDS

    for field_name in required_fields:
        value = variable_data.get(field_name)
        if value is None:
            return ValidationError(
                code="VARIABLE_FIELD_EMPTY",
                message="A required variable field is missing a value.",
                path=f"{variable_name}.{field_name}",
            )
        if isinstance(value, str) and not value.strip():
            return ValidationError(
                code="VARIABLE_FIELD_EMPTY",
                message="A required variable field is empty.",
                path=f"{variable_name}.{field_name}",
            )

    effect_type = variable_data.get("effect_type")
    if effect_type not in ALLOWED_EFFECT_TYPES:
        return ValidationError(
            code="VARIABLE_FIELD_INVALID_VALUE",
            message="effect_type must be one of the allowed values.",
            path=f"{variable_name}.effect_type",
            details={"allowed_values": sorted(ALLOWED_EFFECT_TYPES), "actual_value": effect_type},
        )

    measurement_level = variable_data.get("measurement_level")
    allowed_measurement_levels = (
        ALLOWED_QUANTITATIVE_MEASUREMENT_LEVELS if schema_type == "quantitative" else ALLOWED_CATEGORICAL_MEASUREMENT_LEVELS
    )
    if measurement_level not in allowed_measurement_levels:
        return ValidationError(
            code="VARIABLE_FIELD_INVALID_VALUE",
            message="measurement_level must be one of the allowed values.",
            path=f"{variable_name}.measurement_level",
            details={"allowed_values": sorted(allowed_measurement_levels), "actual_value": measurement_level},
        )

    if schema_type == "quantitative":
        classification = variable_data.get("classification")
        if classification not in ALLOWED_CLASSIFICATIONS:
            return ValidationError(
                code="VARIABLE_FIELD_INVALID_VALUE",
                message="classification must be one of the allowed values.",
                path=f"{variable_name}.classification",
                details={"allowed_values": sorted(ALLOWED_CLASSIFICATIONS), "actual_value": classification},
            )

        skew = variable_data.get("skew")
        if skew not in ALLOWED_SKEWS:
            return ValidationError(
                code="VARIABLE_FIELD_INVALID_VALUE",
                message="skew must be one of the allowed values.",
                path=f"{variable_name}.skew",
                details={"allowed_values": sorted(ALLOWED_SKEWS), "actual_value": skew},
            )

    return None


def validate_variable_schema_consistency(variable_name: str, variable_data: dict[str, Any]) -> ValidationError | None:
    measurement_level = variable_data.get("measurement_level")

    if measurement_level in ALLOWED_CATEGORICAL_MEASUREMENT_LEVELS:
        unexpected_fields = sorted(set(variable_data.keys()) & {"classification", "skew"})
        if unexpected_fields:
            return ValidationError(
                code="VARIABLE_SCHEMA_UNKNOWN_FIELD",
                message="Categorical variables must not include quantitative-only fields.",
                path=variable_name,
                details={"unexpected_fields": unexpected_fields},
            )

    return None


def validate_dag_structure(dag: dict[str, Any], variable_names: set[str]) -> list[ValidationError]:
    errors: list[ValidationError] = []

    nodes = dag.get("nodes")
    edges = dag.get("edges")

    if not isinstance(nodes, dict):
        return [
            ValidationError(
                code="DAG_NODES_INVALID",
                message="DAG nodes must be an object.",
                path="dag.nodes",
                details={"actual_type": type(nodes).__name__},
            )
        ]

    if not isinstance(edges, dict):
        return [
            ValidationError(
                code="DAG_EDGES_INVALID",
                message="DAG edges must be an object.",
                path="dag.edges",
                details={"actual_type": type(edges).__name__},
            )
        ]

    node_names = set(nodes.keys())
    missing_nodes = sorted(variable_names - node_names)
    extra_nodes = sorted(node_names - variable_names)
    if missing_nodes:
        errors.append(
            ValidationError(
                code="DAG_MISSING_NODES",
                message="DAG is missing nodes from the variable list.",
                path="dag.nodes",
                details={"missing_nodes": missing_nodes},
            )
        )
    if extra_nodes:
        errors.append(
            ValidationError(
                code="DAG_EXTRA_NODES",
                message="DAG contains nodes not present in the variable list.",
                path="dag.nodes",
                details={"extra_nodes": extra_nodes},
            )
        )

    for node_name, node_data in nodes.items():
        if not isinstance(node_name, str):
            errors.append(
                ValidationError(
                    code="DAG_NODE_NAME_INVALID",
                    message="DAG node names must be strings.",
                    path=f"dag.nodes.{node_name}",
                    details={"actual_type": type(node_name).__name__},
                )
            )
            continue
        if not isinstance(node_data, dict):
            errors.append(
                ValidationError(
                    code="DAG_NODE_INVALID",
                    message="Each DAG node must be an object.",
                    path=f"dag.nodes.{node_name}",
                    details={"actual_type": type(node_data).__name__},
                )
            )
            continue

        missing_fields = sorted(DAG_NODE_REQUIRED_FIELDS - node_data.keys())
        extra_fields = sorted(set(node_data.keys()) - DAG_NODE_ALLOWED_FIELDS)
        if missing_fields:
            errors.append(
                ValidationError(
                    code="DAG_NODE_MISSING_FIELD",
                    message="A DAG node is missing required fields.",
                    path=f"dag.nodes.{node_name}",
                    details={"missing_fields": missing_fields},
                )
            )
        if extra_fields:
            errors.append(
                ValidationError(
                    code="DAG_NODE_EXTRA_FIELD",
                    message="A DAG node contains unknown fields.",
                    path=f"dag.nodes.{node_name}",
                    details={"unexpected_fields": extra_fields},
                )
            )
        node_type = node_data.get("type")
        if normalize_node_type(node_type) not in DAG_NODE_TYPES:
            errors.append(
                ValidationError(
                    code="DAG_NODE_TYPE_INVALID",
                    message="DAG node type must be Stochastic or Deterministic.",
                    path=f"dag.nodes.{node_name}.type",
                    details={"allowed_values": sorted(DAG_NODE_TYPES), "actual_value": node_type},
                )
            )

    for edge_name, edge_data in edges.items():
        if not isinstance(edge_name, str):
            errors.append(
                ValidationError(
                    code="DAG_EDGE_NAME_INVALID",
                    message="DAG edge names must be strings.",
                    path=f"dag.edges.{edge_name}",
                    details={"actual_type": type(edge_name).__name__},
                )
            )
            continue
        if not isinstance(edge_data, dict):
            errors.append(
                ValidationError(
                    code="DAG_EDGE_INVALID",
                    message="Each DAG edge must be an object.",
                    path=f"dag.edges.{edge_name}",
                    details={"actual_type": type(edge_data).__name__},
                )
            )
            continue
        missing_fields = sorted(DAG_EDGE_REQUIRED_FIELDS - edge_data.keys())
        extra_fields = sorted(set(edge_data.keys()) - DAG_EDGE_ALLOWED_FIELDS)
        if missing_fields:
            errors.append(
                ValidationError(
                    code="DAG_EDGE_MISSING_FIELD",
                    message="A DAG edge is missing required fields.",
                    path=f"dag.edges.{edge_name}",
                    details={"missing_fields": missing_fields},
                )
            )
        if extra_fields:
            errors.append(
                ValidationError(
                    code="DAG_EDGE_EXTRA_FIELD",
                    message="A DAG edge contains unknown fields.",
                    path=f"dag.edges.{edge_name}",
                    details={"unexpected_fields": extra_fields},
                )
            )
        parent = edge_data.get("parent")
        child = edge_data.get("child")
        if not isinstance(parent, str) or not isinstance(child, str):
            errors.append(
                ValidationError(
                    code="DAG_EDGE_ENDPOINT_INVALID",
                    message="DAG edge endpoints must be strings.",
                    path=f"dag.edges.{edge_name}",
                    details={"parent_type": type(parent).__name__, "child_type": type(child).__name__},
                )
            )

    return errors


def validate_distributions(distributions: dict[str, Any], variables: dict[str, dict[str, Any]], dag: dict[str, Any]) -> list[ValidationError]:
    errors: list[ValidationError] = []
    variable_names = set(variables.keys())
    distribution_names = set(distributions.keys())

    missing_variables = sorted(variable_names - distribution_names)
    extra_variables = sorted(distribution_names - variable_names)
    if missing_variables:
        errors.append(
            ValidationError(
                code="DISTRIBUTION_MISSING_VARIABLES",
                message="Distribution list is missing variables from the variable list.",
                path="distributions",
                details={"missing_variables": missing_variables},
            )
        )
    if extra_variables:
        errors.append(
            ValidationError(
                code="DISTRIBUTION_EXTRA_VARIABLES",
                message="Distribution list contains extra variables.",
                path="distributions",
                details={"extra_variables": extra_variables},
            )
        )

    dag_nodes = dag.get("nodes", {}) if isinstance(dag, dict) else {}

    for variable_name in sorted(variable_names & distribution_names):
        entry = distributions[variable_name]
        variable_data = variables.get(variable_name, {})
        dag_node = dag_nodes.get(variable_name, {}) if isinstance(dag_nodes, dict) else {}
        node_type = dag_node.get("type") if isinstance(dag_node, dict) else None
        is_deterministic = node_type == "Deterministic"

        if not isinstance(entry, dict):
            errors.append(
                ValidationError(
                    code="DISTRIBUTION_ENTRY_INVALID",
                    message="Each distribution entry must be an object.",
                    path=f"distributions.{variable_name}",
                    details={"actual_type": type(entry).__name__},
                )
            )
            continue

        distribution = entry.get("distribution")
        if distribution not in ALLOWED_DISTRIBUTIONS:
            errors.append(
                ValidationError(
                    code="DISTRIBUTION_TYPE_INVALID",
                    message="Distribution type is not allowed.",
                    path=f"distributions.{variable_name}.distribution",
                    details={"allowed_values": sorted(ALLOWED_DISTRIBUTIONS), "actual_value": distribution},
                )
            )
            continue

        variable_schema_type, schema_error = classify_variable_schema(variable_name, variable_data)
        if schema_error is not None:
            errors.append(
                ValidationError(
                    code="DISTRIBUTION_VARIABLE_SCHEMA_INVALID",
                    message="Variable schema is invalid for distribution validation.",
                    path=f"variables.{variable_name}",
                    details={"variable_error": error_to_dict(schema_error)},
                )
            )
            continue

        if variable_schema_type == "quantitative":
            classification = variable_data.get("classification")
            allowed_distributions = ALLOWED_DISTRIBUTIONS_BY_VARIABLE_SCHEMA["quantitative"].get(classification, set())
        else:
            allowed_distributions = ALLOWED_DISTRIBUTIONS_BY_VARIABLE_SCHEMA["categorical"]

        if distribution not in allowed_distributions:
            errors.append(
                ValidationError(
                    code="DISTRIBUTION_VARIABLE_SCHEMA_MISMATCH",
                    message="Selected distribution is not allowed for the variable schema.",
                    path=f"distributions.{variable_name}.distribution",
                    details={
                        "variable_schema_type": variable_schema_type,
                        "classification": variable_data.get("classification"),
                        "allowed_values": sorted(allowed_distributions),
                        "actual_value": distribution,
                    },
                )
            )

        if is_deterministic and distribution != "None":
            errors.append(
                ValidationError(
                    code="DISTRIBUTION_DETERMINISTIC_MISMATCH",
                    message="Deterministic DAG nodes must use distribution None.",
                    path=f"distributions.{variable_name}.distribution",
                    details={"dag_node_type": node_type, "actual_distribution": distribution},
                )
            )

        spec = VARIABLE_DISTRIBUTION_FIELDS[distribution]
        required_fields = spec["required"]
        allowed_fields = spec["required"]

        missing_fields = sorted(required_fields - entry.keys())
        extra_fields = sorted(set(entry.keys()) - allowed_fields)
        if missing_fields:
            errors.append(
                ValidationError(
                    code="DISTRIBUTION_MISSING_FIELD",
                    message="A distribution entry is missing required fields.",
                    path=f"distributions.{variable_name}",
                    details={"missing_fields": missing_fields},
                )
            )
        if extra_fields:
            errors.append(
                ValidationError(
                    code="DISTRIBUTION_EXTRA_FIELD",
                    message="A distribution entry contains unknown fields.",
                    path=f"distributions.{variable_name}",
                    details={"unexpected_fields": extra_fields},
                )
            )

        for field_name, expected_type in spec["types"].items():
            value = entry.get(field_name)
            if value is None:
                continue
            if expected_type is float:
                if not isinstance(value, (int, float)):
                    errors.append(
                        ValidationError(
                            code="DISTRIBUTION_FIELD_TYPE_INVALID",
                            message="A distribution field has the wrong type.",
                            path=f"distributions.{variable_name}.{field_name}",
                            details={"expected_type": "number", "actual_type": type(value).__name__},
                        )
                    )
            elif expected_type is int:
                if not isinstance(value, int):
                    errors.append(
                        ValidationError(
                            code="DISTRIBUTION_FIELD_TYPE_INVALID",
                            message="A distribution field has the wrong type.",
                            path=f"distributions.{variable_name}.{field_name}",
                            details={"expected_type": "int", "actual_type": type(value).__name__},
                        )
                    )
            elif expected_type is list:
                if not isinstance(value, list):
                    errors.append(
                        ValidationError(
                            code="DISTRIBUTION_FIELD_TYPE_INVALID",
                            message="A distribution field has the wrong type.",
                            path=f"distributions.{variable_name}.{field_name}",
                            details={"expected_type": "list", "actual_type": type(value).__name__},
                        )
                    )
            elif not isinstance(value, expected_type):
                errors.append(
                    ValidationError(
                        code="DISTRIBUTION_FIELD_TYPE_INVALID",
                        message="A distribution field has the wrong type.",
                        path=f"distributions.{variable_name}.{field_name}",
                        details={"expected_type": expected_type.__name__, "actual_type": type(value).__name__},
                    )
                )

        if distribution in {"Bernoulli", "Binomial", "Geometric"}:
            success_prob = entry.get("success_prob")
            if isinstance(success_prob, float) and not (0.0 <= success_prob <= 1.0):
                errors.append(
                    ValidationError(
                        code="DISTRIBUTION_FIELD_VALUE_INVALID",
                        message="success_prob must be in [0, 1].",
                        path=f"distributions.{variable_name}.success_prob",
                        details={"actual_value": success_prob},
                    )
                )
        if distribution == "Bernoulli":
            if entry.get("success_prob") is not None and type(entry.get("success_prob")) is float:
                pass
        if distribution == "Binomial":
            n_trials = entry.get("n_trials")
            if isinstance(n_trials, int) and n_trials < 1:
                errors.append(
                    ValidationError(
                        code="DISTRIBUTION_FIELD_VALUE_INVALID",
                        message="n_trials must be >= 1.",
                        path=f"distributions.{variable_name}.n_trials",
                        details={"actual_value": n_trials},
                    )
                )
        if distribution in {"Poisson", "Exponential", "Gamma"}:
            rate = entry.get("rate")
            if isinstance(rate, float) and rate <= 0.0:
                errors.append(
                    ValidationError(
                        code="DISTRIBUTION_FIELD_VALUE_INVALID",
                        message="rate must be positive.",
                        path=f"distributions.{variable_name}.rate",
                        details={"actual_value": rate},
                    )
                )
        if distribution == "None" and len(entry.keys()) != 1:
            errors.append(
                ValidationError(
                    code="DISTRIBUTION_NONE_EXTRA_FIELDS",
                    message="None distributions must not include extra fields.",
                    path=f"distributions.{variable_name}",
                    details={"fields": sorted(entry.keys())},
                )
            )

    return errors


def validate_distribution_pipeline(
    variables_path: str | Path,
    dag_path: str | Path,
    distributions_path: str | Path,
) -> ValidationReport:
    errors: list[ValidationError] = []

    variables, variable_errors = load_variables(variables_path)
    errors.extend(variable_errors)
    dag, dag_errors = load_dag(dag_path)
    errors.extend(dag_errors)

    distributions_file = Path(distributions_path)
    if not distributions_file.exists():
        return ValidationReport(ok=True, errors=errors)

    try:
        with distributions_file.open("r", encoding="utf-8") as f:
            distributions = json.load(f)
    except json.JSONDecodeError as exc:
        errors.append(
            ValidationError(
                code="DISTRIBUTIONS_JSON_INVALID",
                message="Distributions file is not valid JSON.",
                path=str(distributions_file),
                details={"line": exc.lineno, "column": exc.colno, "error": exc.msg},
            )
        )
        return ValidationReport(ok=False, errors=errors)

    if not isinstance(distributions, dict):
        errors.append(
            ValidationError(
                code="DISTRIBUTIONS_ROOT_INVALID",
                message="Distributions file must contain a JSON object at the top level.",
                path=str(distributions_file),
                details={"actual_type": type(distributions).__name__},
            )
        )
        return ValidationReport(ok=False, errors=errors)

    if variables is not None and dag is not None:
        errors.extend(validate_dag_structure(dag, set(variables.keys())))
        errors.extend(validate_distributions(distributions, variables, dag))

    return ValidationReport(ok=not errors, errors=errors)


def validate_variable_field_types(variable_name: str, variable_data: dict[str, Any], schema_type: str) -> ValidationError | None:
    if not isinstance(variable_name, str):
        return ValidationError(
            code="VARIABLE_NAME_TYPE_INVALID",
            message="Variable names must be strings.",
            path=variable_name,
            details={"actual_type": type(variable_name).__name__},
        )

    required_fields = QUANTITATIVE_REQUIRED_FIELDS if schema_type == "quantitative" else CATEGORICAL_REQUIRED_FIELDS
    optional_fields = {"classification", "skew"} if schema_type == "quantitative" else set()

    for field_name in required_fields | optional_fields:
        if field_name not in variable_data:
            continue
        value = variable_data[field_name]
        if not isinstance(value, str):
            return ValidationError(
                code="VARIABLE_FIELD_TYPE_INVALID",
                message="A variable field has an invalid type.",
                path=f"{variable_name}.{field_name}",
                details={"expected_type": "str", "actual_type": type(value).__name__},
            )

    return None


def validate_variable_uniqueness(data: dict[str, Any]) -> ValidationError | None:
    seen_names: set[str] = set()
    for variable_name in data.keys():
        if variable_name in seen_names:
            return ValidationError(
                code="VARIABLE_DUPLICATE_NAME",
                message="Duplicate variable name detected.",
                path=variable_name,
                details={"duplicate_name": variable_name},
            )
        seen_names.add(variable_name)

    return None


def normalize_node_type(node_type: Any) -> str:
    return node_type.strip() if isinstance(node_type, str) else ""


def normalize_formula_type(formula: dict[str, Any]) -> str:
    if "intercept" in formula and "predictors" in formula:
        return "quantitative"
    if "reference_category" in formula and "other_categories" in formula and "predictors" in formula and "intercept" not in formula:
        return "categorical ordinal"
    if "reference_category" in formula and "other_categories" in formula:
        return "categorical nominal"
    return ""


def get_dag_parent_map(dag_data: dict[str, Any]) -> dict[str, set[str]]:
    parent_map: dict[str, set[str]] = {node_name: set() for node_name in dag_data.get("nodes", {}) if isinstance(node_name, str)}
    edges = dag_data.get("edges", {}) if isinstance(dag_data.get("edges"), dict) else {}
    for edge_data in edges.values():
        if not isinstance(edge_data, dict):
            continue
        parent = edge_data.get("parent")
        child = edge_data.get("child")
        if isinstance(parent, str) and isinstance(child, str):
            parent_map.setdefault(child, set()).add(parent)
    return parent_map


def validate_formula_root_schema(formula_name: str, formula_data: Any) -> ValidationError | None:
    if not isinstance(formula_data, dict):
        return ValidationError(
            code="FORMULA_SCHEMA_INVALID",
            message="Each formula entry must be an object.",
            path=formula_name,
            details={"actual_type": type(formula_data).__name__},
        )

    formula_type = normalize_formula_type(formula_data)
    if formula_type == "":
        return ValidationError(
            code="FORMULA_TYPE_UNSUPPORTED",
            message="Formula does not match a supported schema.",
            path=formula_name,
            details={"keys": sorted(formula_data.keys())},
        )

    return None


def validate_formula_values(formula_name: str, formula_data: dict[str, Any], formula_type: str) -> list[ValidationError]:
    errors: list[ValidationError] = []

    if formula_type == "quantitative":
        intercept = formula_data.get("intercept")
        if not isinstance(intercept, (int, float)):
            errors.append(ValidationError(code="FORMULA_FIELD_TYPE_INVALID", message="Quantitative intercept must be numeric.", path=f"{formula_name}.intercept", details={"expected_type": "number", "actual_type": type(intercept).__name__}))
        snr = formula_data.get("snr")
        if not (isinstance(snr, (int, float)) or is_nan(snr)):
            errors.append(ValidationError(code="FORMULA_FIELD_TYPE_INVALID", message="Quantitative SNR must be numeric or NaN.", path=f"{formula_name}.snr", details={"expected_type": "number", "actual_type": type(snr).__name__}))
        predictors = formula_data.get("predictors")
        if not isinstance(predictors, dict):
            errors.append(ValidationError(code="FORMULA_FIELD_TYPE_INVALID", message="Quantitative predictors must be an object.", path=f"{formula_name}.predictors", details={"expected_type": "object", "actual_type": type(predictors).__name__}))
        else:
            for predictor_name, predictor_data in predictors.items():
                if not isinstance(predictor_data, dict):
                    errors.append(ValidationError(code="FORMULA_PREDICTOR_INVALID", message="Each predictor must be an object.", path=f"{formula_name}.predictors.{predictor_name}", details={"actual_type": type(predictor_data).__name__}))
                    continue
                if "reference_category" in predictor_data:
                    reference_category = predictor_data.get("reference_category")
                    if not isinstance(reference_category, str) or not reference_category.strip():
                        errors.append(ValidationError(code="FORMULA_FIELD_EMPTY", message="Categorical predictor reference_category must be a non-empty string.", path=f"{formula_name}.predictors.{predictor_name}.reference_category"))
                    other_categories = predictor_data.get("other_categories")
                    if not isinstance(other_categories, dict):
                        errors.append(ValidationError(code="FORMULA_FIELD_TYPE_INVALID", message="Categorical predictor other_categories must be an object.", path=f"{formula_name}.predictors.{predictor_name}.other_categories", details={"expected_type": "object", "actual_type": type(other_categories).__name__}))
                    else:
                        for category_name, category_data in other_categories.items():
                            if not isinstance(category_data, dict):
                                errors.append(ValidationError(code="FORMULA_CATEGORY_INVALID", message="Each categorical predictor category must be an object.", path=f"{formula_name}.predictors.{predictor_name}.other_categories.{category_name}", details={"actual_type": type(category_data).__name__}))
                                continue
                            coefficient = category_data.get("coefficient")
                            if not isinstance(coefficient, (int, float)):
                                errors.append(ValidationError(code="FORMULA_FIELD_TYPE_INVALID", message="Categorical predictor category coefficient must be numeric.", path=f"{formula_name}.predictors.{predictor_name}.other_categories.{category_name}.coefficient", details={"expected_type": "number", "actual_type": type(coefficient).__name__}))
                else:
                    coefficient = predictor_data.get("coefficient")
                    if not isinstance(coefficient, (int, float)):
                        errors.append(ValidationError(code="FORMULA_FIELD_TYPE_INVALID", message="Predictor coefficient must be numeric.", path=f"{formula_name}.predictors.{predictor_name}.coefficient", details={"expected_type": "number", "actual_type": type(coefficient).__name__}))
                    transformation = predictor_data.get("transformation")
                    if transformation not in ALLOWED_FORMULA_TRANSFORMATIONS:
                        errors.append(ValidationError(code="FORMULA_FIELD_INVALID_VALUE", message="Predictor transformation is invalid.", path=f"{formula_name}.predictors.{predictor_name}.transformation", details={"allowed_values": sorted(ALLOWED_FORMULA_TRANSFORMATIONS), "actual_value": transformation}))

    elif formula_type == "categorical nominal":
        reference_category = formula_data.get("reference_category")
        if not isinstance(reference_category, str) or not reference_category.strip():
            errors.append(ValidationError(code="FORMULA_FIELD_EMPTY", message="Nominal reference_category must be a non-empty string.", path=f"{formula_name}.reference_category"))
        other_categories = formula_data.get("other_categories")
        if not isinstance(other_categories, dict):
            errors.append(ValidationError(code="FORMULA_FIELD_TYPE_INVALID", message="Nominal other_categories must be an object.", path=f"{formula_name}.other_categories", details={"expected_type": "object", "actual_type": type(other_categories).__name__}))
        else:
            for category_name, category_data in other_categories.items():
                if not isinstance(category_data, dict):
                    errors.append(ValidationError(code="FORMULA_CATEGORY_INVALID", message="Each nominal category block must be an object.", path=f"{formula_name}.other_categories.{category_name}", details={"actual_type": type(category_data).__name__}))
                    continue
                intercept = category_data.get("intercept")
                if not isinstance(intercept, (int, float)):
                    errors.append(ValidationError(code="FORMULA_FIELD_TYPE_INVALID", message="Nominal category intercept must be numeric.", path=f"{formula_name}.other_categories.{category_name}.intercept", details={"expected_type": "number", "actual_type": type(intercept).__name__}))
                predictors = category_data.get("predictors")
                if predictors is not None:
                    if not isinstance(predictors, dict):
                        errors.append(ValidationError(code="FORMULA_FIELD_TYPE_INVALID", message="Nominal category predictors must be an object when present.", path=f"{formula_name}.other_categories.{category_name}.predictors", details={"expected_type": "object", "actual_type": type(predictors).__name__}))

    elif formula_type == "categorical ordinal":
        reference_category = formula_data.get("reference_category")
        if not isinstance(reference_category, str) or not reference_category.strip():
            errors.append(ValidationError(code="FORMULA_FIELD_EMPTY", message="Ordinal reference_category must be a non-empty string.", path=f"{formula_name}.reference_category"))
        predictors = formula_data.get("predictors")
        if not isinstance(predictors, dict):
            errors.append(ValidationError(code="FORMULA_FIELD_TYPE_INVALID", message="Ordinal predictors must be an object.", path=f"{formula_name}.predictors", details={"expected_type": "object", "actual_type": type(predictors).__name__}))
        other_categories = formula_data.get("other_categories")
        if not isinstance(other_categories, dict):
            errors.append(ValidationError(code="FORMULA_FIELD_TYPE_INVALID", message="Ordinal other_categories must be an object.", path=f"{formula_name}.other_categories", details={"expected_type": "object", "actual_type": type(other_categories).__name__}))
        else:
            for category_name, category_data in other_categories.items():
                if not isinstance(category_data, dict) or not isinstance(category_data.get("intercept"), (int, float, str)):
                    errors.append(ValidationError(code="FORMULA_FIELD_TYPE_INVALID", message="Ordinal thresholds must provide an intercept/threshold value.", path=f"{formula_name}.other_categories.{category_name}.intercept", details={"expected_type": "string_or_number", "actual_type": type(category_data).__name__ if not isinstance(category_data, dict) else type(category_data.get('intercept')).__name__}))

    return errors


def collect_formula_predictors(formula_data: dict[str, Any], formula_type: str) -> set[str]:
    predictors: set[str] = set()
    if formula_type == "quantitative":
        for predictor_name, predictor_data in formula_data.get("predictors", {}).items():
            predictors.add(predictor_name)
            if isinstance(predictor_data, dict) and isinstance(predictor_data.get("other_categories"), dict):
                predictors.update(predictor_data["other_categories"].keys())
    elif formula_type == "categorical nominal":
        for category_data in formula_data.get("other_categories", {}).values():
            if isinstance(category_data, dict):
                for predictor_name, predictor_data in category_data.get("predictors", {}).items():
                    predictors.add(predictor_name)
                    if isinstance(predictor_data, dict) and isinstance(predictor_data.get("other_categories"), dict):
                        predictors.update(predictor_data["other_categories"].keys())
    elif formula_type == "categorical ordinal":
        for predictor_name, predictor_data in formula_data.get("predictors", {}).items():
            predictors.add(predictor_name)
            if isinstance(predictor_data, dict) and isinstance(predictor_data.get("other_categories"), dict):
                predictors.update(predictor_data["other_categories"].keys())
        for category_data in formula_data.get("other_categories", {}).values():
            if isinstance(category_data, dict):
                for predictor_name in category_data.get("predictors", {}).keys():
                    predictors.add(predictor_name)
    return predictors


def validate_formulas_data(formulas_data: dict[str, Any], variables_data: dict[str, Any], dag_data: dict[str, Any], distributions_data: dict[str, Any] | None = None) -> list[ValidationError]:
    errors: list[ValidationError] = []
    parent_map = get_dag_parent_map(dag_data)
    variable_names = set(variables_data.keys())

    missing_formulas = sorted(name for name in variable_names if parent_map.get(name) and name not in formulas_data)
    for variable_name in missing_formulas:
        errors.append(ValidationError(code="FORMULA_MISSING", message="Variables with DAG parents must have formulas.", path=variable_name))

    extra_formulas = sorted(name for name in formulas_data.keys() if not parent_map.get(name))
    for variable_name in extra_formulas:
        errors.append(ValidationError(code="FORMULA_UNEXPECTED", message="Variables without DAG parents must not have formulas.", path=variable_name))

    for variable_name, formula_data in formulas_data.items():
        if variable_name not in variable_names:
            continue

        root_error = validate_formula_root_schema(variable_name, formula_data)
        if root_error is not None:
            errors.append(root_error)
            continue

        assert isinstance(formula_data, dict)
        formula_type = normalize_formula_type(formula_data)
        variable_info = variables_data.get(variable_name, {})
        measurement_level = variable_info.get("measurement_level")
        if measurement_level in ALLOWED_QUANTITATIVE_MEASUREMENT_LEVELS:
            variable_schema_type = "quantitative"
        elif measurement_level == "Nominal":
            variable_schema_type = "categorical nominal"
        elif measurement_level == "Ordinal":
            variable_schema_type = "categorical ordinal"
        else:
            variable_schema_type = "categorical nominal"

        if formula_type != variable_schema_type:
            errors.append(ValidationError(code="FORMULA_TYPE_MISMATCH", message="Formula type does not match the variable definition.", path=variable_name, details={"variable_type": variable_schema_type, "formula_type": formula_type}))

        errors.extend(validate_formula_values(variable_name, formula_data, formula_type))

        if formula_type == "quantitative":
            snr = formula_data.get("snr")
            dag_node = dag_data.get("nodes", {}).get(variable_name, {}) if isinstance(dag_data.get("nodes"), dict) else {}
            if isinstance(dag_node, dict) and normalize_node_type(dag_node.get("type")) == "Deterministic" and not is_nan(snr):
                errors.append(ValidationError(code="FORMULA_DETERMINISTIC_SNR_INVALID", message="Deterministic node formulas must have snr set to NaN.", path=f"{variable_name}.snr", details={"actual_value": snr}))

        expected_parents = parent_map.get(variable_name, set())
        actual_predictors = collect_formula_predictors(formula_data, formula_type)
        missing_parents = sorted(expected_parents - actual_predictors)
        if missing_parents:
            errors.append(ValidationError(code="FORMULA_MISSING_PARENTS", message="All DAG parents must appear in the formula predictors.", path=variable_name, details={"missing_parents": missing_parents}))

    return errors


def validate_dag_schema(dag_data: dict[str, Any]) -> list[ValidationError]:
    errors: list[ValidationError] = []

    nodes = dag_data.get("nodes")
    if not isinstance(nodes, dict):
        errors.append(
            ValidationError(
                code="DAG_NODES_INVALID",
                message='DAG must contain a "nodes" object.',
                path="nodes",
                details={"actual_type": type(nodes).__name__},
            )
        )
        nodes = {}

    edges = dag_data.get("edges")
    if not isinstance(edges, dict):
        errors.append(
            ValidationError(
                code="DAG_EDGES_INVALID",
                message='DAG must contain an "edges" object.',
                path="edges",
                details={"actual_type": type(edges).__name__},
            )
        )
        edges = {}

    for node_name, node_data in nodes.items():
        if not isinstance(node_name, str):
            errors.append(
                ValidationError(
                    code="DAG_NODE_NAME_INVALID",
                    message="Node names must be strings.",
                    path=str(node_name),
                    details={"actual_type": type(node_name).__name__},
                )
            )
            continue

        if not isinstance(node_data, dict):
            errors.append(
                ValidationError(
                    code="DAG_NODE_SCHEMA_INVALID",
                    message="Each node entry must be an object.",
                    path=f"nodes.{node_name}",
                    details={"actual_type": type(node_data).__name__},
                )
            )
            continue

        missing_fields = sorted(DAG_NODE_REQUIRED_FIELDS - node_data.keys())
        if missing_fields:
            errors.append(
                ValidationError(
                    code="DAG_NODE_MISSING_FIELD",
                    message="A required DAG node field is missing.",
                    path=f"nodes.{node_name}",
                    details={"missing_fields": missing_fields},
                )
            )

        unexpected_fields = sorted(set(node_data.keys()) - DAG_NODE_ALLOWED_FIELDS)
        if unexpected_fields:
            errors.append(
                ValidationError(
                    code="DAG_NODE_UNKNOWN_FIELD",
                    message="A DAG node contains unknown fields.",
                    path=f"nodes.{node_name}",
                    details={"unexpected_fields": unexpected_fields},
                )
            )

        node_type = node_data.get("type")
        if not isinstance(node_type, str) or not node_type.strip():
            errors.append(
                ValidationError(
                    code="DAG_NODE_FIELD_EMPTY",
                    message="A DAG node type must be a non-empty string.",
                    path=f"nodes.{node_name}.type",
                )
            )
        elif normalize_node_type(node_type) not in DAG_NODE_TYPES:
            errors.append(
                ValidationError(
                    code="DAG_NODE_FIELD_INVALID_VALUE",
                    message="A DAG node type must be Stochastic or Deterministic.",
                    path=f"nodes.{node_name}.type",
                    details={"allowed_values": ["Stochastic", "Deterministic"], "actual_value": node_type},
                )
            )

    for edge_name, edge_data in edges.items():
        if not isinstance(edge_name, str):
            errors.append(
                ValidationError(
                    code="DAG_EDGE_NAME_INVALID",
                    message="Edge names must be strings.",
                    path=str(edge_name),
                    details={"actual_type": type(edge_name).__name__},
                )
            )
            continue

        if not isinstance(edge_data, dict):
            errors.append(
                ValidationError(
                    code="DAG_EDGE_SCHEMA_INVALID",
                    message="Each edge entry must be an object.",
                    path=f"edges.{edge_name}",
                    details={"actual_type": type(edge_data).__name__},
                )
            )
            continue

        missing_fields = sorted(DAG_EDGE_REQUIRED_FIELDS - edge_data.keys())
        if missing_fields:
            errors.append(
                ValidationError(
                    code="DAG_EDGE_MISSING_FIELD",
                    message="A required DAG edge field is missing.",
                    path=f"edges.{edge_name}",
                    details={"missing_fields": missing_fields},
                )
            )

        unexpected_fields = sorted(set(edge_data.keys()) - DAG_EDGE_ALLOWED_FIELDS)
        if unexpected_fields:
            errors.append(
                ValidationError(
                    code="DAG_EDGE_UNKNOWN_FIELD",
                    message="A DAG edge contains unknown fields.",
                    path=f"edges.{edge_name}",
                    details={"unexpected_fields": unexpected_fields},
                )
            )

        parent = edge_data.get("parent")
        child = edge_data.get("child")
        if not isinstance(parent, str) or not parent.strip():
            errors.append(
                ValidationError(
                    code="DAG_EDGE_FIELD_EMPTY",
                    message="A DAG edge parent must be a non-empty string.",
                    path=f"edges.{edge_name}.parent",
                )
            )
        if not isinstance(child, str) or not child.strip():
            errors.append(
                ValidationError(
                    code="DAG_EDGE_FIELD_EMPTY",
                    message="A DAG edge child must be a non-empty string.",
                    path=f"edges.{edge_name}.child",
                )
            )

    return errors


def validate_dag_against_variables(
    dag_data: dict[str, Any], variables_data: dict[str, Any]
) -> list[ValidationError]:
    errors: list[ValidationError] = []

    nodes = dag_data.get("nodes") if isinstance(dag_data.get("nodes"), dict) else {}
    edges = dag_data.get("edges") if isinstance(dag_data.get("edges"), dict) else {}

    variable_names = set(variables_data.keys())
    node_names = set(nodes.keys())

    missing_nodes = sorted(variable_names - node_names)
    for variable_name in missing_nodes:
        errors.append(
            ValidationError(
                code="DAG_MISSING_VARIABLE_NODE",
                message="Every variable must appear as a DAG node.",
                path=f"nodes.{variable_name}",
                details={"variable_name": variable_name},
            )
        )

    extra_nodes = sorted(node_names - variable_names)
    for node_name in extra_nodes:
        errors.append(
            ValidationError(
                code="DAG_UNKNOWN_NODE",
                message="DAG contains a node that is not in the variable list.",
                path=f"nodes.{node_name}",
                details={"node_name": node_name},
            )
        )

    graph: dict[str, set[str]] = {node_name: set() for node_name in node_names}

    for edge_name, edge_data in edges.items():
        if not isinstance(edge_data, dict):
            continue

        parent = edge_data.get("parent")
        child = edge_data.get("child")
        if not isinstance(parent, str) or not isinstance(child, str):
            continue

        if parent not in node_names:
            errors.append(
                ValidationError(
                    code="DAG_EDGE_UNKNOWN_PARENT",
                    message="DAG edge parent must reference a known node.",
                    path=f"edges.{edge_name}.parent",
                    details={"parent": parent},
                )
            )
            continue

        if child not in node_names:
            errors.append(
                ValidationError(
                    code="DAG_EDGE_UNKNOWN_CHILD",
                    message="DAG edge child must reference a known node.",
                    path=f"edges.{edge_name}.child",
                    details={"child": child},
                )
            )
            continue

        graph[parent].add(child)

    visited: set[str] = set()
    in_stack: set[str] = set()

    def visit(node: str) -> bool:
        if node in in_stack:
            return True
        if node in visited:
            return False

        visited.add(node)
        in_stack.add(node)
        for child in graph.get(node, set()):
            if visit(child):
                return True
        in_stack.remove(node)
        return False

    for node_name in sorted(node_names):
        if visit(node_name):
            errors.append(
                ValidationError(
                    code="DAG_CYCLE_DETECTED",
                    message="The DAG contains a cycle.",
                    path="edges",
                )
            )
            break

    return errors


def validate_variables_data(data: dict[str, Any]) -> list[ValidationError]:
    errors: list[ValidationError] = []

    uniqueness_error = validate_variable_uniqueness(data)
    if uniqueness_error is not None:
        errors.append(uniqueness_error)

    for variable_name, variable_data in data.items():
        if not isinstance(variable_name, str):
            errors.append(
                ValidationError(
                    code="VARIABLE_NAME_TYPE_INVALID",
                    message="Variable names must be strings.",
                    path=str(variable_name),
                    details={"actual_type": type(variable_name).__name__},
                )
            )
            continue

        if not isinstance(variable_data, dict):
            errors.append(
                ValidationError(
                    code="VARIABLE_SCHEMA_INVALID",
                    message="Each variable entry must be an object.",
                    path=variable_name,
                    details={"actual_type": type(variable_data).__name__},
                )
            )
            continue

        schema_type, schema_error = classify_variable_schema(variable_name, variable_data)
        if schema_error is not None or schema_type is None:
            if schema_error is not None:
                errors.append(schema_error)
            continue

        schema_validator = (
            validate_quantitative_variable_schema_fields if schema_type == "quantitative" else validate_categorical_variable_schema_fields
        )
        error = schema_validator(variable_name, variable_data)
        if error is not None:
            errors.append(error)
            continue

        error = validate_variable_field_values(variable_name, variable_data, schema_type)
        if error is not None:
            errors.append(error)
            continue

        error = validate_variable_field_types(variable_name, variable_data, schema_type)
        if error is not None:
            errors.append(error)

        schema_consistency_error = validate_variable_schema_consistency(variable_name, variable_data)
        if schema_consistency_error is not None:
            errors.append(schema_consistency_error)

    return errors


def validate_variables_file(path: str | Path) -> list[ValidationError]:
    data, errors = load_variables(path)
    if errors:
        return errors

    assert data is not None
    return validate_variables_data(data)


def validate_dag_file(path: str | Path) -> list[ValidationError]:
    data, errors = load_dag(path)
    if errors:
        return errors

    assert data is not None
    errors.extend(validate_dag_schema(data))
    return errors


def validate_all_reports(variables_path: str | Path, dag_path: str | Path) -> ValidationReport:
    errors: list[ValidationError] = []

    variables_data, variables_load_errors = load_variables(variables_path)
    errors.extend(variables_load_errors)

    dag_data, dag_load_errors = load_dag(dag_path)
    errors.extend(dag_load_errors)

    formulas_path = Path(variables_path).resolve().parent / "formulas.json"
    formulas_data, formulas_load_errors = load_formulas(formulas_path)
    errors.extend(formulas_load_errors)

    if variables_data is not None:
        errors.extend(validate_variables_data(variables_data))

    if variables_data is not None and dag_data is not None:
        errors.extend(validate_dag_schema(dag_data))
        errors.extend(validate_dag_against_variables(dag_data, variables_data))
        distributions_path = Path(variables_path).resolve().parent / "distributions.json"
        if distributions_path.exists():
            try:
                with distributions_path.open("r", encoding="utf-8") as f:
                    distributions = json.load(f)
            except json.JSONDecodeError as exc:
                errors.append(
                    ValidationError(
                        code="DISTRIBUTIONS_JSON_INVALID",
                        message="Distributions file is not valid JSON.",
                        path=str(distributions_path),
                        details={"line": exc.lineno, "column": exc.colno, "error": exc.msg},
                    )
                )
            else:
                if isinstance(distributions, dict):
                    errors.extend(validate_distributions(distributions, variables_data, dag_data))
                else:
                    errors.append(
                        ValidationError(
                            code="DISTRIBUTIONS_ROOT_INVALID",
                            message="Distributions file must contain a JSON object at the top level.",
                            path=str(distributions_path),
                            details={"actual_type": type(distributions).__name__},
                        )
                    )

        if formulas_data is not None:
            errors.extend(validate_formulas_data(formulas_data, variables_data, dag_data, distributions if 'distributions' in locals() and isinstance(distributions, dict) else None))

    return ValidationReport(ok=not errors, errors=errors)


def validate_variables_report(path: str | Path) -> ValidationReport:
    errors = validate_variables_file(path)
    return ValidationReport(ok=not errors, errors=errors)


def validate_dag_report(path: str | Path) -> ValidationReport:
    errors = validate_dag_file(path)
    return ValidationReport(ok=not errors, errors=errors)


def validate_all_report(variables_path: str | Path, dag_path: str | Path) -> ValidationReport:
    return validate_all_reports(variables_path, dag_path)


def validate_distributions_report(variables_path: str | Path, dag_path: str | Path, distributions_path: str | Path) -> ValidationReport:
    errors: list[ValidationError] = []

    variables_data, variables_load_errors = load_variables(variables_path)
    errors.extend(variables_load_errors)

    dag_data, dag_load_errors = load_dag(dag_path)
    errors.extend(dag_load_errors)

    distributions_file = Path(distributions_path)
    if not distributions_file.exists():
        return ValidationReport(ok=True, errors=errors)

    try:
        with distributions_file.open("r", encoding="utf-8") as f:
            distributions = json.load(f)
    except json.JSONDecodeError as exc:
        errors.append(
            ValidationError(
                code="DISTRIBUTIONS_JSON_INVALID",
                message="Distributions file is not valid JSON.",
                path=str(distributions_file),
                details={"line": exc.lineno, "column": exc.colno, "error": exc.msg},
            )
        )
        return ValidationReport(ok=False, errors=errors)

    if not isinstance(distributions, dict):
        errors.append(
            ValidationError(
                code="DISTRIBUTIONS_ROOT_INVALID",
                message="Distributions file must contain a JSON object at the top level.",
                path=str(distributions_file),
                details={"actual_type": type(distributions).__name__},
            )
        )
        return ValidationReport(ok=False, errors=errors)

    if variables_data is not None and dag_data is not None:
        errors.extend(validate_distributions(distributions, variables_data, dag_data))

    return ValidationReport(ok=not errors, errors=errors)


def validate_formulas_report(variables_path: str | Path, dag_path: str | Path, formulas_path: str | Path) -> ValidationReport:
    errors: list[ValidationError] = []

    variables_data, variables_load_errors = load_variables(variables_path)
    errors.extend(variables_load_errors)

    dag_data, dag_load_errors = load_dag(dag_path)
    errors.extend(dag_load_errors)

    formulas_file = Path(formulas_path)
    if not formulas_file.exists():
        return ValidationReport(ok=True, errors=errors)

    formulas_data, formulas_load_errors = load_formulas(formulas_file)
    errors.extend(formulas_load_errors)

    if variables_data is not None and dag_data is not None and formulas_data is not None:
        errors.extend(validate_formulas_data(formulas_data, variables_data, dag_data))

    return ValidationReport(ok=not errors, errors=errors)


def validate_variables_report_json(path: str | Path) -> str:
    return json.dumps(report_to_dict(validate_variables_report(path)), indent=2)


def validate_all_report_json(variables_path: str | Path, dag_path: str | Path) -> str:
    return json.dumps(report_to_dict(validate_all_reports(variables_path, dag_path)), indent=2)


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    variables_path = repo_root / "synthdata" / "variables.json"
    dag_path = repo_root / "synthdata" / "dag.json"

    report = validate_all_reports(variables_path, dag_path)
    print(json.dumps(report_to_dict(report), indent=2))
    raise SystemExit(0 if report.ok else 1)
