from dataclasses import dataclass
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


def validate_variable_names(variables_data: dict[str, Any]) -> ValidationReport:
    for variable_name in variables_data:
        if not _is_snake_case(variable_name):
            return ValidationReport(
                ok=False,
                errors=[
                    ValidationError(
                        code="invalid_variable_name",
                        message="variable names need to be snake_case_name",
                        path=variable_name,
                    )
                ],
            )

    return ValidationReport(ok=True, errors=[])


def validate_variable_data_types(variables_data: dict[str, Any]) -> ValidationReport:
    errors: list[ValidationError] = []

    for variable_name, variable_data in variables_data.items():
        if not isinstance(variable_data, dict) or "data_type" not in variable_data:
            errors.append(
                ValidationError(
                    code="missing_data_type",
                    message="data_type is missing",
                    path=variable_name,
                )
            )

    return ValidationReport(ok=not errors, errors=errors)


def validate_variable_data_type_values(variables_data: dict[str, Any]) -> ValidationReport:
    errors: list[ValidationError] = []

    for variable_name, variable_data in variables_data.items():
        data_type = variable_data.get("data_type") if isinstance(variable_data, dict) else None
        if data_type not in {"Quantitative", "Categorical"}:
            errors.append(
                ValidationError(
                    code="invalid_data_type",
                    message='data_type must be "Quantitative" or "Categorical"',
                    path=variable_name,
                    details={"data_type": data_type},
                )
            )

    return ValidationReport(ok=not errors, errors=errors)


def validate_variable_required_fields(variables_data: dict[str, Any]) -> ValidationReport:
    errors: list[ValidationError] = []

    quantitative_required_fields = ["description", "justification", "measurement_level", "classification", "skew"]
    categorical_required_fields = ["description", "justification", "measurement_level"]

    for variable_name, variable_data in variables_data.items():
        if not isinstance(variable_data, dict):
            continue

        data_type = variable_data.get("data_type")
        if data_type == "Quantitative":
            missing_fields = [field for field in quantitative_required_fields if field not in variable_data]
        elif data_type == "Categorical":
            missing_fields = [field for field in categorical_required_fields if field not in variable_data]
        else:
            continue

        if missing_fields:
            errors.append(
                ValidationError(
                    code="missing_required_fields",
                    message="required fields are missing",
                    path=variable_name,
                    details={"missing_fields": missing_fields},
                )
            )

    return ValidationReport(ok=not errors, errors=errors)


def validate_variable_field_values_are_strings(variables_data: dict[str, Any]) -> ValidationReport:
    errors: list[ValidationError] = []

    for variable_name, variable_data in variables_data.items():
        if not isinstance(variable_data, dict):
            continue

        for field_name, field_value in variable_data.items():
            if not isinstance(field_value, str):
                errors.append(
                    ValidationError(
                        code="non_string_field_value",
                        message="field values must be strings",
                        path=f"{variable_name}.{field_name}",
                        details={"value": field_value, "type": type(field_value).__name__},
                    )
                )

    return ValidationReport(ok=not errors, errors=errors)


def validate_variable_field_value_constraints(variables_data: dict[str, Any]) -> ValidationReport:
    errors: list[ValidationError] = []

    allowed_values_by_data_type = {
        "Quantitative": {
            "measurement_level": {"Interval", "Ratio"},
            "classification": {"Discrete", "Continuous"},
            "skew": {"Left", "Right", "None"},
        },
        "Categorical": {
            "measurement_level": {"Nominal", "Ordinal"},
        },
    }

    for variable_name, variable_data in variables_data.items():
        if not isinstance(variable_data, dict):
            continue

        data_type = variable_data.get("data_type")
        field_constraints = allowed_values_by_data_type.get(data_type)
        if not field_constraints:
            continue

        for field_name, allowed_values in field_constraints.items():
            field_value = variable_data.get(field_name)
            if field_value not in allowed_values:
                errors.append(
                    ValidationError(
                        code="invalid_field_value",
                        message="invalid value for field",
                        path=f"{variable_name}.{field_name}",
                        details={"value": field_value, "allowed_values": sorted(allowed_values)},
                    )
                )

    return ValidationReport(ok=not errors, errors=errors)


def validate_variable_additional_fields(variables_data: dict[str, Any]) -> ValidationReport:
    errors: list[ValidationError] = []

    allowed_fields_by_data_type = {
        "Quantitative": {"data_type", "description", "justification", "measurement_level", "classification", "skew"},
        "Categorical": {"data_type", "description", "justification", "measurement_level"},
    }

    for variable_name, variable_data in variables_data.items():
        if not isinstance(variable_data, dict):
            continue

        data_type = variable_data.get("data_type")
        allowed_fields = allowed_fields_by_data_type.get(data_type)
        if not allowed_fields:
            continue

        additional_fields = sorted(field for field in variable_data if field not in allowed_fields)
        if additional_fields:
            errors.append(
                ValidationError(
                    code="additional_fields_present",
                    message="unexpected additional fields found",
                    path=variable_name,
                    details={"additional_fields": additional_fields},
                )
            )

    return ValidationReport(ok=not errors, errors=errors)


def _is_snake_case(value: str) -> bool:
    return value.islower() and value.replace("_", "").isalnum() and value[0].islower() and " " not in value and "-" not in value
