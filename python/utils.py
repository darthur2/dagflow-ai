from __future__ import annotations

from typing import Any


ALLOWED_FORMULA_TRANSFORMATIONS = {
    "none",
    "exp",
    "log",
    "sqrt",
    "inverse",
    "square",
    "cubic",
    "quartic",
    "sin",
    "cos",
}


def _extract_predictor_beta_1(predictor_name: str, predictor_data: dict[str, Any]) -> list[float]:
    if "reference_category" in predictor_data:
        other_categories = predictor_data.get("other_categories")
        if not isinstance(other_categories, dict):
            raise ValueError(f"Categorical predictor '{predictor_name}' requires other_categories as an object")

        beta_1: list[float] = []
        for category_name, category_data in other_categories.items():
            if not isinstance(category_data, dict):
                raise ValueError(f"Categorical predictor '{predictor_name}' category '{category_name}' must be an object")
            coefficient = category_data.get("coefficient")
            if not isinstance(coefficient, (int, float)):
                raise ValueError(
                    f"Categorical predictor '{predictor_name}' category '{category_name}' requires numeric coefficient"
                )
            beta_1.append(float(coefficient))
        return beta_1

    coefficient = predictor_data.get("coefficient")
    if not isinstance(coefficient, (int, float)):
        raise ValueError(f"Predictor '{predictor_name}' requires numeric coefficient")

    transformation = predictor_data.get("transformation", "none")
    if transformation not in ALLOWED_FORMULA_TRANSFORMATIONS:
        raise ValueError(f"Predictor '{predictor_name}' has invalid transformation '{transformation}'")

    return [float(coefficient)]


def _extract_predictor_names(predictor_name: str, predictor_data: dict[str, Any]) -> list[str]:
    if "reference_category" in predictor_data:
        other_categories = predictor_data.get("other_categories")
        if not isinstance(other_categories, dict):
            raise ValueError(f"Categorical predictor '{predictor_name}' requires other_categories as an object")

        names: list[str] = []
        for category_name, category_data in other_categories.items():
            if not isinstance(category_data, dict):
                raise ValueError(f"Categorical predictor '{predictor_name}' category '{category_name}' must be an object")
            coefficient = category_data.get("coefficient")
            if not isinstance(coefficient, (int, float)):
                raise ValueError(
                    f"Categorical predictor '{predictor_name}' category '{category_name}' requires numeric coefficient"
                )
            names.append(predictor_name)
        return names

    coefficient = predictor_data.get("coefficient")
    if not isinstance(coefficient, (int, float)):
        raise ValueError(f"Predictor '{predictor_name}' requires numeric coefficient")

    transformation = predictor_data.get("transformation", "none")
    if transformation not in ALLOWED_FORMULA_TRANSFORMATIONS:
        raise ValueError(f"Predictor '{predictor_name}' has invalid transformation '{transformation}'")

    return [predictor_name]


def _extract_predictor_transformation(predictor_name: str, predictor_data: dict[str, Any]) -> list[str]:
    if "reference_category" in predictor_data:
        other_categories = predictor_data.get("other_categories")
        if not isinstance(other_categories, dict):
            raise ValueError(f"Categorical predictor '{predictor_name}' requires other_categories as an object")

        transformations: list[str] = []
        for category_name, category_data in other_categories.items():
            if not isinstance(category_data, dict):
                raise ValueError(f"Categorical predictor '{predictor_name}' category '{category_name}' must be an object")
            coefficient = category_data.get("coefficient")
            if not isinstance(coefficient, (int, float)):
                raise ValueError(
                    f"Categorical predictor '{predictor_name}' category '{category_name}' requires numeric coefficient"
                )
            transformations.append("none")
        return transformations

    coefficient = predictor_data.get("coefficient")
    if not isinstance(coefficient, (int, float)):
        raise ValueError(f"Predictor '{predictor_name}' requires numeric coefficient")

    transformation = predictor_data.get("transformation", "none")
    if transformation not in ALLOWED_FORMULA_TRANSFORMATIONS:
        raise ValueError(f"Predictor '{predictor_name}' has invalid transformation '{transformation}'")

    return [transformation]


def _extract_quantitative_formula_beta_1(formula_data: dict[str, Any]) -> list[float]:
    predictors = formula_data.get("predictors")
    if not isinstance(predictors, dict):
        raise ValueError("Quantitative formula predictors must be an object")

    beta_1: list[float] = []
    for predictor_name, predictor_data in predictors.items():
        if not isinstance(predictor_data, dict):
            raise ValueError(f"Predictor '{predictor_name}' must be an object")
        beta_1.extend(_extract_predictor_beta_1(predictor_name, predictor_data))
    return beta_1


def _extract_quantitative_formula_predictor_names(formula_data: dict[str, Any]) -> list[str]:
    predictors = formula_data.get("predictors")
    if not isinstance(predictors, dict):
        raise ValueError("Quantitative formula predictors must be an object")

    names: list[str] = []
    for predictor_name, predictor_data in predictors.items():
        if not isinstance(predictor_data, dict):
            raise ValueError(f"Predictor '{predictor_name}' must be an object")
        names.extend(_extract_predictor_names(predictor_name, predictor_data))
    return names


def _extract_quantitative_formula_predictor_transformations(formula_data: dict[str, Any]) -> dict[str, str]:
    predictors = formula_data.get("predictors")
    if not isinstance(predictors, dict):
        raise ValueError("Quantitative formula predictors must be an object")

    transformations: dict[str, str] = {}
    for predictor_name, predictor_data in predictors.items():
        if not isinstance(predictor_data, dict):
            raise ValueError(f"Predictor '{predictor_name}' must be an object")

        if "reference_category" in predictor_data:
            other_categories = predictor_data.get("other_categories")
            if not isinstance(other_categories, dict):
                raise ValueError(f"Categorical predictor '{predictor_name}' requires other_categories as an object")
            for category_name, category_data in other_categories.items():
                if not isinstance(category_data, dict):
                    raise ValueError(f"Categorical predictor '{predictor_name}' category '{category_name}' must be an object")
                coefficient = category_data.get("coefficient")
                if not isinstance(coefficient, (int, float)):
                    raise ValueError(
                        f"Categorical predictor '{predictor_name}' category '{category_name}' requires numeric coefficient"
                    )
            continue

        extracted = _extract_predictor_transformation(predictor_name, predictor_data)
        if len(extracted) != 1:
            raise ValueError(f"Predictor '{predictor_name}' produced an invalid transformation payload")

        transformation = extracted[0]
        if transformation != "none":
            transformations[predictor_name] = transformation

    return transformations


def _extract_ordinal_formula_beta_1(formula_data: dict[str, Any]) -> list[float]:
    predictors = formula_data.get("predictors")
    if not isinstance(predictors, dict):
        raise ValueError("Ordinal formula predictors must be an object")

    beta_1: list[float] = []
    for predictor_name, predictor_data in predictors.items():
        if not isinstance(predictor_data, dict):
            raise ValueError(f"Predictor '{predictor_name}' must be an object")
        beta_1.extend(_extract_predictor_beta_1(predictor_name, predictor_data))
    return beta_1


def _extract_ordinal_formula_predictor_names(formula_data: dict[str, Any]) -> list[str]:
    predictors = formula_data.get("predictors")
    if not isinstance(predictors, dict):
        raise ValueError("Ordinal formula predictors must be an object")

    names: list[str] = []
    for predictor_name, predictor_data in predictors.items():
        if not isinstance(predictor_data, dict):
            raise ValueError(f"Predictor '{predictor_name}' must be an object")
        names.extend(_extract_predictor_names(predictor_name, predictor_data))
    return names


def _extract_ordinal_formula_predictor_transformations(formula_data: dict[str, Any]) -> dict[str, str]:
    predictors = formula_data.get("predictors")
    if not isinstance(predictors, dict):
        raise ValueError("Ordinal formula predictors must be an object")

    transformations: dict[str, str] = {}
    for predictor_name, predictor_data in predictors.items():
        if not isinstance(predictor_data, dict):
            raise ValueError(f"Predictor '{predictor_name}' must be an object")

        if "reference_category" in predictor_data:
            other_categories = predictor_data.get("other_categories")
            if not isinstance(other_categories, dict):
                raise ValueError(f"Categorical predictor '{predictor_name}' requires other_categories as an object")
            for category_name, category_data in other_categories.items():
                if not isinstance(category_data, dict):
                    raise ValueError(f"Categorical predictor '{predictor_name}' category '{category_name}' must be an object")
                coefficient = category_data.get("coefficient")
                if not isinstance(coefficient, (int, float)):
                    raise ValueError(
                        f"Categorical predictor '{predictor_name}' category '{category_name}' requires numeric coefficient"
                    )
            continue

        extracted = _extract_predictor_transformation(predictor_name, predictor_data)
        if len(extracted) != 1:
            raise ValueError(f"Predictor '{predictor_name}' produced an invalid transformation payload")

        transformation = extracted[0]
        if transformation != "none":
            transformations[predictor_name] = transformation

    return transformations


def _extract_nominal_formula_beta_1(formula_data: dict[str, Any]) -> dict[str, list[float]]:
    other_categories = formula_data.get("other_categories")
    if not isinstance(other_categories, dict):
        raise ValueError("Nominal formula other_categories must be an object")

    beta_1_by_category: dict[str, list[float]] = {}
    for category_name, category_data in other_categories.items():
        if not isinstance(category_data, dict):
            raise ValueError(f"Nominal category '{category_name}' must be an object")

        predictors = category_data.get("predictors")
        if not isinstance(predictors, dict):
            raise ValueError(f"Nominal category '{category_name}' predictors must be an object")

        beta_1: list[float] = []
        for predictor_name, predictor_data in predictors.items():
            if not isinstance(predictor_data, dict):
                raise ValueError(f"Predictor '{predictor_name}' in nominal category '{category_name}' must be an object")
            beta_1.extend(_extract_predictor_beta_1(predictor_name, predictor_data))

        beta_1_by_category[category_name] = beta_1

    return beta_1_by_category


def _extract_nominal_formula_predictor_names(formula_data: dict[str, Any]) -> dict[str, list[str]]:
    other_categories = formula_data.get("other_categories")
    if not isinstance(other_categories, dict):
        raise ValueError("Nominal formula other_categories must be an object")

    names_by_category: dict[str, list[str]] = {}
    for category_name, category_data in other_categories.items():
        if not isinstance(category_data, dict):
            raise ValueError(f"Nominal category '{category_name}' must be an object")

        predictors = category_data.get("predictors")
        if not isinstance(predictors, dict):
            raise ValueError(f"Nominal category '{category_name}' predictors must be an object")

        names: list[str] = []
        for predictor_name, predictor_data in predictors.items():
            if not isinstance(predictor_data, dict):
                raise ValueError(f"Predictor '{predictor_name}' in nominal category '{category_name}' must be an object")
            names.extend(_extract_predictor_names(predictor_name, predictor_data))

        names_by_category[category_name] = names

    return names_by_category


def _extract_nominal_formula_predictor_transformations(formula_data: dict[str, Any]) -> dict[str, dict[str, str]]:
    other_categories = formula_data.get("other_categories")
    if not isinstance(other_categories, dict):
        raise ValueError("Nominal formula other_categories must be an object")

    transformations_by_category: dict[str, dict[str, str]] = {}
    for category_name, category_data in other_categories.items():
        if not isinstance(category_data, dict):
            raise ValueError(f"Nominal category '{category_name}' must be an object")

        predictors = category_data.get("predictors")
        if not isinstance(predictors, dict):
            raise ValueError(f"Nominal category '{category_name}' predictors must be an object")

        transformations: dict[str, str] = {}
        for predictor_name, predictor_data in predictors.items():
            if not isinstance(predictor_data, dict):
                raise ValueError(f"Predictor '{predictor_name}' in nominal category '{category_name}' must be an object")

            if "reference_category" in predictor_data:
                other_categories = predictor_data.get("other_categories")
                if not isinstance(other_categories, dict):
                    raise ValueError(f"Categorical predictor '{predictor_name}' requires other_categories as an object")
                for subcategory_name, category_data in other_categories.items():
                    if not isinstance(category_data, dict):
                        raise ValueError(
                            f"Categorical predictor '{predictor_name}' category '{subcategory_name}' must be an object"
                        )
                    coefficient = category_data.get("coefficient")
                    if not isinstance(coefficient, (int, float)):
                        raise ValueError(
                            f"Categorical predictor '{predictor_name}' category '{subcategory_name}' requires numeric coefficient"
                        )
                continue

            extracted = _extract_predictor_transformation(predictor_name, predictor_data)
            if len(extracted) != 1:
                raise ValueError(f"Predictor '{predictor_name}' produced an invalid transformation payload")

            transformation = extracted[0]
            if transformation != "none":
                transformations[predictor_name] = transformation

        transformations_by_category[category_name] = transformations

    return transformations_by_category


def extract_beta_1(formula_data: dict[str, Any]) -> list[float] | dict[str, list[float]]:
    if not isinstance(formula_data, dict):
        raise ValueError("Formula must be an object")

    if "intercept" in formula_data and "predictors" in formula_data:
        return _extract_quantitative_formula_beta_1(formula_data)

    if "reference_category" in formula_data and "other_categories" in formula_data and "predictors" in formula_data and "intercept" not in formula_data:
        return _extract_ordinal_formula_beta_1(formula_data)

    if "reference_category" in formula_data and "other_categories" in formula_data:
        return _extract_nominal_formula_beta_1(formula_data)

    raise ValueError("Formula does not match a supported schema")


def extract_beta_1_predictor_names(formula_data: dict[str, Any]) -> list[str] | dict[str, list[str]]:
    if not isinstance(formula_data, dict):
        raise ValueError("Formula must be an object")

    if "intercept" in formula_data and "predictors" in formula_data:
        return _extract_quantitative_formula_predictor_names(formula_data)

    if "reference_category" in formula_data and "other_categories" in formula_data and "predictors" in formula_data and "intercept" not in formula_data:
        return _extract_ordinal_formula_predictor_names(formula_data)

    if "reference_category" in formula_data and "other_categories" in formula_data:
        return _extract_nominal_formula_predictor_names(formula_data)

    raise ValueError("Formula does not match a supported schema")


def extract_predictor_transformations(formula_data: dict[str, Any]) -> dict[str, str] | dict[str, dict[str, str]]:
    if not isinstance(formula_data, dict):
        raise ValueError("Formula must be an object")

    if "intercept" in formula_data and "predictors" in formula_data:
        return _extract_quantitative_formula_predictor_transformations(formula_data)

    if "reference_category" in formula_data and "other_categories" in formula_data and "predictors" in formula_data and "intercept" not in formula_data:
        return _extract_ordinal_formula_predictor_transformations(formula_data)

    if "reference_category" in formula_data and "other_categories" in formula_data:
        return _extract_nominal_formula_predictor_transformations(formula_data)

    raise ValueError("Formula does not match a supported schema")


def extract_target_snr(formula_data: dict[str, Any]) -> float | None:
    if not isinstance(formula_data, dict):
        raise ValueError("Formula must be an object")

    snr = formula_data.get("snr")
    return float(snr) if isinstance(snr, (int, float)) else None
