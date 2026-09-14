import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from distributions import CategoricalNominal, LogNormal, Normal
from regressors import CategoricalNominalRegressor


def test_categorical_nominal_regressor_calibrates_and_samples():
    np.random.seed(0)

    age_years = Normal(mean=52.0, standard_deviation=18.0, min=18.0, max=95.0).sample(100)
    systolic_blood_pressure = Normal(mean=128.0, standard_deviation=16.0, min=85.0, max=220.0).sample(100)
    bmi = LogNormal(log_mean=3.2, log_standard_deviation=0.18, min=16.0, max=60.0).sample(100)
    insurance_type = CategoricalNominal(
        categories=["private", "medicare", "medicaid", "uninsured"],
        probabilities=[0.46, 0.29, 0.18, 0.07],
    ).sample(100)

    medicare = (insurance_type == "medicare").astype(float)
    medicaid = (insurance_type == "medicaid").astype(float)
    uninsured = (insurance_type == "uninsured").astype(float)
    X = np.column_stack([age_years, systolic_blood_pressure, bmi, medicare, medicaid, uninsured])

    target_probabilities = np.array([0.55, 0.25, 0.14, 0.06], dtype=float)
    regressor = CategoricalNominalRegressor(
        target_probabilities=target_probabilities,
        X=X,
        categories=["private", "medicare", "medicaid", "uninsured"],
        beta_1=np.array(
            [
                [0.012, 0.018, 0.028],
                [0.015, 0.022, 0.032],
                [0.02, 0.03, 0.04],
                [0.18, 0.28, 0.35],
                [0.24, 0.22, 0.5],
                [0.12, -0.08, 0.7],
            ],
            dtype=float,
        ),
        predictor_names=[
            "age_years",
            "systolic_blood_pressure",
            "bmi",
            "insurance_type",
            "insurance_type",
            "insurance_type",
        ],
        predictor_transformations={"bmi": "log"},
    )

    calibrated = regressor.calibrate()
    samples = calibrated.sample(2000)

    sample_probabilities = {category: float(np.mean(samples == category)) for category in calibrated.categories}
    target_probabilities = regressor.target_probabilities
    target_probabilities = {category: float(probability) for category, probability in zip(calibrated.categories, target_probabilities)}

    print(f"beta_0: {calibrated.beta_0}")
    print(f"target_probabilities: {target_probabilities}")
    print(f"sample_probabilities: {sample_probabilities}")


def test_categorical_nominal_regressor_municipality_calibrates_and_samples():
    np.random.seed(0)

    metro_region = CategoricalNominal(
        categories=["Midwest", "South", "Northeast", "West"],
        probabilities=[0.28, 0.33, 0.17, 0.22],
    ).sample(100)

    south = (metro_region == "South").astype(float)
    northeast = (metro_region == "Northeast").astype(float)
    west = (metro_region == "West").astype(float)
    X = np.column_stack([south, northeast, west])

    target_probabilities = np.array([0.18, 0.24, 0.22, 0.2, 0.16], dtype=float)
    regressor = CategoricalNominalRegressor(
        target_probabilities=target_probabilities,
        X=X,
        categories=["Northfield", "Maple Grove", "Riverton", "Oak Park", "Cedar Hills"],
        beta_1=np.array(
            [
                [0.25, -0.15, 0.1, 0.0],
                [0.1, -0.05, 0.05, 0.0],
                [0.05, 0.1, 0.12, 0.0],
            ],
            dtype=float,
        ),
        predictor_names=["metro_region", "metro_region", "metro_region"],
    )

    calibrated = regressor.calibrate()
    samples = calibrated.sample(2000)

    sample_probabilities = {category: float(np.mean(samples == category)) for category in calibrated.categories}
    target_probabilities = regressor.target_probabilities
    target_probabilities = {category: float(probability) for category, probability in zip(calibrated.categories, target_probabilities)}

    print(f"beta_0: {calibrated.beta_0}")
    print(f"target_probabilities: {target_probabilities}")
    print(f"sample_probabilities: {sample_probabilities}")


def test_categorical_nominal_regressor_neighborhood_type_calibrates_and_samples():
    np.random.seed(0)

    municipality = CategoricalNominal(
        categories=["Northfield", "Maple Grove", "Riverton", "Oak Park", "Cedar Hills"],
        probabilities=[0.18, 0.24, 0.22, 0.2, 0.16],
    ).sample(100)

    maple_grove = (municipality == "Maple Grove").astype(float)
    riverton = (municipality == "Riverton").astype(float)
    oak_park = (municipality == "Oak Park").astype(float)
    cedar_hills = (municipality == "Cedar Hills").astype(float)
    X = np.column_stack([maple_grove, riverton, oak_park, cedar_hills])

    target_probabilities = np.array([0.34, 0.28, 0.16, 0.22], dtype=float)
    regressor = CategoricalNominalRegressor(
        target_probabilities=target_probabilities,
        X=X,
        categories=["established", "newer development", "exurban", "planned community"],
        beta_1=np.array(
            [
                [0.15, 0.05, 0.08],
                [-0.1, 0.25, -0.05],
                [0.2, -0.05, 0.12],
                [0.12, 0.18, 0.16],
            ],
            dtype=float,
        ),
        predictor_names=["municipality", "municipality", "municipality", "municipality"],
    )

    calibrated = regressor.calibrate()
    samples = calibrated.sample(2000)

    sample_probabilities = {category: float(np.mean(samples == category)) for category in calibrated.categories}
    target_probabilities = regressor.target_probabilities
    target_probabilities = {category: float(probability) for category, probability in zip(calibrated.categories, target_probabilities)}

    print(f"beta_0: {calibrated.beta_0}")
    print(f"target_probabilities: {target_probabilities}")
    print(f"sample_probabilities: {sample_probabilities}")


def test_categorical_nominal_regressor_recent_renovation_calibrates_and_samples():
    np.random.seed(0)

    seller_motivation = CategoricalNominal(
        categories=["standard sale", "relocation", "estate sale", "downsizing"],
        probabilities=[0.58, 0.16, 0.14, 0.12],
    ).sample(100)

    relocation = (seller_motivation == "relocation").astype(float)
    estate_sale = (seller_motivation == "estate sale").astype(float)
    downsizing = (seller_motivation == "downsizing").astype(float)
    X = np.column_stack([relocation, estate_sale, downsizing])

    target_probabilities = np.array([0.78, 0.22], dtype=float)
    regressor = CategoricalNominalRegressor(
        target_probabilities=target_probabilities,
        X=X,
        categories=["no", "yes"],
        beta_1=np.array([[0.12], [-0.2], [0.05]], dtype=float),
        predictor_names=["seller_motivation", "seller_motivation", "seller_motivation"],
    )

    calibrated = regressor.calibrate()
    samples = calibrated.sample(2000)

    sample_probabilities = {category: float(np.mean(samples == category)) for category in calibrated.categories}
    target_probabilities = regressor.target_probabilities
    target_probabilities = {category: float(probability) for category, probability in zip(calibrated.categories, target_probabilities)}

    print(f"beta_0: {calibrated.beta_0}")
    print(f"target_probabilities: {target_probabilities}")
    print(f"sample_probabilities: {sample_probabilities}")


def main() -> None:
    test_categorical_nominal_regressor_calibrates_and_samples()
    test_categorical_nominal_regressor_municipality_calibrates_and_samples()
    test_categorical_nominal_regressor_neighborhood_type_calibrates_and_samples()
    test_categorical_nominal_regressor_recent_renovation_calibrates_and_samples()


if __name__ == "__main__":
    main()
