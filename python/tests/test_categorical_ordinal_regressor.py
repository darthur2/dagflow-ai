import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from distributions import Beta, CategoricalNominal, LogNormal, NegativeBinomial, Poisson
from regressors import CategoricalOrdinalRegressor


def test_categorical_ordinal_regressor_calibrates_and_samples():
    np.random.seed(0)

    mean_time_to_recovery_hours = LogNormal(log_mean=0.75, log_standard_deviation=0.6, min=0.1, max=72.0).sample(100)
    deployments_per_week = Poisson(rate=12.0, min=0, max=40).sample(100)
    open_defect_count = NegativeBinomial(shape=8.0, mean=18.0, min=0, max=120).sample(100)
    deployment_strategy = CategoricalNominal(
        categories=["Blue-Green", "Canary", "Rolling", "Recreate"],
        probabilities=[0.3, 0.25, 0.35, 0.1],
    ).sample(100)

    blue_green = (deployment_strategy == "Blue-Green").astype(float)
    canary = (deployment_strategy == "Canary").astype(float)
    recreate = (deployment_strategy == "Recreate").astype(float)
    X = np.column_stack([mean_time_to_recovery_hours, deployments_per_week, open_defect_count, blue_green, canary, recreate])

    target_probabilities = np.array([0.45, 0.3, 0.18, 0.07], dtype=float)
    regressor = CategoricalOrdinalRegressor(
        target_probabilities=target_probabilities,
        X=X,
        beta_1=np.array([0.45, 0.12, 0.22, -0.18, -0.08, 0.28], dtype=float),
        predictor_names=[
            "mean_time_to_recovery_hours",
            "deployments_per_week",
            "open_defect_count",
            "deployment_strategy",
            "deployment_strategy",
            "deployment_strategy",
        ],
        predictor_transformations={"mean_time_to_recovery_hours": "log", "open_defect_count": "log"},
        categories=["low", "medium", "high", "critical"],
    )

    calibrated = regressor.calibrate()
    samples = calibrated.sample(2000)

    categories = ["low", "medium", "high", "critical"]
    sample_probabilities = {category: float(np.mean(samples == category)) for category in categories}
    target_probabilities = regressor.target_probabilities
    target_probabilities = {category: float(probability) for category, probability in zip(categories, target_probabilities)}

    print(f"beta_0: {calibrated.beta_0}")
    print(f"target_probabilities: {target_probabilities}")
    print(f"sample_probabilities: {sample_probabilities}")


def test_categorical_ordinal_regressor_market_condition_calibrates_and_samples():
    np.random.seed(0)

    metro_region = CategoricalNominal(
        categories=["Midwest", "South", "Northeast", "West"],
        probabilities=[0.28, 0.33, 0.17, 0.22],
    ).sample(100)
    sale_season = CategoricalNominal(
        categories=["winter", "spring", "summer", "fall"],
        probabilities=[0.18, 0.32, 0.28, 0.22],
    ).sample(100)

    south = (metro_region == "South").astype(float)
    northeast = (metro_region == "Northeast").astype(float)
    west = (metro_region == "West").astype(float)
    spring = (sale_season == "spring").astype(float)
    summer = (sale_season == "summer").astype(float)
    fall = (sale_season == "fall").astype(float)
    X = np.column_stack([south, northeast, west, spring, summer, fall])

    target_probabilities = np.array([0.27, 0.46, 0.27], dtype=float)
    regressor = CategoricalOrdinalRegressor(
        target_probabilities=target_probabilities,
        X=X,
        beta_1=np.array([0.15, -0.1, 0.12, 0.45, 0.25, -0.05], dtype=float),
        predictor_names=[
            "metro_region",
            "metro_region",
            "metro_region",
            "sale_season",
            "sale_season",
            "sale_season",
        ],
        categories=["buyer's market", "balanced market", "seller's market"],
    )

    calibrated = regressor.calibrate()
    samples = calibrated.sample(2000)

    categories = ["buyer's market", "balanced market", "seller's market"]
    sample_probabilities = {category: float(np.mean(samples == category)) for category in categories}
    target_probabilities = regressor.target_probabilities
    target_probabilities = {category: float(probability) for category, probability in zip(categories, target_probabilities)}

    print(f"beta_0: {calibrated.beta_0}")
    print(f"target_probabilities: {target_probabilities}")
    print(f"sample_probabilities: {sample_probabilities}")


def test_categorical_ordinal_regressor_sale_season_calibrates_and_samples():
    np.random.seed(0)

    metro_region = CategoricalNominal(
        categories=["Midwest", "South", "Northeast", "West"],
        probabilities=[0.28, 0.33, 0.17, 0.22],
    ).sample(100)

    south = (metro_region == "South").astype(float)
    northeast = (metro_region == "Northeast").astype(float)
    west = (metro_region == "West").astype(float)
    X = np.column_stack([south, northeast, west])

    target_probabilities = np.array([0.18, 0.32, 0.28, 0.22], dtype=float)
    regressor = CategoricalOrdinalRegressor(
        target_probabilities=target_probabilities,
        X=X,
        beta_1=np.array([0.12, -0.08, 0.06], dtype=float),
        predictor_names=["metro_region", "metro_region", "metro_region"],
        categories=["winter", "spring", "summer", "fall"],
    )

    calibrated = regressor.calibrate()
    samples = calibrated.sample(2000)

    categories = ["winter", "spring", "summer", "fall"]
    sample_probabilities = {category: float(np.mean(samples == category)) for category in categories}
    target_probabilities = regressor.target_probabilities
    target_probabilities = {category: float(probability) for category, probability in zip(categories, target_probabilities)}

    print(f"beta_0: {calibrated.beta_0}")
    print(f"target_probabilities: {target_probabilities}")
    print(f"sample_probabilities: {sample_probabilities}")


def test_categorical_ordinal_regressor_school_rating_calibrates_and_samples():
    np.random.seed(0)

    metro_region = CategoricalNominal(
        categories=["Midwest", "South", "Northeast", "West"],
        probabilities=[0.28, 0.33, 0.17, 0.22],
    ).sample(100)
    municipality = CategoricalNominal(
        categories=["Northfield", "Maple Grove", "Riverton", "Oak Park", "Cedar Hills"],
        probabilities=[0.18, 0.24, 0.22, 0.2, 0.16],
    ).sample(100)
    neighborhood_type = CategoricalNominal(
        categories=["established", "newer development", "exurban", "planned community"],
        probabilities=[0.34, 0.28, 0.16, 0.22],
    ).sample(100)
    distance_to_downtown_mi = Beta(shape_1=2.5, shape_2=8.0, min=0.5, max=40.0).sample(100)

    south = (metro_region == "South").astype(float)
    northeast = (metro_region == "Northeast").astype(float)
    west = (metro_region == "West").astype(float)
    maple_grove = (municipality == "Maple Grove").astype(float)
    riverton = (municipality == "Riverton").astype(float)
    oak_park = (municipality == "Oak Park").astype(float)
    cedar_hills = (municipality == "Cedar Hills").astype(float)
    newer_development = (neighborhood_type == "newer development").astype(float)
    exurban = (neighborhood_type == "exurban").astype(float)
    planned_community = (neighborhood_type == "planned community").astype(float)
    X = np.column_stack([
        south,
        northeast,
        west,
        maple_grove,
        riverton,
        oak_park,
        cedar_hills,
        newer_development,
        exurban,
        planned_community,
        distance_to_downtown_mi,
    ])

    target_probabilities = np.array([0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.18, 0.22, 0.18, 0.11], dtype=float)
    regressor = CategoricalOrdinalRegressor(
        target_probabilities=target_probabilities,
        X=X,
        beta_1=np.array([0.16, 0.12, 0.1, 0.3, -0.2, 0.12, 0.22, 0.4, -0.08, 0.45, -0.05], dtype=float),
        predictor_names=[
            "metro_region",
            "metro_region",
            "metro_region",
            "municipality",
            "municipality",
            "municipality",
            "municipality",
            "neighborhood_type",
            "neighborhood_type",
            "neighborhood_type",
            "distance_to_downtown_mi",
        ],
        predictor_transformations={},
        categories=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
    )

    calibrated = regressor.calibrate()
    samples = calibrated.sample(2000)

    categories = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
    sample_probabilities = {category: float(np.mean(samples == category)) for category in categories}
    target_probabilities = regressor.target_probabilities
    target_probabilities = {category: float(probability) for category, probability in zip(categories, target_probabilities)}

    print(f"beta_0: {calibrated.beta_0}")
    print(f"target_probabilities: {target_probabilities}")
    print(f"sample_probabilities: {sample_probabilities}")


def main() -> None:
    test_categorical_ordinal_regressor_calibrates_and_samples()
    test_categorical_ordinal_regressor_market_condition_calibrates_and_samples()
    test_categorical_ordinal_regressor_sale_season_calibrates_and_samples()
    test_categorical_ordinal_regressor_school_rating_calibrates_and_samples()


if __name__ == "__main__":
    main()
