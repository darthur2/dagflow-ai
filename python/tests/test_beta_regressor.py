import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from distributions import Beta, CategoricalNominal, DiscreteUniform, LogNormal
from regressors import BetaRegressor


def test_beta_regressor_calibrates_and_samples():
    np.random.seed(0)

    annual_income = LogNormal(log_mean=10.8, log_standard_deviation=0.45, min=20000.0, max=500000.0).sample(100)
    debt_to_income_ratio = Beta(shape_1=2.0, shape_2=6.5, min=0.0, max=1.0).sample(100)
    credit_score = DiscreteUniform(min=300, max=850).sample(100)
    employment_status = CategoricalNominal(
        categories=["employed", "self_employed", "unemployed", "retired", "student"],
        probabilities=[0.68, 0.12, 0.08, 0.07, 0.05],
    ).sample(100)

    self_employed = (employment_status == "self_employed").astype(float)
    unemployed = (employment_status == "unemployed").astype(float)
    retired = (employment_status == "retired").astype(float)
    student = (employment_status == "student").astype(float)
    X = np.column_stack([annual_income, debt_to_income_ratio, credit_score, self_employed, unemployed, retired, student])

    response = Beta(shape_1=2.2, shape_2=5.8, min=0.0, max=100.0)
    regressor = BetaRegressor(
        target_mean=response.target_mean(),
        target_variance=response.target_variance(),
        min=0.0,
        max=100.0,
        X=X,
        beta_1_init=np.array([0.08, 0.22, -0.0015, 0.06, -0.08, -0.03, 0.04], dtype=float),
        predictor_names=[
            "annual_income",
            "debt_to_income_ratio",
            "credit_score",
            "employment_status",
            "employment_status",
            "employment_status",
            "employment_status",
        ],
        predictor_transformations={"annual_income": "log"},
    )

    calibrated = regressor.calibrate()
    samples = calibrated.sample(2000)

    sample_mean = float(np.mean(samples))
    sample_variance = float(np.var(samples))

    print(f"beta_0: {calibrated.beta_0}")
    print(f"phi: {calibrated.phi}")
    print(f"mean: target={regressor.target_mean} sample={sample_mean}")
    print(f"variance: target={regressor.target_variance} sample={sample_variance}")


def test_beta_regressor_property_tax_rate_calibrates_and_samples():
    np.random.seed(0)

    metro_region = CategoricalNominal(
        categories=["Midwest", "South", "Northeast", "West"],
        probabilities=[0.28, 0.33, 0.17, 0.22],
    ).sample(100)
    municipality = CategoricalNominal(
        categories=["Northfield", "Maple Grove", "Riverton", "Oak Park", "Cedar Hills"],
        probabilities=[0.18, 0.24, 0.22, 0.2, 0.16],
    ).sample(100)

    south = (metro_region == "South").astype(float)
    northeast = (metro_region == "Northeast").astype(float)
    west = (metro_region == "West").astype(float)
    maple_grove = (municipality == "Maple Grove").astype(float)
    riverton = (municipality == "Riverton").astype(float)
    oak_park = (municipality == "Oak Park").astype(float)
    cedar_hills = (municipality == "Cedar Hills").astype(float)
    X = np.column_stack([south, northeast, west, maple_grove, riverton, oak_park, cedar_hills])

    response = Beta(shape_1=2.5, shape_2=8.0, min=0.5, max=3.5)
    regressor = BetaRegressor(
        target_mean=response.target_mean(),
        target_variance=response.target_variance(),
        min=0.5,
        max=3.5,
        X=X,
        beta_1_init=np.array([0.08, 0.18, 0.14, 0.08, -0.06, 0.05, 0.1], dtype=float),
        predictor_names=[
            "metro_region",
            "metro_region",
            "metro_region",
            "municipality",
            "municipality",
            "municipality",
            "municipality",
        ],
    )

    calibrated = regressor.calibrate()
    samples = calibrated.sample(2000)

    sample_mean = float(np.mean(samples))
    sample_variance = float(np.var(samples))

    print(f"beta_0: {calibrated.beta_0}")
    print(f"phi: {calibrated.phi}")
    print(f"mean: target={regressor.target_mean} sample={sample_mean}")
    print(f"variance: target={regressor.target_variance} sample={sample_variance}")


def main() -> None:
    test_beta_regressor_calibrates_and_samples()
    test_beta_regressor_property_tax_rate_calibrates_and_samples()


if __name__ == "__main__":
    main()
