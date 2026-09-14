import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from distributions import Beta, CategoricalNominal, DiscreteUniform, LogNormal
from regressors import BetaRegressor


def _build_x() -> np.ndarray:
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
    return np.column_stack(
        [annual_income, debt_to_income_ratio, credit_score, self_employed, unemployed, retired, student]
    )


def test_beta_regressor_calibrates_and_samples():
    X = _build_x()
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
