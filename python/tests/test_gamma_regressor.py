import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from distributions import CategoricalNominal, Gamma, Normal, Poisson
from regressors import GammaRegressor


def _build_x() -> np.ndarray:
    np.random.seed(0)

    age = Normal(mean=52.0, standard_deviation=16.0, min=18.0, max=90.0).sample(100)
    comorbidity = Gamma(shape=2.0, rate=1.0, min=0.0, max=12.0).sample(100)
    visits = Poisson(rate=4.5, min=0, max=20).sample(100)
    insurance = CategoricalNominal(
        categories=["Private", "Medicare", "Medicaid", "Uninsured"],
        probabilities=[0.58, 0.24, 0.10, 0.08],
    ).sample(100)

    insurance_medicare = (insurance == "Medicare").astype(float)
    insurance_medicaid = (insurance == "Medicaid").astype(float)
    insurance_uninsured = (insurance == "Uninsured").astype(float)
    return np.column_stack([age, comorbidity, visits, insurance_medicare, insurance_medicaid, insurance_uninsured])


def test_gamma_regressor_calibrates_and_samples():
    X = _build_x()
    response = Gamma(shape=2.5, rate=0.00025, min=0.0, max=50000.0)
    regressor = GammaRegressor(
        target_mean=response.target_mean(),
        target_variance=response.target_variance(),
        min=0.0,
        max=50000.0,
        X=X,
        beta_1_init=np.array([0.001, 0.015, 0.005, 0.04, 0.03, -0.02], dtype=float),
        predictor_names=[
            "age_years",
            "comorbidity_score",
            "primary_care_visits",
            "insurance_type",
            "insurance_type",
            "insurance_type",
        ],
        predictor_transformations={},
    )

    calibrated = regressor.calibrate()
    samples = calibrated.sample(2000)

    sample_mean = float(np.mean(samples))
    sample_variance = float(np.var(samples))

    print(f"beta_0: {calibrated.beta_0}")
    print(f"shape: {calibrated.shape}")
    print(f"mean: target={regressor.target_mean} sample={sample_mean}")
    print(f"variance: target={regressor.target_variance} sample={sample_variance}")
