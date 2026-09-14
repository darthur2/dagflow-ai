import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from distributions import CategoricalNominal, LogNormal, Normal
from regressors import CategoricalNominalRegressor


def _build_x() -> np.ndarray:
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
    return np.column_stack([age_years, systolic_blood_pressure, bmi, medicare, medicaid, uninsured])


def test_categorical_nominal_regressor_calibrates_and_samples():
    X = _build_x()
    target_probabilities = np.array([0.55, 0.25, 0.14, 0.06], dtype=float)
    regressor = CategoricalNominalRegressor(
        target_probabilities=target_probabilities,
        X=X,
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

    sample_probabilities = np.bincount(samples.astype(int), minlength=4) / float(len(samples))
    target_probabilities = regressor.target_probabilities

    print(f"beta_0: {calibrated.beta_0}")
    print(f"target_probabilities: {target_probabilities}")
    print(f"sample_probabilities: {sample_probabilities}")
