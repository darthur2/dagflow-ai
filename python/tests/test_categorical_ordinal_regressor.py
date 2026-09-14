import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from distributions import CategoricalNominal, LogNormal, NegativeBinomial, Poisson
from regressors import CategoricalOrdinalRegressor


def _build_x() -> np.ndarray:
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
    return np.column_stack(
        [mean_time_to_recovery_hours, deployments_per_week, open_defect_count, blue_green, canary, recreate]
    )


def test_categorical_ordinal_regressor_calibrates_and_samples():
    X = _build_x()
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
    )

    calibrated = regressor.calibrate()
    samples = calibrated.sample(2000)

    sample_probabilities = np.bincount(samples.astype(int), minlength=4) / float(len(samples))
    target_probabilities = regressor.target_probabilities

    print(f"beta_0: {calibrated.beta_0}")
    print(f"target_probabilities: {target_probabilities}")
    print(f"sample_probabilities: {sample_probabilities}")
