import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from distributions import CategoricalNominal, Gamma, LogNormal, Poisson
from regressors import PoissonRegressor


def _build_x() -> np.ndarray:
    np.random.seed(0)

    codebase_size_kloc = LogNormal(log_mean=4.0, log_standard_deviation=0.6, min=10.0, max=2000.0).sample(100)
    deployment_frequency_per_week = Poisson(rate=5.0, min=0, max=30).sample(100)
    mean_build_time_minutes = Gamma(shape=4.0, rate=0.08, min=5.0, max=180.0).sample(100)
    team_distribution_model = CategoricalNominal(
        categories=["on-site", "hybrid", "remote"],
        probabilities=[0.25, 0.45, 0.30],
    ).sample(100)

    hybrid = (team_distribution_model == "hybrid").astype(float)
    remote = (team_distribution_model == "remote").astype(float)
    return np.column_stack(
        [codebase_size_kloc, deployment_frequency_per_week, mean_build_time_minutes, hybrid, remote]
    )


def test_poisson_regressor_calibrates_and_samples():
    X = _build_x()
    response = Poisson(rate=12.0, min=0, max=60)
    regressor = PoissonRegressor(
        target_mean=response.target_mean(),
        min=0,
        max=60,
        X=X,
        beta_1_init=np.array([0.28, 0.06, 0.015, 0.12, 0.22], dtype=float),
        predictor_names=[
            "codebase_size_kloc",
            "deployment_frequency_per_week",
            "mean_build_time_minutes",
            "team_distribution_model",
            "team_distribution_model",
        ],
        predictor_transformations={"codebase_size_kloc": "log", "mean_build_time_minutes": "log"},
    )

    calibrated = regressor.calibrate()
    samples = calibrated.sample(2000)

    sample_mean = float(np.mean(samples))

    print(f"beta_0: {calibrated.beta_0}")
    print(f"mean: target={regressor.target_mean} sample={sample_mean}")
