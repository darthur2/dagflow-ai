import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from distributions import CategoricalNominal, Gamma, Geometric, LogNormal, Normal
from regressors import GeometricRegressor


def test_geometric_regressor_calibrates_and_samples():
    np.random.seed(0)

    machine_temperature_c = Normal(mean=72.0, standard_deviation=6.0, min=55.0, max=90.0).sample(100)
    line_speed_mpm = LogNormal(log_mean=3.75, log_standard_deviation=0.18, min=10.0, max=100.0).sample(100)
    operator_experience_years = Gamma(shape=4.0, rate=0.5, min=0.0, max=30.0).sample(100)
    shift_type = CategoricalNominal(
        categories=["day", "evening", "night"],
        probabilities=[0.5, 0.3, 0.2],
    ).sample(100)

    evening = (shift_type == "evening").astype(float)
    night = (shift_type == "night").astype(float)
    X = np.column_stack([machine_temperature_c, line_speed_mpm, operator_experience_years, evening, night])

    response = Geometric(success_prob=0.22, min=1, max=25)
    regressor = GeometricRegressor(
        target_mean=response.target_mean(),
        min=1,
        max=25,
        X=X,
        beta_1_init=np.array([0.03, 0.012, -0.08, 0.18, 0.32], dtype=float),
        predictor_names=[
            "machine_temperature_c",
            "line_speed_mpm",
            "operator_experience_years",
            "shift_type",
            "shift_type",
        ],
        predictor_transformations={"line_speed_mpm": "log", "operator_experience_years": "sqrt"},
    )

    calibrated = regressor.calibrate()
    samples = calibrated.sample(2000)

    sample_mean = float(np.mean(samples))

    print(f"beta_0: {calibrated.beta_0}")
    print(f"mean: target={regressor.target_mean} sample={sample_mean}")
