import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from distributions import CategoricalNominal, Exponential, Gamma, Normal
from regressors import ExponentialRegressor


def test_exponential_regressor_calibrates_and_samples():
    np.random.seed(0)

    machine_temperature = Normal(mean=75.0, standard_deviation=4.0, min=60.0, max=90.0).sample(100)
    operator_experience_years = Gamma(shape=4.0, rate=0.4, min=0.0, max=30.0).sample(100)
    conveyor_speed = Normal(mean=18.0, standard_deviation=2.5, min=10.0, max=28.0).sample(100)
    shift_type = CategoricalNominal(
        categories=["day", "evening", "night"],
        probabilities=[0.45, 0.35, 0.2],
    ).sample(100)

    shift_evening = (shift_type == "evening").astype(float)
    shift_night = (shift_type == "night").astype(float)
    X = np.column_stack([machine_temperature, operator_experience_years, conveyor_speed, shift_evening, shift_night])

    response = Exponential(rate=0.37, min=2.0, max=40.0)
    regressor = ExponentialRegressor(
        target_mean=response.target_mean(),
        min=2.0,
        max=40.0,
        X=X,
        beta_1_init=np.array([0.012, -0.04, -0.018, 0.045, 0.09], dtype=float),
        predictor_names=[
            "machine_temperature",
            "operator_experience_years",
            "conveyor_speed",
            "shift_type",
            "shift_type",
        ],
        predictor_transformations={},
    )

    calibrated = regressor.calibrate()
    samples = calibrated.sample(2000)

    sample_mean = float(np.mean(samples))

    print(f"beta_0: {calibrated.beta_0}")
    print(f"mean: target={regressor.target_mean} sample={sample_mean}")
