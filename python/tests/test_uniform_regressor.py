import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from distributions import CategoricalNominal, Gamma, Poisson, Beta, Uniform
from regressors import UniformRegressor


def _build_x() -> np.ndarray:
    np.random.seed(0)

    average_handle_time_minutes = Gamma(shape=4.0, rate=0.4, min=0.0, max=30.0).sample(100)
    queue_length = Poisson(rate=10.0, min=0, max=40).sample(100)
    staff_utilization_rate = Beta(shape_1=5.0, shape_2=2.0, min=0.0, max=1.0).sample(100)
    shift_type = CategoricalNominal(
        categories=["day", "evening", "night"],
        probabilities=[0.5, 0.3, 0.2],
    ).sample(100)

    evening = (shift_type == "evening").astype(float)
    night = (shift_type == "night").astype(float)
    return np.column_stack(
        [average_handle_time_minutes, queue_length, staff_utilization_rate, evening, night]
    )


def test_uniform_regressor_calibrates_and_samples():
    X = _build_x()
    response = Uniform(min=0.0, max=10.0)
    regressor = UniformRegressor(
        target_snr=0.55,
        min=0.0,
        max=10.0,
        X=X,
        beta_1_init=np.array([0.42, 0.24, -1.05, 0.2, 0.45], dtype=float),
        predictor_names=[
            "average_handle_time_minutes",
            "queue_length",
            "staff_utilization_rate",
            "shift_type",
            "shift_type",
        ],
        predictor_transformations={"average_handle_time_minutes": "log", "queue_length": "sqrt"},
    )

    calibrated = regressor.calibrate()
    samples = calibrated.sample(2000)

    sample_mean = float(np.mean(samples))
    sample_variance = float(np.var(samples))

    print(f"beta_0: {calibrated.latent_regressor.beta_0}")
    print(f"mean: target={5.0} sample={sample_mean}")
    print(f"variance: target={((10.0 - 0.0) ** 2) / 12.0} sample={sample_variance}")
