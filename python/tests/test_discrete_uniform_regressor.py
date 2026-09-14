import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from distributions import Beta, CategoricalNominal, DiscreteUniform, Gamma
from regressors import DiscreteUniformRegressor


def test_discrete_uniform_regressor_calibrates_and_samples():
    np.random.seed(0)

    study_hours_per_week = Gamma(shape=4.0, rate=0.5, min=0.0, max=40.0).sample(100)
    attendance_rate = Beta(shape_1=8.0, shape_2=2.5, min=0.0, max=100.0).sample(100)
    previous_gpa = Beta(shape_1=5.0, shape_2=2.0, min=0.0, max=4.0).sample(100)
    instructional_setting = CategoricalNominal(
        categories=["in-person", "hybrid", "online"],
        probabilities=[0.5, 0.25, 0.25],
    ).sample(100)

    hybrid = (instructional_setting == "hybrid").astype(float)
    online = (instructional_setting == "online").astype(float)
    X = np.column_stack([study_hours_per_week, attendance_rate, previous_gpa, hybrid, online])

    response = DiscreteUniform(min=0, max=100)
    regressor = DiscreteUniformRegressor(
        target_snr=0.55,
        min=0,
        max=100,
        X=X,
        beta_1_init=np.array([0.25, 0.05, 0.45, 0.15, 0.35], dtype=float),
        predictor_names=[
            "study_hours_per_week",
            "attendance_rate",
            "previous_gpa",
            "instructional_setting",
            "instructional_setting",
        ],
        predictor_transformations={"study_hours_per_week": "log", "previous_gpa": "sqrt"},
    )

    calibrated = regressor.calibrate()
    samples = calibrated.sample(2000)

    sample_mean = float(np.mean(samples))
    sample_variance = float(np.var(samples))
    target_mean = (0 + 100) / 2.0
    target_variance = (((100 - 0 + 1) ** 2) - 1) / 12.0

    print(f"beta_0: {calibrated.latent_regressor.beta_0}")
    print(f"mean: target={target_mean} sample={sample_mean}")
    print(f"variance: target={target_variance} sample={sample_variance}")
