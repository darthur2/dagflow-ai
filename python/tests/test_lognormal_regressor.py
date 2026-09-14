import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from distributions import Beta, CategoricalNominal, Gamma, LogNormal
from regressors import LogNormalRegressor


def test_lognormal_regressor_calibrates_and_samples():
    np.random.seed(0)

    study_hours_per_week = Gamma(shape=4.0, rate=0.8, min=0.0, max=40.0).sample(100)
    attendance_rate = Beta(shape_1=8.0, shape_2=2.0, min=0.0, max=1.0).sample(100)
    prior_gpa = Beta(shape_1=5.0, shape_2=2.0, min=0.0, max=4.0).sample(100)
    school_type = CategoricalNominal(
        categories=["public", "private", "charter"],
        probabilities=[0.7, 0.2, 0.1],
    ).sample(100)

    school_private = (school_type == "private").astype(float)
    school_charter = (school_type == "charter").astype(float)
    X = np.column_stack([study_hours_per_week, attendance_rate, prior_gpa, school_private, school_charter])

    response = LogNormal(log_mean=4.0, log_standard_deviation=0.18, min=1.0, max=100.0)
    regressor = LogNormalRegressor(
        target_mean=response.target_mean(),
        target_variance=response.target_variance(),
        min=1.0,
        max=100.0,
        X=X,
        beta_1_init=np.array([0.025, 0.35, 0.12, 0.08, 0.04], dtype=float),
        predictor_names=[
            "study_hours_per_week",
            "attendance_rate",
            "prior_gpa",
            "school_type",
            "school_type",
        ],
        predictor_transformations={"study_hours_per_week": "log"},
    )

    calibrated = regressor.calibrate()
    samples = calibrated.sample(2000)

    sample_mean = float(np.mean(samples))
    sample_variance = float(np.var(samples))

    print(f"beta_0: {calibrated.beta_0}")
    print(f"sigma2: {calibrated.sigma2}")
    print(f"mean: target={regressor.target_mean} sample={sample_mean}")
    print(f"variance: target={regressor.target_variance} sample={sample_variance}")
