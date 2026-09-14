import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from distributions import CategoricalNominal, LogNormal, Normal
from regressors import NormalRegressor


def _build_x() -> np.ndarray:
    np.random.seed(0)

    age = Normal(mean=50.0, standard_deviation=8.0, min=18.0, max=80.0).sample(100)
    bmi = LogNormal(log_mean=3.30, log_standard_deviation=0.18, min=15.0, max=45.0).sample(100)
    rhr = Normal(mean=72.0, standard_deviation=11.0, min=40.0, max=120.0).sample(100)
    smoke = CategoricalNominal(categories=["never", "former", "current"], probabilities=[0.5, 0.3, 0.2]).sample(100)

    smoke_former = (smoke == "former").astype(float)
    smoke_current = (smoke == "current").astype(float)
    return np.column_stack([age, bmi, rhr, smoke_former, smoke_current])


def test_normal_regressor_calibrates_and_samples():
    X = _build_x()
    response = Normal(mean=125.0, standard_deviation=15.001355840493375, min=80.0, max=200.0)
    regressor = NormalRegressor(
        target_mean=response.target_mean(),
        target_variance=response.target_variance(),
        target_snr=4.5,
        min=80.0,
        max=200.0,
        X=X,
        beta_1_init=np.array([0.42, 1.15, 0.48, 2.5, 6.5], dtype=float),
        predictor_names=["age", "body_mass_index", "resting_heart_rate", "smoking_status", "smoking_status"],
        predictor_transformations={},
    )

    calibrated = regressor.calibrate()
    samples = calibrated.sample(2000)

    sample_mean = float(np.mean(samples))
    sample_variance = float(np.var(samples))
    cond_mean, cond_var = calibrated._conditional_moments(calibrated.beta_0, calibrated.c, calibrated.sigma2)
    estimated_snr = float(cond_mean.var() / cond_var.mean())

    print(f"beta_0: {calibrated.beta_0}")
    print(f"c: {calibrated.c}")
    print(f"sigma2: {calibrated.sigma2}")
    print(f"mean: target={regressor.target_mean} sample={sample_mean}")
    print(f"variance: target={regressor.target_variance} sample={sample_variance}")
    print(f"snr: target={regressor.target_snr} sample={estimated_snr}")
