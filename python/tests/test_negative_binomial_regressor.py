import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from distributions import CategoricalNominal, Gamma, LogNormal, NegativeBinomial, Poisson
from regressors import NegativeBinomialRegressor


def test_negative_binomial_regressor_calibrates_and_samples():
    np.random.seed(0)

    ad_spend = LogNormal(log_mean=6.0, log_standard_deviation=0.35, min=50.0, max=5000.0).sample(100)
    email_touchpoints = Poisson(rate=18.0, min=0, max=60).sample(100)
    website_sessions = Gamma(shape=5.0, rate=0.08, min=0.0, max=200.0).sample(100)
    customer_segment = CategoricalNominal(
        categories=["new", "active", "loyal"],
        probabilities=[0.35, 0.45, 0.20],
    ).sample(100)

    active = (customer_segment == "active").astype(float)
    loyal = (customer_segment == "loyal").astype(float)
    X = np.column_stack([ad_spend, email_touchpoints, website_sessions, active, loyal])

    response = NegativeBinomial(shape=4.0, mean=12.0, min=0, max=50)
    regressor = NegativeBinomialRegressor(
        target_mean=response.target_mean(),
        target_variance=response.target_variance(),
        min=0,
        max=50,
        X=X,
        beta_1_init=np.array([0.00018, 0.012, 0.004, 0.15, 0.28], dtype=float),
        predictor_names=[
            "ad_spend",
            "email_touchpoints",
            "website_sessions",
            "customer_segment",
            "customer_segment",
        ],
        predictor_transformations={"ad_spend": "log", "website_sessions": "log"},
    )

    calibrated = regressor.calibrate()
    samples = calibrated.sample(2000)

    sample_mean = float(np.mean(samples))
    sample_variance = float(np.var(samples))

    print(f"beta_0: {calibrated.beta_0}")
    print(f"shape: {calibrated.shape}")
    print(f"mean: target={regressor.target_mean} sample={sample_mean}")
    print(f"variance: target={regressor.target_variance} sample={sample_variance}")
