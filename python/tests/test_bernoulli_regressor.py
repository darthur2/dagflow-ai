import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from distributions import CategoricalNominal, Bernoulli, Gamma, LogNormal, Poisson
from regressors import BernoulliRegressor


def test_bernoulli_regressor_calibrates_and_samples():
    np.random.seed(0)

    ad_exposure_count = Poisson(rate=8.0, min=0, max=40).sample(100)
    days_since_last_purchase = Gamma(shape=2.0, rate=0.08, min=0, max=365).sample(100)
    average_order_value = LogNormal(log_mean=3.6, log_standard_deviation=0.5, min=5, max=500).sample(100)
    customer_segment = CategoricalNominal(
        categories=["new", "occasional", "loyal", "high_value"],
        probabilities=[0.35, 0.30, 0.22, 0.13],
    ).sample(100)

    occasional = (customer_segment == "occasional").astype(float)
    loyal = (customer_segment == "loyal").astype(float)
    high_value = (customer_segment == "high_value").astype(float)
    X = np.column_stack([ad_exposure_count, days_since_last_purchase, average_order_value, occasional, loyal, high_value])

    response = Bernoulli(success_prob=0.08)
    regressor = BernoulliRegressor(
        target_mean=response.target_mean(),
        X=X,
        beta_1_init=np.array([0.11, -0.9, 0.35, 0.35, 0.8, 1.1], dtype=float),
        predictor_names=[
            "ad_exposure_count",
            "days_since_last_purchase",
            "average_order_value",
            "customer_segment",
            "customer_segment",
            "customer_segment",
        ],
        predictor_transformations={"days_since_last_purchase": "log", "average_order_value": "log"},
    )

    calibrated = regressor.calibrate()
    samples = calibrated.sample(2000)

    sample_mean = float(np.mean(samples))

    print(f"beta_0: {calibrated.beta_0}")
    print(f"mean: target={regressor.target_mean} sample={sample_mean}")
