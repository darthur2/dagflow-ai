import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from distributions import CategoricalNominal, Gamma, LogNormal, Poisson
from regressors import PoissonRegressor


def test_poisson_regressor_calibrates_and_samples():
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
    X = np.column_stack([codebase_size_kloc, deployment_frequency_per_week, mean_build_time_minutes, hybrid, remote])

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


def test_poisson_regressor_days_on_market_calibrates_and_samples():
    np.random.seed(0)

    market_condition = CategoricalNominal(
        categories=["buyer's market", "balanced market", "seller's market"],
        probabilities=[0.27, 0.46, 0.27],
    ).sample(100)
    sale_season = CategoricalNominal(
        categories=["winter", "spring", "summer", "fall"],
        probabilities=[0.18, 0.32, 0.28, 0.22],
    ).sample(100)
    seller_motivation = CategoricalNominal(
        categories=["standard sale", "relocation", "estate sale", "downsizing"],
        probabilities=[0.58, 0.16, 0.14, 0.12],
    ).sample(100)

    balanced_market = (market_condition == "balanced market").astype(float)
    sellers_market = (market_condition == "seller's market").astype(float)
    spring = (sale_season == "spring").astype(float)
    summer = (sale_season == "summer").astype(float)
    fall = (sale_season == "fall").astype(float)
    relocation = (seller_motivation == "relocation").astype(float)
    estate_sale = (seller_motivation == "estate sale").astype(float)
    downsizing = (seller_motivation == "downsizing").astype(float)
    X = np.column_stack([balanced_market, sellers_market, spring, summer, fall, relocation, estate_sale, downsizing])

    response = Poisson(rate=32.0, min=0, max=180)
    regressor = PoissonRegressor(
        target_mean=response.target_mean(),
        min=0,
        max=180,
        X=X,
        beta_1_init=np.array([-0.12, -0.28, -0.06, -0.02, 0.01, -0.18, 0.12, -0.05], dtype=float),
        predictor_names=[
            "market_condition",
            "market_condition",
            "sale_season",
            "sale_season",
            "sale_season",
            "seller_motivation",
            "seller_motivation",
            "seller_motivation",
        ],
    )

    calibrated = regressor.calibrate()
    samples = calibrated.sample(2000)

    sample_mean = float(np.mean(samples))

    print(f"beta_0: {calibrated.beta_0}")
    print(f"mean: target={regressor.target_mean} sample={sample_mean}")


def test_poisson_regressor_offer_count_calibrates_and_samples():
    np.random.seed(0)

    market_condition = CategoricalNominal(
        categories=["buyer's market", "balanced market", "seller's market"],
        probabilities=[0.27, 0.46, 0.27],
    ).sample(100)
    sale_season = CategoricalNominal(
        categories=["winter", "spring", "summer", "fall"],
        probabilities=[0.18, 0.32, 0.28, 0.22],
    ).sample(100)
    seller_motivation = CategoricalNominal(
        categories=["standard sale", "relocation", "estate sale", "downsizing"],
        probabilities=[0.58, 0.16, 0.14, 0.12],
    ).sample(100)

    balanced_market = (market_condition == "balanced market").astype(float)
    sellers_market = (market_condition == "seller's market").astype(float)
    spring = (sale_season == "spring").astype(float)
    summer = (sale_season == "summer").astype(float)
    fall = (sale_season == "fall").astype(float)
    relocation = (seller_motivation == "relocation").astype(float)
    estate_sale = (seller_motivation == "estate sale").astype(float)
    downsizing = (seller_motivation == "downsizing").astype(float)
    X = np.column_stack([balanced_market, sellers_market, spring, summer, fall, relocation, estate_sale, downsizing])

    response = Poisson(rate=2.1, min=0, max=10)
    regressor = PoissonRegressor(
        target_mean=response.target_mean(),
        min=0,
        max=10,
        X=X,
        beta_1_init=np.array([0.22, 0.48, 0.1, 0.08, -0.04, 0.15, -0.12, 0.05], dtype=float),
        predictor_names=[
            "market_condition",
            "market_condition",
            "sale_season",
            "sale_season",
            "sale_season",
            "seller_motivation",
            "seller_motivation",
            "seller_motivation",
        ],
    )

    calibrated = regressor.calibrate()
    samples = calibrated.sample(2000)

    sample_mean = float(np.mean(samples))

    print(f"beta_0: {calibrated.beta_0}")
    print(f"mean: target={regressor.target_mean} sample={sample_mean}")


def main() -> None:
    test_poisson_regressor_calibrates_and_samples()
    test_poisson_regressor_days_on_market_calibrates_and_samples()
    test_poisson_regressor_offer_count_calibrates_and_samples()


if __name__ == "__main__":
    main()
