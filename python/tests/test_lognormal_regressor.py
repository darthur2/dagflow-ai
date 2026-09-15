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


def test_lognormal_regressor_hoa_fee_monthly_calibrates_and_samples():
    np.random.seed(0)

    municipality = CategoricalNominal(
        categories=["Northfield", "Maple Grove", "Riverton", "Oak Park", "Cedar Hills"],
        probabilities=[0.18, 0.24, 0.22, 0.2, 0.16],
    ).sample(100)

    maple_grove = (municipality == "Maple Grove").astype(float)
    riverton = (municipality == "Riverton").astype(float)
    oak_park = (municipality == "Oak Park").astype(float)
    cedar_hills = (municipality == "Cedar Hills").astype(float)
    X = np.column_stack([maple_grove, riverton, oak_park, cedar_hills])

    response = LogNormal(log_mean=4.0, log_standard_deviation=0.7, min=1e-6, max=1000.0)
    regressor = LogNormalRegressor(
        target_mean=response.target_mean(),
        target_variance=response.target_variance(),
        min=1e-6,
        max=1000.0,
        X=X,
        beta_1_init=np.array([0.35, -0.15, 0.08, 0.2], dtype=float),
        predictor_names=["municipality", "municipality", "municipality", "municipality"],
    )

    calibrated = regressor.calibrate()
    samples = calibrated.sample(2000)

    sample_mean = float(np.mean(samples))
    sample_variance = float(np.var(samples))

    print(f"beta_0: {calibrated.beta_0}")
    print(f"sigma2: {calibrated.sigma2}")
    print(f"mean: target={regressor.target_mean} sample={sample_mean}")
    print(f"variance: target={regressor.target_variance} sample={sample_variance}")


def test_lognormal_regressor_sqft_living_calibrates_and_samples():
    np.random.seed(0)

    neighborhood_type = CategoricalNominal(
        categories=["established", "newer development", "exurban", "planned community"],
        probabilities=[0.34, 0.28, 0.16, 0.22],
    ).sample(100)
    year_built_decade = CategoricalNominal(
        categories=["1950s", "1960s", "1970s", "1980s", "1990s", "2000s", "2010s"],
        probabilities=[0.08, 0.12, 0.18, 0.16, 0.18, 0.16, 0.12],
    ).sample(100)

    newer_development = (neighborhood_type == "newer development").astype(float)
    exurban = (neighborhood_type == "exurban").astype(float)
    planned_community = (neighborhood_type == "planned community").astype(float)

    decade_1960s = (year_built_decade == "1960s").astype(float)
    decade_1970s = (year_built_decade == "1970s").astype(float)
    decade_1980s = (year_built_decade == "1980s").astype(float)
    decade_1990s = (year_built_decade == "1990s").astype(float)
    decade_2000s = (year_built_decade == "2000s").astype(float)
    decade_2010s = (year_built_decade == "2010s").astype(float)

    X = np.column_stack([
        newer_development,
        exurban,
        planned_community,
        decade_1960s,
        decade_1970s,
        decade_1980s,
        decade_1990s,
        decade_2000s,
        decade_2010s,
    ])

    response = LogNormal(log_mean=7.4, log_standard_deviation=0.28, min=500.0, max=6000.0)
    regressor = LogNormalRegressor(
        target_mean=response.target_mean(),
        target_variance=response.target_variance(),
        min=500.0,
        max=6000.0,
        X=X,
        beta_1_init=np.array([0.12, 0.18, 0.1, 0.05, 0.08, 0.12, 0.15, 0.18, 0.2], dtype=float),
        predictor_names=[
            "neighborhood_type",
            "neighborhood_type",
            "neighborhood_type",
            "year_built_decade",
            "year_built_decade",
            "year_built_decade",
            "year_built_decade",
            "year_built_decade",
            "year_built_decade",
        ],
    )

    calibrated = regressor.calibrate()
    samples = calibrated.sample(2000)

    sample_mean = float(np.mean(samples))
    sample_variance = float(np.var(samples))

    print(f"beta_0: {calibrated.beta_0}")
    print(f"sigma2: {calibrated.sigma2}")
    print(f"mean: target={regressor.target_mean} sample={sample_mean}")
    print(f"variance: target={regressor.target_variance} sample={sample_variance}")


def test_lognormal_regressor_list_price_calibrates_and_samples():
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

    response = LogNormal(log_mean=12.62, log_standard_deviation=0.34, min=85000.0, max=2600000.0)
    regressor = LogNormalRegressor(
        target_mean=response.target_mean(),
        target_variance=response.target_variance(),
        min=85000.0,
        max=2600000.0,
        X=X,
        beta_1_init=np.array([0.04, 0.09, 0.02, 0.01, 0.0, -0.03, -0.08, -0.02], dtype=float),
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
    sample_variance = float(np.var(samples))

    print(f"beta_0: {calibrated.beta_0}")
    print(f"sigma2: {calibrated.sigma2}")
    print(f"mean: target={regressor.target_mean} sample={sample_mean}")
    print(f"variance: target={regressor.target_variance} sample={sample_variance}")


def test_lognormal_regressor_sale_price_calibrates_and_samples():
    np.random.seed(0)

    list_price = LogNormal(log_mean=12.62, log_standard_deviation=0.34, min=85000.0, max=2600000.0).sample(100)
    market_condition = CategoricalNominal(
        categories=["buyer's market", "balanced market", "seller's market"],
        probabilities=[0.27, 0.46, 0.27],
    ).sample(100)

    balanced_market = (market_condition == "balanced market").astype(float)
    sellers_market = (market_condition == "seller's market").astype(float)
    X = np.column_stack([list_price, balanced_market, sellers_market])

    response = LogNormal(log_mean=12.6, log_standard_deviation=0.35, min=80000.0, max=2500000.0)
    # The list_price coefficient is deliberately tiny so the lognormal latent scale stays feasible.
    regressor = LogNormalRegressor(
        target_mean=response.target_mean(),
        target_variance=response.target_variance(),
        min=80000.0,
        max=2500000.0,
        X=X,
        beta_1_init=np.array([0.000002, 0.03, 0.08], dtype=float),
        predictor_names=["list_price", "market_condition", "market_condition"],
    )

    calibrated = regressor.calibrate()
    samples = calibrated.sample(2000)

    sample_mean = float(np.mean(samples))
    sample_variance = float(np.var(samples))

    print(f"beta_0: {calibrated.beta_0}")
    print(f"sigma2: {calibrated.sigma2}")
    print(f"mean: target={regressor.target_mean} sample={sample_mean}")
    print(f"variance: target={regressor.target_variance} sample={sample_variance}")


def main() -> None:
    test_lognormal_regressor_calibrates_and_samples()
    test_lognormal_regressor_hoa_fee_monthly_calibrates_and_samples()
    test_lognormal_regressor_sqft_living_calibrates_and_samples()
    test_lognormal_regressor_list_price_calibrates_and_samples()
    test_lognormal_regressor_sale_price_calibrates_and_samples()


if __name__ == "__main__":
    main()
