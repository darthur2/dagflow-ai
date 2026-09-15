import csv
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from distributions import Beta, CategoricalNominal, Gamma, LogNormal, Poisson
from regressors import BetaRegressor, CategoricalNominalRegressor, CategoricalOrdinalRegressor, GammaRegressor, LogNormalRegressor, NoneRegressor, PoissonRegressor


def one_hot(values: np.ndarray, categories: list[str]) -> np.ndarray:
    return np.column_stack([(values == category).astype(float) for category in categories])


def test_data_generator_house_prices():
    np.random.seed(0)

    n = 1000
    state: dict[str, np.ndarray] = {}

    metro_region = CategoricalNominal(
        categories=["Midwest", "South", "Northeast", "West"],
        probabilities=[0.28, 0.33, 0.17, 0.22],
    ).sample(n)
    seller_motivation = CategoricalNominal(
        categories=["standard sale", "relocation", "estate sale", "downsizing"],
        probabilities=[0.58, 0.16, 0.14, 0.12],
    ).sample(n)
    year_built_decade = CategoricalNominal(
        categories=["1950s", "1960s", "1970s", "1980s", "1990s", "2000s", "2010s"],
        probabilities=[0.08, 0.12, 0.18, 0.16, 0.18, 0.16, 0.12],
    ).sample(n)

    state["metro_region"] = metro_region
    state["seller_motivation"] = seller_motivation
    state["year_built_decade"] = year_built_decade

    year_built_decade_X = np.column_stack([
        (year_built_decade == "1960s").astype(float),
        (year_built_decade == "1970s").astype(float),
        (year_built_decade == "1980s").astype(float),
        (year_built_decade == "1990s").astype(float),
        (year_built_decade == "2000s").astype(float),
        (year_built_decade == "2010s").astype(float),
    ])

    home_age = NoneRegressor(
        X=year_built_decade_X,
        beta_0=70,
        beta_1=np.array([-10.0, -20.0, -30.0, -40.0, -50.0, -60.0], dtype=float),
        predictor_names=["year_built_decade"] * 6,
        response_type="quantitative",
    ).sample(n)
    state["home_age"] = home_age

    metro_region_X = one_hot(metro_region, ["Midwest", "South", "Northeast", "West"])

    regressor = CategoricalNominalRegressor(
        target_probabilities=np.array([0.18, 0.24, 0.22, 0.2, 0.16], dtype=float),
        X=metro_region_X[:, 1:],
        categories=["Northfield", "Maple Grove", "Riverton", "Oak Park", "Cedar Hills"],
        beta_1=np.array([[0.25, -0.15, 0.1, 0.0], [0.1, -0.05, 0.05, 0.0], [0.05, 0.1, 0.12, 0.0]], dtype=float),
        predictor_names=["metro_region", "metro_region", "metro_region"],
    )
    municipality = regressor.calibrate().sample(n)
    state["municipality"] = municipality
    municipality_X = one_hot(municipality, ["Northfield", "Maple Grove", "Riverton", "Oak Park", "Cedar Hills"])

    regressor = CategoricalNominalRegressor(
        target_probabilities=np.array([0.34, 0.28, 0.16, 0.22], dtype=float),
        X=municipality_X[:, 1:],
        categories=["established", "newer development", "exurban", "planned community"],
        beta_1=np.array([[0.15, 0.05, 0.08], [-0.1, 0.25, -0.05], [0.2, -0.05, 0.12], [0.12, 0.18, 0.16]], dtype=float),
        predictor_names=["municipality", "municipality", "municipality", "municipality"],
    )
    neighborhood_type = regressor.calibrate().sample(n)
    state["neighborhood_type"] = neighborhood_type
    neighborhood_type_X = one_hot(neighborhood_type, ["established", "newer development", "exurban", "planned community"])

    regressor = CategoricalNominalRegressor(
        target_probabilities=np.array([0.18, 0.32, 0.28, 0.22], dtype=float),
        X=metro_region_X[:, 1:],
        categories=["winter", "spring", "summer", "fall"],
        beta_1=np.array([[0.12, -0.08, 0.06], [0.1, -0.04, 0.05], [0.08, 0.02, 0.04]], dtype=float),
        predictor_names=["metro_region", "metro_region", "metro_region"],
    )
    sale_season = regressor.calibrate().sample(n)
    state["sale_season"] = sale_season
    sale_season_X = one_hot(sale_season, ["winter", "spring", "summer", "fall"])

    regressor = CategoricalOrdinalRegressor(
        target_probabilities=np.array([0.27, 0.46, 0.27], dtype=float),
        X=np.column_stack([metro_region_X[:, 1:], sale_season_X[:, 1:]]),
        categories=["buyer's market", "balanced market", "seller's market"],
        beta_1=np.array([0.15, -0.1, 0.12, 0.45, 0.25, -0.05], dtype=float),
        predictor_names=["metro_region", "metro_region", "metro_region", "sale_season", "sale_season", "sale_season"],
    )
    market_condition = regressor.calibrate().sample(n)
    state["market_condition"] = market_condition
    balanced_market = (market_condition == "balanced market").astype(float)
    sellers_market = (market_condition == "seller's market").astype(float)
    market_condition_X = one_hot(market_condition, ["buyer's market", "balanced market", "seller's market"])

    response = Beta(shape_1=2.5, shape_2=8.0, min=0.5, max=3.5)
    regressor = BetaRegressor(
        target_mean=response.target_mean(),
        target_variance=response.target_variance(),
        min=0.5,
        max=3.5,
        X=np.column_stack([metro_region_X[:, 1:], municipality_X[:, 1:]]),
        beta_1_init=np.array([0.08, 0.18, 0.14, 0.08, -0.06, 0.05, 0.1], dtype=float),
        predictor_names=["metro_region", "metro_region", "metro_region", "municipality", "municipality", "municipality", "municipality"],
    )
    property_tax_rate = regressor.calibrate().sample(n)
    state["property_tax_rate"] = property_tax_rate

    regressor = LogNormalRegressor(
        target_mean=LogNormal(log_mean=4.0, log_standard_deviation=0.7, min=1e-6, max=1000.0).target_mean(),
        target_variance=LogNormal(log_mean=4.0, log_standard_deviation=0.7, min=1e-6, max=1000.0).target_variance(),
        min=1e-6,
        max=1000.0,
        X=municipality_X[:, 1:],
        beta_1_init=np.array([0.35, -0.15, 0.08, 0.2], dtype=float),
        predictor_names=["municipality", "municipality", "municipality", "municipality"],
    )
    hoa_fee_monthly = regressor.calibrate().sample(n)
    state["hoa_fee_monthly"] = hoa_fee_monthly

    regressor = GammaRegressor(
        target_mean=Gamma(shape=3.0, rate=0.25, min=0.5, max=40.0).target_mean(),
        target_variance=Gamma(shape=3.0, rate=0.25, min=0.5, max=40.0).target_variance(),
        min=0.5,
        max=40.0,
        X=municipality_X[:, 1:],
        beta_1_init=np.array([-0.25, 0.2, -0.1, 0.15], dtype=float),
        predictor_names=["municipality", "municipality", "municipality", "municipality"],
    )
    distance_to_downtown_mi = regressor.calibrate().sample(n)
    state["distance_to_downtown_mi"] = distance_to_downtown_mi

    regressor = CategoricalOrdinalRegressor(
        target_probabilities=np.array([0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.18, 0.22, 0.18, 0.11], dtype=float),
        X=np.column_stack([metro_region_X[:, 1:], municipality_X[:, 1:], neighborhood_type_X[:, 1:], distance_to_downtown_mi.reshape(-1, 1)]),
        categories=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
        beta_1=np.array([0.16, 0.12, 0.1, 0.3, -0.2, 0.12, 0.22, 0.4, -0.08, 0.45, -0.05], dtype=float),
        predictor_names=["metro_region", "metro_region", "metro_region", "municipality", "municipality", "municipality", "municipality", "neighborhood_type", "neighborhood_type", "neighborhood_type", "distance_to_downtown_mi"],
    )
    school_rating = regressor.calibrate().sample(n)
    state["school_rating"] = school_rating

    regressor = GammaRegressor(
        target_mean=Gamma(shape=4.0, rate=0.18, min=5.0, max=90.0).target_mean(),
        target_variance=Gamma(shape=4.0, rate=0.18, min=5.0, max=90.0).target_variance(),
        min=5.0,
        max=90.0,
        X=np.column_stack([distance_to_downtown_mi.reshape(-1, 1), municipality_X[:, 1:]]),
        beta_1_init=np.array([0.02, -0.01, 0.02, -0.01, 0.015], dtype=float),
        predictor_names=["distance_to_downtown_mi", "municipality", "municipality", "municipality", "municipality"],
    )
    commute_time_min = regressor.calibrate().sample(n)
    state["commute_time_min"] = commute_time_min

    regressor = LogNormalRegressor(
        target_mean=LogNormal(log_mean=7.4, log_standard_deviation=0.28, min=500.0, max=6000.0).target_mean(),
        target_variance=LogNormal(log_mean=7.4, log_standard_deviation=0.28, min=500.0, max=6000.0).target_variance(),
        min=500.0,
        max=6000.0,
        X=np.column_stack([neighborhood_type_X[:, 1:4], np.column_stack([
            (year_built_decade == "1960s").astype(float),
            (year_built_decade == "1970s").astype(float),
            (year_built_decade == "1980s").astype(float),
            (year_built_decade == "1990s").astype(float),
            (year_built_decade == "2000s").astype(float),
            (year_built_decade == "2010s").astype(float),
        ])]),
        beta_1_init=np.array([0.12, 0.18, 0.1, 0.05, 0.08, 0.12, 0.15, 0.18, 0.2], dtype=float),
        predictor_names=["neighborhood_type", "neighborhood_type", "neighborhood_type", "year_built_decade", "year_built_decade", "year_built_decade", "year_built_decade", "year_built_decade", "year_built_decade"],
    )
    sqft_living = regressor.calibrate().sample(n)
    state["sqft_living"] = sqft_living

    regressor = LogNormalRegressor(
        target_mean=LogNormal(log_mean=8.8, log_standard_deviation=0.75, min=1500.0, max=50000.0).target_mean(),
        target_variance=LogNormal(log_mean=8.8, log_standard_deviation=0.75, min=1500.0, max=50000.0).target_variance(),
        min=1500.0,
        max=50000.0,
        X=np.column_stack([neighborhood_type_X[:, 1:4], municipality_X[:, 1:], np.column_stack([
            (year_built_decade == "1960s").astype(float),
            (year_built_decade == "1970s").astype(float),
            (year_built_decade == "1980s").astype(float),
            (year_built_decade == "1990s").astype(float),
            (year_built_decade == "2000s").astype(float),
            (year_built_decade == "2010s").astype(float),
        ])]),
        beta_1_init=np.array([0.12, -0.2, 0.08, 0.12, 0.22, -0.05, 0.18, -0.05, -0.08, -0.04, 0.03, 0.05, 0.06], dtype=float),
        predictor_names=["neighborhood_type", "neighborhood_type", "neighborhood_type", "municipality", "municipality", "municipality", "municipality", "year_built_decade", "year_built_decade", "year_built_decade", "year_built_decade", "year_built_decade", "year_built_decade"],
    )
    lot_size_sqft = regressor.calibrate().sample(n)
    state["lot_size_sqft"] = lot_size_sqft

    regressor = PoissonRegressor(target_mean=Poisson(rate=3.8, min=1, max=7).target_mean(), min=1, max=7, X=sqft_living.reshape(-1, 1), beta_1_init=np.array([0.28], dtype=float), predictor_names=["sqft_living"], predictor_transformations={"sqft_living": "log"})
    bedrooms = regressor.calibrate().sample(n)
    state["bedrooms"] = bedrooms

    regressor = PoissonRegressor(target_mean=Poisson(rate=2.6, min=1, max=6).target_mean(), min=1, max=6, X=np.column_stack([sqft_living, bedrooms]), beta_1_init=np.array([0.22, 0.3], dtype=float), predictor_names=["sqft_living", "bedrooms"], predictor_transformations={"sqft_living": "log", "bedrooms": "none"})
    bathrooms = regressor.calibrate().sample(n)
    state["bathrooms"] = bathrooms

    regressor = PoissonRegressor(target_mean=Poisson(rate=0.7, min=0, max=3).target_mean(), min=0, max=3, X=np.column_stack([neighborhood_type_X[:, 1:4], np.column_stack([
        (year_built_decade == "1960s").astype(float),
        (year_built_decade == "1970s").astype(float),
        (year_built_decade == "1980s").astype(float),
        (year_built_decade == "1990s").astype(float),
        (year_built_decade == "2000s").astype(float),
        (year_built_decade == "2010s").astype(float),
    ])]), beta_1_init=np.array([0.1, 0.08, 0.18, 0.05, 0.08, 0.12, 0.15, 0.18, 0.2], dtype=float), predictor_names=["neighborhood_type", "neighborhood_type", "neighborhood_type", "year_built_decade", "year_built_decade", "year_built_decade", "year_built_decade", "year_built_decade", "year_built_decade"])
    garage_spaces = regressor.calibrate().sample(n)
    state["garage_spaces"] = garage_spaces

    has_pool = CategoricalNominalRegressor(target_probabilities=np.array([0.82, 0.18], dtype=float), X=np.column_stack([neighborhood_type_X[:, 1:4], sqft_living.reshape(-1, 1)]), categories=["no", "yes"], beta_1=np.array([[0.25], [0.2], [0.3], [0.18]], dtype=float), predictor_names=["neighborhood_type", "neighborhood_type", "neighborhood_type", "sqft_living"], predictor_transformations={"sqft_living": "log"}).calibrate().sample(n)
    state["has_pool"] = has_pool

    has_basement = CategoricalNominalRegressor(target_probabilities=np.array([0.45, 0.55], dtype=float), X=np.column_stack([neighborhood_type_X[:, 1:4], np.column_stack([
        (year_built_decade == "1960s").astype(float),
        (year_built_decade == "1970s").astype(float),
        (year_built_decade == "1980s").astype(float),
        (year_built_decade == "1990s").astype(float),
        (year_built_decade == "2000s").astype(float),
        (year_built_decade == "2010s").astype(float),
    ])]), categories=["no", "yes"], beta_1=np.array([[-0.15], [0.1], [-0.08], [0.1], [0.15], [0.08], [-0.05], [-0.1], [-0.12]], dtype=float), predictor_names=["neighborhood_type", "neighborhood_type", "neighborhood_type", "year_built_decade", "year_built_decade", "year_built_decade", "year_built_decade", "year_built_decade", "year_built_decade"]).calibrate().sample(n)
    state["has_basement"] = has_basement

    recent_renovation = CategoricalNominalRegressor(target_probabilities=np.array([0.78, 0.22], dtype=float), X=np.column_stack([home_age.reshape(-1, 1), np.column_stack([(seller_motivation == "relocation").astype(float), (seller_motivation == "estate sale").astype(float), (seller_motivation == "downsizing").astype(float)])]), categories=["no", "yes"], beta_1=np.array([[0.12], [-0.2], [0.05], [-0.03]], dtype=float), predictor_names=["home_age", "seller_motivation", "seller_motivation", "seller_motivation"]).calibrate().sample(n)
    state["recent_renovation"] = recent_renovation

    list_price = LogNormalRegressor(target_mean=LogNormal(log_mean=12.62, log_standard_deviation=0.34, min=85000.0, max=2600000.0).target_mean(), target_variance=LogNormal(log_mean=12.62, log_standard_deviation=0.34, min=85000.0, max=2600000.0).target_variance(), min=85000.0, max=2600000.0, X=np.column_stack([market_condition_X[:, 1:3], sale_season_X[:, 1:4], np.column_stack([(seller_motivation == "relocation").astype(float), (seller_motivation == "estate sale").astype(float), (seller_motivation == "downsizing").astype(float)]), np.log(lot_size_sqft).reshape(-1, 1)]), beta_1_init=np.array([0.04, 0.09, 0.02, 0.01, 0.0, -0.03, -0.08, -0.02, 0.06], dtype=float), predictor_names=["market_condition", "market_condition", "sale_season", "sale_season", "sale_season", "seller_motivation", "seller_motivation", "seller_motivation", "lot_size_sqft"], predictor_transformations={"lot_size_sqft": "log"}).calibrate().sample(n)
    state["list_price"] = list_price

    sale_price = LogNormalRegressor(target_mean=LogNormal(log_mean=12.6, log_standard_deviation=0.35, min=80000.0, max=2500000.0).target_mean(), target_variance=LogNormal(log_mean=12.6, log_standard_deviation=0.35, min=80000.0, max=2500000.0).target_variance(), min=80000.0, max=2500000.0, X=np.column_stack([list_price.reshape(-1, 1), lot_size_sqft.reshape(-1, 1), balanced_market.reshape(-1, 1), sellers_market.reshape(-1, 1)]), beta_1_init=np.array([0.000002, 0.000001, 0.03, 0.08], dtype=float), predictor_names=["list_price", "lot_size_sqft", "market_condition", "market_condition"], predictor_transformations={"lot_size_sqft": "log"}).calibrate().sample(n)
    state["sale_price"] = sale_price

    days_on_market = PoissonRegressor(target_mean=Poisson(rate=32.0, min=0, max=180).target_mean(), min=0, max=180, X=np.column_stack([balanced_market.reshape(-1, 1), sellers_market.reshape(-1, 1), sale_season_X[:, 1:4], np.column_stack([(seller_motivation == "relocation").astype(float), (seller_motivation == "estate sale").astype(float), (seller_motivation == "downsizing").astype(float)])]), beta_1_init=np.array([-0.12, -0.28, -0.06, -0.02, 0.01, -0.18, 0.12, -0.05], dtype=float), predictor_names=["market_condition", "market_condition", "sale_season", "sale_season", "sale_season", "seller_motivation", "seller_motivation", "seller_motivation"]).calibrate().sample(n)
    state["days_on_market"] = days_on_market

    offer_count = PoissonRegressor(target_mean=Poisson(rate=2.1, min=0, max=10).target_mean(), min=0, max=10, X=np.column_stack([balanced_market.reshape(-1, 1), sellers_market.reshape(-1, 1), sale_season_X[:, 1:4], np.column_stack([(seller_motivation == "relocation").astype(float), (seller_motivation == "estate sale").astype(float), (seller_motivation == "downsizing").astype(float)])]), beta_1_init=np.array([0.22, 0.48, 0.1, 0.08, -0.04, 0.15, -0.12, 0.05], dtype=float), predictor_names=["market_condition", "market_condition", "sale_season", "sale_season", "sale_season", "seller_motivation", "seller_motivation", "seller_motivation"]).calibrate().sample(n)
    state["offer_count"] = offer_count

    above_ask_sale = NoneRegressor(
        X=np.column_stack([
            np.log(sale_price),
            np.log(list_price),
            offer_count.astype(float),
            (market_condition == "balanced market").astype(float),
            (market_condition == "seller's market").astype(float),
        ]),
        beta_0=-1.0,
        beta_1=np.array([0.2, -0.24, 0.18, 0.2, 0.45], dtype=float),
        predictor_names=["sale_price", "list_price", "offer_count", "market_condition", "market_condition"],
        predictor_transformations={"sale_price": "log", "list_price": "log"},
        response_type="categorical_nominal",
        categories=["no", "yes"],
    ).sample(n)
    state["above_ask_sale"] = above_ask_sale

    output_path = Path(__file__).resolve().parents[2] / "synthdata" / "generated_data.csv"
    fieldnames = [
        "metro_region",
        "municipality",
        "neighborhood_type",
        "market_condition",
        "sale_season",
        "seller_motivation",
        "property_tax_rate",
        "hoa_fee_monthly",
        "school_rating",
        "distance_to_downtown_mi",
        "commute_time_min",
        "year_built_decade",
        "home_age",
        "sqft_living",
        "lot_size_sqft",
        "bedrooms",
        "bathrooms",
        "garage_spaces",
        "has_pool",
        "has_basement",
        "recent_renovation",
        "list_price",
        "sale_price",
        "days_on_market",
        "offer_count",
        "above_ask_sale",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(n):
            writer.writerow({name: state[name][idx] for name in fieldnames})

    print(f"Wrote {output_path}")


def main() -> None:
    test_data_generator_house_prices()


if __name__ == "__main__":
    main()
