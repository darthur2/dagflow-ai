import numpy as np
import pytest
from scipy import stats

from distributions import Beta, Bernoulli, DiscreteUniform, Exponential, Gamma, Geometric, LogNormal, NegativeBinomial, Normal, Poisson, Uniform
from regressors import ExponentialRegressor, NormalRegressor


FIXTURE_DISTRIBUTIONS = (
    Normal(mean=0.0, standard_deviation=1.0, min=-2.0, max=2.0),
    Exponential(rate=1.5, min=0.0, max=4.0),
    Gamma(shape=2.0, rate=1.2, min=0.0, max=8.0),
    Beta(shape_1=2.0, shape_2=5.0, min=-1.0, max=3.0),
    LogNormal(log_mean=0.0, log_standard_deviation=0.5, min=0.1, max=5.0),
    Uniform(min=-3.0, max=3.0),
    DiscreteUniform(min=1, max=8),
    Bernoulli(success_prob=0.35),
    stats.binom(n=6, p=0.4),
    Poisson(rate=3.0, min=0, max=12),
    Geometric(success_prob=0.25, min=1, max=15),
    NegativeBinomial(shape=4.0, mean=6.0, min=0, max=20),
)


NORMAL_SCENARIOS = [
    {
        "name": "baseline",
        "n": 2000,
        "p": 6,
        "beta_0": 0.4,
        "beta_1_init": np.array([-1.0, 0.6, 0.25, -0.9, 1.4, 0.8], dtype=float),
        "c": 1.1,
        "sigma2": 0.65,
        "min_value": -1.5,
        "max_value": 3.5,
        "target_snr": 3.2,
        "seed_x": 123,
        "seed_y": 321,
    },
    {
        "name": "wide_bounds",
        "n": 1800,
        "p": 6,
        "beta_0": -0.85,
        "beta_1_init": np.array([0.5, 1.2, -0.8, 0.35, 0.9, -1.1], dtype=float),
        "c": 0.75,
        "sigma2": 1.35,
        "min_value": -3.5,
        "max_value": 4.5,
        "target_snr": 1.8,
        "seed_x": 456,
        "seed_y": 654,
    },
    {
        "name": "moderate_signal",
        "n": 2200,
        "p": 6,
        "beta_0": 1.25,
        "beta_1_init": np.array([0.8, -0.4, 1.1, 0.3, -0.7, 0.5], dtype=float),
        "c": 1.4,
        "sigma2": 0.9,
        "min_value": -2.0,
        "max_value": 5.0,
        "target_snr": 2.5,
        "seed_x": 789,
        "seed_y": 987,
    },
    {
        "name": "higher_signal",
        "n": 1600,
        "p": 6,
        "beta_0": -0.2,
        "beta_1_init": np.array([1.1, -0.6, 0.4, 0.7, -1.3, 0.2], dtype=float),
        "c": 1.25,
        "sigma2": 0.8,
        "min_value": -6.0,
        "max_value": 4.0,
        "target_snr": 4.0,
        "seed_x": 321,
        "seed_y": 123,
    },
    {
        "name": "effectively_untruncated_high_snr",
        "n": 1600,
        "p": 6,
        "beta_0": 0.15,
        "beta_1_init": np.array([0.7, -0.9, 0.5, 1.0, -0.4, 0.8], dtype=float),
        "c": 1.3,
        "sigma2": 0.55,
        "min_value": -20.0,
        "max_value": 20.0,
        "target_snr": 6.0,
        "seed_x": 222,
        "seed_y": 444,
    },
]


EXPONENTIAL_SCENARIOS = [
    {
        "name": "exp_low_snr_01",
        "n": 1800,
        "p": 6,
        "beta_0": 0.15,
        "beta_1_init": np.array([0.7, -0.5, 1.0, 0.2, -0.8, 0.4], dtype=float),
        "c": 0.6,
        "target_snr": 0.1,
        "seed_x": 111,
        "seed_y": 222,
    },
    {
        "name": "exp_low_snr_025",
        "n": 1800,
        "p": 6,
        "beta_0": -0.25,
        "beta_1_init": np.array([-0.4, 0.9, 0.6, -0.7, 0.8, -1.0], dtype=float),
        "c": 0.55,
        "target_snr": 0.25,
        "seed_x": 333,
        "seed_y": 444,
    },
    {
        "name": "exp_low_snr_05",
        "n": 1800,
        "p": 6,
        "beta_0": 0.4,
        "beta_1_init": np.array([1.0, -0.3, 0.5, 0.8, -0.6, 0.2], dtype=float),
        "c": 0.45,
        "target_snr": 0.5,
        "seed_x": 555,
        "seed_y": 666,
    },
    {
        "name": "exp_low_snr_075",
        "n": 1800,
        "p": 6,
        "beta_0": -0.1,
        "beta_1_init": np.array([0.2, 1.1, -0.9, 0.4, 0.7, -0.5], dtype=float),
        "c": 0.35,
        "target_snr": 0.75,
        "seed_x": 777,
        "seed_y": 888,
    },
]


def _build_x_matrix(n: int, p: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    columns = []
    for idx in range(p):
        distribution = FIXTURE_DISTRIBUTIONS[idx % len(FIXTURE_DISTRIBUTIONS)]
        if hasattr(distribution, "sample"):
            columns.append(np.asarray(distribution.sample(n), dtype=float))
        else:
            columns.append(np.asarray(distribution.rvs(size=n, random_state=rng), dtype=float))
    return np.column_stack(columns)


def _sample_truncated_normal_regression(x: np.ndarray, beta_0: float, beta_1_init: np.ndarray, c: float, sigma2: float, min_value: float, max_value: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sigma = np.sqrt(sigma2)
    eta = beta_0 + x @ (c * beta_1_init)
    a = (min_value - eta) / sigma
    b = (max_value - eta) / sigma
    return stats.truncnorm.rvs(a, b, loc=eta, scale=sigma, random_state=rng)


def _sample_truncated_exponential_regression(x: np.ndarray, beta_0: float, beta_1_init: np.ndarray, c: float, min_value: float, max_value: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eta = beta_0 + x @ (c * beta_1_init)
    lam = np.exp(-eta)
    samples = np.empty(len(x), dtype=float)
    for idx, current_lambda in enumerate(lam):
        u = rng.random()
        lower = np.exp(-current_lambda * min_value)
        upper = np.exp(-current_lambda * max_value)
        samples[idx] = -np.log(lower - u * (lower - upper)) / current_lambda
    return samples


def _run_normal_scenario(scenario: dict) -> dict[str, float]:
    x = _build_x_matrix(scenario["n"], scenario["p"], seed=scenario["seed_x"])
    samples = _sample_truncated_normal_regression(
        x,
        scenario["beta_0"],
        scenario["beta_1_init"],
        scenario["c"],
        scenario["sigma2"],
        scenario["min_value"],
        scenario["max_value"],
        seed=scenario["seed_y"],
    )

    target_mean = float(np.mean(samples))
    target_variance = float(np.var(samples))

    regressor = NormalRegressor(
        target_mean=target_mean,
        target_variance=target_variance,
        min=scenario["min_value"],
        max=scenario["max_value"],
        X=x,
        beta_1_init=scenario["beta_1_init"],
        target_snr=scenario["target_snr"],
    )

    fitted = regressor.calibrate()
    fitted_samples = fitted.sample(scenario["n"])

    try:
        cond_mean, cond_var = fitted._conditional_moments(fitted.beta_0, fitted.c, fitted.sigma2)
        model_snr = float(np.var(cond_mean) / np.mean(cond_var))
        model_mean = float(fitted.target_mean_value(fitted.beta_0, fitted.c, fitted.sigma2))
        model_variance = float(fitted.target_variance_value(fitted.beta_0, fitted.c, fitted.sigma2))
    except ValueError:
        cond_mean = None
        cond_var = None
        model_snr = float("nan")
        model_mean = float("nan")
        model_variance = float("nan")

    return {
        "target_mean": target_mean,
        "target_variance": target_variance,
        "target_snr": scenario["target_snr"],
        "model_mean": model_mean,
        "model_variance": model_variance,
        "model_snr": model_snr,
        "sample_mean": float(np.mean(fitted_samples)),
        "sample_variance": float(np.var(fitted_samples)),
        "fitted_beta_0": float(fitted.beta_0),
        "fitted_c": float(fitted.c),
        "fitted_sigma2": float(fitted.sigma2),
        "model_valid": cond_mean is not None and cond_var is not None,
    }


def _run_exponential_scenario(scenario: dict) -> dict[str, float]:
    x = _build_x_matrix(scenario["n"], scenario["p"], seed=scenario["seed_x"])
    samples = _sample_truncated_exponential_regression(
        x,
        scenario["beta_0"],
        scenario["beta_1_init"],
        scenario["c"],
        0.0,
        1000.0,
        seed=scenario["seed_y"],
    )
    target_mean = float(np.mean(samples))

    regressor = ExponentialRegressor(
        target_mean=target_mean,
        target_snr=scenario["target_snr"],
        X=x,
        beta_1_init=scenario["beta_1_init"],
    )

    fitted = regressor.calibrate()
    fitted_samples = fitted.sample(scenario["n"])
    model_mean = float(fitted.target_mean_value(fitted.beta_0, fitted.c))
    model_snr = float(fitted.target_snr_value(fitted.beta_0, fitted.c))

    return {
        "target_mean": target_mean,
        "target_snr": scenario["target_snr"],
        "model_mean": model_mean,
        "model_snr": model_snr,
        "sample_mean": float(np.mean(fitted_samples)),
        "fitted_beta_0": float(fitted.beta_0),
        "fitted_c": float(fitted.c),
    }


@pytest.mark.parametrize("scenario", NORMAL_SCENARIOS, ids=[scenario["name"] for scenario in NORMAL_SCENARIOS])
def test_normal_regressor_synthetic_calibration(scenario):
    results = _run_normal_scenario(scenario)

    print(f"NormalRegressor synthetic calibration summary [{scenario['name']}]")
    print(f"  target mean      : {results['target_mean']:.6f}")
    print(f"  model mean       : {results['model_mean']:.6f}")
    print(f"  sample mean      : {results['sample_mean']:.6f}")
    print(f"  target variance  : {results['target_variance']:.6f}")
    print(f"  model variance   : {results['model_variance']:.6f}")
    print(f"  sample variance  : {results['sample_variance']:.6f}")
    print(f"  target SNR       : {results['target_snr']:.6f}")
    print(f"  model SNR        : {results['model_snr']:.6f}")
    print(f"  fitted beta_0    : {results['fitted_beta_0']:.6f}")
    print(f"  fitted c         : {results['fitted_c']:.6f}")
    print(f"  fitted sigma2    : {results['fitted_sigma2']:.6f}")

    assert np.isclose(results["sample_mean"], results["target_mean"], rtol=0.08, atol=0.08)
    assert np.isclose(results["sample_variance"], results["target_variance"], rtol=0.15, atol=0.15)
    if results["model_valid"]:
        assert np.isclose(results["model_mean"], results["target_mean"], rtol=1e-6, atol=1e-6)
        assert np.isclose(results["model_variance"], results["target_variance"], rtol=1e-6, atol=1e-6)
        assert np.isclose(results["model_snr"], results["target_snr"], rtol=1e-6, atol=1e-6)
    assert np.isfinite(results["fitted_beta_0"])
    assert np.isfinite(results["fitted_c"])
    assert np.isfinite(results["fitted_sigma2"])


@pytest.mark.parametrize("scenario", EXPONENTIAL_SCENARIOS, ids=[scenario["name"] for scenario in EXPONENTIAL_SCENARIOS])
def test_exponential_regressor_synthetic_calibration(scenario):
    results = _run_exponential_scenario(scenario)

    print(f"ExponentialRegressor synthetic calibration summary [{scenario['name']}]")
    print(f"  target mean      : {results['target_mean']:.6f}")
    print(f"  model mean       : {results['model_mean']:.6f}")
    print(f"  sample mean      : {results['sample_mean']:.6f}")
    print(f"  target SNR       : {results['target_snr']:.6f}")
    print(f"  model SNR        : {results['model_snr']:.6f}")
    print(f"  fitted beta_0    : {results['fitted_beta_0']:.6f}")
    print(f"  fitted c         : {results['fitted_c']:.6f}")

    assert np.isclose(results["model_mean"], results["target_mean"], rtol=1e-6, atol=1e-6)
    assert np.isclose(results["model_snr"], results["target_snr"], rtol=1e-6, atol=1e-6)
    assert np.isfinite(results["fitted_beta_0"])
    assert np.isfinite(results["fitted_c"])


if __name__ == "__main__":
    for scenario in NORMAL_SCENARIOS:
        test_normal_regressor_synthetic_calibration(scenario)
    for scenario in EXPONENTIAL_SCENARIOS:
        test_exponential_regressor_synthetic_calibration(scenario)
