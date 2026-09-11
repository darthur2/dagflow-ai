import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from regressors import NormalRegressor

from tests.regressor_test_helpers import build_x_matrix


NORMAL_SCENARIOS = [
    {
        "name": "baseline",
        "n": 2000,
        "p": 6,
        "beta_0": 0.4,
        "beta_1_init": np.array([-1.0, 0.6, 0.25, -0.9, 1.4, 0.8], dtype=float),
        "c": 1.1,
        "sigma2": 0.65,
        "min_value": -2,
        "max_value": 5,
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
]


def _sample_truncated_normal_regression(x: np.ndarray, beta_0: float, beta_1_init: np.ndarray, c: float, sigma2: float, min_value: float, max_value: float, seed: int = 0) -> np.ndarray:
    from scipy import stats

    rng = np.random.default_rng(seed)
    sigma = np.sqrt(sigma2)
    eta = beta_0 + x @ (c * beta_1_init)
    a = (min_value - eta) / sigma
    b = (max_value - eta) / sigma
    return stats.truncnorm.rvs(a, b, loc=eta, scale=sigma, random_state=rng)


def test_normal_regressor_synthetic_calibration():
    from scipy import stats

    for scenario in NORMAL_SCENARIOS:
        x = build_x_matrix(scenario["n"], scenario["p"], seed=scenario["seed_x"])
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

        cond_mean, cond_var = fitted._conditional_moments(fitted.beta_0, fitted.c, fitted.sigma2)
        model_snr = float(np.var(cond_mean) / np.mean(cond_var))
        model_mean = float(fitted.target_mean_value(fitted.beta_0, fitted.c, fitted.sigma2))
        model_variance = float(fitted.target_variance_value(fitted.beta_0, fitted.c, fitted.sigma2))

        print(f"NormalRegressor synthetic calibration summary [{scenario['name']}]")
        print(f"  target mean      : {target_mean:.6f}")
        print(f"  model mean       : {model_mean:.6f}")
        print(f"  sample mean      : {float(np.mean(fitted_samples)):.6f}")
        print(f"  target variance  : {target_variance:.6f}")
        print(f"  model variance   : {model_variance:.6f}")
        print(f"  sample variance  : {float(np.var(fitted_samples)):.6f}")
        print(f"  target SNR       : {scenario['target_snr']:.6f}")
        print(f"  model SNR        : {model_snr:.6f}")

        assert np.isclose(float(np.mean(fitted_samples)), target_mean, rtol=0.08, atol=0.08)
        assert np.isclose(float(np.var(fitted_samples)), target_variance, rtol=0.15, atol=0.15)
        assert np.isclose(model_mean, target_mean, rtol=1e-6, atol=1e-6)
        assert np.isclose(model_variance, target_variance, rtol=1e-6, atol=1e-6)
        assert np.isclose(model_snr, scenario["target_snr"], rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    test_normal_regressor_synthetic_calibration()
