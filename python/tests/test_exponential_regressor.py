import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from regressors import ExponentialRegressor

from tests.regressor_test_helpers import build_x_matrix


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
]


def _sample_truncated_exponential_regression(x: np.ndarray, beta_0: float, beta_1_init: np.ndarray, c: float, min_value: float, max_value: float, seed: int = 0) -> np.ndarray:
    from scipy import stats

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


def test_exponential_regressor_synthetic_calibration():
    for scenario in EXPONENTIAL_SCENARIOS:
        x = build_x_matrix(scenario["n"], scenario["p"], seed=scenario["seed_x"])
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

        print(f"ExponentialRegressor synthetic calibration summary [{scenario['name']}]")
        print(f"  target mean      : {target_mean:.6f}")
        print(f"  model mean       : {model_mean:.6f}")
        print(f"  sample mean      : {float(np.mean(fitted_samples)):.6f}")
        print(f"  target SNR       : {scenario['target_snr']:.6f}")
        print(f"  model SNR        : {model_snr:.6f}")

        assert np.isclose(model_mean, target_mean, rtol=1e-6, atol=1e-6)
        assert np.isclose(model_snr, scenario["target_snr"], rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    test_exponential_regressor_synthetic_calibration()
