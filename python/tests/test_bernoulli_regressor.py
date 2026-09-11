import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from regressors import BernoulliRegressor

from tests.regressor_test_helpers import build_x_matrix


BERNOULLI_SCENARIOS = [
    {
        "name": "bernoulli_mid_snr",
        "n": 1800,
        "p": 6,
        "beta_0": -0.15,
        "beta_1_init": np.array([0.9, -0.6, 0.4, 0.7, -0.3, 1.0], dtype=float),
        "c": 0.8,
        "target_snr": 0.9,
        "seed_x": 42424,
        "seed_y": 52525,
    },
    {
        "name": "bernoulli_lower_snr",
        "n": 1800,
        "p": 6,
        "beta_0": 0.1,
        "beta_1_init": np.array([-0.7, 0.5, -0.4, 0.8, 0.3, -0.9], dtype=float),
        "c": 0.55,
        "target_snr": 0.4,
        "seed_x": 63636,
        "seed_y": 74747,
    },
    {
        "name": "bernoulli_higher_snr",
        "n": 1800,
        "p": 6,
        "beta_0": -0.25,
        "beta_1_init": np.array([1.0, -0.8, 0.6, 0.4, -0.5, 0.9], dtype=float),
        "c": 1.0,
        "target_snr": 1.6,
        "seed_x": 85858,
        "seed_y": 96969,
    },
]


def _sample_bernoulli_regression(x: np.ndarray, beta_0: float, beta_1_init: np.ndarray, c: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eta = beta_0 + x @ (c * beta_1_init)
    p = 1.0 / (1.0 + np.exp(-eta))
    return np.asarray(np.random.default_rng(rng.integers(0, 2**32 - 1)).binomial(1, p), dtype=float)


def run_bernoulli_scenario(scenario: dict) -> dict[str, float]:
    x = build_x_matrix(scenario["n"], scenario["p"], seed=scenario["seed_x"])
    samples = _sample_bernoulli_regression(
        x,
        scenario["beta_0"],
        scenario["beta_1_init"],
        scenario["c"],
        seed=scenario["seed_y"],
    )
    target_mean = float(np.mean(samples))

    regressor = BernoulliRegressor(
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
        "sample_binary": bool(np.all(np.logical_or(fitted_samples == 0, fitted_samples == 1))),
        "fitted_beta_0": float(fitted.beta_0),
        "fitted_c": float(fitted.c),
    }


def test_bernoulli_regressor_synthetic_calibration():
    for scenario in BERNOULLI_SCENARIOS:
        results = run_bernoulli_scenario(scenario)

        print(f"BernoulliRegressor synthetic calibration summary [{scenario['name']}]")
        print(f"  target mean      : {results['target_mean']:.6f}")
        print(f"  model mean       : {results['model_mean']:.6f}")
        print(f"  sample mean      : {results['sample_mean']:.6f}")
        print(f"  target SNR       : {results['target_snr']:.6f}")
        print(f"  model SNR        : {results['model_snr']:.6f}")
        print(f"  fitted beta_0    : {results['fitted_beta_0']:.6f}")
        print(f"  fitted c         : {results['fitted_c']:.6f}")

        assert np.isclose(results["sample_mean"], results["target_mean"], rtol=0.12, atol=0.12)
        assert np.isclose(results["model_mean"], results["target_mean"], rtol=0.04, atol=0.04)
        assert np.isclose(results["model_snr"], results["target_snr"], rtol=0.25, atol=0.25)
        assert results["sample_binary"]
        assert np.isfinite(results["fitted_beta_0"])
        assert np.isfinite(results["fitted_c"])


if __name__ == "__main__":
    test_bernoulli_regressor_synthetic_calibration()
