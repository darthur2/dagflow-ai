import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from regressors import BinomialRegressor

from tests.regressor_test_helpers import build_x_matrix, sample_truncated_binomial_regression


BINOMIAL_SCENARIOS = [
    {
        "name": "binomial_mid_snr",
        "n": 1800,
        "p": 6,
        "n_trials": 8,
        "beta_0": -0.1,
        "beta_1_init": np.array([0.8, -0.5, 0.6, 0.2, -0.7, 0.4], dtype=float),
        "c": 0.75,
        "min": 1,
        "max": 7,
        "target_snr": 1.0,
        "seed_x": 11111,
        "seed_y": 22222,
    },
    {
        "name": "binomial_lower_snr",
        "n": 1800,
        "p": 6,
        "n_trials": 10,
        "beta_0": 0.2,
        "beta_1_init": np.array([-0.7, 0.4, -0.5, 0.8, 0.3, -0.9], dtype=float),
        "c": 0.6,
        "min": 4,
        "max": 10,
        "target_snr": 0.6,
        "seed_x": 33333,
        "seed_y": 44444,
    },
    {
        "name": "binomial_higher_snr",
        "n": 1800,
        "p": 6,
        "n_trials": 12,
        "beta_0": -0.25,
        "beta_1_init": np.array([1.0, -0.8, 0.5, 0.4, -0.3, 0.9], dtype=float),
        "c": 0.9,
        "min": 1,
        "max": 10,
        "target_snr": 1.4,
        "seed_x": 55555,
        "seed_y": 66666,
    },
]


def run_binomial_scenario(scenario: dict) -> dict[str, float]:
    x = build_x_matrix(scenario["n"], scenario["p"], seed=scenario["seed_x"])
    samples = sample_truncated_binomial_regression(
        x,
        scenario["beta_0"],
        scenario["beta_1_init"],
        scenario["c"],
        scenario["n_trials"],
        scenario["min"],
        scenario["max"],
        seed=scenario["seed_y"],
    )
    target_mean = float(np.mean(samples))

    regressor = BinomialRegressor(
        n_trials=scenario["n_trials"],
        target_mean=target_mean,
        target_snr=scenario["target_snr"],
        X=x,
        beta_1_init=scenario["beta_1_init"],
        min=scenario["min"],
        max=scenario["max"],
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
        "sample_min": float(np.min(fitted_samples)),
        "sample_max": float(np.max(fitted_samples)),
        "sample_integer": bool(np.all(np.equal(np.mod(fitted_samples, 1), 0))),
        "fitted_beta_0": float(fitted.beta_0),
        "fitted_c": float(fitted.c),
    }


def test_binomial_regressor_synthetic_calibration():
    for scenario in BINOMIAL_SCENARIOS:
        results = run_binomial_scenario(scenario)

        print(f"BinomialRegressor synthetic calibration summary [{scenario['name']}]")
        print(f"  target mean      : {results['target_mean']:.6f}")
        print(f"  model mean       : {results['model_mean']:.6f}")
        print(f"  sample mean      : {results['sample_mean']:.6f}")
        print(f"  target SNR       : {results['target_snr']:.6f}")
        print(f"  model SNR        : {results['model_snr']:.6f}")
        print(f"  fitted beta_0    : {results['fitted_beta_0']:.6f}")
        print(f"  fitted c         : {results['fitted_c']:.6f}")

        assert np.isclose(results["sample_mean"], results["target_mean"], rtol=0.18, atol=0.18)
        assert np.isclose(results["model_mean"], results["target_mean"], rtol=0.04, atol=0.04)
        assert np.isclose(results["model_snr"], results["target_snr"], rtol=0.25, atol=0.25)
        assert results["sample_integer"]
        assert results["sample_min"] >= scenario["min"]
        assert results["sample_max"] <= scenario["max"]
        assert np.isfinite(results["fitted_beta_0"])
        assert np.isfinite(results["fitted_c"])


if __name__ == "__main__":
    test_binomial_regressor_synthetic_calibration()
