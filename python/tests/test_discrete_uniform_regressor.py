import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from regressors import DiscreteUniformRegressor

from tests.regressor_test_helpers import build_x_matrix


DISCRETE_UNIFORM_SCENARIOS = [
    {
        "name": "du_mid_snr",
        "n": 1800,
        "p": 6,
        "beta_0": 0.2,
        "beta_1_init": np.array([0.7, -0.5, 0.6, 0.2, -0.7, 0.4], dtype=float),
        "c": 0.9,
        "sigma2": 0.65,
        "target_snr": 1.8,
        "seed_x": 9090,
        "seed_y": 9191,
        "min": 0,
        "max": 10,
    },
    {
        "name": "du_low_snr",
        "n": 1800,
        "p": 6,
        "beta_0": -0.1,
        "beta_1_init": np.array([-0.6, 0.8, -0.4, 0.5, 0.3, -0.9], dtype=float),
        "c": 0.8,
        "sigma2": 0.9,
        "target_snr": 1.1,
        "seed_x": 9292,
        "seed_y": 9393,
        "min": 0,
        "max": 10,
    },
]


def _run_discrete_uniform_scenario(scenario: dict) -> dict[str, float]:
    x = build_x_matrix(scenario["n"], scenario["p"], seed=scenario["seed_x"])
    regressor = DiscreteUniformRegressor(
        target_snr=scenario["target_snr"],
        X=x,
        beta_1_init=scenario["beta_1_init"],
        min=scenario["min"],
        max=scenario["max"],
    )

    fitted = regressor.calibrate()
    fitted_samples = fitted.sample(scenario["n"])

    return {
        "target_mean": float((scenario["min"] + scenario["max"]) / 2.0),
        "target_variance": float(((scenario["max"] - scenario["min"] + 1) ** 2 - 1) / 12.0),
        "target_snr": scenario["target_snr"],
        "sample_mean": float(np.mean(fitted_samples)),
        "sample_variance": float(np.var(fitted_samples)),
        "sample_min": float(np.min(fitted_samples)),
        "sample_max": float(np.max(fitted_samples)),
        "sample_integer": bool(np.all(np.equal(np.mod(fitted_samples, 1), 0))),
        "fitted_latent": fitted._latent is not None,
    }


def test_discrete_uniform_regressor_synthetic_calibration():
    for scenario in DISCRETE_UNIFORM_SCENARIOS:
        results = _run_discrete_uniform_scenario(scenario)

        print(f"DiscreteUniformRegressor synthetic calibration summary [{scenario['name']}]")
        print(f"  target mean      : {results['target_mean']:.6f}")
        print(f"  sample mean      : {results['sample_mean']:.6f}")
        print(f"  target variance  : {results['target_variance']:.6f}")
        print(f"  sample variance  : {results['sample_variance']:.6f}")
        print(f"  target SNR       : {results['target_snr']:.6f}")
        print(f"  sample min       : {results['sample_min']:.6f}")
        print(f"  sample max       : {results['sample_max']:.6f}")

        assert np.isclose(results["sample_mean"], results["target_mean"], rtol=0.2, atol=0.2)
        assert np.isclose(results["sample_variance"], results["target_variance"], rtol=0.3, atol=0.3)
        assert results["sample_min"] >= scenario["min"]
        assert results["sample_max"] <= scenario["max"]
        assert results["sample_integer"]
        assert results["fitted_latent"]


if __name__ == "__main__":
    test_discrete_uniform_regressor_synthetic_calibration()
