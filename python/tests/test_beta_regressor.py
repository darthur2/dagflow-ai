import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from regressors import BetaRegressor

from tests.regressor_test_helpers import build_x_matrix


BETA_SCENARIOS = [
    {
        "name": "beta_mid_snr",
        "n": 1800,
        "p": 6,
        "beta_0": -0.2,
        "beta_1_init": np.array([0.8, -0.5, 0.6, 0.2, -0.7, 0.4], dtype=float),
        "c": 0.5,
        "phi": 7.0,
        "target_snr": 2.0,
        "seed_x": 1111,
        "seed_y": 2222,
    },
]


def _sample_beta_regression(x: np.ndarray, beta_0: float, beta_1_init: np.ndarray, c: float, phi: float, seed: int = 0) -> np.ndarray:
    from scipy import stats

    rng = np.random.default_rng(seed)
    eta = beta_0 + x @ (c * beta_1_init)
    mu = 1.0 / (1.0 + np.exp(-eta))
    return stats.beta.rvs(phi * mu, phi * (1.0 - mu), random_state=rng)


def test_beta_regressor_synthetic_calibration():
    for scenario in BETA_SCENARIOS:
        x = build_x_matrix(scenario["n"], scenario["p"], seed=scenario["seed_x"])
        samples = _sample_beta_regression(
            x,
            scenario["beta_0"],
            scenario["beta_1_init"],
            scenario["c"],
            scenario["phi"],
            seed=scenario["seed_y"],
        )
        target_mean = float(np.mean(samples))
        target_variance = float(np.var(samples))

        regressor = BetaRegressor(
            target_mean=target_mean,
            target_variance=target_variance,
            target_snr=scenario["target_snr"],
            X=x,
            beta_1_init=scenario["beta_1_init"],
        )

        fitted = regressor.calibrate()
        fitted_samples = fitted.sample(scenario["n"])
        model_mean = float(fitted.target_mean_value(fitted.beta_0, fitted.c, fitted.phi))
        model_variance = float(fitted.target_variance_value(fitted.beta_0, fitted.c, fitted.phi))
        model_snr = float(fitted.target_snr_value(fitted.beta_0, fitted.c, fitted.phi))

        print(f"BetaRegressor synthetic calibration summary [{scenario['name']}]")
        print(f"  target mean      : {target_mean:.6f}")
        print(f"  model mean       : {model_mean:.6f}")
        print(f"  sample mean      : {float(np.mean(fitted_samples)):.6f}")
        print(f"  target variance  : {target_variance:.6f}")
        print(f"  model variance   : {model_variance:.6f}")
        print(f"  sample variance  : {float(np.var(fitted_samples)):.6f}")
        print(f"  target SNR       : {scenario['target_snr']:.6f}")
        print(f"  model SNR        : {model_snr:.6f}")

        assert np.isclose(model_mean, target_mean, rtol=1e-6, atol=1e-6)
        assert np.isclose(model_variance, target_variance, rtol=1e-6, atol=1e-6)
        assert np.isclose(model_snr, scenario["target_snr"], rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    test_beta_regressor_synthetic_calibration()
