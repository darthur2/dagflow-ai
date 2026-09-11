import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from regressors import GammaRegressor

from tests.regressor_test_helpers import build_x_matrix


GAMMA_SCENARIOS = [
    {
        "name": "gamma_mid_snr",
        "n": 1800,
        "p": 6,
        "beta_0": 0.25,
        "beta_1_init": np.array([0.8, -0.4, 0.7, 0.3, -0.6, 0.5], dtype=float),
        "c": 0.5,
        "shape": 4.0,
        "target_snr": 1.8,
        "seed_x": 999,
        "seed_y": 111,
    },
]


def _sample_truncated_gamma_regression(x: np.ndarray, beta_0: float, beta_1_init: np.ndarray, c: float, shape: float, seed: int = 0) -> np.ndarray:
    from scipy import stats

    rng = np.random.default_rng(seed)
    eta = beta_0 + x @ (c * beta_1_init)
    rate = np.exp(-eta)
    return stats.gamma.rvs(a=shape, scale=1.0 / rate, random_state=rng)


def test_gamma_regressor_synthetic_calibration():
    for scenario in GAMMA_SCENARIOS:
        x = build_x_matrix(scenario["n"], scenario["p"], seed=scenario["seed_x"])
        samples = _sample_truncated_gamma_regression(
            x,
            scenario["beta_0"],
            scenario["beta_1_init"],
            scenario["c"],
            scenario["shape"],
            seed=scenario["seed_y"],
        )
        target_mean = float(np.mean(samples))
        target_variance = float(np.var(samples))

        regressor = GammaRegressor(
            target_mean=target_mean,
            target_variance=target_variance,
            target_snr=scenario["target_snr"],
            X=x,
            beta_1_init=scenario["beta_1_init"],
        )

        fitted = regressor.calibrate()
        fitted_samples = fitted.sample(scenario["n"])
        model_mean = float(fitted.target_mean_value(fitted.beta_0, fitted.c, fitted.shape))
        model_variance = float(fitted.target_variance_value(fitted.beta_0, fitted.c, fitted.shape))
        model_snr = float(fitted.target_snr_value(fitted.beta_0, fitted.c, fitted.shape))

        print(f"GammaRegressor synthetic calibration summary [{scenario['name']}]")
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
    test_gamma_regressor_synthetic_calibration()
