from dataclasses import dataclass, replace

import numpy as np
from scipy import optimize, stats

try:
    from distributions import Normal
except ImportError:  # pragma: no cover
    from .distributions import Normal


def _validate_bounds(min_value, max_value) -> None:
    if min_value > max_value:
        raise ValueError("min must be less than or equal to max")


def _as_1d_array(values) -> np.ndarray:
    return np.asarray(values).reshape(-1)


@dataclass(frozen=True)
class NormalRegressor:
    target_mean: float
    target_variance: float
    min: float
    max: float
    X: np.ndarray
    beta_1_init: np.ndarray
    target_snr: float
    beta_0: float | None = None
    c: float | None = None
    sigma2: float | None = None

    def __post_init__(self) -> None:
        _validate_bounds(self.min, self.max)
        X = np.asarray(self.X, dtype=float)
        beta_1_init = np.asarray(self.beta_1_init, dtype=float).reshape(-1)
        if X.ndim != 2:
            raise ValueError("X must be a 2D regression matrix")
        if X.shape[1] != beta_1_init.shape[0]:
            raise ValueError("beta_1_init must have one coefficient per column in X")
        if self.target_variance <= 0:
            raise ValueError("target_variance must be positive")
        if self.target_snr < 0:
            raise ValueError("target_snr must be non-negative")
        object.__setattr__(self, "X", X)
        object.__setattr__(self, "beta_1_init", beta_1_init)

    def _linear_predictor(self, beta_0: float, c: float) -> np.ndarray:
        return beta_0 + self.X @ (c * self.beta_1_init)

    def _row_moments(self, mean: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
        a = (self.min - mean) / sigma
        b = (self.max - mean) / sigma
        z = stats.norm.cdf(b) - stats.norm.cdf(a)
        if np.any(z <= 0):
            raise ValueError("truncation interval has zero probability mass")

        alpha = stats.norm.pdf(a)
        beta = stats.norm.pdf(b)
        cond_mean = mean + sigma * (alpha - beta) / z
        cond_var = sigma**2 * (1.0 + (a * alpha - b * beta) / z - ((alpha - beta) / z) ** 2)
        return cond_mean, cond_var

    def _conditional_moments(self, beta_0: float, c: float, sigma2: float) -> tuple[np.ndarray, np.ndarray]:
        sigma = np.sqrt(sigma2)
        eta = self._linear_predictor(beta_0, c)
        return self._row_moments(eta, sigma)

    def target_mean_value(self, beta_0: float, c: float, sigma2: float) -> float:
        cond_mean, _ = self._conditional_moments(beta_0, c, sigma2)
        return float(cond_mean.mean())

    def target_variance_value(self, beta_0: float, c: float, sigma2: float) -> float:
        cond_mean, cond_var = self._conditional_moments(beta_0, c, sigma2)
        return float(cond_mean.var() + cond_var.mean())

    def target_snr_value(self, beta_0: float, c: float, sigma2: float) -> float:
        cond_mean, cond_var = self._conditional_moments(beta_0, c, sigma2)
        within = float(cond_var.mean())
        if within <= 0:
            raise ValueError("Conditional variance is non-positive")
        return float(cond_mean.var() / within)

    def calibrate(self) -> "NormalRegressor":
        if self.beta_0 is not None and self.c is not None and self.sigma2 is not None:
            return self

        initial_beta_0 = float(self.target_mean)
        initial_c = 1.0
        initial_log_sigma2 = float(np.log(self.target_variance))

        def residuals(params: np.ndarray) -> np.ndarray:
            beta_0, c, log_sigma2 = params
            sigma2 = float(np.exp(log_sigma2))
            try:
                mean_residual = self.target_mean_value(beta_0, c, sigma2) - self.target_mean
                variance_residual = self.target_variance_value(beta_0, c, sigma2) - self.target_variance
                snr_residual = self.target_snr_value(beta_0, c, sigma2) - self.target_snr
                return np.array([mean_residual, variance_residual, snr_residual], dtype=float)
            except ValueError:
                return np.array([1e6, 1e6, 1e6], dtype=float)

        result = optimize.least_squares(
            residuals,
            x0=np.array([initial_beta_0, initial_c, initial_log_sigma2], dtype=float),
            bounds=([-np.inf, 0.0, -np.inf], [np.inf, np.inf, np.inf]),
        )

        if not result.success:
            raise ValueError(f"Unable to calibrate NormalRegressor: {result.message}")

        beta_0, c, log_sigma2 = result.x
        sigma2 = float(np.exp(log_sigma2))
        return replace(self, beta_0=float(beta_0), c=float(c), sigma2=sigma2)

    def sample(self, n: int) -> np.ndarray:
        if self.beta_0 is None or self.c is None or self.sigma2 is None:
            raise ValueError("NormalRegressor must be calibrated before sampling")

        sigma = np.sqrt(self.sigma2)
        eta = self._linear_predictor(self.beta_0, self.c)
        mean = np.repeat(eta, int(np.ceil(n / len(eta))))[:n]
        a = (self.min - mean) / sigma
        b = (self.max - mean) / sigma
        samples = stats.truncnorm.rvs(a, b, loc=mean, scale=sigma, size=n)
        return _as_1d_array(samples)


@dataclass(frozen=True)
class ExponentialRegressor:
    target_mean: float
    target_snr: float
    X: np.ndarray
    beta_1_init: np.ndarray
    beta_0: float | None = None
    c: float | None = None

    def __post_init__(self) -> None:
        X = np.asarray(self.X, dtype=float)
        beta_1_init = np.asarray(self.beta_1_init, dtype=float).reshape(-1)
        if X.ndim != 2:
            raise ValueError("X must be a 2D regression matrix")
        if X.shape[1] != beta_1_init.shape[0]:
            raise ValueError("beta_1_init must have one coefficient per column in X")
        if self.target_mean <= 0:
            raise ValueError("target_mean must be positive")
        if not 0.0 < self.target_snr < 1.0:
            raise ValueError("target_snr must be in (0, 1)")
        object.__setattr__(self, "X", X)
        object.__setattr__(self, "beta_1_init", beta_1_init)

    def _linear_predictor(self, beta_0: float, c: float) -> np.ndarray:
        return beta_0 + self.X @ (c * self.beta_1_init)

    def _row_moments(self, beta_0: float, c: float) -> tuple[np.ndarray, np.ndarray]:
        eta = self._linear_predictor(beta_0, c)
        lam = np.exp(-eta)
        alpha = 0.0
        beta = 1000.0
        z = np.exp(-lam * alpha) - np.exp(-lam * beta)
        if np.any(z <= 0):
            raise ValueError("truncation interval has zero probability mass")

        exp_a = np.exp(-lam * alpha)
        exp_b = np.exp(-lam * beta)
        mean = (1.0 / lam) + (alpha * exp_a - beta * exp_b) / z
        second_moment = (2.0 / (lam**2)) + (
            (alpha**2) * exp_a - (beta**2) * exp_b - 2.0 * (alpha * exp_a - beta * exp_b) / lam
        ) / z
        var = second_moment - mean**2
        return mean, var

    def target_mean_value(self, beta_0: float, c: float) -> float:
        cond_mean, _ = self._row_moments(beta_0, c)
        return float(cond_mean.mean())

    def target_variance_value(self, beta_0: float, c: float) -> float:
        cond_mean, cond_var = self._row_moments(beta_0, c)
        return float(cond_mean.var() + cond_var.mean())

    def target_snr_value(self, beta_0: float, c: float) -> float:
        cond_mean, cond_var = self._row_moments(beta_0, c)
        within = float(cond_var.mean())
        if within <= 0:
            raise ValueError("Conditional variance is non-positive")
        return float(cond_mean.var() / within)

    def _feasible_initial_guess(self) -> tuple[float, float]:
        x_mean = np.mean(self.X @ self.beta_1_init)
        beta_0 = np.log(self.target_mean) - x_mean
        c = 1.0
        return beta_0, c

    def calibrate(self) -> "ExponentialRegressor":
        if self.beta_0 is not None and self.c is not None:
            return self

        initial_beta_0, initial_c = self._feasible_initial_guess()

        def residuals(params: np.ndarray) -> np.ndarray:
            beta_0, c = params
            try:
                mean_residual = self.target_mean_value(beta_0, c) - self.target_mean
                snr_residual = self.target_snr_value(beta_0, c) - self.target_snr
                return np.array([mean_residual, snr_residual], dtype=float)
            except ValueError:
                return np.array([1e6, 1e6], dtype=float)

        result = optimize.least_squares(
            residuals,
            x0=np.array([initial_beta_0, initial_c], dtype=float),
            bounds=([-np.inf, 0.0], [np.inf, np.inf]),
        )

        if not result.success:
            raise ValueError(f"Unable to calibrate ExponentialRegressor: {result.message}")

        beta_0, c = result.x
        return replace(self, beta_0=float(beta_0), c=float(c))

    def sample(self, n: int) -> np.ndarray:
        if self.beta_0 is None or self.c is None:
            raise ValueError("ExponentialRegressor must be calibrated before sampling")

        eta = self._linear_predictor(self.beta_0, self.c)
        lam = np.exp(-np.repeat(eta, int(np.ceil(n / len(eta))))[:n])
        samples = np.empty(n, dtype=float)
        rng = np.random.default_rng()
        for idx, current_lambda in enumerate(lam):
            u = rng.random()
            lower = np.exp(-current_lambda * 0.0)
            upper = np.exp(-current_lambda * 1000.0)
            samples[idx] = -np.log(lower - u * (lower - upper)) / current_lambda
        return _as_1d_array(samples)


@dataclass(frozen=True)
class GammaRegressor:
    target_mean: float
    target_variance: float
    target_snr: float
    X: np.ndarray
    beta_1_init: np.ndarray
    beta_0: float | None = None
    c: float | None = None
    shape: float | None = None

    def __post_init__(self) -> None:
        X = np.asarray(self.X, dtype=float)
        beta_1_init = np.asarray(self.beta_1_init, dtype=float).reshape(-1)
        if X.ndim != 2:
            raise ValueError("X must be a 2D regression matrix")
        if X.shape[1] != beta_1_init.shape[0]:
            raise ValueError("beta_1_init must have one coefficient per column in X")
        if self.target_mean <= 0:
            raise ValueError("target_mean must be positive")
        if self.target_variance <= 0:
            raise ValueError("target_variance must be positive")
        if self.target_snr <= 0:
            raise ValueError("target_snr must be positive")
        object.__setattr__(self, "X", X)
        object.__setattr__(self, "beta_1_init", beta_1_init)

    def _linear_predictor(self, beta_0: float, c: float) -> np.ndarray:
        return beta_0 + self.X @ (c * self.beta_1_init)

    def _row_moments(self, beta_0: float, c: float, shape: float) -> tuple[np.ndarray, np.ndarray]:
        eta = self._linear_predictor(beta_0, c)
        rate = np.exp(-eta)
        mean = shape / rate
        var = shape / (rate**2)
        return mean, var

    def target_mean_value(self, beta_0: float, c: float, shape: float) -> float:
        cond_mean, _ = self._row_moments(beta_0, c, shape)
        return float(cond_mean.mean())

    def target_variance_value(self, beta_0: float, c: float, shape: float) -> float:
        cond_mean, cond_var = self._row_moments(beta_0, c, shape)
        return float(cond_mean.var() + cond_var.mean())

    def target_snr_value(self, beta_0: float, c: float, shape: float) -> float:
        cond_mean, cond_var = self._row_moments(beta_0, c, shape)
        within = float(cond_var.mean())
        if within <= 0:
            raise ValueError("Conditional variance is non-positive")
        return float(cond_mean.var() / within)

    def _feasible_initial_guess(self) -> tuple[float, float, float]:
        x_mean = np.mean(self.X @ self.beta_1_init)
        beta_0 = np.log(self.target_mean / self.target_snr) - x_mean
        c = 1.0
        shape = max(self.target_snr, 1.0)
        return beta_0, c, shape

    def calibrate(self) -> "GammaRegressor":
        if self.beta_0 is not None and self.c is not None and self.shape is not None:
            return self

        initial_beta_0, initial_c, initial_shape = self._feasible_initial_guess()

        def residuals(params: np.ndarray) -> np.ndarray:
            beta_0, c, shape = params
            try:
                mean_residual = self.target_mean_value(beta_0, c, shape) - self.target_mean
                variance_residual = self.target_variance_value(beta_0, c, shape) - self.target_variance
                snr_residual = self.target_snr_value(beta_0, c, shape) - self.target_snr
                return np.array([mean_residual, variance_residual, snr_residual], dtype=float)
            except ValueError:
                return np.array([1e6, 1e6, 1e6], dtype=float)

        result = optimize.least_squares(
            residuals,
            x0=np.array([initial_beta_0, initial_c, initial_shape], dtype=float),
            bounds=([-np.inf, 0.0, np.finfo(float).tiny], [np.inf, np.inf, np.inf]),
        )

        if not result.success:
            raise ValueError(f"Unable to calibrate GammaRegressor: {result.message}")

        beta_0, c, shape = result.x
        return replace(self, beta_0=float(beta_0), c=float(c), shape=float(shape))

    def sample(self, n: int) -> np.ndarray:
        if self.beta_0 is None or self.c is None or self.shape is None:
            raise ValueError("GammaRegressor must be calibrated before sampling")

        eta = self._linear_predictor(self.beta_0, self.c)
        rate = np.exp(-np.repeat(eta, int(np.ceil(n / len(eta))))[:n])
        samples = stats.gamma.rvs(a=self.shape, scale=1.0 / rate, size=n)
        return _as_1d_array(samples)


@dataclass(frozen=True)
class LogNormalRegressor:
    target_mean: float
    target_variance: float
    target_snr: float
    X: np.ndarray
    beta_1_init: np.ndarray
    beta_0: float | None = None
    c: float | None = None
    sigma2: float | None = None

    def __post_init__(self) -> None:
        X = np.asarray(self.X, dtype=float)
        beta_1_init = np.asarray(self.beta_1_init, dtype=float).reshape(-1)
        if X.ndim != 2:
            raise ValueError("X must be a 2D regression matrix")
        if X.shape[1] != beta_1_init.shape[0]:
            raise ValueError("beta_1_init must have one coefficient per column in X")
        if self.target_mean <= 0:
            raise ValueError("target_mean must be positive")
        if self.target_variance <= 0:
            raise ValueError("target_variance must be positive")
        if self.target_snr <= 0:
            raise ValueError("target_snr must be positive")
        object.__setattr__(self, "X", X)
        object.__setattr__(self, "beta_1_init", beta_1_init)

    def _linear_predictor(self, beta_0: float, c: float) -> np.ndarray:
        return beta_0 + self.X @ (c * self.beta_1_init)

    def _row_moments(self, beta_0: float, c: float, sigma2: float) -> tuple[np.ndarray, np.ndarray]:
        eta = self._linear_predictor(beta_0, c)
        mean = np.exp(eta + 0.5 * sigma2)
        var = (np.exp(sigma2) - 1.0) * np.exp(2.0 * eta + sigma2)
        return mean, var

    def target_mean_value(self, beta_0: float, c: float, sigma2: float) -> float:
        cond_mean, _ = self._row_moments(beta_0, c, sigma2)
        return float(cond_mean.mean())

    def target_variance_value(self, beta_0: float, c: float, sigma2: float) -> float:
        cond_mean, cond_var = self._row_moments(beta_0, c, sigma2)
        return float(cond_mean.var() + cond_var.mean())

    def target_snr_value(self, beta_0: float, c: float, sigma2: float) -> float:
        cond_mean, cond_var = self._row_moments(beta_0, c, sigma2)
        within = float(cond_var.mean())
        if within <= 0:
            raise ValueError("Conditional variance is non-positive")
        return float(cond_mean.var() / within)

    def _feasible_initial_guess(self) -> tuple[float, float, float]:
        x_mean = np.mean(self.X @ self.beta_1_init)
        beta_0 = np.log(self.target_mean) - x_mean
        c = 1.0
        sigma2 = 0.5
        return beta_0, c, sigma2

    def calibrate(self) -> "LogNormalRegressor":
        if self.beta_0 is not None and self.c is not None and self.sigma2 is not None:
            return self

        initial_beta_0, initial_c, initial_sigma2 = self._feasible_initial_guess()

        def residuals(params: np.ndarray) -> np.ndarray:
            beta_0, c, log_sigma2 = params
            sigma2 = float(np.exp(log_sigma2))
            try:
                mean_residual = self.target_mean_value(beta_0, c, sigma2) - self.target_mean
                variance_residual = self.target_variance_value(beta_0, c, sigma2) - self.target_variance
                snr_residual = self.target_snr_value(beta_0, c, sigma2) - self.target_snr
                return np.array([mean_residual, variance_residual, snr_residual], dtype=float)
            except ValueError:
                return np.array([1e6, 1e6, 1e6], dtype=float)

        result = optimize.least_squares(
            residuals,
            x0=np.array([initial_beta_0, initial_c, np.log(initial_sigma2)], dtype=float),
            bounds=([-np.inf, 0.0, -np.inf], [np.inf, np.inf, np.inf]),
        )

        if not result.success:
            raise ValueError(f"Unable to calibrate LogNormalRegressor: {result.message}")

        beta_0, c, log_sigma2 = result.x
        sigma2 = float(np.exp(log_sigma2))
        return replace(self, beta_0=float(beta_0), c=float(c), sigma2=sigma2)

    def sample(self, n: int) -> np.ndarray:
        if self.beta_0 is None or self.c is None or self.sigma2 is None:
            raise ValueError("LogNormalRegressor must be calibrated before sampling")

        eta = self._linear_predictor(self.beta_0, self.c)
        mean = np.repeat(eta, int(np.ceil(n / len(eta))))[:n]
        samples = stats.lognorm.rvs(s=np.sqrt(self.sigma2), scale=np.exp(mean), size=n)
        return _as_1d_array(samples)
