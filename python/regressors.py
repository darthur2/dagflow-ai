from dataclasses import dataclass, replace

import numpy as np
from scipy import optimize, special, stats

try:
    from distributions import Geometric, NegativeBinomial, Normal, Poisson, Binomial
except ImportError:  # pragma: no cover
    from .distributions import Geometric, NegativeBinomial, Normal, Poisson, Binomial


def _validate_bounds(min_value, max_value) -> None:
    if min_value > max_value:
        raise ValueError("min must be less than or equal to max")


def _as_1d_array(values) -> np.ndarray:
    return np.asarray(values).reshape(-1)


def _apply_single_transformation(values: np.ndarray, transformation: str) -> np.ndarray:
    if transformation == "none":
        return values
    if transformation == "exp":
        return np.exp(values)
    if transformation == "log":
        if np.any(values <= 0):
            raise ValueError("log transformation requires positive values")
        return np.log(values)
    if transformation == "sqrt":
        if np.any(values < 0):
            raise ValueError("sqrt transformation requires non-negative values")
        return np.sqrt(values)
    if transformation == "inverse":
        if np.any(values == 0):
            raise ValueError("inverse transformation requires non-zero values")
        return 1.0 / values
    if transformation == "square":
        return values**2
    if transformation == "cubic":
        return values**3
    if transformation == "quartic":
        return values**4
    if transformation == "sin":
        return np.sin(values)
    if transformation == "cos":
        return np.cos(values)
    raise ValueError(f"Unsupported transformation: {transformation}")


def _transform_predictors(
    X: np.ndarray,
    predictor_names: list[str],
    predictor_transformations: dict[str, str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    if predictor_transformations is None:
        return X, predictor_names
    if X.ndim != 2:
        raise ValueError("X must be a 2D regression matrix")
    if len(predictor_names) != X.shape[1]:
        raise ValueError("predictor_names must have one name per column in X")

    transformed_columns = []
    transformed_names = []
    for idx, name in enumerate(predictor_names):
        current = X[:, idx]
        if name in predictor_transformations:
            transformation = predictor_transformations[name]
            current = _apply_single_transformation(current, transformation)
        else:
            pass
        transformed_columns.append(np.asarray(current, dtype=float))
        transformed_names.append(name)
    return np.column_stack(transformed_columns), transformed_names


@dataclass(frozen=True)
class NormalRegressor:
    target_mean: float
    target_variance: float
    target_snr: float
    min: float
    max: float
    X: np.ndarray
    beta_1_init: np.ndarray
    predictor_names: list[str] | None = None
    predictor_transformations: dict[str, str] | None = None
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
        predictor_names = self.predictor_names or [f"x{i}" for i in range(X.shape[1])]
        X, predictor_names = _transform_predictors(X, predictor_names, self.predictor_transformations)
        object.__setattr__(self, "X", X)
        object.__setattr__(self, "beta_1_init", beta_1_init)
        object.__setattr__(self, "predictor_names", predictor_names)

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
    min: float
    max: float
    X: np.ndarray
    beta_1_init: np.ndarray
    predictor_names: list[str] | None = None
    predictor_transformations: dict[str, str] | None = None
    beta_0: float | None = None
    c: float | None = None

    def __post_init__(self) -> None:
        _validate_bounds(self.min, self.max)
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
        predictor_names = self.predictor_names or [f"x{i}" for i in range(X.shape[1])]
        X, predictor_names = _transform_predictors(X, predictor_names, self.predictor_transformations)
        object.__setattr__(self, "X", X)
        object.__setattr__(self, "beta_1_init", beta_1_init)
        object.__setattr__(self, "predictor_names", predictor_names)

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
    min: float
    max: float
    X: np.ndarray
    beta_1_init: np.ndarray
    predictor_names: list[str] | None = None
    predictor_transformations: dict[str, str] | None = None
    beta_0: float | None = None
    shape: float | None = None

    def __post_init__(self) -> None:
        _validate_bounds(self.min, self.max)
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
        predictor_names = self.predictor_names or [f"x{i}" for i in range(X.shape[1])]
        X, predictor_names = _transform_predictors(X, predictor_names, self.predictor_transformations)
        object.__setattr__(self, "X", X)
        object.__setattr__(self, "beta_1_init", beta_1_init)
        object.__setattr__(self, "predictor_names", predictor_names)

    def _linear_predictor(self, beta_0: float) -> np.ndarray:
        return beta_0 + self.X @ self.beta_1_init

    def _row_moments(self, beta_0: float, shape: float) -> tuple[np.ndarray, np.ndarray]:
        eta = self._linear_predictor(beta_0)
        rate = np.exp(-eta)
        a = shape
        lower_scaled = rate * self.min
        upper_scaled = rate * self.max
        denom = special.gammainc(a, upper_scaled) - special.gammainc(a, lower_scaled)
        if np.any(denom <= 0):
            raise ValueError("truncation interval has zero probability mass")

        mean = (special.gamma(a + 1.0) / (rate * special.gamma(a))) * (
            (special.gammainc(a + 1.0, upper_scaled) - special.gammainc(a + 1.0, lower_scaled)) / denom
        )
        second_moment = (special.gamma(a + 2.0) / (rate**2 * special.gamma(a))) * (
            (special.gammainc(a + 2.0, upper_scaled) - special.gammainc(a + 2.0, lower_scaled)) / denom
        )
        var = second_moment - mean**2
        return np.asarray(mean, dtype=float), np.asarray(var, dtype=float)

    def target_mean_value(self, beta_0: float, shape: float) -> float:
        cond_mean, _ = self._row_moments(beta_0, shape)
        return float(cond_mean.mean())

    def target_variance_value(self, beta_0: float, shape: float) -> float:
        cond_mean, cond_var = self._row_moments(beta_0, shape)
        return float(cond_mean.var() + cond_var.mean())

    def _feasible_initial_guess(self) -> tuple[float, float]:
        x_mean = np.mean(self.X @ self.beta_1_init)
        beta_0 = np.log(max(self.target_mean, np.finfo(float).tiny)) - x_mean
        shape = max(self.target_mean**2 / max(self.target_variance, np.finfo(float).tiny), np.finfo(float).tiny)
        return beta_0, shape

    def calibrate_mean_variance(self, shape: float) -> "GammaRegressor":
        if self.beta_0 is not None and self.shape is not None:
            return self

        initial_beta_0 = float(np.log(max(self.target_mean, np.finfo(float).tiny)) - np.mean(self.X @ self.beta_1_init))

        def residuals(params: np.ndarray) -> np.ndarray:
            beta_0 = params[0]
            try:
                mean_residual = self.target_mean_value(beta_0, shape) - self.target_mean
                variance_residual = self.target_variance_value(beta_0, shape) - self.target_variance
                return np.array([mean_residual, variance_residual], dtype=float)
            except ValueError:
                return np.array([1e6, 1e6], dtype=float)

        result = optimize.least_squares(
            residuals,
            x0=np.array([initial_beta_0], dtype=float),
            bounds=([-np.inf], [np.inf]),
        )

        if not result.success:
            raise ValueError(f"Unable to calibrate GammaRegressor: {result.message}")

        beta_0 = float(result.x[0])
        return replace(self, beta_0=beta_0, shape=float(shape))

    def calibrate(self) -> "GammaRegressor":
        if self.beta_0 is not None and self.shape is not None:
            return self

        initial_beta_0, initial_shape = self._feasible_initial_guess()

        def residuals(params: np.ndarray) -> np.ndarray:
            beta_0, shape = params
            try:
                mean_residual = self.target_mean_value(beta_0, shape) - self.target_mean
                variance_residual = self.target_variance_value(beta_0, shape) - self.target_variance
                return np.array([mean_residual, variance_residual], dtype=float)
            except ValueError:
                return np.array([1e6, 1e6], dtype=float)

        result = optimize.least_squares(
            residuals,
            x0=np.array([initial_beta_0, initial_shape], dtype=float),
            bounds=([-np.inf, np.finfo(float).tiny], [np.inf, np.inf]),
        )

        if not result.success:
            raise ValueError(f"Unable to calibrate GammaRegressor: {result.message}")

        beta_0, shape = result.x
        return replace(self, beta_0=float(beta_0), shape=float(shape))

    def sample(self, n: int) -> np.ndarray:
        if self.beta_0 is None or self.shape is None:
            raise ValueError("GammaRegressor must be calibrated before sampling")

        eta = self._linear_predictor(self.beta_0)
        rate = np.exp(-np.repeat(eta, int(np.ceil(n / len(eta))))[:n])
        samples = stats.gamma.rvs(a=self.shape, scale=1.0 / rate, size=n)
        return _as_1d_array(samples)


@dataclass(frozen=True)
class LogNormalRegressor:
    target_mean: float
    target_variance: float
    target_snr: float
    min: float
    max: float
    X: np.ndarray
    beta_1_init: np.ndarray
    predictor_names: list[str] | None = None
    predictor_transformations: dict[str, str] | None = None
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
        if self.target_mean <= 0:
            raise ValueError("target_mean must be positive")
        if self.target_variance <= 0:
            raise ValueError("target_variance must be positive")
        if self.target_snr <= 0:
            raise ValueError("target_snr must be positive")
        predictor_names = self.predictor_names or [f"x{i}" for i in range(X.shape[1])]
        X, predictor_names = _transform_predictors(X, predictor_names, self.predictor_transformations)
        object.__setattr__(self, "X", X)
        object.__setattr__(self, "beta_1_init", beta_1_init)
        object.__setattr__(self, "predictor_names", predictor_names)

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


@dataclass(frozen=True)
class BetaRegressor:
    target_mean: float
    target_variance: float
    target_snr: float
    min: float
    max: float
    X: np.ndarray
    beta_1_init: np.ndarray
    predictor_names: list[str] | None = None
    predictor_transformations: dict[str, str] | None = None
    beta_0: float | None = None
    c: float | None = None
    phi: float | None = None

    def __post_init__(self) -> None:
        _validate_bounds(self.min, self.max)
        X = np.asarray(self.X, dtype=float)
        beta_1_init = np.asarray(self.beta_1_init, dtype=float).reshape(-1)
        if X.ndim != 2:
            raise ValueError("X must be a 2D regression matrix")
        if X.shape[1] != beta_1_init.shape[0]:
            raise ValueError("beta_1_init must have one coefficient per column in X")
        if not 0.0 < self.target_mean < 1.0:
            raise ValueError("target_mean must be in (0, 1)")
        if self.target_variance <= 0:
            raise ValueError("target_variance must be positive")
        if self.target_snr <= 0:
            raise ValueError("target_snr must be positive")
        predictor_names = self.predictor_names or [f"x{i}" for i in range(X.shape[1])]
        X, predictor_names = _transform_predictors(X, predictor_names, self.predictor_transformations)
        object.__setattr__(self, "X", X)
        object.__setattr__(self, "beta_1_init", beta_1_init)
        object.__setattr__(self, "predictor_names", predictor_names)

    def _linear_predictor(self, beta_0: float, c: float) -> np.ndarray:
        return beta_0 + self.X @ (c * self.beta_1_init)

    def _mu(self, beta_0: float, c: float) -> np.ndarray:
        eta = self._linear_predictor(beta_0, c)
        return 1.0 / (1.0 + np.exp(-eta))

    def _row_moments(self, beta_0: float, c: float, phi: float) -> tuple[np.ndarray, np.ndarray]:
        mu = self._mu(beta_0, c)
        mean = mu
        var = mu * (1.0 - mu) / (1.0 + phi)
        return mean, var

    def target_mean_value(self, beta_0: float, c: float, phi: float) -> float:
        cond_mean, _ = self._row_moments(beta_0, c, phi)
        return float(cond_mean.mean())

    def target_variance_value(self, beta_0: float, c: float, phi: float) -> float:
        cond_mean, cond_var = self._row_moments(beta_0, c, phi)
        return float(cond_mean.var() + cond_var.mean())

    def target_snr_value(self, beta_0: float, c: float, phi: float) -> float:
        cond_mean, cond_var = self._row_moments(beta_0, c, phi)
        within = float(cond_var.mean())
        if within <= 0:
            raise ValueError("Conditional variance is non-positive")
        return float(cond_mean.var() / within)

    def _feasible_initial_guess(self) -> tuple[float, float, float]:
        mu = np.clip(self.target_mean, 1e-3, 1.0 - 1e-3)
        x_mean = np.mean(self.X @ self.beta_1_init)
        beta_0 = np.log(mu / (1.0 - mu)) - x_mean
        c = 1.0
        phi = max(self.target_snr, 1.0)
        return beta_0, c, phi

    def calibrate(self) -> "BetaRegressor":
        if self.beta_0 is not None and self.c is not None and self.phi is not None:
            return self

        initial_beta_0, initial_c, initial_phi = self._feasible_initial_guess()

        def residuals(params: np.ndarray) -> np.ndarray:
            beta_0, c, log_phi = params
            phi = float(np.exp(log_phi))
            try:
                mean_residual = self.target_mean_value(beta_0, c, phi) - self.target_mean
                variance_residual = self.target_variance_value(beta_0, c, phi) - self.target_variance
                snr_residual = self.target_snr_value(beta_0, c, phi) - self.target_snr
                return np.array([mean_residual, variance_residual, snr_residual], dtype=float)
            except ValueError:
                return np.array([1e6, 1e6, 1e6], dtype=float)

        result = optimize.least_squares(
            residuals,
            x0=np.array([initial_beta_0, initial_c, np.log(initial_phi)], dtype=float),
            bounds=([-np.inf, 0.0, -np.inf], [np.inf, np.inf, np.inf]),
        )

        if not result.success:
            raise ValueError(f"Unable to calibrate BetaRegressor: {result.message}")

        beta_0, c, log_phi = result.x
        phi = float(np.exp(log_phi))
        return replace(self, beta_0=float(beta_0), c=float(c), phi=phi)

    def sample(self, n: int) -> np.ndarray:
        if self.beta_0 is None or self.c is None or self.phi is None:
            raise ValueError("BetaRegressor must be calibrated before sampling")

        mu = np.repeat(self._mu(self.beta_0, self.c), int(np.ceil(n / len(self.X))))[:n]
        samples = stats.beta.rvs(self.phi * mu, self.phi * (1.0 - mu), size=n)
        return _as_1d_array(samples)


@dataclass(frozen=True)
class UniformRegressor:
    target_snr: float
    X: np.ndarray
    beta_1_init: np.ndarray
    predictor_names: list[str] | None = None
    predictor_transformations: dict[str, str] | None = None
    min: float = -10.0
    max: float = 10.0
    _latent: NormalRegressor | None = None

    def __post_init__(self) -> None:
        _validate_bounds(self.min, self.max)
        X = np.asarray(self.X, dtype=float)
        beta_1_init = np.asarray(self.beta_1_init, dtype=float).reshape(-1)
        if X.ndim != 2:
            raise ValueError("X must be a 2D regression matrix")
        if X.shape[1] != beta_1_init.shape[0]:
            raise ValueError("beta_1_init must have one coefficient per column in X")
        if self.target_snr <= 0:
            raise ValueError("target_snr must be positive")
        predictor_names = self.predictor_names or [f"x{i}" for i in range(X.shape[1])]
        X, predictor_names = _transform_predictors(X, predictor_names, self.predictor_transformations)
        object.__setattr__(self, "X", X)
        object.__setattr__(self, "beta_1_init", beta_1_init)
        object.__setattr__(self, "predictor_names", predictor_names)

    def calibrate(self) -> "UniformRegressor":
        if self._latent is not None:
            return self

        latent = NormalRegressor(
            target_mean=0.0,
            target_variance=1.0,
            min=self.min,
            max=self.max,
            X=self.X,
            beta_1_init=self.beta_1_init,
            target_snr=self.target_snr,
        ).calibrate()
        return replace(self, _latent=latent)

    def sample(self, n: int) -> np.ndarray:
        if self._latent is None:
            raise ValueError("UniformRegressor must be calibrated before sampling")

        latent_samples = self._latent.sample(n)
        uniforms = stats.norm.cdf(latent_samples)
        return _as_1d_array(self.min + (self.max - self.min) * uniforms)


@dataclass(frozen=True)
class DiscreteUniformRegressor:
    target_snr: float
    X: np.ndarray
    beta_1_init: np.ndarray
    predictor_names: list[str] | None = None
    predictor_transformations: dict[str, str] | None = None
    min: int = 0
    max: int = 10
    _latent: UniformRegressor | None = None
    _latent_min: float | None = None
    _latent_max: float | None = None

    def __post_init__(self) -> None:
        _validate_bounds(self.min, self.max)
        X = np.asarray(self.X, dtype=float)
        beta_1_init = np.asarray(self.beta_1_init, dtype=float).reshape(-1)
        if X.ndim != 2:
            raise ValueError("X must be a 2D regression matrix")
        if X.shape[1] != beta_1_init.shape[0]:
            raise ValueError("beta_1_init must have one coefficient per column in X")
        if self.target_snr <= 0:
            raise ValueError("target_snr must be positive")
        predictor_names = self.predictor_names or [f"x{i}" for i in range(X.shape[1])]
        X, predictor_names = _transform_predictors(X, predictor_names, self.predictor_transformations)
        object.__setattr__(self, "X", X)
        object.__setattr__(self, "beta_1_init", beta_1_init)
        object.__setattr__(self, "predictor_names", predictor_names)

    def calibrate(self) -> "DiscreteUniformRegressor":
        if self._latent is not None:
            return self

        latent_min = self.min - 0.5
        latent_max = self.max + 0.5
        latent = UniformRegressor(
            target_snr=self.target_snr,
            min=latent_min,
            max=latent_max,
            X=self.X,
            beta_1_init=self.beta_1_init,
        ).calibrate()
        return replace(self, _latent=latent, _latent_min=latent_min, _latent_max=latent_max)

    def sample(self, n: int) -> np.ndarray:
        if self._latent is None or self._latent_min is None or self._latent_max is None:
            raise ValueError("DiscreteUniformRegressor must be calibrated before sampling")

        samples = self._latent.sample(n)
        rounded = np.rint(samples).astype(int)
        return _as_1d_array(np.clip(rounded, self.min, self.max))


@dataclass(frozen=True)
class BernoulliRegressor:
    target_mean: float
    target_snr: float
    X: np.ndarray
    beta_1_init: np.ndarray
    predictor_names: list[str] | None = None
    predictor_transformations: dict[str, str] | None = None
    min: float = 0.0
    max: float = 1.0
    beta_0: float | None = None
    c: float | None = None

    def __post_init__(self) -> None:
        X = np.asarray(self.X, dtype=float)
        beta_1_init = np.asarray(self.beta_1_init, dtype=float).reshape(-1)
        if X.ndim != 2:
            raise ValueError("X must be a 2D regression matrix")
        if X.shape[1] != beta_1_init.shape[0]:
            raise ValueError("beta_1_init must have one coefficient per column in X")
        if not 0.0 < self.target_mean < 1.0:
            raise ValueError("target_mean must be in (0, 1)")
        if self.target_snr <= 0:
            raise ValueError("target_snr must be positive")
        predictor_names = self.predictor_names or [f"x{i}" for i in range(X.shape[1])]
        X, predictor_names = _transform_predictors(X, predictor_names, self.predictor_transformations)
        object.__setattr__(self, "X", X)
        object.__setattr__(self, "beta_1_init", beta_1_init)
        object.__setattr__(self, "predictor_names", predictor_names)

    def _linear_predictor(self, beta_0: float, c: float) -> np.ndarray:
        return beta_0 + self.X @ (c * self.beta_1_init)

    def _probability(self, beta_0: float, c: float) -> np.ndarray:
        eta = self._linear_predictor(beta_0, c)
        return 1.0 / (1.0 + np.exp(-eta))

    def target_mean_value(self, beta_0: float, c: float) -> float:
        return float(self._probability(beta_0, c).mean())

    def target_snr_value(self, beta_0: float, c: float) -> float:
        p = self._probability(beta_0, c)
        within = float(np.mean(p * (1.0 - p)))
        if within <= 0:
            raise ValueError("Conditional variance is non-positive")
        return float(p.var() / within)

    def _feasible_initial_guess(self) -> tuple[float, float]:
        x_mean = np.mean(self.X @ self.beta_1_init)
        beta_0 = np.log(self.target_mean / (1.0 - self.target_mean)) - x_mean
        c = 1.0
        return beta_0, c

    def calibrate(self) -> "BernoulliRegressor":
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
            raise ValueError(f"Unable to calibrate BernoulliRegressor: {result.message}")

        beta_0, c = result.x
        return replace(self, beta_0=float(beta_0), c=float(c))

    def sample(self, n: int) -> np.ndarray:
        if self.beta_0 is None or self.c is None:
            raise ValueError("BernoulliRegressor must be calibrated before sampling")

        p = np.repeat(self._probability(self.beta_0, self.c), int(np.ceil(n / len(self.X))))[:n]
        samples = stats.bernoulli.rvs(p, size=n)
        return _as_1d_array(samples)


@dataclass(frozen=True)
class BinomialRegressor:
    target_mean: float
    target_snr: float
    n_trials: int
    X: np.ndarray
    beta_1_init: np.ndarray
    min: int
    max: int
    predictor_names: list[str] | None = None
    predictor_transformations: dict[str, str] | None = None
    beta_0: float | None = None
    c: float | None = None
    _support: np.ndarray | None = None

    def __post_init__(self) -> None:
        _validate_bounds(self.min, self.max)
        X = np.asarray(self.X, dtype=float)
        beta_1_init = np.asarray(self.beta_1_init, dtype=float).reshape(-1)
        if X.ndim != 2:
            raise ValueError("X must be a 2D regression matrix")
        if X.shape[1] != beta_1_init.shape[0]:
            raise ValueError("beta_1_init must have one coefficient per column in X")
        if self.n_trials < 1:
            raise ValueError("n_trials must be >= 1")
        if self.min < 0 or self.max > self.n_trials:
            raise ValueError("min and max must lie within [0, n_trials]")
        if self.target_snr <= 0:
            raise ValueError("target_snr must be positive")
        predictor_names = self.predictor_names or [f"x{i}" for i in range(X.shape[1])]
        X, predictor_names = _transform_predictors(X, predictor_names, self.predictor_transformations)
        object.__setattr__(self, "X", X)
        object.__setattr__(self, "beta_1_init", beta_1_init)
        object.__setattr__(self, "predictor_names", predictor_names)
        object.__setattr__(self, "_support", np.arange(self.min, self.max + 1, dtype=float))

    def _linear_predictor(self, beta_0: float, c: float) -> np.ndarray:
        return beta_0 + self.X @ (c * self.beta_1_init)

    def _probability(self, beta_0: float, c: float) -> np.ndarray:
        eta = self._linear_predictor(beta_0, c)
        return 1.0 / (1.0 + np.exp(-eta))

    def target_mean_value(self, beta_0: float, c: float) -> float:
        p = self._probability(beta_0, c)
        means = np.empty(len(p), dtype=float)
        for idx, current_p in enumerate(p):
            means[idx] = Binomial(self.n_trials, float(current_p), self.min, self.max).target_mean()
        return float(means.mean())

    def target_snr_value(self, beta_0: float, c: float) -> float:
        p = self._probability(beta_0, c)
        cond_mean = np.empty(len(p), dtype=float)
        cond_var = np.empty(len(p), dtype=float)
        for idx, current_p in enumerate(p):
            dist = stats.binom(self.n_trials, float(current_p))
            lower = dist.cdf(self.min - 1)
            upper = dist.cdf(self.max)
            if lower >= upper:
                raise ValueError("truncation interval has zero probability mass")
            pmf = dist.pmf(self._support)
            pmf = pmf / float(np.sum(pmf))
            current_mean = float(np.sum(self._support * pmf))
            current_second = float(np.sum((self._support**2) * pmf))
            cond_mean[idx] = current_mean
            cond_var[idx] = current_second - current_mean**2
        within = float(cond_var.mean())
        if within <= 0:
            raise ValueError("Conditional variance is non-positive")
        return float(cond_mean.var() / within)

    def _feasible_initial_guess(self) -> tuple[float, float]:
        mean_prob = np.clip(self.target_mean / self.n_trials, 1e-3, 1.0 - 1e-3)
        x_mean = np.mean(self.X @ self.beta_1_init)
        beta_0 = np.log(mean_prob / (1.0 - mean_prob)) - x_mean
        c = 1.0
        return beta_0, c

    def calibrate(self) -> "BinomialRegressor":
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
            raise ValueError(f"Unable to calibrate BinomialRegressor: {result.message}")

        beta_0, c = result.x
        return replace(self, beta_0=float(beta_0), c=float(c))

    def sample(self, n: int) -> np.ndarray:
        if self.beta_0 is None or self.c is None:
            raise ValueError("BinomialRegressor must be calibrated before sampling")

        p = np.repeat(self._probability(self.beta_0, self.c), int(np.ceil(n / len(self.X))))[:n]
        samples = np.empty(n, dtype=float)
        rng = np.random.default_rng()
        for idx, current_p in enumerate(p):
            dist = stats.binom(self.n_trials, current_p)
            lower = dist.cdf(self.min - 1)
            upper = dist.cdf(self.max)
            u = rng.uniform(lower, upper)
            samples[idx] = dist.ppf(u)
        return _as_1d_array(np.clip(samples, self.min, self.max))


@dataclass(frozen=True)
class PoissonRegressor:
    target_mean: float
    target_snr: float
    X: np.ndarray
    beta_1_init: np.ndarray
    min: int
    max: int
    predictor_names: list[str] | None = None
    predictor_transformations: dict[str, str] | None = None
    beta_0: float | None = None
    c: float | None = None
    _support: np.ndarray | None = None

    def __post_init__(self) -> None:
        _validate_bounds(self.min, self.max)
        X = np.asarray(self.X, dtype=float)
        beta_1_init = np.asarray(self.beta_1_init, dtype=float).reshape(-1)
        if X.ndim != 2:
            raise ValueError("X must be a 2D regression matrix")
        if X.shape[1] != beta_1_init.shape[0]:
            raise ValueError("beta_1_init must have one coefficient per column in X")
        if self.min < 0:
            raise ValueError("min must be >= 0")
        if self.target_snr <= 0:
            raise ValueError("target_snr must be positive")
        predictor_names = self.predictor_names or [f"x{i}" for i in range(X.shape[1])]
        X, predictor_names = _transform_predictors(X, predictor_names, self.predictor_transformations)
        object.__setattr__(self, "X", X)
        object.__setattr__(self, "beta_1_init", beta_1_init)
        object.__setattr__(self, "predictor_names", predictor_names)
        object.__setattr__(self, "_support", np.arange(self.min, self.max + 1, dtype=float))

    def _linear_predictor(self, beta_0: float, c: float) -> np.ndarray:
        return beta_0 + self.X @ (c * self.beta_1_init)

    def _rate(self, beta_0: float, c: float) -> np.ndarray:
        return np.exp(self._linear_predictor(beta_0, c))

    def target_mean_value(self, beta_0: float, c: float) -> float:
        rates = self._rate(beta_0, c)
        means = np.empty(len(rates), dtype=float)
        for idx, current_rate in enumerate(rates):
            means[idx] = Poisson(float(current_rate), self.min, self.max).target_mean()
        return float(means.mean())

    def target_snr_value(self, beta_0: float, c: float) -> float:
        rates = self._rate(beta_0, c)
        cond_mean = np.empty(len(rates), dtype=float)
        cond_var = np.empty(len(rates), dtype=float)
        for idx, current_rate in enumerate(rates):
            dist = stats.poisson(float(current_rate))
            lower = dist.cdf(self.min - 1)
            upper = dist.cdf(self.max)
            if lower >= upper:
                raise ValueError("truncation interval has zero probability mass")
            pmf = dist.pmf(self._support)
            pmf = pmf / float(np.sum(pmf))
            current_mean = float(np.sum(self._support * pmf))
            current_second = float(np.sum((self._support**2) * pmf))
            cond_mean[idx] = current_mean
            cond_var[idx] = current_second - current_mean**2
        within = float(cond_var.mean())
        if within <= 0:
            raise ValueError("Conditional variance is non-positive")
        return float(cond_mean.var() / within)

    def _feasible_initial_guess(self) -> tuple[float, float]:
        beta_0 = float(np.log(max(self.target_mean, 1e-6)))
        c = 1.0
        return beta_0, c

    def calibrate(self) -> "PoissonRegressor":
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
            raise ValueError(f"Unable to calibrate PoissonRegressor: {result.message}")

        beta_0, c = result.x
        return replace(self, beta_0=float(beta_0), c=float(c))

    def sample(self, n: int) -> np.ndarray:
        if self.beta_0 is None or self.c is None:
            raise ValueError("PoissonRegressor must be calibrated before sampling")

        lam = np.repeat(self._rate(self.beta_0, self.c), int(np.ceil(n / len(self.X))))[:n]
        samples = np.empty(n, dtype=float)
        rng = np.random.default_rng()
        for idx, current_lam in enumerate(lam):
            dist = stats.poisson(current_lam)
            lower = dist.cdf(self.min - 1)
            upper = dist.cdf(self.max)
            u = rng.uniform(lower, upper)
            samples[idx] = dist.ppf(u)
        return _as_1d_array(np.clip(samples, self.min, self.max))


@dataclass(frozen=True)
class GeometricRegressor:
    target_mean: float
    target_snr: float
    X: np.ndarray
    beta_1_init: np.ndarray
    predictor_names: list[str] | None = None
    predictor_transformations: dict[str, str] | None = None
    min: int = 0
    max: int = 20
    beta_0: float | None = None
    c: float | None = None

    def __post_init__(self) -> None:
        _validate_bounds(self.min, self.max)
        X = np.asarray(self.X, dtype=float)
        beta_1_init = np.asarray(self.beta_1_init, dtype=float).reshape(-1)
        if X.ndim != 2:
            raise ValueError("X must be a 2D regression matrix")
        if X.shape[1] != beta_1_init.shape[0]:
            raise ValueError("beta_1_init must have one coefficient per column in X")
        if self.min < 0:
            raise ValueError("min must be >= 0")
        if self.target_mean <= 0:
            raise ValueError("target_mean must be positive")
        if self.target_snr <= 0:
            raise ValueError("target_snr must be positive")
        predictor_names = self.predictor_names or [f"x{i}" for i in range(X.shape[1])]
        X, predictor_names = _transform_predictors(X, predictor_names, self.predictor_transformations)
        object.__setattr__(self, "X", X)
        object.__setattr__(self, "beta_1_init", beta_1_init)
        object.__setattr__(self, "predictor_names", predictor_names)

    def _linear_predictor(self, beta_0: float, c: float) -> np.ndarray:
        return beta_0 + self.X @ (c * self.beta_1_init)

    def _probability(self, beta_0: float, c: float) -> np.ndarray:
        eta = self._linear_predictor(beta_0, c)
        return 1.0 / (1.0 + np.exp(-eta))

    def target_mean_value(self, beta_0: float, c: float) -> float:
        probs = self._probability(beta_0, c)
        means = np.empty(len(probs), dtype=float)
        for idx, current_p in enumerate(probs):
            means[idx] = Geometric(float(current_p), self.min, self.max).target_mean()
        return float(means.mean())

    def target_snr_value(self, beta_0: float, c: float) -> float:
        probs = self._probability(beta_0, c)
        cond_mean = np.empty(len(probs), dtype=float)
        cond_var = np.empty(len(probs), dtype=float)
        for idx, current_p in enumerate(probs):
            dist = stats.nbinom(1, float(current_p))
            lower = dist.cdf(self.min - 1)
            upper = dist.cdf(self.max)
            if lower >= upper:
                raise ValueError("truncation interval has zero probability mass")
            pmf = dist.pmf(np.arange(self.min, self.max + 1, dtype=float))
            pmf = pmf / float(np.sum(pmf))
            support = np.arange(self.min, self.max + 1, dtype=float)
            current_mean = float(np.sum(support * pmf))
            current_second = float(np.sum((support**2) * pmf))
            cond_mean[idx] = current_mean
            cond_var[idx] = current_second - current_mean**2
        within = float(cond_var.mean())
        if within <= 0:
            raise ValueError("Conditional variance is non-positive")
        return float(cond_mean.var() / within)

    def _feasible_initial_guess(self) -> tuple[float, float]:
        p = np.clip(1.0 / (1.0 + self.target_mean), 1e-4, 1.0 - 1e-4)
        x_mean = np.mean(self.X @ self.beta_1_init)
        beta_0 = np.log(p / (1.0 - p)) - x_mean
        c = 1.0
        return beta_0, c

    def calibrate(self) -> "GeometricRegressor":
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
            raise ValueError(f"Unable to calibrate GeometricRegressor: {result.message}")

        beta_0, c = result.x
        return replace(self, beta_0=float(beta_0), c=float(c))

    def sample(self, n: int) -> np.ndarray:
        if self.beta_0 is None or self.c is None:
            raise ValueError("GeometricRegressor must be calibrated before sampling")

        p = np.repeat(self._probability(self.beta_0, self.c), int(np.ceil(n / len(self.X))))[:n]
        samples = np.empty(n, dtype=float)
        rng = np.random.default_rng()
        for idx, current_p in enumerate(p):
            dist = stats.nbinom(1, current_p)
            lower = dist.cdf(self.min - 1)
            upper = dist.cdf(self.max)
            u = rng.uniform(lower, upper)
            samples[idx] = dist.ppf(u)
        return _as_1d_array(np.clip(samples, self.min, self.max))


@dataclass(frozen=True)
class NegativeBinomialRegressor:
    target_mean: float
    target_variance: float
    target_snr: float
    min: int
    max: int
    X: np.ndarray
    beta_1_init: np.ndarray
    predictor_names: list[str] | None = None
    predictor_transformations: dict[str, str] | None = None
    shape: float | None = None
    beta_0: float | None = None
    c: float | None = None

    def __post_init__(self) -> None:
        _validate_bounds(self.min, self.max)
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
        predictor_names = self.predictor_names or [f"x{i}" for i in range(X.shape[1])]
        X, predictor_names = _transform_predictors(X, predictor_names, self.predictor_transformations)
        object.__setattr__(self, "X", X)
        object.__setattr__(self, "beta_1_init", beta_1_init)
        object.__setattr__(self, "predictor_names", predictor_names)

    def _linear_predictor(self, beta_0: float, c: float) -> np.ndarray:
        return beta_0 + self.X @ (c * self.beta_1_init)

    def _mean_param(self, beta_0: float, c: float) -> np.ndarray:
        return np.exp(self._linear_predictor(beta_0, c))

    def _row_moments(self, beta_0: float, c: float, shape: float) -> tuple[np.ndarray, np.ndarray]:
        mean_param = self._mean_param(beta_0, c)
        cond_mean = np.empty(len(mean_param), dtype=float)
        cond_var = np.empty(len(mean_param), dtype=float)
        for idx, current_mean in enumerate(mean_param):
            p = shape / (shape + current_mean)
            dist = stats.nbinom(shape, p)
            lower = dist.cdf(self.min - 1)
            upper = dist.cdf(self.max)
            if lower >= upper:
                raise ValueError("truncation interval has zero probability mass")
            support = np.arange(self.min, self.max + 1, dtype=float)
            pmf = dist.pmf(support)
            pmf = pmf / float(np.sum(pmf))
            row_mean = float(np.sum(support * pmf))
            row_second = float(np.sum((support**2) * pmf))
            cond_mean[idx] = row_mean
            cond_var[idx] = row_second - row_mean**2
        return cond_mean, cond_var

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
        shape = max(self.target_snr + 1.0, 1.0 + 1e-3)
        mean_param = max(self.target_mean, 1e-6)
        beta_0 = float(np.log(mean_param))
        c = 1.0
        return beta_0, c, shape

    def calibrate(self) -> "NegativeBinomialRegressor":
        if self.beta_0 is not None and self.c is not None and self.shape is not None:
            return self

        initial_beta_0, initial_c, initial_shape = self._feasible_initial_guess()

        def residuals(params: np.ndarray) -> np.ndarray:
            beta_0, c, log_shape = params
            shape = float(np.exp(log_shape))
            try:
                mean_residual = self.target_mean_value(beta_0, c, shape) - self.target_mean
                variance_residual = self.target_variance_value(beta_0, c, shape) - self.target_variance
                snr_residual = self.target_snr_value(beta_0, c, shape) - self.target_snr
                return np.array([mean_residual, variance_residual, snr_residual], dtype=float)
            except ValueError:
                return np.array([1e6, 1e6, 1e6], dtype=float)

        result = optimize.least_squares(
            residuals,
            x0=np.array([initial_beta_0, initial_c, np.log(initial_shape)], dtype=float),
            bounds=([-np.inf, 0.0, -np.inf], [np.inf, np.inf, np.inf]),
        )

        if not result.success:
            raise ValueError(f"Unable to calibrate NegativeBinomialRegressor: {result.message}")

        beta_0, c, log_shape = result.x
        shape = float(np.exp(log_shape))
        return replace(self, beta_0=float(beta_0), c=float(c), shape=shape)

    def sample(self, n: int) -> np.ndarray:
        if self.beta_0 is None or self.c is None or self.shape is None:
            raise ValueError("NegativeBinomialRegressor must be calibrated before sampling")

        mean_param = np.repeat(self._mean_param(self.beta_0, self.c), int(np.ceil(n / len(self.X))))[:n]
        samples = np.empty(n, dtype=float)
        rng = np.random.default_rng()
        for idx, current_mean in enumerate(mean_param):
            p = self.shape / (self.shape + current_mean)
            dist = stats.nbinom(self.shape, p)
            lower = dist.cdf(self.min - 1)
            upper = dist.cdf(self.max)
            u = rng.uniform(lower, upper)
            samples[idx] = dist.ppf(u)
        return _as_1d_array(np.clip(samples, self.min, self.max))


@dataclass(frozen=True)
class CategoricalNominalRegressor:
    target_probabilities: np.ndarray
    X: np.ndarray
    beta_1: np.ndarray
    predictor_names: list[str] | None = None
    predictor_transformations: dict[str, str] | None = None
    beta_0: np.ndarray | None = None

    def __post_init__(self) -> None:
        probabilities = np.asarray(self.target_probabilities, dtype=float).reshape(-1)
        X = np.asarray(self.X, dtype=float)
        beta_1 = np.asarray(self.beta_1, dtype=float)
        if probabilities.ndim != 1 or probabilities.size < 2:
            raise ValueError("target_probabilities must be a 1D vector with at least 2 categories")
        if np.any(probabilities < 0):
            raise ValueError("target_probabilities must be non-negative")
        total = float(probabilities.sum())
        if total <= 0:
            raise ValueError("target_probabilities must sum to a positive value")
        probabilities = probabilities / total
        if X.ndim != 2:
            raise ValueError("X must be a 2D regression matrix")
        if beta_1.ndim != 2:
            raise ValueError("beta_1 must be a 2D matrix")
        if X.shape[1] != beta_1.shape[0]:
            raise ValueError("beta_1 must have one row per column in X")
        if beta_1.shape[1] != probabilities.size - 1:
            raise ValueError("beta_1 must have K-1 columns for K response categories")
        predictor_names = self.predictor_names or [f"x{i}" for i in range(X.shape[1])]
        X, predictor_names = _transform_predictors(X, predictor_names, self.predictor_transformations)
        object.__setattr__(self, "target_probabilities", probabilities)
        object.__setattr__(self, "X", X)
        object.__setattr__(self, "beta_1", beta_1)
        object.__setattr__(self, "predictor_names", predictor_names)

    def _linear_predictor(self, beta_0: np.ndarray) -> np.ndarray:
        return beta_0 + self.X @ self.beta_1

    def _probabilities(self, beta_0: np.ndarray) -> np.ndarray:
        eta = self._linear_predictor(beta_0)
        logits = np.column_stack([eta, np.zeros(len(self.X), dtype=float)])
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)

    def target_probabilities_value(self, beta_0: np.ndarray) -> np.ndarray:
        return self._probabilities(beta_0).mean(axis=0)

    def _feasible_initial_guess(self) -> np.ndarray:
        target = np.clip(self.target_probabilities[:-1], 1e-6, 1.0 - 1e-6)
        baseline = np.clip(self.target_probabilities[-1], 1e-6, 1.0 - 1e-6)
        beta_0 = np.log(target / baseline)
        return beta_0

    def calibrate(self) -> "CategoricalNominalRegressor":
        if self.beta_0 is not None:
            return self

        initial_beta_0 = self._feasible_initial_guess()

        def residuals(beta_0: np.ndarray) -> np.ndarray:
            try:
                model = self.target_probabilities_value(beta_0)
                return model[:-1] - self.target_probabilities[:-1]
            except ValueError:
                return np.full(self.target_probabilities.size - 1, 1e6, dtype=float)

        result = optimize.least_squares(residuals, x0=initial_beta_0, bounds=(-np.inf, np.inf))

        if not result.success:
            raise ValueError(f"Unable to calibrate CategoricalNominalRegressor: {result.message}")

        return replace(self, beta_0=np.asarray(result.x, dtype=float))

    def sample(self, n: int) -> np.ndarray:
        if self.beta_0 is None:
            raise ValueError("CategoricalNominalRegressor must be calibrated before sampling")

        probabilities = self._probabilities(self.beta_0)
        row_probs = np.repeat(probabilities, int(np.ceil(n / len(probabilities))), axis=0)[:n]
        rng = np.random.default_rng()
        categories = np.arange(probabilities.shape[1])
        samples = np.empty(n, dtype=int)
        for idx, probs in enumerate(row_probs):
            samples[idx] = rng.choice(categories, p=probs)
        return _as_1d_array(samples)


@dataclass(frozen=True)
class CategoricalOrdinalRegressor:
    target_probabilities: np.ndarray
    X: np.ndarray
    beta_1: np.ndarray
    predictor_names: list[str] | None = None
    predictor_transformations: dict[str, str] | None = None
    beta_0: np.ndarray | None = None

    def __post_init__(self) -> None:
        probabilities = np.asarray(self.target_probabilities, dtype=float).reshape(-1)
        X = np.asarray(self.X, dtype=float)
        beta_1 = np.asarray(self.beta_1, dtype=float).reshape(-1)
        if probabilities.ndim != 1 or probabilities.size < 2:
            raise ValueError("target_probabilities must be a 1D vector with at least 2 categories")
        if np.any(probabilities < 0):
            raise ValueError("target_probabilities must be non-negative")
        total = float(probabilities.sum())
        if total <= 0:
            raise ValueError("target_probabilities must sum to a positive value")
        probabilities = probabilities / total
        if X.ndim != 2:
            raise ValueError("X must be a 2D regression matrix")
        if X.shape[1] != beta_1.shape[0]:
            raise ValueError("beta_1 must have one coefficient per column in X")
        predictor_names = self.predictor_names or [f"x{i}" for i in range(X.shape[1])]
        X, predictor_names = _transform_predictors(X, predictor_names, self.predictor_transformations)
        object.__setattr__(self, "target_probabilities", probabilities)
        object.__setattr__(self, "X", X)
        object.__setattr__(self, "beta_1", beta_1)
        object.__setattr__(self, "predictor_names", predictor_names)

    def _linear_predictor(self) -> np.ndarray:
        return self.X @ self.beta_1

    def _thresholds(self, beta_0: np.ndarray) -> np.ndarray:
        beta_0 = np.asarray(beta_0, dtype=float).reshape(-1)
        if beta_0.size != self.target_probabilities.size - 1:
            raise ValueError("beta_0 must have K-1 thresholds for K categories")
        if np.any(np.diff(beta_0) <= 0):
            raise ValueError("beta_0 thresholds must be strictly increasing")
        return beta_0

    def _cumulative_probabilities(self, beta_0: np.ndarray) -> np.ndarray:
        thresholds = self._thresholds(beta_0)
        eta = self._linear_predictor()
        cumulative = 1.0 / (1.0 + np.exp(-(thresholds[None, :] - eta[:, None])))
        first_k = cumulative
        last = np.ones((len(self.X), 1), dtype=float)
        return np.column_stack([first_k, last])

    def _category_probabilities(self, beta_0: np.ndarray) -> np.ndarray:
        cumulative = self._cumulative_probabilities(beta_0)
        probs = np.empty((len(self.X), self.target_probabilities.size), dtype=float)
        probs[:, 0] = cumulative[:, 0]
        for idx in range(1, self.target_probabilities.size - 1):
            probs[:, idx] = cumulative[:, idx] - cumulative[:, idx - 1]
        probs[:, -1] = 1.0 - cumulative[:, -2]
        return probs

    def target_probabilities_value(self, beta_0: np.ndarray) -> np.ndarray:
        return self._category_probabilities(beta_0).mean(axis=0)

    def _feasible_initial_guess(self) -> np.ndarray:
        cumulative = np.cumsum(self.target_probabilities[:-1])
        cumulative = np.clip(cumulative, 1e-6, 1.0 - 1e-6)
        eta_mean = float(np.mean(self._linear_predictor()))
        thresholds = np.log(cumulative / (1.0 - cumulative)) + eta_mean
        return np.maximum.accumulate(thresholds)

    def calibrate(self) -> "CategoricalOrdinalRegressor":
        if self.beta_0 is not None:
            return self

        initial_beta_0 = self._feasible_initial_guess()

        def residuals(beta_0: np.ndarray) -> np.ndarray:
            try:
                model = self.target_probabilities_value(beta_0)
                return model - self.target_probabilities
            except ValueError:
                return np.full(self.target_probabilities.size, 1e6, dtype=float)

        lower = np.full(self.target_probabilities.size - 1, -np.inf, dtype=float)
        upper = np.full(self.target_probabilities.size - 1, np.inf, dtype=float)
        result = optimize.least_squares(residuals, x0=initial_beta_0, bounds=(lower, upper))

        if not result.success:
            raise ValueError(f"Unable to calibrate CategoricalOrdinalRegressor: {result.message}")

        return replace(self, beta_0=np.asarray(np.maximum.accumulate(result.x), dtype=float))

    def sample(self, n: int) -> np.ndarray:
        if self.beta_0 is None:
            raise ValueError("CategoricalOrdinalRegressor must be calibrated before sampling")

        probabilities = self._category_probabilities(self.beta_0)
        row_probs = np.repeat(probabilities, int(np.ceil(n / len(probabilities))), axis=0)[:n]
        rng = np.random.default_rng()
        categories = np.arange(probabilities.shape[1])
        samples = np.empty(n, dtype=int)
        for idx, probs in enumerate(row_probs):
            samples[idx] = rng.choice(categories, p=probs)
        return _as_1d_array(samples)
