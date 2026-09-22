from dataclasses import dataclass, replace

import numpy as np
from scipy import optimize, stats

try:
    from distributions import Beta, Binomial, Bernoulli, Gamma, LogNormal, NegativeBinomial, Normal, Poisson
except ImportError:  # pragma: no cover
    from .distributions import Beta, Binomial, Bernoulli, Gamma, LogNormal, NegativeBinomial, Normal, Poisson


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
class NoneRegressor:
    X: np.ndarray
    beta_0: float | np.ndarray | None = None
    beta_1: np.ndarray | None = None
    predictor_names: list[str] | None = None
    predictor_transformations: dict[str, str] | None = None
    response_type: str = "quantitative"
    categories: list[str] | None = None

    def calibrate(self) -> "NoneRegressor":
        return self

    def sample(self, n: int) -> np.ndarray:
        n = int(n)
        if self.response_type == "quantitative":
            if self.X.ndim == 2 and self.X.shape[0] > 0:
                row = self.X[0]
                if self.beta_1 is not None and np.asarray(self.beta_1).size == row.size:
                    return (np.asarray(self.beta_0 if self.beta_0 is not None else 0.0, dtype=float) + self.X @ np.asarray(self.beta_1, dtype=float).reshape(-1)).reshape(-1)[:n]
            return np.full(n, float(self.beta_0 if self.beta_0 is not None else 0.0), dtype=float)

        if self.categories is None:
            raise ValueError("categories are required for categorical NoneRegressor")
        if self.response_type == "categorical_nominal":
            return np.asarray([self.categories[0]] * n, dtype=object)
        if self.response_type == "categorical_ordinal":
            return np.asarray([self.categories[0]] * n, dtype=object)
        raise ValueError(f"Unsupported response_type: {self.response_type}")


@dataclass(frozen=True, kw_only=True)
class NormalRegressor(Normal):
    X: np.ndarray
    beta_1: np.ndarray
    predictor_names: list[str] | None = None
    predictor_transformations: dict[str, str] | None = None
    beta_0: float | None = None
    sigma2: float | None = None

    def __post_init__(self) -> None:
        _validate_bounds(self.min, self.max)
        X = np.asarray(self.X, dtype=float)
        beta_1 = np.asarray(self.beta_1, dtype=float).reshape(-1)
        if X.ndim != 2:
            raise ValueError("X must be a 2D regression matrix")
        if X.shape[1] != beta_1.shape[0]:
            raise ValueError("beta_1 must have one coefficient per column in X")
        predictor_names = self.predictor_names or [f"x{i}" for i in range(X.shape[1])]
        X, predictor_names = _transform_predictors(X, predictor_names, self.predictor_transformations)
        object.__setattr__(self, "X", X)
        object.__setattr__(self, "beta_1", beta_1)
        object.__setattr__(self, "predictor_names", predictor_names)

    def calibrate(self) -> "NormalRegressor":
        if self.truncated:
            return self._calibrate_truncated()
        return self._calibrate_untruncated()

    def _calibrate_untruncated(self) -> "NormalRegressor":
        self._validate_truncation()

        target_mean = self.get_mean()
        target_variance = self.get_variance()
        predictor_mean = float(np.mean(self.X @ self.beta_1))
        predictor_variance = float(np.var(self.X @ self.beta_1))
        sigma2 = float(target_variance - predictor_variance)
        if sigma2 <= 0:
            raise ValueError(
                "Unable to calibrate NormalRegressor: implied sigma2 must be positive; "
                f"target_mean={target_mean:.6g}, target_variance={target_variance:.6g}, "
                f"predictor_mean={predictor_mean:.6g}, predictor_variance={predictor_variance:.6g}, "
                f"sigma2={sigma2:.6g}, X_shape={self.X.shape}, beta_1_shape={self.beta_1.shape}"
            )

        beta_0 = float(target_mean - predictor_mean)
        return NormalRegressor(
            mean=float(beta_0),
            standard_deviation=float(np.sqrt(sigma2)),
            min=self.min,
            max=self.max,
            truncated=self.truncated,
            X=self.X,
            beta_1=self.beta_1,
            predictor_names=self.predictor_names,
            predictor_transformations=None,
            beta_0=float(beta_0),
            sigma2=sigma2,
        )

    def _calibrate_truncated(self) -> "NormalRegressor":
        raise NotImplementedError("NormalRegressor truncated calibration not yet implemented")

    def _linear_predictor(self, beta_0: float) -> np.ndarray:
        return beta_0 + self.X @ self.beta_1

    def sample(self, n: int) -> np.ndarray:
        if self.beta_0 is None or self.sigma2 is None:
            raise ValueError("NormalRegressor must be calibrated before sampling")

        self._validate_truncation()
        sigma = np.sqrt(self.sigma2)
        eta = self._linear_predictor(self.beta_0)
        mean = np.repeat(eta, int(np.ceil(n / len(eta))))[:n]
        a = (self.min - mean) / sigma
        b = (self.max - mean) / sigma
        return _as_1d_array(stats.truncnorm.rvs(a, b, loc=mean, scale=sigma, size=n))


@dataclass(frozen=True, kw_only=True)
class GammaRegressor(Gamma):
    X: np.ndarray
    shape: float
    beta_0: float
    beta_1: np.ndarray
    predictor_names: list[str] | None = None
    predictor_transformations: dict[str, str] | None = None

    def __post_init__(self) -> None:
        _validate_bounds(self.min, self.max)
        X = np.asarray(self.X, dtype=float)
        beta_1 = np.asarray(self.beta_1, dtype=float).reshape(-1)
        if X.ndim != 2:
            raise ValueError("X must be a 2D regression matrix")
        if X.shape[1] != beta_1.shape[0]:
            raise ValueError("beta_1 must have one coefficient per column in X")
        if self.shape <= 0:
            raise ValueError("shape must be positive")
        predictor_names = self.predictor_names or [f"x{i}" for i in range(X.shape[1])]
        X, predictor_names = _transform_predictors(X, predictor_names, self.predictor_transformations)
        object.__setattr__(self, "X", X)
        object.__setattr__(self, "beta_1", beta_1)
        object.__setattr__(self, "predictor_names", predictor_names)

    def calibrate(self) -> "GammaRegressor":
        if self.truncated:
            return self._calibrate_truncated()
        return self._calibrate_untruncated()

    def _mu(self, beta_0: float) -> np.ndarray:
        return np.exp(beta_0 + self.X @ self.beta_1)

    def _calibrate_untruncated(self) -> "GammaRegressor":
        self._validate_truncation()

        target_mean = self.get_mean()
        target_variance = self.get_variance()
        g = self.X @ self.beta_1
        exp_g = np.exp(g)
        m1 = float(np.mean(exp_g))
        m2 = float(np.mean(exp_g**2))

        eps = np.finfo(float).tiny
        if not np.isfinite(m1) or m1 <= 0:
            raise ValueError("Unable to calibrate GammaRegressor: mean(exp(X @ beta_1)) must be finite and positive")
        if not np.isfinite(m2) or m2 <= 0:
            raise ValueError("Unable to calibrate GammaRegressor: mean(exp(2 * X @ beta_1)) must be finite and positive")

        beta_0 = float(np.log(max(target_mean, eps)) - np.log(m1))

        ratio = m2 / (m1 * m1)
        if not np.isfinite(ratio) or ratio < 1.0:
            raise ValueError("Unable to calibrate GammaRegressor: predictor moment ratio must be finite and at least 1")
        denominator = target_variance - target_mean * target_mean * (ratio - 1.0)
        if denominator <= 0:
            raise ValueError(
                "Unable to calibrate GammaRegressor: target_variance is too small for the observed predictor moments; "
                f"target_mean={target_mean:.6g}, target_variance={target_variance:.6g}, ratio={ratio:.6g}, denominator={denominator:.6g}"
            )

        shape = float(target_mean * target_mean * ratio / denominator)
        if shape <= 0:
            raise ValueError(
                "Unable to calibrate GammaRegressor: implied shape must be positive; "
                f"target_mean={target_mean:.6g}, target_variance={target_variance:.6g}, ratio={ratio:.6g}, shape={shape:.6g}"
            )
        return GammaRegressor(
            shape=shape,
            rate=shape / float(np.exp(beta_0)),
            min=self.min,
            max=self.max,
            truncated=self.truncated,
            X=self.X,
            beta_1=self.beta_1,
            predictor_names=self.predictor_names,
            predictor_transformations=None,
            beta_0=float(beta_0),
        )

    def _calibrate_truncated(self) -> "GammaRegressor":
        raise NotImplementedError("GammaRegressor truncated calibration not yet implemented")

    def sample(self, n: int) -> np.ndarray:
        mu = np.repeat(self._mu(self.beta_0), int(np.ceil(n / len(self.X))))[:n]
        scale = mu / self.shape
        dist = stats.gamma(a=self.shape, scale=scale)
        lower = dist.cdf(self.min)
        upper = dist.cdf(self.max)
        if np.any(lower >= upper):
            raise ValueError("truncation interval has zero probability mass")
        uniforms = np.random.uniform(lower, upper, size=n)
        return _as_1d_array(dist.ppf(uniforms))


@dataclass(frozen=True, kw_only=True)
class LogNormalRegressor(LogNormal):
    X: np.ndarray
    log_mean: float
    beta_1: np.ndarray
    log_standard_deviation: float
    predictor_names: list[str] | None = None
    predictor_transformations: dict[str, str] | None = None
    beta_0: float | None = None

    def __post_init__(self) -> None:
        _validate_bounds(self.min, self.max)
        X = np.asarray(self.X, dtype=float)
        beta_1 = np.asarray(self.beta_1, dtype=float).reshape(-1)
        if X.ndim != 2:
            raise ValueError("X must be a 2D regression matrix")
        if X.shape[1] != beta_1.shape[0]:
            raise ValueError("beta_1 must have one coefficient per column in X")
        if self.log_standard_deviation <= 0:
            raise ValueError("log_standard_deviation must be positive")
        predictor_names = self.predictor_names or [f"x{i}" for i in range(X.shape[1])]
        X, predictor_names = _transform_predictors(X, predictor_names, self.predictor_transformations)
        object.__setattr__(self, "X", X)
        object.__setattr__(self, "beta_1", beta_1)
        object.__setattr__(self, "predictor_names", predictor_names)

    def calibrate(self) -> "LogNormalRegressor":
        if self.truncated:
            return self._calibrate_truncated()
        return self._calibrate_untruncated()

    def _log_mean(self, beta_0: float) -> np.ndarray:
        return beta_0 + self.X @ self.beta_1

    def _calibrate_untruncated(self) -> "LogNormalRegressor":
        self._validate_truncation()

        target_mean = self.get_mean()
        target_variance = self.get_variance()
        g = self.X @ self.beta_1
        exp_g = np.exp(g)
        m1 = float(np.mean(exp_g))
        m2 = float(np.mean(exp_g**2))

        eps = np.finfo(float).tiny
        if not np.isfinite(m1) or m1 <= 0:
            raise ValueError("Unable to calibrate LogNormalRegressor: mean(exp(X @ beta_1)) must be finite and positive")
        if not np.isfinite(m2) or m2 <= 0:
            raise ValueError("Unable to calibrate LogNormalRegressor: mean(exp(2 * X @ beta_1)) must be finite and positive")

        mean_ratio = target_mean / max(m1, eps)
        sigma2_argument = ((target_variance / max(target_mean * target_mean, eps)) + 1.0) * (m1 * m1 / m2)
        if not np.isfinite(sigma2_argument) or sigma2_argument <= 0:
            raise ValueError(
                "Unable to calibrate LogNormalRegressor: sigma^2 argument must be positive; "
                f"target_mean={target_mean:.6g}, target_variance={target_variance:.6g}, m1={m1:.6g}, m2={m2:.6g}, sigma2_argument={sigma2_argument:.6g}"
            )

        sigma2 = float(max(0.0, np.log(sigma2_argument)))
        sigma = float(np.sqrt(sigma2))
        beta_0 = float(np.log(max(mean_ratio, eps)) - 0.5 * sigma2)
        return LogNormalRegressor(
            log_mean=float(beta_0),
            log_standard_deviation=sigma,
            min=self.min,
            max=self.max,
            truncated=self.truncated,
            X=self.X,
            beta_1=self.beta_1,
            predictor_names=self.predictor_names,
            predictor_transformations=None,
            beta_0=float(beta_0),
        )

    def _calibrate_truncated(self) -> "LogNormalRegressor":
        raise NotImplementedError("LogNormalRegressor truncated calibration not yet implemented")

    def sample(self, n: int) -> np.ndarray:
        if self.beta_0 is None:
            raise ValueError("LogNormalRegressor must be calibrated before sampling")

        log_mean = np.repeat(self._log_mean(self.beta_0), int(np.ceil(n / len(self.X))))[:n]
        dist = stats.lognorm(s=self.log_standard_deviation, scale=np.exp(log_mean))
        lower = dist.cdf(self.min)
        upper = dist.cdf(self.max)
        if np.any(lower >= upper):
            raise ValueError("truncation interval has zero probability mass")
        uniforms = np.random.uniform(lower, upper, size=n)
        return _as_1d_array(dist.ppf(uniforms))


@dataclass(frozen=True, kw_only=True)
class BetaRegressor(Beta):
    X: np.ndarray
    beta_1: np.ndarray
    predictor_names: list[str] | None = None
    predictor_transformations: dict[str, str] | None = None
    beta_0: float | None = None
    phi: float | None = None

    def __post_init__(self) -> None:
        _validate_bounds(self.min, self.max)
        X = np.asarray(self.X, dtype=float)
        beta_1 = np.asarray(self.beta_1, dtype=float).reshape(-1)
        if X.ndim != 2:
            raise ValueError("X must be a 2D regression matrix")
        if X.shape[1] != beta_1.shape[0]:
            raise ValueError("beta_1 must have one coefficient per column in X")
        predictor_names = self.predictor_names or [f"x{i}" for i in range(X.shape[1])]
        X, predictor_names = _transform_predictors(X, predictor_names, self.predictor_transformations)
        object.__setattr__(self, "X", X)
        object.__setattr__(self, "beta_1", beta_1)
        object.__setattr__(self, "predictor_names", predictor_names)

    def calibrate(self) -> "BetaRegressor":
        return self._calibrate_untruncated()

    def _mu(self, beta_0: float) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(beta_0 + self.X @ self.beta_1)))

    def _calibrate_untruncated(self) -> "BetaRegressor":
        target_mean = self.get_unit_mean()
        target_variance = self.get_unit_variance()

        if not np.isfinite(target_mean) or target_mean <= 0.0 or target_mean >= 1.0:
            raise ValueError(
                "Unable to calibrate BetaRegressor: target mean must be finite and strictly between 0 and 1; "
                f"target_mean={target_mean:.6g}"
            )
        if not np.isfinite(target_variance) or target_variance <= 0.0:
            raise ValueError(
                "Unable to calibrate BetaRegressor: target variance must be finite and positive; "
                f"target_mean={target_mean:.6g}, target_variance={target_variance:.6g}"
            )

        predictor_linear = self.X @ self.beta_1
        if not np.all(np.isfinite(predictor_linear)):
            raise ValueError(
                "Unable to calibrate BetaRegressor: X @ beta_1 must be finite; "
                f"X_shape={self.X.shape}, beta_1_shape={self.beta_1.shape}"
            )

        beta_0_guess = float(np.log(target_mean / max(1.0 - target_mean, np.finfo(float).tiny)))
        phi_guess = max(target_mean * (1.0 - target_mean) / max(target_variance, np.finfo(float).tiny) - 1.0, 1e-6)

        def moments(params: np.ndarray) -> np.ndarray:
            beta_0, log_phi = params
            phi = float(np.exp(log_phi))
            mu = self._mu(beta_0)
            mean_mu = float(mu.mean())
            var_mu = float(mu.var())
            e_var = float((mu * (1.0 - mu)).mean() / (phi + 1.0))
            total_var = var_mu + e_var
            if not np.all(np.isfinite([mean_mu, var_mu, e_var, total_var])):
                return np.array([1e6, 1e6], dtype=float)
            return np.array(
                [
                    mean_mu - target_mean,
                    total_var - target_variance,
                ],
                dtype=float,
            )

        result = optimize.least_squares(
            moments,
            x0=np.array([beta_0_guess, np.log(phi_guess)], dtype=float),
            bounds=([-np.inf, -np.inf], [np.inf, np.inf]),
        )

        if not result.success:
            raise ValueError(
                "Unable to calibrate BetaRegressor: nonlinear solve failed; "
                f"message={result.message}; target_mean={target_mean:.6g}, target_variance={target_variance:.6g}, "
                f"X_shape={self.X.shape}, beta_1_shape={self.beta_1.shape}, residual_norm={np.linalg.norm(result.fun):.6g}"
            )

        beta_0, log_phi = result.x
        return BetaRegressor(
            shape_1=self.shape_1,
            shape_2=self.shape_2,
            min=self.min,
            max=self.max,
            X=self.X,
            beta_1=self.beta_1,
            predictor_names=self.predictor_names,
            predictor_transformations=None,
            beta_0=float(beta_0),
            phi=float(np.exp(log_phi)),
        )

    def sample(self, n: int) -> np.ndarray:
        if self.beta_0 is None or self.phi is None:
            raise ValueError("BetaRegressor must be calibrated before sampling")

        mu = np.repeat(self._mu(self.beta_0), int(np.ceil(n / len(self.X))))[:n]
        alpha = self.phi * mu
        beta_param = self.phi * (1.0 - mu)
        samples = stats.beta.rvs(alpha, beta_param, size=n)
        return _as_1d_array(self.min + (self.max - self.min) * samples)


@dataclass(frozen=True, kw_only=True)
class BinomialRegressor(Binomial):
    success_prob: float
    X: np.ndarray
    beta_1: np.ndarray
    predictor_names: list[str] | None = None
    predictor_transformations: dict[str, str] | None = None
    beta_0: float | None = None

    def __post_init__(self) -> None:
        _validate_bounds(self.min, self.max)
        X = np.asarray(self.X, dtype=float)
        beta_1 = np.asarray(self.beta_1, dtype=float).reshape(-1)
        if X.ndim != 2:
            raise ValueError("X must be a 2D regression matrix")
        if X.shape[1] != beta_1.shape[0]:
            raise ValueError("beta_1 must have one coefficient per column in X")
        if self.n_trials < 1:
            raise ValueError("n_trials must be >= 1")
        predictor_names = self.predictor_names or [f"x{i}" for i in range(X.shape[1])]
        X, predictor_names = _transform_predictors(X, predictor_names, self.predictor_transformations)
        object.__setattr__(self, "X", X)
        object.__setattr__(self, "beta_1", beta_1)
        object.__setattr__(self, "predictor_names", predictor_names)

    def calibrate(self) -> "BinomialRegressor":
        return self._calibrate_truncated() if self.truncated else self._calibrate_untruncated()

    def _mu(self, beta_0: float) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(beta_0 + self.X @ self.beta_1)))

    def _calibrate_untruncated(self) -> "BinomialRegressor":
        target_mean = self.get_mean()
        if not np.isfinite(target_mean):
            raise ValueError(f"Unable to calibrate BinomialRegressor: target_mean must be finite; target_mean={target_mean:.6g}")
        if target_mean <= 0.0 or target_mean >= self.n_trials:
            raise ValueError(
                "Unable to calibrate BinomialRegressor: target_mean must be strictly between 0 and n_trials; "
                f"target_mean={target_mean:.6g}, n_trials={self.n_trials}"
            )

        predictor_linear = self.X @ self.beta_1
        if not np.all(np.isfinite(predictor_linear)):
            raise ValueError(
                "Unable to calibrate BinomialRegressor: X @ beta_1 must be finite; "
                f"X_shape={self.X.shape}, beta_1_shape={self.beta_1.shape}"
            )

        mean_prob = np.clip(target_mean / self.n_trials, 1e-6, 1.0 - 1e-6)
        beta_0_guess = float(np.log(mean_prob / (1.0 - mean_prob)))

        def moments(params: np.ndarray) -> np.ndarray:
            beta_0 = float(params[0])
            mu = self._mu(beta_0)
            mean_y = float((self.n_trials * mu).mean())
            if not np.isfinite(mean_y):
                return np.array([1e6], dtype=float)
            return np.array([mean_y - target_mean], dtype=float)

        result = optimize.least_squares(
            moments,
            x0=np.array([beta_0_guess], dtype=float),
            bounds=([-np.inf], [np.inf]),
        )

        if not result.success:
            raise ValueError(
                "Unable to calibrate BinomialRegressor: nonlinear solve failed; "
                f"message={result.message}; target_mean={target_mean:.6g}, n_trials={self.n_trials}, "
                f"X_shape={self.X.shape}, beta_1_shape={self.beta_1.shape}, residual_norm={np.linalg.norm(result.fun):.6g}"
            )

        beta_0 = result.x[0]
        return BinomialRegressor(
            success_prob=self.success_prob,
            n_trials=self.n_trials,
            min=self.min,
            max=self.max,
            X=self.X,
            beta_1=self.beta_1,
            predictor_names=self.predictor_names,
            predictor_transformations=None,
            beta_0=float(beta_0),
            truncated=self.truncated,
        )

    def _calibrate_truncated(self) -> "BinomialRegressor":
        raise NotImplementedError("BinomialRegressor truncated calibration not yet implemented")

    def sample(self, n: int) -> np.ndarray:
        if self.beta_0 is None:
            raise ValueError("BinomialRegressor must be calibrated before sampling")

        mu = np.repeat(self._mu(self.beta_0), int(np.ceil(n / len(self.X))))[:n]
        lower = stats.binom.cdf(self.min - 1, self.n_trials, mu)
        upper = stats.binom.cdf(self.max, self.n_trials, mu)
        if np.any(lower >= upper):
            raise ValueError("truncation interval has zero probability mass")
        uniforms = np.random.uniform(lower, upper, size=n)
        return _as_1d_array(stats.binom.ppf(uniforms, self.n_trials, mu))


@dataclass(frozen=True, kw_only=True)
class BernoulliRegressor(Bernoulli):
    X: np.ndarray
    beta_1: np.ndarray
    predictor_names: list[str] | None = None
    predictor_transformations: dict[str, str] | None = None
    beta_0: float | None = None

    def __post_init__(self) -> None:
        X = np.asarray(self.X, dtype=float)
        beta_1 = np.asarray(self.beta_1, dtype=float).reshape(-1)
        if X.ndim != 2:
            raise ValueError("X must be a 2D regression matrix")
        if X.shape[1] != beta_1.shape[0]:
            raise ValueError("beta_1 must have one coefficient per column in X")
        predictor_names = self.predictor_names or [f"x{i}" for i in range(X.shape[1])]
        X, predictor_names = _transform_predictors(X, predictor_names, self.predictor_transformations)
        object.__setattr__(self, "X", X)
        object.__setattr__(self, "beta_1", beta_1)
        object.__setattr__(self, "predictor_names", predictor_names)

    def calibrate(self) -> "BernoulliRegressor":
        return self._calibrate_untruncated()

    def _mu(self, beta_0: float) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(beta_0 + self.X @ self.beta_1)))

    def _calibrate_untruncated(self) -> "BernoulliRegressor":
        target_mean = self.success_prob
        if not np.isfinite(target_mean) or target_mean <= 0.0 or target_mean >= 1.0:
            raise ValueError(
                "Unable to calibrate BernoulliRegressor: success_prob must be finite and strictly between 0 and 1; "
                f"success_prob={target_mean:.6g}"
            )

        predictor_linear = self.X @ self.beta_1
        if not np.all(np.isfinite(predictor_linear)):
            raise ValueError(
                "Unable to calibrate BernoulliRegressor: X @ beta_1 must be finite; "
                f"X_shape={self.X.shape}, beta_1_shape={self.beta_1.shape}"
            )

        beta_0_guess = float(np.log(target_mean / max(1.0 - target_mean, np.finfo(float).tiny)))

        def moments(params: np.ndarray) -> np.ndarray:
            beta_0 = float(params[0])
            mu = self._mu(beta_0)
            mean_mu = float(mu.mean())
            if not np.isfinite(mean_mu):
                return np.array([1e6], dtype=float)
            return np.array([mean_mu - target_mean], dtype=float)

        result = optimize.least_squares(
            moments,
            x0=np.array([beta_0_guess], dtype=float),
            bounds=([-np.inf], [np.inf]),
        )

        if not result.success:
            raise ValueError(
                "Unable to calibrate BernoulliRegressor: nonlinear solve failed; "
                f"message={result.message}; success_prob={target_mean:.6g}, "
                f"X_shape={self.X.shape}, beta_1_shape={self.beta_1.shape}, residual_norm={np.linalg.norm(result.fun):.6g}"
            )

        beta_0 = result.x[0]
        return BernoulliRegressor(
            success_prob=self.success_prob,
            X=self.X,
            beta_1=self.beta_1,
            predictor_names=self.predictor_names,
            predictor_transformations=None,
            beta_0=float(beta_0),
        )

    def sample(self, n: int) -> np.ndarray:
        if self.beta_0 is None:
            raise ValueError("BernoulliRegressor must be calibrated before sampling")

        mu = np.repeat(self._mu(self.beta_0), int(np.ceil(n / len(self.X))))[:n]
        return _as_1d_array(stats.bernoulli.rvs(mu, size=n))


@dataclass(frozen=True, kw_only=True)
class PoissonRegressor(Poisson):
    X: np.ndarray
    beta_1: np.ndarray
    predictor_names: list[str] | None = None
    predictor_transformations: dict[str, str] | None = None
    beta_0: float | None = None

    def __post_init__(self) -> None:
        _validate_bounds(self.min, self.max)
        X = np.asarray(self.X, dtype=float)
        beta_1 = np.asarray(self.beta_1, dtype=float).reshape(-1)
        if X.ndim != 2:
            raise ValueError("X must be a 2D regression matrix")
        if X.shape[1] != beta_1.shape[0]:
            raise ValueError("beta_1 must have one coefficient per column in X")
        predictor_names = self.predictor_names or [f"x{i}" for i in range(X.shape[1])]
        X, predictor_names = _transform_predictors(X, predictor_names, self.predictor_transformations)
        object.__setattr__(self, "X", X)
        object.__setattr__(self, "beta_1", beta_1)
        object.__setattr__(self, "predictor_names", predictor_names)

    def calibrate(self) -> "PoissonRegressor":
        return self._calibrate_truncated() if self.truncated else self._calibrate_untruncated()

    def _lambda(self, beta_0: float) -> np.ndarray:
        return np.exp(beta_0 + self.X @ self.beta_1)

    def _calibrate_untruncated(self) -> "PoissonRegressor":
        target_mean = self.get_mean()
        if not np.isfinite(target_mean) or target_mean <= 0.0:
            raise ValueError(
                "Unable to calibrate PoissonRegressor: target_mean must be finite and positive; "
                f"target_mean={target_mean:.6g}"
            )

        predictor_linear = self.X @ self.beta_1
        if not np.all(np.isfinite(predictor_linear)):
            raise ValueError(
                "Unable to calibrate PoissonRegressor: X @ beta_1 must be finite; "
                f"X_shape={self.X.shape}, beta_1_shape={self.beta_1.shape}"
            )

        mean_rate = max(target_mean, np.finfo(float).tiny)
        beta_0_guess = float(np.log(mean_rate))

        def moments(params: np.ndarray) -> np.ndarray:
            beta_0 = float(params[0])
            lam = self._lambda(beta_0)
            mean_lam = float(lam.mean())
            if not np.isfinite(mean_lam):
                return np.array([1e6], dtype=float)
            return np.array([mean_lam - target_mean], dtype=float)

        result = optimize.least_squares(
            moments,
            x0=np.array([beta_0_guess], dtype=float),
            bounds=([-np.inf], [np.inf]),
        )

        if not result.success:
            raise ValueError(
                "Unable to calibrate PoissonRegressor: nonlinear solve failed; "
                f"message={result.message}; target_mean={target_mean:.6g}, "
                f"X_shape={self.X.shape}, beta_1_shape={self.beta_1.shape}, residual_norm={np.linalg.norm(result.fun):.6g}"
            )

        beta_0 = result.x[0]
        return PoissonRegressor(
            min=self.min,
            max=self.max,
            truncated=self.truncated,
            X=self.X,
            beta_1=self.beta_1,
            predictor_names=self.predictor_names,
            predictor_transformations=None,
            beta_0=float(beta_0),
            rate=self.rate,
        )

    def _calibrate_truncated(self) -> "PoissonRegressor":
        raise NotImplementedError("PoissonRegressor truncated calibration not yet implemented")

    def sample(self, n: int) -> np.ndarray:
        if self.beta_0 is None:
            raise ValueError("PoissonRegressor must be calibrated before sampling")

        lam = np.repeat(self._lambda(self.beta_0), int(np.ceil(n / len(self.X))))[:n]
        lower = stats.poisson.cdf(self.min - 1, lam)
        upper = stats.poisson.cdf(self.max, lam)
        if np.any(lower >= upper):
            raise ValueError("truncation interval has zero probability mass")
        uniforms = np.random.uniform(lower, upper, size=n)
        return _as_1d_array(stats.poisson.ppf(uniforms, lam))


@dataclass(frozen=True, kw_only=True)
class NegativeBinomialRegressor(NegativeBinomial):
    X: np.ndarray
    beta_1: np.ndarray
    predictor_names: list[str] | None = None
    predictor_transformations: dict[str, str] | None = None
    beta_0: float | None = None

    def __post_init__(self) -> None:
        _validate_bounds(self.min, self.max)
        X = np.asarray(self.X, dtype=float)
        beta_1 = np.asarray(self.beta_1, dtype=float).reshape(-1)
        if X.ndim != 2:
            raise ValueError("X must be a 2D regression matrix")
        if X.shape[1] != beta_1.shape[0]:
            raise ValueError("beta_1 must have one coefficient per column in X")
        predictor_names = self.predictor_names or [f"x{i}" for i in range(X.shape[1])]
        X, predictor_names = _transform_predictors(X, predictor_names, self.predictor_transformations)
        object.__setattr__(self, "X", X)
        object.__setattr__(self, "beta_1", beta_1)
        object.__setattr__(self, "predictor_names", predictor_names)

    def calibrate(self) -> "NegativeBinomialRegressor":
        return self._calibrate_truncated() if self.truncated else self._calibrate_untruncated()

    def _mu(self, beta_0: float) -> np.ndarray:
        return np.exp(beta_0 + self.X @ self.beta_1)

    def _calibrate_untruncated(self) -> "NegativeBinomialRegressor":
        target_mean = self.get_mean()
        target_variance = self.get_variance()

        if not np.isfinite(target_mean) or target_mean <= 0.0:
            raise ValueError(
                "Unable to calibrate NegativeBinomialRegressor: target_mean must be finite and positive; "
                f"target_mean={target_mean:.6g}"
            )
        if not np.isfinite(target_variance) or target_variance <= target_mean:
            raise ValueError(
                "Unable to calibrate NegativeBinomialRegressor: target_variance must be finite and greater than target_mean; "
                f"target_mean={target_mean:.6g}, target_variance={target_variance:.6g}"
            )

        predictor_linear = self.X @ self.beta_1
        if not np.all(np.isfinite(predictor_linear)):
            raise ValueError(
                "Unable to calibrate NegativeBinomialRegressor: X @ beta_1 must be finite; "
                f"X_shape={self.X.shape}, beta_1_shape={self.beta_1.shape}"
            )

        beta_0_guess = float(np.log(max(target_mean, np.finfo(float).tiny)))
        shape_guess = max(target_mean * target_mean / max(target_variance - target_mean, np.finfo(float).tiny), 1e-6)

        def moments(params: np.ndarray) -> np.ndarray:
            beta_0, log_shape = params
            shape = float(np.exp(log_shape))
            mu = self._mu(beta_0)
            mean_mu = float(mu.mean())
            mean_mu2 = float((mu**2).mean())
            total_var = mean_mu + (1.0 + 1.0 / shape) * mean_mu2 - mean_mu**2
            if not np.all(np.isfinite([mean_mu, total_var])):
                return np.array([1e6, 1e6], dtype=float)
            return np.array(
                [
                    mean_mu - target_mean,
                    total_var - target_variance,
                ],
                dtype=float,
            )

        result = optimize.least_squares(
            moments,
            x0=np.array([beta_0_guess, np.log(shape_guess)], dtype=float),
            bounds=([-np.inf, -np.inf], [np.inf, np.inf]),
        )

        if not result.success:
            raise ValueError(
                "Unable to calibrate NegativeBinomialRegressor: nonlinear solve failed; "
                f"message={result.message}; target_mean={target_mean:.6g}, target_variance={target_variance:.6g}, "
                f"X_shape={self.X.shape}, beta_1_shape={self.beta_1.shape}, residual_norm={np.linalg.norm(result.fun):.6g}"
            )

        beta_0, log_shape = result.x
        return NegativeBinomialRegressor(
            shape=float(np.exp(log_shape)),
            mean=float(np.exp(beta_0)),
            min=self.min,
            max=self.max,
            truncated=self.truncated,
            X=self.X,
            beta_1=self.beta_1,
            predictor_names=self.predictor_names,
            predictor_transformations=None,
            beta_0=float(beta_0),
        )

    def _calibrate_truncated(self) -> "NegativeBinomialRegressor":
        raise NotImplementedError("NegativeBinomialRegressor truncated calibration not yet implemented")

    def sample(self, n: int) -> np.ndarray:
        if self.beta_0 is None:
            raise ValueError("NegativeBinomialRegressor must be calibrated before sampling")

        mu = np.repeat(self._mu(self.beta_0), int(np.ceil(n / len(self.X))))[:n]
        p = np.where(mu > 0, self.shape / (self.shape + mu), 1.0)
        lower = stats.nbinom.cdf(self.min - 1, self.shape, p)
        upper = stats.nbinom.cdf(self.max, self.shape, p)
        if np.any(lower >= upper):
            raise ValueError("truncation interval has zero probability mass")
        uniforms = np.random.uniform(lower, upper, size=n)
        return _as_1d_array(stats.nbinom.ppf(uniforms, self.shape, p))


@dataclass(frozen=True)
class CategoricalNominalRegressor:
    target_probabilities: np.ndarray
    X: np.ndarray
    beta_1: np.ndarray
    predictor_names: list[str] | None = None
    predictor_transformations: dict[str, str] | None = None
    categories: list[str] | None = None
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

        return CategoricalNominalRegressor(
            target_probabilities=self.target_probabilities,
            X=self.X,
            beta_1=self.beta_1,
            predictor_names=self.predictor_names,
            predictor_transformations=None,
            categories=self.categories,
            beta_0=np.asarray(result.x, dtype=float),
        )

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
        return _as_1d_array([self.categories[idx] for idx in samples])


@dataclass(frozen=True)
class CategoricalOrdinalRegressor:
    target_probabilities: np.ndarray
    X: np.ndarray
    beta_1: np.ndarray
    predictor_names: list[str] | None = None
    predictor_transformations: dict[str, str] | None = None
    categories: list[str] | None = None
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
        categories = self.categories or [str(i) for i in range(probabilities.size)]
        if len(categories) != probabilities.size:
            raise ValueError("categories must have one label per response category")
        predictor_names = self.predictor_names or [f"x{i}" for i in range(X.shape[1])]
        X, predictor_names = _transform_predictors(X, predictor_names, self.predictor_transformations)
        object.__setattr__(self, "target_probabilities", probabilities)
        object.__setattr__(self, "X", X)
        object.__setattr__(self, "beta_1", beta_1)
        object.__setattr__(self, "predictor_names", predictor_names)
        object.__setattr__(self, "categories", categories)

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

        return CategoricalOrdinalRegressor(
            target_probabilities=self.target_probabilities,
            X=self.X,
            beta_1=self.beta_1,
            predictor_names=self.predictor_names,
            predictor_transformations=None,
            categories=self.categories,
            beta_0=np.asarray(np.maximum.accumulate(result.x), dtype=float),
        )

    def sample(self, n: int) -> np.ndarray:
        if self.beta_0 is None:
            raise ValueError("CategoricalOrdinalRegressor must be calibrated before sampling")

        probabilities = self._category_probabilities(self.beta_0)
        row_probs = np.repeat(probabilities, int(np.ceil(n / len(probabilities))), axis=0)[:n]
        rng = np.random.default_rng()
        samples = np.empty(n, dtype=int)
        for idx, probs in enumerate(row_probs):
            cumulative = np.cumsum(probs)
            draw = rng.random()
            samples[idx] = int(np.searchsorted(cumulative, draw, side="right"))
        return _as_1d_array([self.categories[idx] for idx in samples])
