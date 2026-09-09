"""Distribution helpers for solving and sampling truncated distributions.

This module is intentionally self-contained. It currently provides a truncated
normal implementation and is structured to host additional distributions later.
It only requires SciPy at runtime for the default solver and sampler
implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import optimize, stats


@dataclass
class TruncatedNormal:
    target_mean: float
    target_std_deviation: float
    min: float
    max: float
    mu: Optional[float] = None
    sigma: Optional[float] = None

    def solve(self) -> tuple[float, float]:
        """Solve for the underlying distribution parameters.

        Returns:
            (mu, sigma) for the untruncated normal whose truncation to
            [min, max] matches the requested population mean and variance as
            closely as the numerical solver can find.
        """
        self._validate_inputs()
        self._validate_feasibility()

        initial_mu = float(self.target_mean)
        initial_sigma = max((self.max - self.min) / 6.0, 1e-3)

        result = optimize.root(
            self._root_equations,
            x0=[initial_mu, math.log(initial_sigma)],
            method="hybr",
        )

        if not result.success:
            raise RuntimeError(
                f"Failed to solve truncated normal parameters: {result.message}"
            )

        self.mu = float(result.x[0])
        self.sigma = float(math.exp(result.x[1]))

        fitted_mean, fitted_variance = self._truncated_moments(self.mu, self.sigma)
        mean_error = abs(fitted_mean - self.target_mean)
        variance_error = abs(fitted_variance - self.target_std_deviation**2)
        tolerance = 1e-8
        if mean_error > tolerance or variance_error > tolerance:
            raise RuntimeError(
                "Solved parameters do not match target moments within tolerance "
                f"(mean_error={mean_error:.3e}, variance_error={variance_error:.3e})"
            )

        return self.mu, self.sigma

    def sample(self, n: int, random_state=None):
        """Sample n values from the fitted truncated normal.

        If solve() has not been called yet, this will call it first.
        """
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer")

        if self.mu is None or self.sigma is None:
            self.solve()

        assert self.mu is not None
        assert self.sigma is not None

        dist = stats.truncnorm(
            (self.min - self.mu) / self.sigma,
            (self.max - self.mu) / self.sigma,
            loc=self.mu,
            scale=self.sigma,
        )
        return dist.rvs(size=n, random_state=random_state)

    def _validate_inputs(self) -> None:
        if not math.isfinite(self.target_mean):
            raise ValueError("target_mean must be finite")
        if not math.isfinite(self.target_std_deviation) or self.target_std_deviation <= 0:
            raise ValueError("target_std_deviation must be a positive finite number")
        if not math.isfinite(self.min) or not math.isfinite(self.max):
            raise ValueError("min and max must be finite")
        if self.min >= self.max:
            raise ValueError("min must be less than max")
        if self.target_mean < self.min or self.target_mean > self.max:
            raise ValueError("target_mean must lie within [min, max]")

    def _validate_feasibility(self) -> None:
        max_possible_variance = (self.max - self.min) ** 2 / 4.0
        if self.target_std_deviation**2 > max_possible_variance:
            raise ValueError(
                "target_std_deviation is not feasible for the given bounds "
                f"(max allowed is {max_possible_variance})"
            )

    def _root_equations(self, params):
        mu = float(params[0])
        sigma = float(math.exp(params[1]))
        moments = self._truncated_moments(mu, sigma)
        if moments is None:
            return [1e6, 1e6]

        fitted_mean, fitted_variance = moments
        return [
            fitted_mean - self.target_mean,
            fitted_variance - self.target_std_deviation**2,
        ]

    def _truncated_moments(self, mu: float, sigma: float):
        alpha = (self.min - mu) / sigma
        beta = (self.max - mu) / sigma
        z = stats.norm.cdf(beta) - stats.norm.cdf(alpha)
        if z <= 0.0:
            return None

        pa = stats.norm.pdf(alpha)
        pb = stats.norm.pdf(beta)
        delta = (pa - pb) / z

        fitted_mean = mu + sigma * delta
        fitted_variance = sigma**2 * (1.0 + (alpha * pa - beta * pb) / z - delta**2)
        return fitted_mean, fitted_variance


@dataclass
class TruncatedGamma:
    target_q25: float
    target_median: float
    target_q75: float
    min: float
    max: float
    shape: Optional[float] = None
    rate: Optional[float] = None

    def solve(self) -> dict:
        """Solve for the truncated gamma shape and rate.

        Returns:
            A dictionary containing the fitted shape, rate, and population
            quantiles at 25%, 50%, and 75%.
        """
        self._validate_inputs()

        initial_shape = 2.0
        initial_rate = max(initial_shape / max(self._center(), 1e-3), 1e-3)

        result = optimize.minimize(
            self._quantile_objective,
            x0=[math.log(initial_shape), math.log(initial_rate)],
            method="Nelder-Mead",
            options={"maxiter": 5000, "xatol": 1e-10, "fatol": 1e-10},
        )

        if not result.success:
            raise RuntimeError(
                f"Failed to solve truncated gamma parameters: {result.message}"
            )

        self.shape = float(math.exp(result.x[0]))
        self.rate = float(math.exp(result.x[1]))

        fitted_q25, fitted_median, fitted_q75 = self._truncated_quantiles(
            self.shape, self.rate
        )

        return {
            "shape": self.shape,
            "rate": self.rate,
            "fitted_q25": float(fitted_q25),
            "fitted_median": float(fitted_median),
            "fitted_q75": float(fitted_q75),
        }

    def sample(self, n: int, random_state=None):
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer")

        if self.shape is None or self.rate is None:
            self.solve()

        assert self.shape is not None
        assert self.rate is not None

        dist = stats.gamma(a=self.shape, scale=1.0 / self.rate)
        low = dist.cdf(self.min)
        high = dist.cdf(self.max)
        u = stats.uniform.rvs(size=n, loc=0.0, scale=1.0, random_state=random_state)
        return dist.ppf(low + u * (high - low))

    def _validate_inputs(self) -> None:
        for name, value in [
            ("target_q25", self.target_q25),
            ("target_median", self.target_median),
            ("target_q75", self.target_q75),
            ("min", self.min),
            ("max", self.max),
        ]:
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")

        if self.min >= self.max:
            raise ValueError("min must be less than max")
        if self.min < 0:
            raise ValueError("min must be >= 0 for a gamma distribution")
        if self.target_q25 > self.target_median or self.target_median > self.target_q75:
            raise ValueError("target quantiles must satisfy q25 <= median <= q75")
        if not (self.min <= self.target_q25 <= self.max):
            raise ValueError("target_q25 must lie within [min, max]")
        if not (self.min <= self.target_median <= self.max):
            raise ValueError("target_median must lie within [min, max]")
        if not (self.min <= self.target_q75 <= self.max):
            raise ValueError("target_q75 must lie within [min, max]")

    def _center(self) -> float:
        return (self.target_q25 + self.target_median + self.target_q75) / 3.0

    def _quantile_objective(self, params):
        shape = float(math.exp(params[0]))
        rate = float(math.exp(params[1]))
        fitted_q25, fitted_median, fitted_q75 = self._truncated_quantiles(shape, rate)
        return (
            (fitted_q25 - self.target_q25) ** 2
            + (fitted_median - self.target_median) ** 2
            + (fitted_q75 - self.target_q75) ** 2
        )

    def _truncated_quantiles(self, shape: float, rate: float):
        dist = stats.gamma(a=shape, scale=1.0 / rate)
        p_low = dist.cdf(self.min)
        p_high = dist.cdf(self.max)
        if p_high <= p_low:
            raise ValueError("invalid gamma truncation interval")

        probs = [0.25, 0.5, 0.75]
        quantiles = [dist.ppf(p_low + p * (p_high - p_low)) for p in probs]
        return quantiles


@dataclass
class TruncatedLogNormal:
    target_q25: float
    target_median: float
    target_q75: float
    min: float
    max: float
    meanlog: Optional[float] = None
    sdlog: Optional[float] = None

    def solve(self) -> dict:
        """Solve for the truncated lognormal meanlog and sdlog.

        Returns:
            A dictionary containing the fitted meanlog, sdlog, and population
            quantiles at 25%, 50%, and 75%.
        """
        self._validate_inputs()

        initial_meanlog = math.log(max(self._center(), 1e-3))
        initial_sdlog = 1.0

        result = optimize.minimize(
            self._quantile_objective,
            x0=[initial_meanlog, math.log(initial_sdlog)],
            method="Nelder-Mead",
            options={"maxiter": 5000, "xatol": 1e-10, "fatol": 1e-10},
        )

        if not result.success:
            raise RuntimeError(
                f"Failed to solve truncated lognormal parameters: {result.message}"
            )

        self.meanlog = float(result.x[0])
        self.sdlog = float(math.exp(result.x[1]))

        fitted_q25, fitted_median, fitted_q75 = self._truncated_quantiles(
            self.meanlog, self.sdlog
        )

        return {
            "meanlog": self.meanlog,
            "sdlog": self.sdlog,
            "fitted_q25": float(fitted_q25),
            "fitted_median": float(fitted_median),
            "fitted_q75": float(fitted_q75),
        }

    def sample(self, n: int, random_state=None):
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer")

        if self.meanlog is None or self.sdlog is None:
            self.solve()

        assert self.meanlog is not None
        assert self.sdlog is not None

        dist = stats.lognorm(s=self.sdlog, scale=math.exp(self.meanlog))
        low = dist.cdf(self.min)
        high = dist.cdf(self.max)
        u = stats.uniform.rvs(size=n, loc=0.0, scale=1.0, random_state=random_state)
        return dist.ppf(low + u * (high - low))

    def _validate_inputs(self) -> None:
        for name, value in [
            ("target_q25", self.target_q25),
            ("target_median", self.target_median),
            ("target_q75", self.target_q75),
            ("min", self.min),
            ("max", self.max),
        ]:
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")

        if self.min >= self.max:
            raise ValueError("min must be less than max")
        if self.max <= 0:
            raise ValueError("max must be > 0 for a lognormal distribution")
        if self.min < 0:
            raise ValueError("min must be >= 0 for a lognormal distribution")
        if self.target_q25 > self.target_median or self.target_median > self.target_q75:
            raise ValueError("target quantiles must satisfy q25 <= median <= q75")
        if self.target_q25 <= 0 or self.target_median <= 0 or self.target_q75 <= 0:
            raise ValueError("target quantiles must be positive for a lognormal distribution")
        if not (self.min <= self.target_q25 <= self.max):
            raise ValueError("target_q25 must lie within [min, max]")
        if not (self.min <= self.target_median <= self.max):
            raise ValueError("target_median must lie within [min, max]")
        if not (self.min <= self.target_q75 <= self.max):
            raise ValueError("target_q75 must lie within [min, max]")

    def _center(self) -> float:
        return (self.target_q25 + self.target_median + self.target_q75) / 3.0

    def _quantile_objective(self, params):
        meanlog = float(params[0])
        sdlog = float(math.exp(params[1]))
        fitted_q25, fitted_median, fitted_q75 = self._truncated_quantiles(meanlog, sdlog)
        return (
            (fitted_q25 - self.target_q25) ** 2
            + (fitted_median - self.target_median) ** 2
            + (fitted_q75 - self.target_q75) ** 2
        )

    def _truncated_quantiles(self, meanlog: float, sdlog: float):
        dist = stats.lognorm(s=sdlog, scale=math.exp(meanlog))
        p_low = dist.cdf(self.min)
        p_high = dist.cdf(self.max)
        if p_high <= p_low:
            raise ValueError("invalid lognormal truncation interval")

        probs = [0.25, 0.5, 0.75]
        quantiles = [dist.ppf(p_low + p * (p_high - p_low)) for p in probs]
        return quantiles


@dataclass
class TruncatedBeta:
    target_q25: float
    target_median: float
    target_q75: float
    min: float
    max: float
    shape1: Optional[float] = None
    shape2: Optional[float] = None

    def solve(self) -> dict:
        """Solve for the truncated beta shape parameters.

        Returns:
            A dictionary containing the fitted shape1, shape2, and population
            quantiles at 25%, 50%, and 75%.
        """
        self._validate_inputs()

        initial_shape1 = 2.0
        initial_shape2 = 2.0

        result = optimize.minimize(
            self._quantile_objective,
            x0=[math.log(initial_shape1), math.log(initial_shape2)],
            method="Nelder-Mead",
            options={"maxiter": 5000, "xatol": 1e-10, "fatol": 1e-10},
        )

        if not result.success:
            raise RuntimeError(
                f"Failed to solve truncated beta parameters: {result.message}"
            )

        self.shape1 = float(math.exp(result.x[0]))
        self.shape2 = float(math.exp(result.x[1]))

        fitted_q25, fitted_median, fitted_q75 = self._truncated_quantiles(
            self.shape1, self.shape2
        )

        return {
            "shape1": self.shape1,
            "shape2": self.shape2,
            "fitted_q25": float(fitted_q25),
            "fitted_median": float(fitted_median),
            "fitted_q75": float(fitted_q75),
        }

    def sample(self, n: int, random_state=None):
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer")

        if self.shape1 is None or self.shape2 is None:
            self.solve()

        assert self.shape1 is not None
        assert self.shape2 is not None

        dist = stats.beta(a=self.shape1, b=self.shape2)
        u = stats.uniform.rvs(size=n, loc=0.0, scale=1.0, random_state=random_state)
        x = dist.ppf(u)
        return self.min + (self.max - self.min) * x

    def _validate_inputs(self) -> None:
        for name, value in [
            ("target_q25", self.target_q25),
            ("target_median", self.target_median),
            ("target_q75", self.target_q75),
            ("min", self.min),
            ("max", self.max),
        ]:
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")

        if self.min >= self.max:
            raise ValueError("min must be less than max")
        if self.target_q25 > self.target_median or self.target_median > self.target_q75:
            raise ValueError("target quantiles must satisfy q25 <= median <= q75")
        if not (self.min <= self.target_q25 <= self.max):
            raise ValueError("target_q25 must lie within [min, max]")
        if not (self.min <= self.target_median <= self.max):
            raise ValueError("target_median must lie within [min, max]")
        if not (self.min <= self.target_q75 <= self.max):
            raise ValueError("target_q75 must lie within [min, max]")

    def _quantile_objective(self, params):
        shape1 = float(math.exp(params[0]))
        shape2 = float(math.exp(params[1]))
        fitted_q25, fitted_median, fitted_q75 = self._truncated_quantiles(shape1, shape2)
        return (
            (fitted_q25 - self.target_q25) ** 2
            + (fitted_median - self.target_median) ** 2
            + (fitted_q75 - self.target_q75) ** 2
        )

    def _truncated_quantiles(self, shape1: float, shape2: float):
        dist = stats.beta(a=shape1, b=shape2)
        probs = [0.25, 0.5, 0.75]
        quantiles = [self.min + (self.max - self.min) * dist.ppf(p) for p in probs]
        return quantiles


@dataclass
class Uniform:
    min: float
    max: float

    def sample(self, n: int, random_state=None):
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer")
        if not math.isfinite(self.min) or not math.isfinite(self.max):
            raise ValueError("min and max must be finite")
        if self.min >= self.max:
            raise ValueError("min must be less than max")

        return stats.uniform.rvs(
            loc=self.min,
            scale=self.max - self.min,
            size=n,
            random_state=random_state,
        )


@dataclass
class Bernoulli:
    prob: float

    def sample(self, n: int, random_state=None):
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer")
        if not math.isfinite(self.prob):
            raise ValueError("prob must be finite")
        if self.prob < 0 or self.prob > 1:
            raise ValueError("prob must be between 0 and 1")

        return stats.bernoulli.rvs(p=self.prob, size=n, random_state=random_state)


@dataclass
class DiscreteUniform:
    min: int
    max: int

    def sample(self, n: int, random_state=None):
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer")
        if not isinstance(self.min, int) or not isinstance(self.max, int):
            raise ValueError("min and max must be integers")
        if self.min > self.max:
            raise ValueError("min must be less than or equal to max")

        return stats.randint.rvs(
            low=self.min,
            high=self.max + 1,
            size=n,
            random_state=random_state,
        )


@dataclass
class CategoricalNominal:
    categories: list
    probabilities: list[float]

    def sample(self, n: int, random_state=None):
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer")
        if not isinstance(self.categories, list) or len(self.categories) == 0:
            raise ValueError("categories must be a non-empty list")
        if not isinstance(self.probabilities, list):
            raise ValueError("probabilities must be a list")
        if len(self.categories) != len(self.probabilities):
            raise ValueError("categories and probabilities must have the same length")

        probs = [float(p) for p in self.probabilities]
        if any((not math.isfinite(p)) or p < 0 for p in probs):
            raise ValueError("probabilities must be finite and non-negative")
        if abs(sum(probs) - 1.0) > 1e-8:
            raise ValueError("probabilities must sum to 1")

        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(self.categories), size=n, p=probs)
        return [self.categories[i] for i in idx]


@dataclass
class CategoricalOrdinal:
    categories: list
    probabilities: list[float]

    def sample(self, n: int, random_state=None):
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer")
        if not isinstance(self.categories, list) or len(self.categories) == 0:
            raise ValueError("categories must be a non-empty list")
        if not isinstance(self.probabilities, list):
            raise ValueError("probabilities must be a list")
        if len(self.categories) != len(self.probabilities):
            raise ValueError("categories and probabilities must have the same length")

        probs = [float(p) for p in self.probabilities]
        if any((not math.isfinite(p)) or p < 0 for p in probs):
            raise ValueError("probabilities must be finite and non-negative")
        if abs(sum(probs) - 1.0) > 1e-8:
            raise ValueError("probabilities must sum to 1")

        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(self.categories), size=n, p=probs)
        return [self.categories[i] for i in idx]


@dataclass
class TruncatedBinomial:
    target_q25: float
    target_median: float
    target_q75: float
    n_trials: int
    min: float
    max: float
    prob: Optional[float] = None

    def solve(self) -> dict:
        """Solve for the truncated binomial probability.

        Returns:
            A dictionary containing the fitted probability and population
            quantiles at 25%, 50%, and 75%.
        """
        self._validate_inputs()

        initial_prob = 0.5

        result = optimize.minimize_scalar(
            self._quantile_objective,
            bounds=(1e-12, 1.0 - 1e-12),
            method="bounded",
            options={"xatol": 1e-12},
        )

        if not result.success:
            raise RuntimeError(
                f"Failed to solve truncated binomial parameters: {result.message}"
            )

        self.prob = float(result.x)

        fitted_q25, fitted_median, fitted_q75 = self._truncated_quantiles(self.prob)

        return {
            "prob": self.prob,
            "fitted_q25": float(fitted_q25),
            "fitted_median": float(fitted_median),
            "fitted_q75": float(fitted_q75),
        }

    def sample(self, n: int, random_state=None):
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer")

        if self.prob is None:
            self.solve()

        assert self.prob is not None

        dist = stats.binom(self.n_trials, self.prob)
        valid_min = int(math.ceil(max(0, self.min)))
        valid_max = int(math.floor(min(self.n_trials, self.max)))
        if valid_min > valid_max:
            raise ValueError("No support in the interval [min, max] for this binomial")

        p_low = dist.cdf(valid_min - 1)
        p_high = dist.cdf(valid_max)
        u = stats.uniform.rvs(size=n, loc=0.0, scale=1.0, random_state=random_state)
        return dist.ppf(p_low + u * (p_high - p_low))

    def _validate_inputs(self) -> None:
        for name, value in [
            ("target_q25", self.target_q25),
            ("target_median", self.target_median),
            ("target_q75", self.target_q75),
            ("min", self.min),
            ("max", self.max),
        ]:
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")

        if not isinstance(self.n_trials, int) or self.n_trials < 2:
            raise ValueError("n_trials must be an integer >= 2")
        if self.min >= self.max:
            raise ValueError("min must be less than max")
        if self.min < 0:
            raise ValueError("min must be >= 0 for a binomial distribution")
        if self.max > self.n_trials:
            raise ValueError("max must be <= n_trials for a binomial distribution")
        if self.target_q25 > self.target_median or self.target_median > self.target_q75:
            raise ValueError("target quantiles must satisfy q25 <= median <= q75")
        if not (self.min <= self.target_q25 <= self.max):
            raise ValueError("target_q25 must lie within [min, max]")
        if not (self.min <= self.target_median <= self.max):
            raise ValueError("target_median must lie within [min, max]")
        if not (self.min <= self.target_q75 <= self.max):
            raise ValueError("target_q75 must lie within [min, max]")
        if any(float(v) != math.floor(float(v)) for v in [self.target_q25, self.target_median, self.target_q75]):
            raise ValueError("target quantiles must be integers for a binomial distribution")

    def _quantile_objective(self, prob: float):
        fitted_q25, fitted_median, fitted_q75 = self._truncated_quantiles(prob)
        return (
            (fitted_q25 - self.target_q25) ** 2
            + (fitted_median - self.target_median) ** 2
            + (fitted_q75 - self.target_q75) ** 2
        )

    def _truncated_quantiles(self, prob: float):
        dist = stats.binom(self.n_trials, prob)
        valid_min = int(math.ceil(max(0, self.min)))
        valid_max = int(math.floor(min(self.n_trials, self.max)))
        if valid_min > valid_max:
            raise ValueError("No support in the interval [min, max] for this binomial")

        p_low = dist.cdf(valid_min - 1)
        p_high = dist.cdf(valid_max)
        probs = [0.25, 0.5, 0.75]
        return [dist.ppf(p_low + p * (p_high - p_low)) for p in probs]


@dataclass
class TruncatedPoisson:
    target_q25: float
    target_median: float
    target_q75: float
    min: float
    max: float
    lambda_: Optional[float] = None

    def solve(self) -> dict:
        """Solve for the truncated Poisson rate parameter.

        Returns:
            A dictionary containing the fitted lambda and population quantiles
            at 25%, 50%, and 75%.
        """
        self._validate_inputs()

        result = optimize.minimize_scalar(
            self._quantile_objective,
            bounds=(1e-12, max(self._center() * 4.0, 1e-6)),
            method="bounded",
            options={"xatol": 1e-12},
        )

        if not result.success:
            raise RuntimeError(
                f"Failed to solve truncated poisson parameters: {result.message}"
            )

        self.lambda_ = float(result.x)

        fitted_q25, fitted_median, fitted_q75 = self._truncated_quantiles(self.lambda_)

        return {
            "lambda": self.lambda_,
            "fitted_q25": float(fitted_q25),
            "fitted_median": float(fitted_median),
            "fitted_q75": float(fitted_q75),
        }

    def sample(self, n: int, random_state=None):
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer")

        if self.lambda_ is None:
            self.solve()

        assert self.lambda_ is not None

        dist = stats.poisson(mu=self.lambda_)
        valid_min = int(math.ceil(max(0, self.min)))
        valid_max = int(math.floor(self.max))
        if valid_min > valid_max:
            raise ValueError("No support in the interval [min, max] for this poisson")

        p_low = dist.cdf(valid_min - 1)
        p_high = dist.cdf(valid_max)
        u = stats.uniform.rvs(size=n, loc=0.0, scale=1.0, random_state=random_state)
        return dist.ppf(p_low + u * (p_high - p_low))

    def _validate_inputs(self) -> None:
        for name, value in [
            ("target_q25", self.target_q25),
            ("target_median", self.target_median),
            ("target_q75", self.target_q75),
            ("min", self.min),
            ("max", self.max),
        ]:
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")

        if self.min >= self.max:
            raise ValueError("min must be less than max")
        if self.min < 0:
            raise ValueError("min must be >= 0 for a poisson distribution")
        if not (self.min <= self.target_q25 <= self.max):
            raise ValueError("target_q25 must lie within [min, max]")
        if not (self.min <= self.target_median <= self.max):
            raise ValueError("target_median must lie within [min, max]")
        if not (self.min <= self.target_q75 <= self.max):
            raise ValueError("target_q75 must lie within [min, max]")
        if self.target_q25 > self.target_median or self.target_median > self.target_q75:
            raise ValueError("target quantiles must satisfy q25 <= median <= q75")
        if any(float(v) != math.floor(float(v)) for v in [self.target_q25, self.target_median, self.target_q75]):
            raise ValueError("target quantiles must be integers for a poisson distribution")

    def _center(self) -> float:
        return (self.target_q25 + self.target_median + self.target_q75) / 3.0

    def _quantile_objective(self, lambda_: float):
        fitted_q25, fitted_median, fitted_q75 = self._truncated_quantiles(lambda_)
        return (
            (fitted_q25 - self.target_q25) ** 2
            + (fitted_median - self.target_median) ** 2
            + (fitted_q75 - self.target_q75) ** 2
        )

    def _truncated_quantiles(self, lambda_: float):
        dist = stats.poisson(mu=lambda_)
        valid_min = int(math.ceil(max(0, self.min)))
        valid_max = int(math.floor(self.max))
        if valid_min > valid_max:
            raise ValueError("No support in the interval [min, max] for this poisson")

        p_low = dist.cdf(valid_min - 1)
        p_high = dist.cdf(valid_max)
        probs = [0.25, 0.5, 0.75]
        return [dist.ppf(p_low + p * (p_high - p_low)) for p in probs]


@dataclass
class TruncatedNegativeBinomial:
    target_q25: float
    target_median: float
    target_q75: float
    min: float
    max: float
    size: Optional[float] = None
    mu: Optional[float] = None

    def solve(self) -> dict:
        """Solve for the truncated negative binomial size and mu.

        Returns:
            A dictionary containing the fitted size, mu, and population
            quantiles at 25%, 50%, and 75%.
        """
        self._validate_inputs()

        result = optimize.minimize(
            self._quantile_objective,
            x0=[math.log(2.0), math.log(max(self._center(), 1e-3))],
            method="Nelder-Mead",
            options={"maxiter": 5000, "xatol": 1e-10, "fatol": 1e-10},
        )

        if not result.success:
            raise RuntimeError(
                f"Failed to solve truncated negative binomial parameters: {result.message}"
            )

        self.size = float(math.exp(result.x[0]))
        self.mu = float(math.exp(result.x[1]))

        fitted_q25, fitted_median, fitted_q75 = self._truncated_quantiles(
            self.size, self.mu
        )

        return {
            "size": self.size,
            "mu": self.mu,
            "fitted_q25": float(fitted_q25),
            "fitted_median": float(fitted_median),
            "fitted_q75": float(fitted_q75),
        }

    def sample(self, n: int, random_state=None):
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer")

        if self.size is None or self.mu is None:
            self.solve()

        assert self.size is not None
        assert self.mu is not None

        dist = stats.nbinom(n=self.size, p=self.size / (self.size + self.mu))
        valid_min = int(math.ceil(max(0, self.min)))
        valid_max = int(math.floor(self.max))
        if valid_min > valid_max:
            raise ValueError("No support in the interval [min, max] for this negative binomial")

        p_low = dist.cdf(valid_min - 1)
        p_high = dist.cdf(valid_max)
        u = stats.uniform.rvs(size=n, loc=0.0, scale=1.0, random_state=random_state)
        return dist.ppf(p_low + u * (p_high - p_low))

    def _validate_inputs(self) -> None:
        for name, value in [
            ("target_q25", self.target_q25),
            ("target_median", self.target_median),
            ("target_q75", self.target_q75),
            ("min", self.min),
            ("max", self.max),
        ]:
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")

        if self.min >= self.max:
            raise ValueError("min must be less than max")
        if self.min < 0:
            raise ValueError("min must be >= 0 for a negative binomial distribution")
        if self.target_q25 > self.target_median or self.target_median > self.target_q75:
            raise ValueError("target quantiles must satisfy q25 <= median <= q75")
        if not (self.min <= self.target_q25 <= self.max):
            raise ValueError("target_q25 must lie within [min, max]")
        if not (self.min <= self.target_median <= self.max):
            raise ValueError("target_median must lie within [min, max]")
        if not (self.min <= self.target_q75 <= self.max):
            raise ValueError("target_q75 must lie within [min, max]")
        if any(float(v) != math.floor(float(v)) for v in [self.target_q25, self.target_median, self.target_q75]):
            raise ValueError("target quantiles must be integers for a negative binomial distribution")

    def _center(self) -> float:
        return (self.target_q25 + self.target_median + self.target_q75) / 3.0

    def _quantile_objective(self, params):
        size = float(math.exp(params[0]))
        mu = float(math.exp(params[1]))
        fitted_q25, fitted_median, fitted_q75 = self._truncated_quantiles(size, mu)
        return (
            (fitted_q25 - self.target_q25) ** 2
            + (fitted_median - self.target_median) ** 2
            + (fitted_q75 - self.target_q75) ** 2
        )

    def _truncated_quantiles(self, size: float, mu: float):
        dist = stats.nbinom(n=size, p=size / (size + mu))
        valid_min = int(math.ceil(max(0, self.min)))
        valid_max = int(math.floor(self.max))
        if valid_min > valid_max:
            raise ValueError("No support in the interval [min, max] for this negative binomial")

        p_low = dist.cdf(valid_min - 1)
        p_high = dist.cdf(valid_max)
        probs = [0.25, 0.5, 0.75]
        return [dist.ppf(p_low + p * (p_high - p_low)) for p in probs]
