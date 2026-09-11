from dataclasses import dataclass

import numpy as np
from scipy import stats


def _as_1d_array(values) -> np.ndarray:
    return np.asarray(values).reshape(-1)


def _validate_bounds(min_value, max_value) -> None:
    if min_value > max_value:
        raise ValueError("min must be less than or equal to max")


@dataclass(frozen=True)
class Normal:
    mean: float
    standard_deviation: float
    min: float
    max: float

    def _distribution(self):
        if self.standard_deviation <= 0:
            raise ValueError("standard_deviation must be positive")
        _validate_bounds(self.min, self.max)
        a = (self.min - self.mean) / self.standard_deviation
        b = (self.max - self.mean) / self.standard_deviation
        return stats.truncnorm(a, b, loc=self.mean, scale=self.standard_deviation)

    def target_mean(self) -> float:
        return float(self._distribution().mean())

    def target_variance(self) -> float:
        return float(self._distribution().var())

    def sample(self, n: int) -> np.ndarray:
        return _as_1d_array(self._distribution().rvs(size=n))


@dataclass(frozen=True)
class Exponential:
    rate: float
    min: float
    max: float

    def _distribution(self):
        if self.rate <= 0:
            raise ValueError("rate must be positive")
        _validate_bounds(self.min, self.max)
        return stats.truncexpon(b=(self.max - self.min) * self.rate, loc=self.min, scale=1.0 / self.rate)

    def target_mean(self) -> float:
        return float(self._distribution().mean())

    def sample(self, n: int) -> np.ndarray:
        return _as_1d_array(self._distribution().rvs(size=n))


@dataclass(frozen=True)
class Gamma:
    shape: float
    rate: float
    min: float
    max: float

    def sample(self, n: int) -> np.ndarray:
        if self.shape <= 0:
            raise ValueError("shape must be positive")
        if self.rate <= 0:
            raise ValueError("rate must be positive")
        _validate_bounds(self.min, self.max)

        dist = stats.gamma(a=self.shape, scale=1.0 / self.rate)
        lower = dist.cdf(self.min)
        upper = dist.cdf(self.max)
        if lower >= upper:
            raise ValueError("truncation interval has zero probability mass")

        uniforms = np.random.uniform(lower, upper, size=n)
        return _as_1d_array(dist.ppf(uniforms))

    def target_mean(self) -> float:
        if self.shape <= 0:
            raise ValueError("shape must be positive")
        if self.rate <= 0:
            raise ValueError("rate must be positive")
        _validate_bounds(self.min, self.max)

        dist = stats.gamma(a=self.shape, scale=1.0 / self.rate)
        lower = dist.cdf(self.min)
        upper = dist.cdf(self.max)
        if lower >= upper:
            raise ValueError("truncation interval has zero probability mass")
        return float(dist.expect(lambda x: x, lb=self.min, ub=self.max, conditional=True))

    def target_variance(self) -> float:
        if self.shape <= 0:
            raise ValueError("shape must be positive")
        if self.rate <= 0:
            raise ValueError("rate must be positive")
        _validate_bounds(self.min, self.max)

        dist = stats.gamma(a=self.shape, scale=1.0 / self.rate)
        lower = dist.cdf(self.min)
        upper = dist.cdf(self.max)
        if lower >= upper:
            raise ValueError("truncation interval has zero probability mass")
        return float(dist.expect(lambda x: x * x, lb=self.min, ub=self.max, conditional=True) - dist.expect(lambda x: x, lb=self.min, ub=self.max, conditional=True) ** 2)


@dataclass(frozen=True)
class LogNormal:
    log_mean: float
    log_standard_deviation: float
    min: float
    max: float

    def sample(self, n: int) -> np.ndarray:
        if self.log_standard_deviation <= 0:
            raise ValueError("log_standard_deviation must be positive")
        _validate_bounds(self.min, self.max)

        dist = stats.lognorm(s=self.log_standard_deviation, scale=np.exp(self.log_mean))
        lower = dist.cdf(self.min)
        upper = dist.cdf(self.max)
        if lower >= upper:
            raise ValueError("truncation interval has zero probability mass")

        uniforms = np.random.uniform(lower, upper, size=n)
        return _as_1d_array(dist.ppf(uniforms))


@dataclass(frozen=True)
class Beta:
    shape_1: float
    shape_2: float
    min: float
    max: float

    def sample(self, n: int) -> np.ndarray:
        if self.shape_1 <= 0 or self.shape_2 <= 0:
            raise ValueError("shape parameters must be positive")
        _validate_bounds(self.min, self.max)

        samples = stats.beta.rvs(self.shape_1, self.shape_2, size=n)
        return _as_1d_array(self.min + (self.max - self.min) * samples)


@dataclass(frozen=True)
class Uniform:
    min: float
    max: float

    def sample(self, n: int) -> np.ndarray:
        _validate_bounds(self.min, self.max)
        return _as_1d_array(stats.uniform.rvs(loc=self.min, scale=self.max - self.min, size=n))


@dataclass(frozen=True)
class DiscreteUniform:
    min: int
    max: int

    def sample(self, n: int) -> np.ndarray:
        _validate_bounds(self.min, self.max)
        return _as_1d_array(stats.randint.rvs(low=self.min, high=self.max + 1, size=n))


@dataclass(frozen=True)
class Bernoulli:
    success_prob: float

    def sample(self, n: int) -> np.ndarray:
        if not 0.0 <= self.success_prob <= 1.0:
            raise ValueError("success_prob must be in [0, 1]")
        return _as_1d_array(stats.bernoulli.rvs(self.success_prob, size=n))


@dataclass(frozen=True)
class Binomial:
    n_trials: int
    success_prob: float
    min: int
    max: int

    def sample(self, n: int) -> np.ndarray:
        if self.n_trials < 1:
            raise ValueError("n_trials must be >= 1")
        if not 0.0 <= self.success_prob <= 1.0:
            raise ValueError("success_prob must be in [0, 1]")
        _validate_bounds(self.min, self.max)

        dist = stats.binom(self.n_trials, self.success_prob)
        lower = dist.cdf(self.min - 1)
        upper = dist.cdf(self.max)
        if lower >= upper:
            raise ValueError("truncation interval has zero probability mass")

        uniforms = np.random.uniform(lower, upper, size=n)
        return _as_1d_array(dist.ppf(uniforms))


@dataclass(frozen=True)
class Poisson:
    rate: float
    min: int
    max: int

    def sample(self, n: int) -> np.ndarray:
        if self.rate <= 0:
            raise ValueError("rate must be positive")
        _validate_bounds(self.min, self.max)

        dist = stats.poisson(self.rate)
        lower = dist.cdf(self.min - 1)
        upper = dist.cdf(self.max)
        if lower >= upper:
            raise ValueError("truncation interval has zero probability mass")

        uniforms = np.random.uniform(lower, upper, size=n)
        return _as_1d_array(dist.ppf(uniforms))


@dataclass(frozen=True)
class Geometric:
    success_prob: float
    min: int
    max: int

    def sample(self, n: int) -> np.ndarray:
        if not 0.0 < self.success_prob <= 1.0:
            raise ValueError("success_prob must be in (0, 1]")
        _validate_bounds(self.min, self.max)

        dist = stats.geom(self.success_prob)
        lower = dist.cdf(self.min - 1)
        upper = dist.cdf(self.max)
        if lower >= upper:
            raise ValueError("truncation interval has zero probability mass")

        uniforms = np.random.uniform(lower, upper, size=n)
        return _as_1d_array(dist.ppf(uniforms))


@dataclass(frozen=True)
class NegativeBinomial:
    shape: float
    mean: float
    min: int
    max: int

    def sample(self, n: int) -> np.ndarray:
        if self.shape <= 0:
            raise ValueError("shape must be positive")
        if self.mean < 0:
            raise ValueError("mean must be non-negative")
        _validate_bounds(self.min, self.max)

        p = self.shape / (self.shape + self.mean) if self.mean > 0 else 1.0
        dist = stats.nbinom(self.shape, p)
        lower = dist.cdf(self.min - 1)
        upper = dist.cdf(self.max)
        if lower >= upper:
            raise ValueError("truncation interval has zero probability mass")

        uniforms = np.random.uniform(lower, upper, size=n)
        return _as_1d_array(dist.ppf(uniforms))


@dataclass(frozen=True)
class CategoricalNominal:
    categories: list[str]
    probabilities: list[float]

    def sample(self, n: int) -> np.ndarray:
        if len(self.categories) == 0:
            raise ValueError("categories must not be empty")
        if len(self.categories) != len(self.probabilities):
            raise ValueError("categories and probabilities must have the same length")
        probabilities = np.asarray(self.probabilities, dtype=float)
        if np.any(probabilities < 0):
            raise ValueError("probabilities must be non-negative")
        total = probabilities.sum()
        if total <= 0:
            raise ValueError("probabilities must sum to a positive value")
        probabilities = probabilities / total
        return _as_1d_array(np.random.choice(self.categories, size=n, p=probabilities))


@dataclass(frozen=True)
class CategoricalOrdinal(CategoricalNominal):
    pass


@dataclass(frozen=True)
class NoneDistribution:
    def sample(self, n: int) -> np.ndarray:
        return _as_1d_array(np.array([None] * n, dtype=object))
