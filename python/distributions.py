from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.stats import norm
from scipy.special import gammainc, gamma as gamma_function


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
    truncated: bool = False
    truncation_tolerance: float = 0.05

    def _get_untruncated_mean(self) -> float:
        if self.standard_deviation <= 0:
            raise ValueError("standard_deviation must be positive")
        return float(self.mean)

    def _get_untruncated_variance(self) -> float:
        if self.standard_deviation <= 0:
            raise ValueError("standard_deviation must be positive")
        return float(self.standard_deviation**2)

    def _get_truncated_mean(self) -> float:
        if self.standard_deviation <= 0:
            raise ValueError("standard_deviation must be positive")
        _validate_bounds(self.min, self.max)
        a = (self.min - self.mean) / self.standard_deviation
        b = (self.max - self.mean) / self.standard_deviation
        return float(stats.truncnorm(a, b, loc=self.mean, scale=self.standard_deviation).mean())

    def _get_truncated_variance(self) -> float:
        if self.standard_deviation <= 0:
            raise ValueError("standard_deviation must be positive")
        _validate_bounds(self.min, self.max)
        a = (self.min - self.mean) / self.standard_deviation
        b = (self.max - self.mean) / self.standard_deviation
        return float(stats.truncnorm(a, b, loc=self.mean, scale=self.standard_deviation).var())

    def _validate_truncation(self) -> None:
        if self.standard_deviation <= 0:
            raise ValueError("standard_deviation must be positive")
        _validate_bounds(self.min, self.max)

        untruncated_mean = self._get_untruncated_mean()
        untruncated_variance = self._get_untruncated_variance()
        truncated_mean = self._get_truncated_mean()
        truncated_variance = self._get_truncated_variance()

        eps = np.finfo(float).tiny
        mean_shift = abs(truncated_mean - untruncated_mean) / max(abs(untruncated_mean), eps)
        variance_shift = abs(truncated_variance - untruncated_variance) / max(abs(untruncated_variance), eps)
        if mean_shift > self.truncation_tolerance or variance_shift > self.truncation_tolerance:
            raise ValueError(
                "truncation too severe; adjust parameters or min/max to avoid serious truncation; "
                f"mean shift={mean_shift:.6f}, variance shift={variance_shift:.6f}, "
                f"tolerance={self.truncation_tolerance:.6f}, "
                f"min={self.min}, max={self.max}"
            )

    def get_mean(self) -> float:
        self._validate_truncation()
        return self._get_truncated_mean() if self.truncated else self._get_untruncated_mean()

    def get_variance(self) -> float:
        self._validate_truncation()
        return self._get_truncated_variance() if self.truncated else self._get_untruncated_variance()

    def sample(self, n: int) -> np.ndarray:
        if self.standard_deviation <= 0:
            raise ValueError("standard_deviation must be positive")
        _validate_bounds(self.min, self.max)
        self._validate_truncation()
        a = (self.min - self.mean) / self.standard_deviation
        b = (self.max - self.mean) / self.standard_deviation
        return _as_1d_array(stats.truncnorm(a, b, loc=self.mean, scale=self.standard_deviation).rvs(size=n))


@dataclass(frozen=True)
class Gamma:
    shape: float
    rate: float
    min: float
    max: float
    truncated: bool = False
    truncation_tolerance: float = 0.05

    def _truncated_raw_moment(self, k: int) -> float:
        if self.shape <= 0:
            raise ValueError("shape must be positive")
        if self.rate <= 0:
            raise ValueError("rate must be positive")
        _validate_bounds(self.min, self.max)

        lower = gammainc(self.shape, self.rate * self.min)
        upper = gammainc(self.shape, self.rate * self.max)
        mass = upper - lower
        if mass <= 0:
            raise ValueError("truncation interval has zero probability mass")

        numerator = gammainc(self.shape + k, self.rate * self.max) - gammainc(self.shape + k, self.rate * self.min)
        return float((self.rate ** (-k)) * (gamma_function(self.shape + k) / gamma_function(self.shape)) * (numerator / mass))

    def _get_untruncated_mean(self) -> float:
        if self.shape <= 0:
            raise ValueError("shape must be positive")
        if self.rate <= 0:
            raise ValueError("rate must be positive")
        return float(self.shape / self.rate)

    def _get_untruncated_variance(self) -> float:
        if self.shape <= 0:
            raise ValueError("shape must be positive")
        if self.rate <= 0:
            raise ValueError("rate must be positive")
        return float(self.shape / (self.rate**2))

    def _get_truncated_mean(self) -> float:
        return self._truncated_raw_moment(1)

    def _get_truncated_variance(self) -> float:
        mean = self._truncated_raw_moment(1)
        second_moment = self._truncated_raw_moment(2)
        return float(second_moment - mean**2)

    def _validate_truncation(self) -> None:
        if self.shape <= 0:
            raise ValueError("shape must be positive")
        if self.rate <= 0:
            raise ValueError("rate must be positive")
        _validate_bounds(self.min, self.max)
        untruncated_mean = self._get_untruncated_mean()
        untruncated_variance = self._get_untruncated_variance()
        truncated_mean = self._get_truncated_mean()
        truncated_variance = self._get_truncated_variance()
        eps = np.finfo(float).tiny
        mean_shift = abs(truncated_mean - untruncated_mean) / max(abs(untruncated_mean), eps)
        variance_shift = abs(truncated_variance - untruncated_variance) / max(abs(untruncated_variance), eps)
        if mean_shift > self.truncation_tolerance or variance_shift > self.truncation_tolerance:
            raise ValueError(
                "truncation too severe; adjust parameters or min/max to avoid serious truncation; "
                f"mean shift={mean_shift:.6f}, variance shift={variance_shift:.6f}, "
                f"tolerance={self.truncation_tolerance:.6f}, "
                f"min={self.min}, max={self.max}"
            )

    def get_mean(self) -> float:
        self._validate_truncation()
        return self._get_truncated_mean() if self.truncated else self._get_untruncated_mean()

    def get_variance(self) -> float:
        self._validate_truncation()
        return self._get_truncated_variance() if self.truncated else self._get_untruncated_variance()

    def sample(self, n: int) -> np.ndarray:
        if self.shape <= 0:
            raise ValueError("shape must be positive")
        if self.rate <= 0:
            raise ValueError("rate must be positive")
        _validate_bounds(self.min, self.max)
        self._validate_truncation()
        dist = stats.gamma(a=self.shape, scale=1.0 / self.rate)
        lower = dist.cdf(self.min)
        upper = dist.cdf(self.max)
        if lower >= upper:
            raise ValueError("truncation interval has zero probability mass")
        uniforms = np.random.uniform(lower, upper, size=n)
        return _as_1d_array(dist.ppf(uniforms))

    
@dataclass(frozen=True)
class LogNormal:
    log_mean: float
    log_standard_deviation: float
    min: float
    max: float
    truncated: bool = False
    truncation_tolerance: float = 0.05

    def _truncated_raw_moment(self, r: int) -> float:
        if self.log_standard_deviation <= 0:
            raise ValueError("log_standard_deviation must be positive")
        _validate_bounds(self.min, self.max)

        sigma = float(self.log_standard_deviation)
        mu = float(self.log_mean)
        lower = norm.cdf((np.log(self.min) - mu) / sigma) if self.min > 0 else 0.0
        upper = norm.cdf((np.log(self.max) - mu) / sigma) if np.isfinite(self.max) else 1.0
        mass = upper - lower
        if mass <= 0:
            raise ValueError("truncation interval has zero probability mass")

        shifted_lower = norm.cdf((np.log(self.min) - mu - r * sigma * sigma) / sigma) if self.min > 0 else 0.0
        shifted_upper = norm.cdf((np.log(self.max) - mu - r * sigma * sigma) / sigma) if np.isfinite(self.max) else 1.0
        numerator = shifted_upper - shifted_lower
        return float(np.exp(r * mu + 0.5 * (r**2) * sigma * sigma) * (numerator / mass))

    def _get_untruncated_mean(self) -> float:
        if self.log_standard_deviation <= 0:
            raise ValueError("log_standard_deviation must be positive")
        return float(np.exp(self.log_mean + 0.5 * self.log_standard_deviation**2))

    def _get_untruncated_variance(self) -> float:
        if self.log_standard_deviation <= 0:
            raise ValueError("log_standard_deviation must be positive")
        mean = np.exp(self.log_mean + 0.5 * self.log_standard_deviation**2)
        return float((np.exp(self.log_standard_deviation**2) - 1.0) * mean**2)

    def _get_truncated_mean(self) -> float:
        return self._truncated_raw_moment(1)

    def _get_truncated_variance(self) -> float:
        mean = self._truncated_raw_moment(1)
        second_moment = self._truncated_raw_moment(2)
        return float(second_moment - mean**2)

    def _validate_truncation(self) -> None:
        if self.log_standard_deviation <= 0:
            raise ValueError("log_standard_deviation must be positive")
        _validate_bounds(self.min, self.max)
        untruncated_mean = self._get_untruncated_mean()
        untruncated_variance = self._get_untruncated_variance()
        truncated_mean = self._get_truncated_mean()
        truncated_variance = self._get_truncated_variance()
        eps = np.finfo(float).tiny
        mean_shift = abs(truncated_mean - untruncated_mean) / max(abs(untruncated_mean), eps)
        variance_shift = abs(truncated_variance - untruncated_variance) / max(abs(untruncated_variance), eps)
        if mean_shift > self.truncation_tolerance or variance_shift > self.truncation_tolerance:
            raise ValueError(
                "truncation too severe; adjust parameters or min/max to avoid serious truncation; "
                f"mean shift={mean_shift:.6f}, variance shift={variance_shift:.6f}, "
                f"tolerance={self.truncation_tolerance:.6f}, "
                f"min={self.min}, max={self.max}"
            )

    def get_mean(self) -> float:
        self._validate_truncation()
        return self._get_truncated_mean() if self.truncated else self._get_untruncated_mean()

    def get_variance(self) -> float:
        self._validate_truncation()
        return self._get_truncated_variance() if self.truncated else self._get_untruncated_variance()

    def sample(self, n: int) -> np.ndarray:
        if self.log_standard_deviation <= 0:
            raise ValueError("log_standard_deviation must be positive")
        _validate_bounds(self.min, self.max)
        self._validate_truncation()
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

    def get_unit_mean(self) -> float:
        if self.shape_1 <= 0 or self.shape_2 <= 0:
            raise ValueError("shape parameters must be positive")
        return float(self.shape_1 / (self.shape_1 + self.shape_2))

    def get_unit_variance(self) -> float:
        if self.shape_1 <= 0 or self.shape_2 <= 0:
            raise ValueError("shape parameters must be positive")
        total = self.shape_1 + self.shape_2
        return float(self.shape_1 * self.shape_2 / (total**2 * (total + 1.0)))

    def get_scaled_mean(self) -> float:
        _validate_bounds(self.min, self.max)
        return float(self.min + (self.max - self.min) * self.get_unit_mean())

    def get_scaled_variance(self) -> float:
        _validate_bounds(self.min, self.max)
        return float((self.max - self.min) ** 2 * self.get_unit_variance())

    def get_mean(self) -> float:
        return self.get_scaled_mean()

    def get_variance(self) -> float:
        return self.get_scaled_variance()

    def sample(self, n: int) -> np.ndarray:
        if self.shape_1 <= 0 or self.shape_2 <= 0:
            raise ValueError("shape parameters must be positive")
        _validate_bounds(self.min, self.max)

        samples = stats.beta.rvs(self.shape_1, self.shape_2, size=n)
        return _as_1d_array(self.min + (self.max - self.min) * samples)


@dataclass(frozen=True)
class Binomial:
    n_trials: int
    success_prob: float
    min: int
    max: int
    truncated: bool = False
    truncation_tolerance: float = 0.05

    def _get_untruncated_mean(self) -> float:
        if self.n_trials < 1:
            raise ValueError("n_trials must be >= 1")
        if not 0.0 <= self.success_prob <= 1.0:
            raise ValueError("success_prob must be in [0, 1]")
        return float(self.n_trials * self.success_prob)

    def _get_untruncated_variance(self) -> float:
        if self.n_trials < 1:
            raise ValueError("n_trials must be >= 1")
        if not 0.0 <= self.success_prob <= 1.0:
            raise ValueError("success_prob must be in [0, 1]")
        return float(self.n_trials * self.success_prob * (1.0 - self.success_prob))

    def _get_truncated_mean(self) -> float:
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
        support = np.arange(self.min, self.max + 1)
        pmf = dist.pmf(support)
        mass = float(np.sum(pmf))
        if mass <= 0.0:
            raise ValueError("truncation interval has zero probability mass")
        return float(np.sum(support * pmf) / mass)

    def _get_truncated_variance(self) -> float:
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
        support = np.arange(self.min, self.max + 1)
        pmf = dist.pmf(support)
        mass = float(np.sum(pmf))
        if mass <= 0.0:
            raise ValueError("truncation interval has zero probability mass")
        mean = float(np.sum(support * pmf) / mass)
        second_moment = float(np.sum((support**2) * pmf) / mass)
        return second_moment - mean**2

    def _validate_truncation(self) -> None:
        if self.n_trials < 1:
            raise ValueError("n_trials must be >= 1")
        if not 0.0 <= self.success_prob <= 1.0:
            raise ValueError("success_prob must be in [0, 1]")
        _validate_bounds(self.min, self.max)
        untruncated_mean = self._get_untruncated_mean()
        untruncated_variance = self._get_untruncated_variance()
        truncated_mean = self._get_truncated_mean()
        truncated_variance = self._get_truncated_variance()
        eps = np.finfo(float).tiny
        mean_shift = abs(truncated_mean - untruncated_mean) / max(abs(untruncated_mean), eps)
        variance_shift = abs(truncated_variance - untruncated_variance) / max(abs(untruncated_variance), eps)
        if mean_shift > self.truncation_tolerance or variance_shift > self.truncation_tolerance:
            raise ValueError(
                "truncation too severe; adjust parameters or min/max to avoid serious truncation; "
                f"mean shift={mean_shift:.6f}, variance shift={variance_shift:.6f}, "
                f"tolerance={self.truncation_tolerance:.6f}, "
                f"min={self.min}, max={self.max}"
            )

    def get_mean(self) -> float:
        self._validate_truncation()
        return self._get_truncated_mean() if self.truncated else self._get_untruncated_mean()

    def get_variance(self) -> float:
        self._validate_truncation()
        return self._get_truncated_variance() if self.truncated else self._get_untruncated_variance()

    def sample(self, n: int) -> np.ndarray:
        if self.n_trials < 1:
            raise ValueError("n_trials must be >= 1")
        if not 0.0 <= self.success_prob <= 1.0:
            raise ValueError("success_prob must be in [0, 1]")
        _validate_bounds(self.min, self.max)
        self._validate_truncation()
        dist = stats.binom(self.n_trials, self.success_prob)
        lower = dist.cdf(self.min - 1)
        upper = dist.cdf(self.max)
        if lower >= upper:
            raise ValueError("truncation interval has zero probability mass")
        uniforms = np.random.uniform(lower, upper, size=n)
        return _as_1d_array(dist.ppf(uniforms))


@dataclass(frozen=True)
class Bernoulli:
    success_prob: float

    def get_mean(self) -> float:
        if not 0.0 <= self.success_prob <= 1.0:
            raise ValueError("success_prob must be in [0, 1]")
        return float(self.success_prob)

    def get_variance(self) -> float:
        if not 0.0 <= self.success_prob <= 1.0:
            raise ValueError("success_prob must be in [0, 1]")
        return float(self.success_prob * (1.0 - self.success_prob))

    def sample(self, n: int) -> np.ndarray:
        if not 0.0 <= self.success_prob <= 1.0:
            raise ValueError("success_prob must be in [0, 1]")
        return _as_1d_array(stats.bernoulli.rvs(self.success_prob, size=n))

    
@dataclass(frozen=True)
class Poisson:
    rate: float
    min: int
    max: int
    truncated: bool = False
    truncation_tolerance: float = 0.05

    def _get_untruncated_mean(self) -> float:
        if self.rate <= 0:
            raise ValueError("rate must be positive")
        return float(self.rate)

    def _get_untruncated_variance(self) -> float:
        if self.rate <= 0:
            raise ValueError("rate must be positive")
        return float(self.rate)

    def _get_truncated_mean(self) -> float:
        if self.rate <= 0:
            raise ValueError("rate must be positive")
        _validate_bounds(self.min, self.max)
        dist = stats.poisson(self.rate)
        lower = dist.cdf(self.min - 1)
        upper = dist.cdf(self.max)
        if lower >= upper:
            raise ValueError("truncation interval has zero probability mass")
        support = np.arange(self.min, self.max + 1)
        pmf = dist.pmf(support)
        mass = float(np.sum(pmf))
        if mass <= 0.0:
            raise ValueError("truncation interval has zero probability mass")
        return float(np.sum(support * pmf) / mass)

    def _get_truncated_variance(self) -> float:
        if self.rate <= 0:
            raise ValueError("rate must be positive")
        _validate_bounds(self.min, self.max)
        dist = stats.poisson(self.rate)
        lower = dist.cdf(self.min - 1)
        upper = dist.cdf(self.max)
        if lower >= upper:
            raise ValueError("truncation interval has zero probability mass")
        support = np.arange(self.min, self.max + 1)
        pmf = dist.pmf(support)
        mass = float(np.sum(pmf))
        if mass <= 0.0:
            raise ValueError("truncation interval has zero probability mass")
        mean = float(np.sum(support * pmf) / mass)
        second_moment = float(np.sum((support**2) * pmf) / mass)
        return second_moment - mean**2

    def _validate_truncation(self) -> None:
        if self.rate <= 0:
            raise ValueError("rate must be positive")
        _validate_bounds(self.min, self.max)
        untruncated_mean = self._get_untruncated_mean()
        untruncated_variance = self._get_untruncated_variance()
        truncated_mean = self._get_truncated_mean()
        truncated_variance = self._get_truncated_variance()
        eps = np.finfo(float).tiny
        mean_shift = abs(truncated_mean - untruncated_mean) / max(abs(untruncated_mean), eps)
        variance_shift = abs(truncated_variance - untruncated_variance) / max(abs(untruncated_variance), eps)
        if mean_shift > self.truncation_tolerance or variance_shift > self.truncation_tolerance:
            raise ValueError(
                "truncation too severe; adjust parameters or min/max to avoid serious truncation; "
                f"mean shift={mean_shift:.6f}, variance shift={variance_shift:.6f}, "
                f"tolerance={self.truncation_tolerance:.6f}, "
                f"min={self.min}, max={self.max}"
            )

    def get_mean(self) -> float:
        self._validate_truncation()
        return self._get_truncated_mean() if self.truncated else self._get_untruncated_mean()

    def get_variance(self) -> float:
        self._validate_truncation()
        return self._get_truncated_variance() if self.truncated else self._get_untruncated_variance()

    def sample(self, n: int) -> np.ndarray:
        if self.rate <= 0:
            raise ValueError("rate must be positive")
        _validate_bounds(self.min, self.max)
        self._validate_truncation()
        dist = stats.poisson(self.rate)
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
    truncated: bool = False
    truncation_tolerance: float = 0.05

    def _get_untruncated_mean(self) -> float:
        if self.shape <= 0:
            raise ValueError("shape must be positive")
        if self.mean < 0:
            raise ValueError("mean must be non-negative")
        return float(self.mean)

    def _get_untruncated_variance(self) -> float:
        if self.shape <= 0:
            raise ValueError("shape must be positive")
        if self.mean < 0:
            raise ValueError("mean must be non-negative")
        p = self.shape / (self.shape + self.mean) if self.mean > 0 else 1.0
        return float(self.shape * (1.0 - p) / (p**2))

    def _get_truncated_mean(self) -> float:
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
        support = np.arange(self.min, self.max + 1)
        pmf = dist.pmf(support)
        mass = float(np.sum(pmf))
        if mass <= 0.0:
            raise ValueError("truncation interval has zero probability mass")
        return float(np.sum(support * pmf) / mass)

    def _get_truncated_variance(self) -> float:
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
        support = np.arange(self.min, self.max + 1)
        pmf = dist.pmf(support)
        mass = float(np.sum(pmf))
        if mass <= 0.0:
            raise ValueError("truncation interval has zero probability mass")
        mean = float(np.sum(support * pmf) / mass)
        second_moment = float(np.sum((support**2) * pmf) / mass)
        return second_moment - mean**2

    def _validate_truncation(self) -> None:
        if self.shape <= 0:
            raise ValueError("shape must be positive")
        if self.mean < 0:
            raise ValueError("mean must be non-negative")
        _validate_bounds(self.min, self.max)
        untruncated_mean = self._get_untruncated_mean()
        untruncated_variance = self._get_untruncated_variance()
        truncated_mean = self._get_truncated_mean()
        truncated_variance = self._get_truncated_variance()
        eps = np.finfo(float).tiny
        mean_shift = abs(truncated_mean - untruncated_mean) / max(abs(untruncated_mean), eps)
        variance_shift = abs(truncated_variance - untruncated_variance) / max(abs(untruncated_variance), eps)
        if mean_shift > self.truncation_tolerance or variance_shift > self.truncation_tolerance:
            raise ValueError(
                "truncation too severe; adjust parameters or min/max to avoid serious truncation; "
                f"mean shift={mean_shift:.6f}, variance shift={variance_shift:.6f}, "
                f"tolerance={self.truncation_tolerance:.6f}, "
                f"min={self.min}, max={self.max}"
            )

    def get_mean(self) -> float:
        self._validate_truncation()
        return self._get_truncated_mean() if self.truncated else self._get_untruncated_mean()

    def get_variance(self) -> float:
        self._validate_truncation()
        return self._get_truncated_variance() if self.truncated else self._get_untruncated_variance()

    def sample(self, n: int) -> np.ndarray:
        if self.shape <= 0:
            raise ValueError("shape must be positive")
        if self.mean < 0:
            raise ValueError("mean must be non-negative")
        _validate_bounds(self.min, self.max)
        self._validate_truncation()
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
