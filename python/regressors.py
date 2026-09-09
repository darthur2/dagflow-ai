"""Regression calibration helpers for truncated normal models.

This module provides a calibration helper for a truncated normal regression
model with a fixed coefficient direction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import optimize, stats


@dataclass
class TruncatedNormalRegression:
    X: np.ndarray
    beta_1_init: np.ndarray
    target_mean: float
    target_std_deviation: float
    min: float
    max: float
    target_snr: float
    beta_0: Optional[float] = None
    c: Optional[float] = None
    sigma_regression: Optional[float] = None

    def solve(self) -> dict:
        """Solve for beta_0, c, and sigma_regression.

        Returns:
            A dictionary with the fitted parameters and the realized moments
            under the truncated normal regression model.
        """
        self._validate_inputs()

        d = self._linear_predictor_direction()
        d_std = float(np.std(d))
        if d_std < np.finfo(float).eps:
            raise ValueError("linear predictor has no variation")

        sigma2_total = self.target_std_deviation**2
        sigma2_signal = sigma2_total * self.target_snr / (self.target_snr + 1.0)
        sigma2_noise = sigma2_total / (self.target_snr + 1.0)

        c0 = math.sqrt(max(sigma2_signal / float(np.var(d)), 1e-12))
        beta0_0 = self.target_mean - c0 * float(np.mean(d))
        sigma0 = math.sqrt(max(sigma2_noise, 1e-12))

        result = optimize.root(
            self._root_equations,
            x0=[beta0_0, math.log(c0), math.log(sigma0)],
            method="hybr",
        )

        if not result.success:
            raise RuntimeError(
                f"Failed to solve truncated normal regression parameters: {result.message}"
            )

        self.beta_0 = float(result.x[0])
        self.c = float(math.exp(result.x[1]))
        self.sigma_regression = float(math.exp(result.x[2]))

        metrics = self._fitted_metrics(self.beta_0, self.c, self.sigma_regression)
        mean_error = abs(metrics["fitted_mean"] - self.target_mean)
        std_error = abs(math.sqrt(metrics["fitted_total_variance"]) - self.target_std_deviation)
        snr_error = abs(metrics["fitted_snr"] - self.target_snr)
        tolerance = 1e-8
        if mean_error > tolerance or std_error > tolerance or snr_error > tolerance:
            raise RuntimeError(
                "Solved parameters do not match target moments within tolerance "
                f"(mean_error={mean_error:.3e}, std_error={std_error:.3e}, snr_error={snr_error:.3e})"
            )

        return {
            "beta_0": self.beta_0,
            "c": self.c,
            "sigma_regression": self.sigma_regression,
            **metrics,
        }

    def sample(self, n: int, random_state=None):
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer")

        if self.beta_0 is None or self.c is None or self.sigma_regression is None:
            self.solve()

        assert self.beta_0 is not None
        assert self.c is not None
        assert self.sigma_regression is not None

        d = self._linear_predictor_direction()
        mu = self.beta_0 + self.c * d
        rng = np.random.default_rng(random_state)
        u = rng.uniform(size=len(mu))
        samples = np.empty(len(mu))

        for i, mu_i in enumerate(mu):
            alpha = (self.min - mu_i) / self.sigma_regression
            beta = (self.max - mu_i) / self.sigma_regression
            p_low = stats.norm.cdf(alpha)
            p_high = stats.norm.cdf(beta)
            samples[i] = stats.norm.ppf(p_low + u[i] * (p_high - p_low), loc=mu_i, scale=self.sigma_regression)

        return samples

    def _validate_inputs(self) -> None:
        self.X = np.asarray(self.X, dtype=float)
        self.beta_1_init = np.asarray(self.beta_1_init, dtype=float)

        if self.X.ndim != 2:
            raise ValueError("X must be a 2D numeric matrix")
        if self.beta_1_init.ndim != 1 or self.beta_1_init.shape[0] != self.X.shape[1]:
            raise ValueError("beta_1_init must be a 1D numeric vector of length ncol(X)")
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
        if not math.isfinite(self.target_snr) or self.target_snr < 0:
            raise ValueError("target_snr must be a non-negative finite number")

    def _linear_predictor_direction(self) -> np.ndarray:
        return self.X @ self.beta_1_init

    def _root_equations(self, params):
        beta_0 = float(params[0])
        c = float(math.exp(params[1]))
        sigma_regression = float(math.exp(params[2]))
        metrics = self._fitted_metrics(beta_0, c, sigma_regression)

        return [
            metrics["fitted_mean"] - self.target_mean,
            metrics["fitted_total_variance"] - self.target_std_deviation**2,
            metrics["fitted_snr"] - self.target_snr,
        ]

    def _fitted_metrics(self, beta_0: float, c: float, sigma_regression: float) -> dict:
        d = self._linear_predictor_direction()
        mu = beta_0 + c * d

        alpha = (self.min - mu) / sigma_regression
        beta = (self.max - mu) / sigma_regression
        z = stats.norm.cdf(beta) - stats.norm.cdf(alpha)
        if np.any(z <= 0):
            raise ValueError("No support in the interval [min, max] for this truncated normal regression")

        pa = stats.norm.pdf(alpha)
        pb = stats.norm.pdf(beta)
        delta = (pa - pb) / z

        m = mu + sigma_regression * delta
        v = sigma_regression**2 * (1.0 + (alpha * pa - beta * pb) / z - delta**2)

        fitted_mean = float(np.mean(m))
        fitted_within = float(np.mean(v))
        fitted_between = float(np.var(m))
        fitted_total = fitted_between + fitted_within
        fitted_snr = float(fitted_between / fitted_within) if fitted_within > 0 else math.inf

        return {
            "fitted_mean": fitted_mean,
            "fitted_between_variance": fitted_between,
            "fitted_within_variance": fitted_within,
            "fitted_total_variance": fitted_total,
            "fitted_snr": fitted_snr,
        }


@dataclass
class TruncatedGammaRegression:
    """Best-fit calibrator for a truncated gamma regression model.

    The solver matches target quantiles and target SNR as closely as possible
    under a fixed-direction log-link gamma regression model.
    """

    X: np.ndarray
    beta_1_init: np.ndarray
    target_q25: float
    target_median: float
    target_q75: float
    min: float
    max: float
    target_snr: float
    beta_0: Optional[float] = None
    c: Optional[float] = None
    phi: Optional[float] = None

    def solve(self) -> dict:
        """Solve for beta_0, c, and phi.

        Returns:
            A dictionary with fitted parameters and realized quantile/SNR
            diagnostics under the truncated gamma regression model.
        """
        self._validate_inputs()

        d = self._linear_predictor_direction()
        d_std = float(np.std(d))
        if d_std < np.finfo(float).eps:
            raise ValueError("linear predictor has no variation")

        q_target = float(np.mean([self.target_q25, self.target_median, self.target_q75]))
        beta0_0 = math.log(max(q_target, 1e-6)) - float(np.mean(d))
        c0 = 1.0
        phi0 = max(1.0, float(np.var(d)) / max(self.target_snr, 1e-6))

        result = optimize.minimize(
            self._objective,
            x0=[beta0_0, math.log(c0), math.log(phi0)],
            method="Nelder-Mead",
            options={"maxiter": 10000, "xatol": 1e-10, "fatol": 1e-10},
        )

        if not result.success:
            raise RuntimeError(
                f"Failed to solve truncated gamma regression parameters: {result.message}"
            )

        self.beta_0 = float(result.x[0])
        self.c = float(math.exp(result.x[1]))
        self.phi = float(math.exp(result.x[2]))

        metrics = self._fitted_metrics(self.beta_0, self.c, self.phi)

        return {
            "beta_0": self.beta_0,
            "c": self.c,
            "phi": self.phi,
            **metrics,
        }

    def sample(self, n: int, random_state=None):
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer")

        if self.beta_0 is None or self.c is None or self.phi is None:
            self.solve()

        assert self.beta_0 is not None
        assert self.c is not None
        assert self.phi is not None

        d = self._linear_predictor_direction()
        mu = np.exp(self.beta_0 + self.c * d)
        shape = self.phi
        rate = shape / mu

        rng = np.random.default_rng(random_state)
        u = rng.uniform(size=len(mu))
        samples = np.empty(len(mu))

        for i, (shape_i, rate_i) in enumerate(zip(np.repeat(shape, len(mu)), rate)):
            dist = stats.gamma(a=shape_i, scale=1.0 / rate_i)
            p_low = dist.cdf(self.min)
            p_high = dist.cdf(self.max)
            samples[i] = dist.ppf(p_low + u[i] * (p_high - p_low))

        return samples

    def _validate_inputs(self) -> None:
        self.X = np.asarray(self.X, dtype=float)
        self.beta_1_init = np.asarray(self.beta_1_init, dtype=float)

        if self.X.ndim != 2:
            raise ValueError("X must be a 2D numeric matrix")
        if self.beta_1_init.ndim != 1 or self.beta_1_init.shape[0] != self.X.shape[1]:
            raise ValueError("beta_1_init must be a 1D numeric vector of length ncol(X)")
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
        if any(v <= 0 for v in [self.target_q25, self.target_median, self.target_q75]):
            raise ValueError("target quantiles must be positive for a gamma distribution")
        if not math.isfinite(self.target_snr) or self.target_snr < 0:
            raise ValueError("target_snr must be a non-negative finite number")

    def _linear_predictor_direction(self) -> np.ndarray:
        return self.X @ self.beta_1_init

    def _objective(self, params):
        beta_0 = float(params[0])
        c = float(math.exp(params[1]))
        phi = float(math.exp(params[2]))
        metrics = self._fitted_metrics(beta_0, c, phi)

        q25 = (metrics["fitted_q25"] - self.target_q25) ** 2
        med = (metrics["fitted_median"] - self.target_median) ** 2
        q75 = (metrics["fitted_q75"] - self.target_q75) ** 2
        snr = (metrics["fitted_snr"] - self.target_snr) ** 2
        return q25 + med + q75 + snr

    def _fitted_metrics(self, beta_0: float, c: float, phi: float) -> dict:
        d = self._linear_predictor_direction()
        mu = np.exp(beta_0 + c * d)
        shape = phi
        rate = shape / mu

        q25 = np.empty(len(mu))
        q50 = np.empty(len(mu))
        q75 = np.empty(len(mu))
        within = np.empty(len(mu))

        for i, rate_i in enumerate(rate):
            dist = stats.gamma(a=shape, scale=1.0 / rate_i)
            p_low = dist.cdf(self.min)
            p_high = dist.cdf(self.max)
            if p_high <= p_low:
                raise ValueError("No support in the interval [min, max] for this truncated gamma regression")

            q25[i] = dist.ppf(p_low + 0.25 * (p_high - p_low))
            q50[i] = dist.ppf(p_low + 0.50 * (p_high - p_low))
            q75[i] = dist.ppf(p_low + 0.75 * (p_high - p_low))
            within[i] = float(np.var([q25[i], q50[i], q75[i]]))

        fitted_q25 = float(np.mean(q25))
        fitted_median = float(np.mean(q50))
        fitted_q75 = float(np.mean(q75))

        fitted_between = float(np.var(q50))
        fitted_within = float(np.mean(within))
        fitted_snr = float(fitted_between / fitted_within) if fitted_within > 0 else math.inf

        return {
            "fitted_q25": fitted_q25,
            "fitted_median": fitted_median,
            "fitted_q75": fitted_q75,
            "fitted_between_variance": fitted_between,
            "fitted_within_variance": fitted_within,
            "fitted_snr": fitted_snr,
        }


@dataclass
class TruncatedLogNormalRegression:
    """Best-fit calibrator for a truncated lognormal regression model.

    The solver matches target quantiles and target SNR as closely as possible
    under a fixed-direction log-link lognormal regression model.
    """

    X: np.ndarray
    beta_1_init: np.ndarray
    target_q25: float
    target_median: float
    target_q75: float
    min: float
    max: float
    target_snr: float
    beta_0: Optional[float] = None
    c: Optional[float] = None
    sdlog: Optional[float] = None

    def solve(self) -> dict:
        self._validate_inputs()

        d = self._linear_predictor_direction()
        d_std = float(np.std(d))
        if d_std < np.finfo(float).eps:
            raise ValueError("linear predictor has no variation")

        q_target = float(np.mean([self.target_q25, self.target_median, self.target_q75]))
        beta0_0 = math.log(max(q_target, 1e-6)) - float(np.mean(d))
        c0 = 1.0
        sdlog0 = 1.0

        result = optimize.minimize(
            self._objective,
            x0=[beta0_0, math.log(c0), math.log(sdlog0)],
            method="Nelder-Mead",
            options={"maxiter": 10000, "xatol": 1e-10, "fatol": 1e-10},
        )

        if not result.success:
            raise RuntimeError(
                f"Failed to solve truncated lognormal regression parameters: {result.message}"
            )

        self.beta_0 = float(result.x[0])
        self.c = float(math.exp(result.x[1]))
        self.sdlog = float(math.exp(result.x[2]))

        metrics = self._fitted_metrics(self.beta_0, self.c, self.sdlog)
        return {"beta_0": self.beta_0, "c": self.c, "sdlog": self.sdlog, **metrics}

    def sample(self, n: int, random_state=None):
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer")

        if self.beta_0 is None or self.c is None or self.sdlog is None:
            self.solve()

        assert self.beta_0 is not None
        assert self.c is not None
        assert self.sdlog is not None

        d = self._linear_predictor_direction()
        meanlog = self.beta_0 + self.c * d
        rng = np.random.default_rng(random_state)
        u = rng.uniform(size=len(meanlog))
        samples = np.empty(len(meanlog))

        for i, mlog in enumerate(meanlog):
            dist = stats.lognorm(s=self.sdlog, scale=math.exp(mlog))
            p_low = dist.cdf(self.min)
            p_high = dist.cdf(self.max)
            samples[i] = dist.ppf(p_low + u[i] * (p_high - p_low))

        return samples

    def _validate_inputs(self) -> None:
        self.X = np.asarray(self.X, dtype=float)
        self.beta_1_init = np.asarray(self.beta_1_init, dtype=float)

        if self.X.ndim != 2:
            raise ValueError("X must be a 2D numeric matrix")
        if self.beta_1_init.ndim != 1 or self.beta_1_init.shape[0] != self.X.shape[1]:
            raise ValueError("beta_1_init must be a 1D numeric vector of length ncol(X)")
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
            raise ValueError("min must be >= 0 for a lognormal distribution")
        if self.max <= 0:
            raise ValueError("max must be > 0 for a lognormal distribution")
        if self.target_q25 > self.target_median or self.target_median > self.target_q75:
            raise ValueError("target quantiles must satisfy q25 <= median <= q75")
        if any(v <= 0 for v in [self.target_q25, self.target_median, self.target_q75]):
            raise ValueError("target quantiles must be positive for a lognormal distribution")
        if not (self.min <= self.target_q25 <= self.max):
            raise ValueError("target_q25 must lie within [min, max]")
        if not (self.min <= self.target_median <= self.max):
            raise ValueError("target_median must lie within [min, max]")
        if not (self.min <= self.target_q75 <= self.max):
            raise ValueError("target_q75 must lie within [min, max]")
        if not math.isfinite(self.target_snr) or self.target_snr < 0:
            raise ValueError("target_snr must be a non-negative finite number")

    def _linear_predictor_direction(self) -> np.ndarray:
        return self.X @ self.beta_1_init

    def _objective(self, params):
        beta_0 = float(params[0])
        c = float(math.exp(params[1]))
        sdlog = float(math.exp(params[2]))
        metrics = self._fitted_metrics(beta_0, c, sdlog)

        q25 = (metrics["fitted_q25"] - self.target_q25) ** 2
        med = (metrics["fitted_median"] - self.target_median) ** 2
        q75 = (metrics["fitted_q75"] - self.target_q75) ** 2
        snr = (metrics["fitted_snr"] - self.target_snr) ** 2
        return q25 + med + q75 + snr

    def _fitted_metrics(self, beta_0: float, c: float, sdlog: float) -> dict:
        d = self._linear_predictor_direction()
        meanlog = beta_0 + c * d

        q25 = np.empty(len(meanlog))
        q50 = np.empty(len(meanlog))
        q75 = np.empty(len(meanlog))
        within = np.empty(len(meanlog))

        for i, mlog in enumerate(meanlog):
            dist = stats.lognorm(s=sdlog, scale=math.exp(mlog))
            p_low = dist.cdf(self.min)
            p_high = dist.cdf(self.max)
            if p_high <= p_low:
                raise ValueError("No support in the interval [min, max] for this truncated lognormal regression")
            q25[i] = dist.ppf(p_low + 0.25 * (p_high - p_low))
            q50[i] = dist.ppf(p_low + 0.50 * (p_high - p_low))
            q75[i] = dist.ppf(p_low + 0.75 * (p_high - p_low))
            within[i] = float(np.var([q25[i], q50[i], q75[i]]))

        fitted_q25 = float(np.mean(q25))
        fitted_median = float(np.mean(q50))
        fitted_q75 = float(np.mean(q75))
        fitted_between = float(np.var(q50))
        fitted_within = float(np.mean(within))
        fitted_snr = float(fitted_between / fitted_within) if fitted_within > 0 else math.inf

        return {
            "fitted_q25": fitted_q25,
            "fitted_median": fitted_median,
            "fitted_q75": fitted_q75,
            "fitted_between_variance": fitted_between,
            "fitted_within_variance": fitted_within,
            "fitted_snr": fitted_snr,
        }


@dataclass
class TruncatedBetaRegression:
    """Best-fit calibrator for a truncated beta regression model.

    The solver matches target quantiles and target SNR as closely as possible
    under a fixed-direction logit-link beta regression model.
    """

    X: np.ndarray
    beta_1_init: np.ndarray
    target_q25: float
    target_median: float
    target_q75: float
    min: float
    max: float
    target_snr: float
    beta_0: Optional[float] = None
    c: Optional[float] = None
    phi: Optional[float] = None

    def solve(self) -> dict:
        self._validate_inputs()

        d = self._linear_predictor_direction()
        d_std = float(np.std(d))
        if d_std < np.finfo(float).eps:
            raise ValueError("linear predictor has no variation")

        q_target = float(np.mean([self.target_q25, self.target_median, self.target_q75]))
        mu_target = (q_target - self.min) / (self.max - self.min)
        mu_target = min(max(mu_target, 1e-6), 1 - 1e-6)
        beta0_0 = math.log(mu_target / (1.0 - mu_target)) - float(np.mean(d))
        c0 = 1.0
        phi0 = max(1.0, float(np.var(d)) / max(self.target_snr, 1e-6))

        result = optimize.minimize(
            self._objective,
            x0=[beta0_0, math.log(c0), math.log(phi0)],
            method="Nelder-Mead",
            options={"maxiter": 10000, "xatol": 1e-10, "fatol": 1e-10},
        )

        if not result.success:
            raise RuntimeError(
                f"Failed to solve truncated beta regression parameters: {result.message}"
            )

        self.beta_0 = float(result.x[0])
        self.c = float(math.exp(result.x[1]))
        self.phi = float(math.exp(result.x[2]))

        metrics = self._fitted_metrics(self.beta_0, self.c, self.phi)
        return {"beta_0": self.beta_0, "c": self.c, "phi": self.phi, **metrics}

    def sample(self, n: int, random_state=None):
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer")

        if self.beta_0 is None or self.c is None or self.phi is None:
            self.solve()

        assert self.beta_0 is not None
        assert self.c is not None
        assert self.phi is not None

        d = self._linear_predictor_direction()
        mu = 1.0 / (1.0 + np.exp(-(self.beta_0 + self.c * d)))
        shape1 = mu * self.phi
        shape2 = (1.0 - mu) * self.phi

        rng = np.random.default_rng(random_state)
        u = rng.uniform(size=len(mu))
        samples = np.empty(len(mu))

        for i, (s1, s2) in enumerate(zip(shape1, shape2)):
            dist = stats.beta(a=s1, b=s2)
            p_low = dist.cdf((self.min - self.min) / (self.max - self.min))
            p_high = dist.cdf((self.max - self.min) / (self.max - self.min))
            samples[i] = self.min + (self.max - self.min) * dist.ppf(p_low + u[i] * (p_high - p_low))

        return samples

    def _validate_inputs(self) -> None:
        self.X = np.asarray(self.X, dtype=float)
        self.beta_1_init = np.asarray(self.beta_1_init, dtype=float)

        if self.X.ndim != 2:
            raise ValueError("X must be a 2D numeric matrix")
        if self.beta_1_init.ndim != 1 or self.beta_1_init.shape[0] != self.X.shape[1]:
            raise ValueError("beta_1_init must be a 1D numeric vector of length ncol(X)")
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
        if any(v <= 0 for v in [self.target_q25, self.target_median, self.target_q75]):
            raise ValueError("target quantiles must be positive for a beta distribution")
        if not math.isfinite(self.target_snr) or self.target_snr < 0:
            raise ValueError("target_snr must be a non-negative finite number")

    def _linear_predictor_direction(self) -> np.ndarray:
        return self.X @ self.beta_1_init

    def _objective(self, params):
        beta_0 = float(params[0])
        c = float(math.exp(params[1]))
        phi = float(math.exp(params[2]))
        metrics = self._fitted_metrics(beta_0, c, phi)

        q25 = (metrics["fitted_q25"] - self.target_q25) ** 2
        med = (metrics["fitted_median"] - self.target_median) ** 2
        q75 = (metrics["fitted_q75"] - self.target_q75) ** 2
        snr = (metrics["fitted_snr"] - self.target_snr) ** 2
        return q25 + med + q75 + snr

    def _fitted_metrics(self, beta_0: float, c: float, phi: float) -> dict:
        d = self._linear_predictor_direction()
        mu = 1.0 / (1.0 + np.exp(-(beta_0 + c * d)))
        shape1 = mu * phi
        shape2 = (1.0 - mu) * phi

        q25 = np.empty(len(mu))
        q50 = np.empty(len(mu))
        q75 = np.empty(len(mu))
        within = np.empty(len(mu))
        scale = self.max - self.min

        for i, (s1, s2) in enumerate(zip(shape1, shape2)):
            dist = stats.beta(a=s1, b=s2)
            q25[i] = self.min + scale * dist.ppf(0.25)
            q50[i] = self.min + scale * dist.ppf(0.50)
            q75[i] = self.min + scale * dist.ppf(0.75)
            within[i] = float(np.var([q25[i], q50[i], q75[i]]))

        fitted_q25 = float(np.mean(q25))
        fitted_median = float(np.mean(q50))
        fitted_q75 = float(np.mean(q75))
        fitted_between = float(np.var(q50))
        fitted_within = float(np.mean(within))
        fitted_snr = float(fitted_between / fitted_within) if fitted_within > 0 else math.inf

        return {
            "fitted_q25": fitted_q25,
            "fitted_median": fitted_median,
            "fitted_q75": fitted_q75,
            "fitted_between_variance": fitted_between,
            "fitted_within_variance": fitted_within,
            "fitted_snr": fitted_snr,
        }


@dataclass
class UniformRegression:
    """Wrapper around a latent truncated normal regression model.

    Samples are transformed through the latent truncated-normal CDF and then
    linearly scaled to [min, max], producing an exact uniform output.
    """

    X: np.ndarray
    beta_1_init: np.ndarray
    min: float
    max: float
    target_snr: float
    beta_0: Optional[float] = None
    c: Optional[float] = None
    sigma_regression: Optional[float] = None
    latent_beta_0: Optional[float] = None
    latent_c: Optional[float] = None
    latent_sigma_regression: Optional[float] = None

    def solve(self) -> dict:
        self._validate_inputs()

        latent = TruncatedNormalRegression(
            X=self.X,
            beta_1_init=self.beta_1_init,
            target_mean=0.0,
            target_std_deviation=1.0,
            min=-4.0,
            max=4.0,
            target_snr=self.target_snr,
        )
        latent_result = latent.solve()

        self.latent_beta_0 = latent_result["beta_0"]
        self.latent_c = latent_result["c"]
        self.latent_sigma_regression = latent_result["sigma_regression"]

        self.beta_0 = self.latent_beta_0
        self.c = self.latent_c
        self.sigma_regression = self.latent_sigma_regression

        return {
            "beta_0": self.beta_0,
            "c": self.c,
            "sigma_regression": self.sigma_regression,
            "latent_beta_0": self.latent_beta_0,
            "latent_c": self.latent_c,
            "latent_sigma_regression": self.latent_sigma_regression,
            "min": self.min,
            "max": self.max,
            "target_snr": self.target_snr,
        }

    def sample(self, n: int, random_state=None):
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer")

        if self.latent_beta_0 is None or self.latent_c is None or self.latent_sigma_regression is None:
            self.solve()

        assert self.latent_beta_0 is not None
        assert self.latent_c is not None
        assert self.latent_sigma_regression is not None

        latent = TruncatedNormalRegression(
            X=self.X,
            beta_1_init=self.beta_1_init,
            target_mean=0.0,
            target_std_deviation=1.0,
            min=-4.0,
            max=4.0,
            target_snr=self.target_snr,
            beta_0=self.latent_beta_0,
            c=self.latent_c,
            sigma_regression=self.latent_sigma_regression,
        )

        d = latent._linear_predictor_direction()
        mu = self.latent_beta_0 + self.latent_c * d
        sigma = self.latent_sigma_regression
        rng = np.random.default_rng(random_state)
        u = rng.uniform(size=len(mu))
        samples = np.empty(len(mu))

        for i, mu_i in enumerate(mu):
            alpha = (-4.0 - mu_i) / sigma
            beta = (4.0 - mu_i) / sigma
            p_low = stats.norm.cdf(alpha)
            p_high = stats.norm.cdf(beta)
            z = stats.norm.ppf(p_low + u[i] * (p_high - p_low), loc=mu_i, scale=sigma)
            latent_cdf = (stats.norm.cdf((z - mu_i) / sigma) - p_low) / (p_high - p_low)
            samples[i] = self.min + (self.max - self.min) * latent_cdf

        return samples

    def _validate_inputs(self) -> None:
        self.X = np.asarray(self.X, dtype=float)
        self.beta_1_init = np.asarray(self.beta_1_init, dtype=float)

        if self.X.ndim != 2:
            raise ValueError("X must be a 2D numeric matrix")
        if self.beta_1_init.ndim != 1 or self.beta_1_init.shape[0] != self.X.shape[1]:
            raise ValueError("beta_1_init must be a 1D numeric vector of length ncol(X)")
        if not math.isfinite(self.min) or not math.isfinite(self.max):
            raise ValueError("min and max must be finite")
        if self.min >= self.max:
            raise ValueError("min must be less than max")
        if not math.isfinite(self.target_snr) or self.target_snr < 0:
            raise ValueError("target_snr must be a non-negative finite number")
