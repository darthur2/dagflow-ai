import numpy as np
from scipy import stats

from distributions import Beta, Bernoulli, DiscreteUniform, Exponential, Gamma, Geometric, LogNormal, NegativeBinomial, Normal, Poisson, Uniform


FIXTURE_DISTRIBUTIONS = (
    Normal(mean=0.0, standard_deviation=1.0, min=-2.0, max=2.0),
    Exponential(rate=1.5, min=0.0, max=4.0),
    Gamma(shape=2.0, rate=1.2, min=0.0, max=8.0),
    Beta(shape_1=2.0, shape_2=5.0, min=-1.0, max=3.0),
    LogNormal(log_mean=0.0, log_standard_deviation=0.5, min=0.1, max=5.0),
    Uniform(min=-3.0, max=3.0),
    DiscreteUniform(min=1, max=8),
    Bernoulli(success_prob=0.35),
    stats.binom(n=6, p=0.4),
    Poisson(rate=3.0, min=0, max=12),
    Geometric(success_prob=0.25, min=1, max=15),
    NegativeBinomial(shape=4.0, mean=6.0, min=0, max=20),
)


def build_x_matrix(n: int, p: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    columns = []
    for idx in range(p):
        distribution = FIXTURE_DISTRIBUTIONS[idx % len(FIXTURE_DISTRIBUTIONS)]
        if hasattr(distribution, "sample"):
            columns.append(np.asarray(distribution.sample(n), dtype=float))
        else:
            columns.append(np.asarray(distribution.rvs(size=n, random_state=rng), dtype=float))
    return np.column_stack(columns)


def sample_truncated_binomial_regression(x: np.ndarray, beta_0: float, beta_1_init: np.ndarray, c: float, n_trials: int, min_value: int, max_value: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eta = beta_0 + x @ (c * beta_1_init)
    p = 1.0 / (1.0 + np.exp(-eta))
    samples = np.empty(len(x), dtype=float)
    values = np.arange(min_value, max_value + 1)
    for idx, current_p in enumerate(p):
        dist = stats.binom(n_trials, current_p)
        lower = dist.cdf(min_value - 1)
        upper = dist.cdf(max_value)
        u = rng.uniform(lower, upper)
        samples[idx] = dist.ppf(u)
    return samples
