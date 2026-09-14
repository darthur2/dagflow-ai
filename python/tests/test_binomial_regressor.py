import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from distributions import Binomial, CategoricalNominal, Normal
from regressors import BinomialRegressor


def _build_x() -> np.ndarray:
    np.random.seed(0)

    test_anxiety_score = Normal(mean=45.0, standard_deviation=10.0, min=15.0, max=75.0).sample(100)
    working_memory_capacity = Normal(mean=0.0, standard_deviation=1.0, min=-3.0, max=3.0).sample(100)
    sleep_duration_hours = Normal(mean=7.2, standard_deviation=1.1, min=3.5, max=10.5).sample(100)
    intervention_group = CategoricalNominal(
        categories=["control", "mindfulness"],
        probabilities=[0.5, 0.5],
    ).sample(100)

    mindfulness = (intervention_group == "mindfulness").astype(float)
    return np.column_stack([test_anxiety_score, working_memory_capacity, sleep_duration_hours, mindfulness])


def test_binomial_regressor_calibrates_and_samples():
    X = _build_x()
    response = Binomial(n_trials=20, success_prob=0.6, min=1, max=18)
    regressor = BinomialRegressor(
        target_mean=response.target_mean(),
        n_trials=20,
        min=1,
        max=18,
        X=X,
        beta_1_init=np.array([-0.025, 0.55, 0.18, 0.22], dtype=float),
        predictor_names=[
            "test_anxiety_score",
            "working_memory_capacity",
            "sleep_duration_hours",
            "intervention_group",
        ],
        predictor_transformations={},
    )

    calibrated = regressor.calibrate()
    samples = calibrated.sample(2000)

    sample_mean = float(np.mean(samples))

    print(f"beta_0: {calibrated.beta_0}")
    print(f"mean: target={regressor.target_mean} sample={sample_mean}")
