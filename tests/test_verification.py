"""Unit tests for contribution evaluation, anomaly detection, and Shapley valuation."""

import numpy as np
import pytest

from agent_federated_learning.models import ModelUpdate
from agent_federated_learning.verification import ContributionEvaluator


def test_cosine_similarity():
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.0, 1.0])
    v3 = np.array([-1.0, 0.0])
    v_zero = np.array([0.0, 0.0])

    assert np.isclose(ContributionEvaluator.compute_cosine_similarity(v1, v1), 1.0)
    assert np.isclose(ContributionEvaluator.compute_cosine_similarity(v1, v2), 0.0)
    assert np.isclose(ContributionEvaluator.compute_cosine_similarity(v1, v3), -1.0)
    assert np.isclose(ContributionEvaluator.compute_cosine_similarity(v1, v_zero), 0.0)


def test_evaluate_updates_anomalies():
    # 3 honest updates pointing in direction [1.0, 1.0]
    # 1 sign-inverting adversary [-1.0, -1.0]
    # 1 norm-exploding adversary [100.0, 100.0]
    updates = [
        ModelUpdate(agent_id="h1", round_id=1, parameters={"w": [1.0, 1.0]}),
        ModelUpdate(agent_id="h2", round_id=1, parameters={"w": [1.05, 0.95]}),
        ModelUpdate(agent_id="h3", round_id=1, parameters={"w": [0.95, 1.05]}),
        ModelUpdate(agent_id="adv_flip", round_id=1, parameters={"w": [-1.0, -1.0]}),
        ModelUpdate(agent_id="adv_norm", round_id=1, parameters={"w": [100.0, 100.0]}),
    ]

    contributions = ContributionEvaluator.evaluate_updates(updates)
    assert len(contributions) == 5

    res_map = {c.agent_id: c for c in contributions}
    assert res_map["h1"].is_byzantine is False
    assert res_map["h1"].cosine_similarity > 0.9

    # Directional inversion check
    assert res_map["adv_flip"].is_byzantine is True
    assert "Negative directional alignment" in (res_map["adv_flip"].rejection_reason or "")

    # Norm explosion check
    assert res_map["adv_norm"].is_byzantine is True
    assert "Gradient norm explosion" in (res_map["adv_norm"].rejection_reason or "")


def test_evaluate_updates_empty():
    assert ContributionEvaluator.evaluate_updates([]) == []


def test_calculate_shapley_values():
    u1 = ModelUpdate(agent_id="h1", round_id=1, parameters={"w": [1.0, 0.0]})
    u2 = ModelUpdate(agent_id="h2", round_id=1, parameters={"w": [0.0, 1.0]})
    u_adv = ModelUpdate(agent_id="adv", round_id=1, parameters={"w": [-1.0, -1.0]})

    target = {"w": [1.0, 1.0]}

    shapley = ContributionEvaluator.calculate_shapley_values(
        [u1, u2, u_adv],
        target_gradient=target,
        num_monte_carlo_samples=50,
        seed=42,
    )

    assert "h1" in shapley
    assert "h2" in shapley
    assert "adv" in shapley

    # Honest agents contributing positively toward [1, 1] should have higher Shapley value than adversary
    assert shapley["h1"] > shapley["adv"]
    assert shapley["h2"] > shapley["adv"]

    # Edge cases
    assert ContributionEvaluator.calculate_shapley_values([], target) == {}
    single_res = ContributionEvaluator.calculate_shapley_values([u1], target)
    assert single_res == {"h1": 1.0}
