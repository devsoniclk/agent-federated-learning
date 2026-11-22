"""Unit tests for FederatedSwarmCoordinator lifecycle, quarantine, and round orchestration."""

import pytest

from agent_federated_learning.coordinator import FederatedSwarmCoordinator
from agent_federated_learning.models import (
    ByzantineDefenseType,
    DifferentialPrivacyConfig,
    ModelUpdate,
)


def test_coordinator_init():
    coord = FederatedSwarmCoordinator(model_id="test-model")
    assert coord.model_id == "test-model"
    assert coord.current_round_id == 0
    assert "dense.weight" in coord.global_parameters


def test_start_round_and_submit():
    coord = FederatedSwarmCoordinator()
    rnd = coord.start_round(
        defense=ByzantineDefenseType.KRUM, min_required_updates=3
    )
    assert rnd.round_id == 1
    assert rnd.status == "COLLECTING"

    # Submit updates
    u1 = ModelUpdate(agent_id="a1", round_id=1, parameters={"dense.weight": [1.0] * 10, "dense.bias": [0.1] * 10})
    res1 = coord.submit_update(u1)
    assert res1["accepted"] is True
    assert res1["total_received"] == 1


def test_submit_to_nonexistent_or_closed_round():
    coord = FederatedSwarmCoordinator()
    u = ModelUpdate(agent_id="a1", round_id=99, parameters={"w": [1.0]})
    with pytest.raises(ValueError):
        coord.submit_update(u)


def test_quarantine_enforcement():
    coord = FederatedSwarmCoordinator()
    coord.quarantine_agent("bad-node", reason="Malicious sign inversion")

    status = coord.get_status()
    assert status["quarantined_agents_count"] == 1
    assert "bad-node" in coord.quarantined_agents

    coord.start_round()
    u = ModelUpdate(agent_id="bad-node", round_id=1, parameters={"dense.weight": [1.0] * 10, "dense.bias": [0.1] * 10})
    res = coord.submit_update(u)
    assert res["accepted"] is False
    assert "quarantined" in res["reason"]

    # Unquarantine
    coord.unquarantine_agent("bad-node")
    res2 = coord.submit_update(u)
    assert res2["accepted"] is True


def test_aggregate_round_lifecycle():
    coord = FederatedSwarmCoordinator(
        initial_parameters={"w": [0.0, 0.0]},
        default_defense=ByzantineDefenseType.KRUM,
    )
    coord.start_round(min_required_updates=3)

    # 3 honest updates
    for i in range(1, 4):
        coord.submit_update(
            ModelUpdate(
                agent_id=f"honest-{i}",
                round_id=1,
                parameters={"w": [1.0, 1.0 + 0.1 * i]},
                sample_count=10,
            )
        )

    agg = coord.aggregate_round(round_id=1, byzantine_tolerance=1)
    assert agg.round_id == 1
    assert len(agg.participating_agents) == 3
    assert coord.rounds[1].status == "COMPLETED"
    assert len(coord.completed_rounds) == 1
    assert coord.get_global_model()["w"] is not None


def test_aggregate_insufficient_updates():
    coord = FederatedSwarmCoordinator()
    coord.start_round(min_required_updates=5)
    coord.submit_update(
        ModelUpdate(
            agent_id="a1",
            round_id=1,
            parameters={"dense.weight": [1.0] * 10, "dense.bias": [0.1] * 10},
        )
    )
    with pytest.raises(ValueError, match="Insufficient updates"):
        coord.aggregate_round(round_id=1)


def test_aggregate_with_byzantine_rejection():
    coord = FederatedSwarmCoordinator()
    coord.start_round(min_required_updates=3)

    # 2 honest
    coord.submit_update(ModelUpdate(agent_id="h1", round_id=1, parameters={"dense.weight": [1.0] * 10, "dense.bias": [0.1] * 10}))
    coord.submit_update(ModelUpdate(agent_id="h2", round_id=1, parameters={"dense.weight": [1.05] * 10, "dense.bias": [0.1] * 10}))
    # 1 inverted adversary
    coord.submit_update(ModelUpdate(agent_id="adv", round_id=1, parameters={"dense.weight": [-10.0] * 10, "dense.bias": [-10.0] * 10}))

    agg = coord.aggregate_round(round_id=1)
    assert "adv" in agg.rejected_agents
    assert "adv" in coord.quarantined_agents


def test_differential_privacy_round_tracking():
    dp_cfg = DifferentialPrivacyConfig(
        enabled=True, epsilon=1.0, delta=1e-5, clip_norm=1.0, noise_multiplier=0.5
    )
    coord = FederatedSwarmCoordinator(default_dp=dp_cfg)
    coord.start_round(min_required_updates=2)

    coord.submit_update(ModelUpdate(agent_id="h1", round_id=1, parameters={"dense.weight": [1.0] * 10, "dense.bias": [0.1] * 10}))
    coord.submit_update(ModelUpdate(agent_id="h2", round_id=1, parameters={"dense.weight": [1.2] * 10, "dense.bias": [0.1] * 10}))

    agg = coord.aggregate_round(round_id=1)
    assert agg.cumulative_epsilon > 0.0
    assert agg.cumulative_delta == 1e-5
