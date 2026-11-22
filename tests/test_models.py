"""Unit tests for agent_federated_learning data models."""

import pytest
from agent_federated_learning.models import (
    AggregationResult,
    AgentContribution,
    ByzantineDefenseType,
    DifferentialPrivacyConfig,
    ModelUpdate,
    SMPCConfig,
    TrainingRound,
)


def test_byzantine_defense_type_enum():
    assert ByzantineDefenseType.FEDAVG.value == "fedavg"
    assert ByzantineDefenseType.KRUM.value == "krum"
    assert ByzantineDefenseType.MULTI_KRUM.value == "multi_krum"
    assert ByzantineDefenseType.TRIMMED_MEAN.value == "trimmed_mean"
    assert ByzantineDefenseType.COORDINATE_MEDIAN.value == "coordinate_median"
    assert ByzantineDefenseType.BULYAN.value == "bulyan"


def test_dp_config_validation():
    cfg = DifferentialPrivacyConfig(
        enabled=True, epsilon=1.5, delta=1e-5, clip_norm=2.0, noise_multiplier=0.8
    )
    assert cfg.enabled is True
    assert cfg.epsilon == 1.5
    assert cfg.delta == 1e-5
    assert cfg.clip_norm == 2.0
    assert cfg.noise_multiplier == 0.8

    # Test constraints
    with pytest.raises(Exception):
        DifferentialPrivacyConfig(epsilon=0.0001)  # ge=0.01

    with pytest.raises(Exception):
        DifferentialPrivacyConfig(delta=0.5)  # le=1e-3


def test_smpc_config_validation():
    cfg = SMPCConfig(enabled=True, num_parties=5, modulus_bound=2e6)
    assert cfg.num_parties == 5
    assert cfg.modulus_bound == 2e6

    with pytest.raises(Exception):
        SMPCConfig(num_parties=1)  # ge=2


def test_model_update_creation():
    update = ModelUpdate(
        agent_id="agent-01",
        round_id=1,
        parameters={"w": [0.1, 0.2, 0.3], "b": [0.05]},
        sample_count=50,
        loss=0.12,
    )
    assert update.agent_id == "agent-01"
    assert update.round_id == 1
    assert update.sample_count == 50
    assert update.loss == 0.12
    assert update.dp_applied is False
    assert len(update.parameters["w"]) == 3


def test_agent_contribution_model():
    contrib = AgentContribution(
        agent_id="agent-02",
        round_id=3,
        sample_count=100,
        cosine_similarity=0.95,
        shapley_value=0.25,
        reputation_score=0.98,
        is_byzantine=False,
    )
    assert contrib.cosine_similarity == 0.95
    assert contrib.is_byzantine is False
    assert contrib.rejection_reason is None


def test_training_round_and_aggregation_result():
    rnd = TrainingRound(
        round_id=10,
        target_model_id="agent-v1",
        defense_type=ByzantineDefenseType.BULYAN,
        min_required_updates=4,
    )
    assert rnd.status == "PENDING"
    assert rnd.defense_type == ByzantineDefenseType.BULYAN

    res = AggregationResult(
        round_id=10,
        aggregated_parameters={"w": [1.0, 2.0]},
        defense_used=ByzantineDefenseType.BULYAN,
        participating_agents=["agent-01", "agent-02"],
        rejected_agents=["bad-agent-99"],
        cumulative_epsilon=0.85,
        cumulative_delta=1e-5,
        duration_ms=12.5,
    )
    assert res.round_id == 10
    assert len(res.participating_agents) == 2
    assert res.rejected_agents == ["bad-agent-99"]
