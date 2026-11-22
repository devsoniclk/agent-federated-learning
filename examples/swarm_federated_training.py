"""Demonstration of Byzantine-resilient federated learning across an AI agent swarm."""

from __future__ import annotations

import numpy as np
from agent_federated_learning import (
    ByzantineDefenseType,
    DifferentialPrivacyConfig,
    FederatedSwarmCoordinator,
    ModelUpdate,
)


def main() -> None:
    print("=== Agent Swarm Federated Learning Demo ===")

    # Initialize coordinator with Bulyan defense and Differential Privacy
    dp_config = DifferentialPrivacyConfig(
        enabled=True,
        epsilon=1.0,
        clip_norm=1.0,
        noise_multiplier=0.5,
    )
    coordinator = FederatedSwarmCoordinator(
        model_id="agent-decision-policy-v1",
        default_defense=ByzantineDefenseType.BULYAN,
        default_dp=dp_config,
    )

    training_round = coordinator.start_round(min_required_updates=5)
    print(f"Started Round {training_round.round_id} with Bulyan defense & DP.")

    # 4 Honest Agents submitting aligned gradients
    for i in range(4):
        honest_params = {
            "policy.weights": [float(w) for w in (np.ones(8) * 0.5 + np.random.normal(0, 0.05, 8))],
            "policy.bias": [0.05, 0.05],
        }
        update = ModelUpdate(
            agent_id=f"agent-worker-{i+1}",
            round_id=training_round.round_id,
            parameters=honest_params,
            sample_count=100,
        )
        coordinator.submit_update(update)
        print(f"  [+] Agent worker {i+1} submitted update.")

    # 1 Byzantine Attacker attempting gradient poisoning
    poison_params = {
        "policy.weights": [-100.0] * 8,
        "policy.bias": [-50.0, -50.0],
    }
    byzantine_update = ModelUpdate(
        agent_id="agent-adversary",
        round_id=training_round.round_id,
        parameters=poison_params,
        sample_count=100,
    )
    coordinator.submit_update(byzantine_update)
    print("  [!] Adversary injected poison update (weights: -100.0).")

    # Aggregate round with tolerance m=1
    result = coordinator.aggregate_round(round_id=training_round.round_id, byzantine_tolerance=1)

    print("\n--- Aggregation Completed ---")
    print(f"Defense Strategy: {result.defense_used.value}")
    print(f"Participating Agents: {result.participating_agents}")
    print(f"Rejected Adversaries: {result.rejected_agents}")
    print(f"Aggregated Weights (sample): {result.aggregated_parameters['policy.weights'][:4]}")
    print(f"Duration: {result.duration_ms:.2f} ms")


if __name__ == "__main__":
    main()
