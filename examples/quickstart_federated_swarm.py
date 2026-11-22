"""End-to-end simulation of a federated agent swarm with Byzantine defense and differential privacy."""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from agent_federated_learning.coordinator import FederatedSwarmCoordinator
from agent_federated_learning.models import (
    ByzantineDefenseType,
    DifferentialPrivacyConfig,
    ModelUpdate,
)


def run_simulation() -> None:
    print("=== Initializing Federated Swarm Coordinator ===")
    dp_cfg = DifferentialPrivacyConfig(
        enabled=True,
        epsilon=2.0,
        delta=1e-5,
        clip_norm=1.0,
        noise_multiplier=0.4,
    )
    coord = FederatedSwarmCoordinator(
        model_id="swarm-linear-regressor",
        initial_parameters={"weights": [0.0, 0.0, 0.0], "bias": [0.0]},
        default_defense=ByzantineDefenseType.BULYAN,
        default_dp=dp_cfg,
    )

    true_target = np.array([2.5, -1.0, 0.5])

    for round_idx in range(1, 4):
        print(f"\n--- Starting Round {round_idx} ---")
        rnd = coord.start_round(min_required_updates=5)

        # Generate updates from 4 honest agents
        for i in range(1, 5):
            noise = np.random.normal(0, 0.1, size=3)
            agent_grad = (true_target + noise).tolist()
            update = ModelUpdate(
                agent_id=f"agent-worker-{i}",
                round_id=round_idx,
                parameters={"weights": agent_grad, "bias": [0.1 * i]},
                sample_count=100,
                loss=0.15,
            )
            res = coord.submit_update(update)
            print(f"Submitted honest update from agent-worker-{i}: accepted={res['accepted']}")

        # Generate update from 1 Byzantine adversary (extreme sign-inversion attack)
        byz_update = ModelUpdate(
            agent_id="adversary-agent-99",
            round_id=round_idx,
            parameters={"weights": [-50.0, 20.0, -10.0], "bias": [-100.0]},
            sample_count=500,
            loss=99.9,
        )
        res_byz = coord.submit_update(byz_update)
        print(f"Submitted adversary update: accepted={res_byz['accepted']}")

        # Aggregate round
        try:
            agg = coord.aggregate_round(round_id=round_idx, byzantine_tolerance=1)
            print(f"Round {round_idx} Aggregated Successfully in {agg.duration_ms:.2f}ms!")
            print(f"  Defense Used: {agg.defense_used.value}")
            print(f"  Accepted Agents: {agg.participating_agents}")
            print(f"  Rejected / Quarantined: {agg.rejected_agents}")
            print(f"  New Global Weights: {coord.get_global_model()['weights']}")
            print(f"  Cumulative DP Epsilon: {agg.cumulative_epsilon:.4f}")
        except Exception as e:
            print(f"Round {round_idx} aggregation error: {e}")

    print("\n=== Final Swarm Status ===")
    status = coord.get_status()
    print(status)


if __name__ == "__main__":
    run_simulation()
