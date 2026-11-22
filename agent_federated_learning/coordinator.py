"""Federated learning coordinator orchestrating rounds, verification, and model state."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Set
import numpy as np

from agent_federated_learning.models import (
    AggregationResult,
    ByzantineDefenseType,
    DifferentialPrivacyConfig,
    ModelUpdate,
    TrainingRound,
)
from agent_federated_learning.aggregation import aggregate, flatten_params, unflatten_params
from agent_federated_learning.privacy import DifferentialPrivacyEngine
from agent_federated_learning.verification import ContributionEvaluator


class FederatedSwarmCoordinator:
    """Central lifecycle coordinator for autonomous agent federated learning rounds."""

    def __init__(
        self,
        model_id: str = "agent-base-v1",
        initial_parameters: Optional[Dict[str, List[float]]] = None,
        default_defense: ByzantineDefenseType = ByzantineDefenseType.KRUM,
        default_dp: Optional[DifferentialPrivacyConfig] = None,
    ) -> None:
        self.model_id = model_id
        self.default_defense = default_defense
        self.default_dp = default_dp or DifferentialPrivacyConfig()

        # Initialize global parameters
        if initial_parameters is not None:
            self.global_parameters = initial_parameters
        else:
            # Default single layer weights for testing/scaffolding
            self.global_parameters = {"dense.weight": [0.0] * 10, "dense.bias": [0.0] * 10}

        self.current_round_id = 0
        self.rounds: Dict[int, TrainingRound] = {}
        self.pending_updates: Dict[int, List[ModelUpdate]] = {}
        self.quarantined_agents: Dict[str, str] = {}
        self.reputation_scores: Dict[str, float] = {}
        self.completed_rounds: List[AggregationResult] = []

    def start_round(
        self,
        defense: Optional[ByzantineDefenseType] = None,
        dp_config: Optional[DifferentialPrivacyConfig] = None,
        min_required_updates: int = 3,
    ) -> TrainingRound:
        """Initialize a new federated training round."""
        self.current_round_id += 1
        rid = self.current_round_id

        round_obj = TrainingRound(
            round_id=rid,
            status="COLLECTING",
            target_model_id=self.model_id,
            defense_type=defense or self.default_defense,
            dp_config=dp_config or self.default_dp,
            min_required_updates=min_required_updates,
        )

        self.rounds[rid] = round_obj
        self.pending_updates[rid] = []
        return round_obj

    def submit_update(self, update: ModelUpdate) -> Dict[str, Any]:
        """Validate and accept a local model gradient update from an agent."""
        rid = update.round_id
        if rid not in self.rounds:
            raise ValueError(f"Round {rid} does not exist.")

        round_obj = self.rounds[rid]
        if round_obj.status != "COLLECTING":
            raise ValueError(f"Round {rid} is in '{round_obj.status}' state, not COLLECTING.")

        # Check quarantine
        if update.agent_id in self.quarantined_agents:
            return {
                "accepted": False,
                "reason": f"Agent {update.agent_id} is quarantined: {self.quarantined_agents[update.agent_id]}",
            }

        # Apply Differential Privacy if enabled for this round and not applied locally
        if round_obj.dp_config.enabled and not update.dp_applied:
            perturbed_params, _ = DifferentialPrivacyEngine.apply_dp(
                update.parameters, round_obj.dp_config
            )
            update = ModelUpdate(
                agent_id=update.agent_id,
                round_id=update.round_id,
                parameters=perturbed_params,
                sample_count=update.sample_count,
                loss=update.loss,
                timestamp=update.timestamp,
                dp_applied=True,
                metadata=update.metadata,
            )

        self.pending_updates[rid].append(update)
        round_obj.num_updates_received = len(self.pending_updates[rid])

        return {
            "accepted": True,
            "round_id": rid,
            "agent_id": update.agent_id,
            "total_received": round_obj.num_updates_received,
        }

    def aggregate_round(
        self,
        round_id: int,
        byzantine_tolerance: int = 1
    ) -> AggregationResult:
        """Execute Byzantine-robust aggregation for the designated round and update global weights."""
        if round_id not in self.rounds:
            raise ValueError(f"Round {round_id} does not exist.")

        round_obj = self.rounds[round_id]
        if round_obj.status != "COLLECTING":
            raise ValueError(f"Round {round_id} status is '{round_obj.status}', expected COLLECTING.")

        updates = self.pending_updates.get(round_id, [])
        if len(updates) < round_obj.min_required_updates:
            raise ValueError(
                f"Insufficient updates for round {round_id}: {len(updates)} received, "
                f"minimum required {round_obj.min_required_updates}."
            )

        t_start = time.perf_counter()
        round_obj.status = "AGGREGATING"

        # Step 1: Pre-aggregation direction and norm sanity evaluation
        contributions = ContributionEvaluator.evaluate_updates(updates)
        valid_updates = []
        for c, u in zip(contributions, updates):
            if c.is_byzantine:
                self.quarantine_agent(c.agent_id, c.rejection_reason or "Byzantine anomaly detected")
                round_obj.num_byzantine_rejected += 1
            else:
                valid_updates.append(u)

        if not valid_updates:
            round_obj.status = "FAILED"
            raise RuntimeError(f"All submitted updates in round {round_id} were rejected as Byzantine.")

        # Step 2: Algorithmic Aggregation
        aggregated_weights, accepted_agents, rejected_agents = aggregate(
            valid_updates,
            defense=round_obj.defense_type,
            num_byzantine=byzantine_tolerance
        )

        # Step 3: Update global parameters with aggregated delta
        # If parameters represent deltas or full weights; here we apply new weights directly
        self.global_parameters = aggregated_weights
        round_obj.num_updates_accepted = len(accepted_agents)
        round_obj.status = "COMPLETED"
        round_obj.end_time = time.time()

        # Step 4: Privacy budget tracking
        cum_eps = 0.0
        cum_delta = 0.0
        if round_obj.dp_config.enabled:
            cum_eps = DifferentialPrivacyEngine.calculate_epsilon(
                num_rounds=round_id,
                noise_multiplier=round_obj.dp_config.noise_multiplier,
                target_delta=round_obj.dp_config.delta
            )
            cum_delta = round_obj.dp_config.delta

        duration_ms = (time.perf_counter() - t_start) * 1000.0

        result = AggregationResult(
            round_id=round_id,
            aggregated_parameters=aggregated_weights,
            defense_used=round_obj.defense_type,
            participating_agents=[u.agent_id for u in updates],
            rejected_agents=list(set(rejected_agents + [c.agent_id for c in contributions if c.is_byzantine])),
            cumulative_epsilon=cum_eps,
            cumulative_delta=cum_delta,
            duration_ms=duration_ms,
            metadata={"accepted_count": len(accepted_agents)},
        )

        self.completed_rounds.append(result)
        return result

    def quarantine_agent(self, agent_id: str, reason: str) -> None:
        """Place an adversarial or malfunctioning agent on the quarantine blacklist."""
        self.quarantined_agents[agent_id] = reason

    def unquarantine_agent(self, agent_id: str) -> None:
        """Restore an agent from quarantine."""
        self.quarantined_agents.pop(agent_id, None)

    def get_global_model(self) -> Dict[str, List[float]]:
        """Retrieve the current global model weights."""
        return self.global_parameters

    def get_status(self) -> Dict[str, Any]:
        """Summarize current coordinator health, active round, and swarm state."""
        return {
            "model_id": self.model_id,
            "current_round": self.current_round_id,
            "active_rounds": sum(1 for r in self.rounds.values() if r.status == "COLLECTING"),
            "completed_rounds": len(self.completed_rounds),
            "quarantined_agents_count": len(self.quarantined_agents),
            "quarantined_agents": self.quarantined_agents,
            "default_defense": self.default_defense.value,
        }
