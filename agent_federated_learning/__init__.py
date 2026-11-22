"""Privacy-Preserving Federated Learning & Byzantine-Robust Swarm Model Aggregation Engine for Autonomous AI Agents."""

from __future__ import annotations

from agent_federated_learning.models import (
    AggregationResult,
    AgentContribution,
    ByzantineDefenseType,
    DifferentialPrivacyConfig,
    ModelUpdate,
    SMPCConfig,
    TrainingRound,
)
from agent_federated_learning.aggregation import (
    aggregate,
    bulyan,
    compute_pairwise_distances,
    coordinate_median,
    fedavg,
    flatten_params,
    krum,
    krum_select_indices,
    multi_krum,
    trimmed_mean,
    unflatten_params,
)
from agent_federated_learning.privacy import (
    AdditiveSecretSharing,
    DifferentialPrivacyEngine,
)
from agent_federated_learning.verification import ContributionEvaluator
from agent_federated_learning.coordinator import FederatedSwarmCoordinator

__version__ = "0.1.0"
__author__ = "Lakshitha Dananjaya (devsoniclk)"
__all__ = [
    "AggregationResult",
    "AgentContribution",
    "ByzantineDefenseType",
    "DifferentialPrivacyConfig",
    "ModelUpdate",
    "SMPCConfig",
    "TrainingRound",
    "aggregate",
    "bulyan",
    "compute_pairwise_distances",
    "coordinate_median",
    "fedavg",
    "flatten_params",
    "krum",
    "krum_select_indices",
    "multi_krum",
    "trimmed_mean",
    "unflatten_params",
    "AdditiveSecretSharing",
    "DifferentialPrivacyEngine",
    "ContributionEvaluator",
    "FederatedSwarmCoordinator",
    "__version__",
]
