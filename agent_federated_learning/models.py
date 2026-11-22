"""Data models for privacy-preserving federated agent learning."""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ByzantineDefenseType(str, Enum):
    """Supported aggregation defense mechanisms against Byzantine adversaries."""
    FEDAVG = "fedavg"
    KRUM = "krum"
    MULTI_KRUM = "multi_krum"
    TRIMMED_MEAN = "trimmed_mean"
    COORDINATE_MEDIAN = "coordinate_median"
    BULYAN = "bulyan"


class DifferentialPrivacyConfig(BaseModel):
    """Configuration for local and central differential privacy guarantees."""
    enabled: bool = False
    epsilon: float = Field(default=1.0, ge=0.01, description="Target privacy parameter epsilon")
    delta: float = Field(default=1e-5, ge=1e-9, le=1e-3, description="Target privacy parameter delta")
    clip_norm: float = Field(default=1.0, ge=0.001, description="L2 gradient clipping bound")
    noise_multiplier: float = Field(default=0.5, ge=0.0, description="Noise multiplier sigma = noise_std / clip_norm")


class SMPCConfig(BaseModel):
    """Configuration for Secure Multiparty Computation additive masking."""
    enabled: bool = False
    num_parties: int = Field(default=3, ge=2, description="Number of participating agents in secret sharing")
    modulus_bound: float = Field(default=1e6, ge=1e3, description="Additive noise ceiling bound")


class ModelUpdate(BaseModel):
    """Payload representing an individual agent's model gradient or parameter update."""
    agent_id: str
    round_id: int
    parameters: Dict[str, List[float]]
    sample_count: int = Field(default=1, ge=1)
    loss: Optional[float] = None
    timestamp: float = Field(default_factory=time.time)
    dp_applied: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentContribution(BaseModel):
    """Evaluated contribution score and trust telemetry for an agent update."""
    agent_id: str
    round_id: int
    sample_count: int
    cosine_similarity: float = 1.0
    shapley_value: float = 0.0
    reputation_score: float = 1.0
    is_byzantine: bool = False
    rejection_reason: Optional[str] = None


class TrainingRound(BaseModel):
    """Metadata tracking a federated training epoch across the agent swarm."""
    round_id: int
    status: str = "PENDING"  # PENDING, COLLECTING, AGGREGATING, COMPLETED, FAILED
    target_model_id: str
    defense_type: ByzantineDefenseType = ByzantineDefenseType.KRUM
    dp_config: DifferentialPrivacyConfig = Field(default_factory=DifferentialPrivacyConfig)
    start_time: float = Field(default_factory=time.time)
    end_time: Optional[float] = None
    min_required_updates: int = 3
    num_updates_received: int = 0
    num_updates_accepted: int = 0
    num_byzantine_rejected: int = 0


class AggregationResult(BaseModel):
    """Outcome of a completed federated aggregation step."""
    round_id: int
    aggregated_parameters: Dict[str, List[float]]
    defense_used: ByzantineDefenseType
    participating_agents: List[str]
    rejected_agents: List[str]
    cumulative_epsilon: float = 0.0
    cumulative_delta: float = 0.0
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
