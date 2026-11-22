"""FastMCP stdio JSON-RPC 2.0 server exposing federated learning tools to AI agents."""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, Optional

from agent_federated_learning.coordinator import FederatedSwarmCoordinator
from agent_federated_learning.models import (
    ByzantineDefenseType,
    DifferentialPrivacyConfig,
    ModelUpdate,
)
from agent_federated_learning.verification import ContributionEvaluator

# Singleton coordinator instance for the MCP server session
_COORDINATOR = FederatedSwarmCoordinator()


def get_coordinator() -> FederatedSwarmCoordinator:
    """Return the active singleton coordinator."""
    return _COORDINATOR


def fl_create_round(
    target_model_id: str = "agent-base-v1",
    defense: Optional[str] = None,
    defense_type: Optional[str] = None,
    epsilon: float = 1.0,
    clip_norm: float = 1.0,
    noise_multiplier: float = 0.5,
    min_required_updates: int = 3,
    dp_enabled: bool = False,
) -> Dict[str, Any]:
    """Initialize a new federated training round for the agent swarm."""
    coord = get_coordinator()
    coord.model_id = target_model_id

    defense_name = defense or defense_type or "krum"
    try:
        def_type = ByzantineDefenseType(defense_name.lower())
    except ValueError:
        def_type = ByzantineDefenseType.KRUM

    dp_cfg = DifferentialPrivacyConfig(
        enabled=dp_enabled,
        epsilon=epsilon,
        clip_norm=clip_norm,
        noise_multiplier=noise_multiplier,
    )

    round_obj = coord.start_round(
        defense=def_type,
        dp_config=dp_cfg,
        min_required_updates=min_required_updates,
    )

    return {
        "status": "success",
        "round_id": round_obj.round_id,
        "target_model_id": round_obj.target_model_id,
        "defense_type": round_obj.defense_type.value,
        "dp_enabled": round_obj.dp_config.enabled,
        "min_required_updates": round_obj.min_required_updates,
    }


def fl_submit_update(
    round_id: int,
    agent_id: str,
    parameters: Dict[str, List[float]],
    sample_count: int = 1,
    loss: Optional[float] = None,
) -> Dict[str, Any]:
    """Submit an agent's local model gradient or parameter update to an active round."""
    coord = get_coordinator()
    update = ModelUpdate(
        agent_id=agent_id,
        round_id=round_id,
        parameters=parameters,
        sample_count=sample_count,
        loss=loss,
    )
    result = coord.submit_update(update)
    return result


def fl_aggregate_round(
    round_id: int,
    byzantine_tolerance: int = 1,
) -> Dict[str, Any]:
    """Execute Byzantine-robust aggregation on all accepted updates and update global model."""
    coord = get_coordinator()
    try:
        agg_res = coord.aggregate_round(round_id=round_id, byzantine_tolerance=byzantine_tolerance)
        return {
            "status": "success",
            "round_id": agg_res.round_id,
            "defense_used": agg_res.defense_used.value,
            "participating_agents": agg_res.participating_agents,
            "rejected_agents": agg_res.rejected_agents,
            "duration_ms": round(agg_res.duration_ms, 2),
            "cumulative_epsilon": agg_res.cumulative_epsilon,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def fl_get_global_model() -> Dict[str, Any]:
    """Retrieve the current aggregated global model weights."""
    coord = get_coordinator()
    weights = coord.get_global_model()
    return {
        "model_id": coord.model_id,
        "layer_names": list(weights.keys()),
        "parameters": weights,
    }


def fl_evaluate_contributions(round_id: int) -> Dict[str, Any]:
    """Audit submitted updates for gradient inversion and norm explosions."""
    coord = get_coordinator()
    updates = coord.pending_updates.get(round_id, [])
    if not updates:
        return {"status": "error", "message": f"No pending updates found for round {round_id}"}

    contributions = ContributionEvaluator.evaluate_updates(updates)
    return {
        "round_id": round_id,
        "evaluated_count": len(contributions),
        "contributions": [c.model_dump() for c in contributions],
    }


def fl_verify_privacy_budget() -> Dict[str, Any]:
    """Audit the cumulative differential privacy spending across all completed rounds."""
    coord = get_coordinator()
    total_rounds = len(coord.completed_rounds)
    latest_eps = coord.completed_rounds[-1].cumulative_epsilon if total_rounds > 0 else 0.0
    latest_delta = coord.completed_rounds[-1].cumulative_delta if total_rounds > 0 else 0.0
    return {
        "completed_rounds": total_rounds,
        "cumulative_epsilon": latest_eps,
        "cumulative_delta": latest_delta,
        "dp_guarantee_valid": latest_eps < 10.0 if latest_eps > 0 else True,
    }


def fl_get_swarm_status() -> Dict[str, Any]:
    """Query current federated coordinator status, active rounds, and quarantined agents."""
    coord = get_coordinator()
    return coord.get_status()


# Tool registry mapping
TOOLS = {
    "fl_create_round": fl_create_round,
    "fl_submit_update": fl_submit_update,
    "fl_aggregate_round": fl_aggregate_round,
    "fl_get_global_model": fl_get_global_model,
    "fl_evaluate_contributions": fl_evaluate_contributions,
    "fl_verify_privacy_budget": fl_verify_privacy_budget,
    "fl_get_swarm_status": fl_get_swarm_status,
}


def run_stdio_server() -> None:
    """Run JSON-RPC 2.0 stdio server handling tool requests."""
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            method = req.get("method")
            params = req.get("params", {})
            req_id = req.get("id")

            if method == "tools/list":
                tools_list = [
                    {"name": k, "description": v.__doc__}
                    for k, v in TOOLS.items()
                ]
                resp = {"jsonrpc": "2.0", "result": {"tools": tools_list}, "id": req_id}
            elif method in TOOLS:
                tool_func = TOOLS[method]
                result = tool_func(**params)
                resp = {"jsonrpc": "2.0", "result": result, "id": req_id}
            else:
                resp = {
                    "jsonrpc": "2.0",
                    "error": {"code": -32601, "message": f"Method {method} not found"},
                    "id": req_id,
                }
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
        except Exception as err:
            err_resp = {
                "jsonrpc": "2.0",
                "error": {"code": -32000, "message": str(err)},
                "id": None,
            }
            sys.stdout.write(json.dumps(err_resp) + "\n")
            sys.stdout.flush()
