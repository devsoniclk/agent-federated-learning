"""Command-line interface for the Agent Federated Learning engine."""

from __future__ import annotations

import json
import sys
from typing import Optional
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from agent_federated_learning import (
    __version__,
    ByzantineDefenseType,
    DifferentialPrivacyConfig,
    FederatedSwarmCoordinator,
    ModelUpdate,
)
from agent_federated_learning.mcp_server import get_coordinator, run_stdio_server
from agent_federated_learning.verification import ContributionEvaluator

console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="agent-fl")
def main() -> None:
    """Agent Federated Learning CLI - Privacy-preserving swarm model aggregation."""
    pass


@main.command(name="mcp")
def run_mcp() -> None:
    """Run the FastMCP JSON-RPC 2.0 stdio server for AI agent integrations."""
    run_stdio_server()


@main.command(name="status")
def status() -> None:
    """Display current status and supported Byzantine defenses."""
    coord = get_coordinator()
    st = coord.get_status()
    console.print(Panel(
        f"[bold green]Federated Swarm Coordinator Status[/bold green]\n"
        f"Model ID: {st.get('model_id')}\n"
        f"Current Round: {st.get('current_round')}\n"
        f"Active Rounds: {st.get('active_rounds')}\n"
        f"Completed Rounds: {st.get('completed_rounds')}\n"
        f"Default Defense: {st.get('default_defense')}\n"
        f"Quarantined Agents: {st.get('quarantined_agents_count')}",
        title="Federated Swarm Coordinator Status",
        border_style="blue"
    ))


@main.group(name="round")
def round_group() -> None:
    """Manage federated training rounds."""
    pass


@round_group.command(name="create")
@click.option("--target-model", default="agent-base-v1", help="Target model identifier.")
@click.option("--defense", default="krum", help="Byzantine aggregation defense.")
@click.option("--min-updates", default=3, help="Minimum required updates before aggregation.")
@click.option("--dp-enabled", is_flag=True, default=False, help="Enable differential privacy.")
@click.option("--epsilon", default=1.0, help="DP privacy budget epsilon.")
@click.option("--clip-norm", default=1.0, help="DP gradient clipping norm.")
def round_create(target_model: str, defense: str, min_updates: int, dp_enabled: bool, epsilon: float, clip_norm: float) -> None:
    """Create a new federated training round."""
    coord = get_coordinator()
    coord.model_id = target_model
    try:
        def_type = ByzantineDefenseType(defense.lower())
    except ValueError:
        def_type = ByzantineDefenseType.KRUM

    dp_cfg = DifferentialPrivacyConfig(
        enabled=dp_enabled,
        epsilon=epsilon,
        clip_norm=clip_norm,
    )
    rnd = coord.start_round(defense=def_type, dp_config=dp_cfg, min_required_updates=min_updates)
    console.print(f"[bold green]Federated Round #{rnd.round_id} Created[/bold green] (defense: {rnd.defense_type.value}, min_updates: {rnd.min_required_updates})")


@round_group.command(name="submit")
@click.option("--round-id", required=True, type=int, help="Round ID.")
@click.option("--agent-id", required=True, help="Submitting agent ID.")
@click.option("--params-json", required=True, help="JSON string of parameter dictionary.")
@click.option("--sample-count", default=1, help="Number of training samples.")
def round_submit(round_id: int, agent_id: str, params_json: str, sample_count: int) -> None:
    """Submit local parameter update to an active round."""
    coord = get_coordinator()
    params = json.loads(params_json)
    update = ModelUpdate(
        agent_id=agent_id,
        round_id=round_id,
        parameters=params,
        sample_count=sample_count,
    )
    res = coord.submit_update(update)
    if res["accepted"]:
        console.print(f"[green]Update accepted[/green] for round #{round_id} from agent [bold]{agent_id}[/bold]")
    else:
        console.print(f"[red]Update rejected[/red]: {res.get('reason')}")


@round_group.command(name="aggregate")
@click.option("--round-id", required=True, type=int, help="Round ID to aggregate.")
@click.option("--byzantine-tolerance", default=1, help="Estimated number of Byzantine adversaries.")
def round_aggregate(round_id: int, byzantine_tolerance: int) -> None:
    """Trigger Byzantine-robust aggregation for a round."""
    coord = get_coordinator()
    res = coord.aggregate_round(round_id=round_id, byzantine_tolerance=byzantine_tolerance)
    console.print(Panel(
        f"[bold green]Round #{round_id} Aggregation Summary[/bold green]\n"
        f"Defense Used: {res.defense_used.value}\n"
        f"Accepted Count: {res.metadata.get('accepted_count')}\n"
        f"Participating: {res.participating_agents}\n"
        f"Rejected: {res.rejected_agents}\n"
        f"Duration: {res.duration_ms:.2f} ms",
        title=f"Round #{round_id} Aggregation Summary",
        border_style="green"
    ))


@main.command(name="evaluate")
@click.option("--round-id", required=True, type=int, help="Round ID to evaluate.")
def evaluate(round_id: int) -> None:
    """Audit pending updates in a round for gradient inversion and anomalies."""
    coord = get_coordinator()
    updates = coord.pending_updates.get(round_id, [])
    if not updates:
        console.print(f"[yellow]No updates found for round #{round_id}[/yellow]")
        return
    contribs = ContributionEvaluator.evaluate_updates(updates)
    console.print(f"[bold cyan]Update Audit for Round #{round_id}[/bold cyan] ({len(contribs)} updates evaluated)")
    for c in contribs:
        status = "[red]BYZANTINE[/red]" if c.is_byzantine else "[green]VALID[/green]"
        console.print(f"  {status} Agent: {c.agent_id} | Cosine: {c.cosine_similarity:.3f} | Reputation: {c.reputation_score:.2f}")


@main.group(name="model")
def model_group() -> None:
    """Inspect and manage global swarm model weights."""
    pass


@model_group.command(name="get")
@click.option("--json-out", is_flag=True, default=False, help="Output raw JSON.")
def model_get(json_out: bool) -> None:
    """Retrieve current global model weights."""
    coord = get_coordinator()
    weights = coord.get_global_model()
    if json_out:
        console.print(json.dumps(weights))
    else:
        console.print(Panel(
            f"[bold]Global Model: {coord.model_id}[/bold]\n"
            f"Layers: {list(weights.keys())}\n"
            f"Parameters: {sum(len(v) for v in weights.values())} total coordinates",
            title="Global Model"
        ))


@main.group(name="privacy")
def privacy_group() -> None:
    """Audit privacy budget and Differential Privacy telemetry."""
    pass


@privacy_group.command(name="audit")
def privacy_audit() -> None:
    """Audit cumulative differential privacy spending across rounds."""
    coord = get_coordinator()
    total_rounds = len(coord.completed_rounds)
    latest_eps = coord.completed_rounds[-1].cumulative_epsilon if total_rounds > 0 else 0.0
    console.print(Panel(
        f"[bold]Swarm Privacy Audit[/bold]\n"
        f"Completed Rounds: {total_rounds}\n"
        f"Cumulative Epsilon: {latest_eps:.4f}",
        title="Swarm Privacy Audit"
    ))


@main.command(name="demo")
@click.option("--agents", default=5, help="Number of participating agents.")
@click.option("--byzantine", default=1, help="Number of Byzantine attackers.")
@click.option("--defense", default="krum", help="Defense rule.")
def demo_cmd(agents: int, byzantine: int, defense: str) -> None:
    """Run self-contained swarm simulation."""
    import numpy as np
    from agent_federated_learning import ByzantineDefenseType, FederatedSwarmCoordinator, ModelUpdate
    coord = FederatedSwarmCoordinator(default_defense=ByzantineDefenseType(defense))
    rnd = coord.start_round(min_required_updates=agents)
    for i in range(agents - byzantine):
        coord.submit_update(ModelUpdate(agent_id=f"honest-{i}", round_id=rnd.round_id, parameters={"w": [1.0] * 5}))
    for j in range(byzantine):
        coord.submit_update(ModelUpdate(agent_id=f"byzantine-{j}", round_id=rnd.round_id, parameters={"w": [-100.0] * 5}))
    res = coord.aggregate_round(round_id=rnd.round_id, byzantine_tolerance=byzantine)
    console.print(f"Simulation completed: {res.metadata.get('accepted_count')} updates accepted, {res.rejected_agents} rejected.")


if __name__ == "__main__":
    main()
