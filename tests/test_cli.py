"""Unit tests for agent-fl Click CLI commands."""

import json
from click.testing import CliRunner

from agent_federated_learning.cli import main
from agent_federated_learning.mcp_server import get_coordinator


def test_cli_version():
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "agent-fl" in result.output


def test_cli_status():
    runner = CliRunner()
    result = runner.invoke(main, ["status"])
    assert result.exit_code == 0
    assert "Federated Swarm Coordinator Status" in result.output


def test_cli_round_lifecycle():
    runner = CliRunner()
    coord = get_coordinator()
    coord.rounds.clear()
    coord.pending_updates.clear()
    coord.completed_rounds.clear()
    coord.current_round_id = 0

    # Create round
    res_create = runner.invoke(
        main,
        ["round", "create", "--defense", "krum", "--min-updates", "2", "--dp-enabled", "--epsilon", "1.5"],
    )
    assert res_create.exit_code == 0
    assert "Federated Round #1 Created" in res_create.output

    # Submit update via JSON string
    params_str = json.dumps({"dense.weight": [1.0] * 10, "dense.bias": [0.1] * 10})
    res_sub1 = runner.invoke(
        main,
        ["round", "submit", "--round-id", "1", "--agent-id", "agent-1", "--params-json", params_str],
    )
    assert res_sub1.exit_code == 0
    assert "Update accepted" in res_sub1.output

    # Submit second update
    res_sub2 = runner.invoke(
        main,
        ["round", "submit", "--round-id", "1", "--agent-id", "agent-2", "--params-json", params_str],
    )
    assert res_sub2.exit_code == 0

    # Evaluate updates
    res_eval = runner.invoke(main, ["evaluate", "--round-id", "1"])
    assert res_eval.exit_code == 0
    assert "Update Audit for Round #1" in res_eval.output

    # Aggregate round
    res_agg = runner.invoke(main, ["round", "aggregate", "--round-id", "1"])
    assert res_agg.exit_code == 0
    assert "Round #1 Aggregation Summary" in res_agg.output

    # Inspect global model
    res_model = runner.invoke(main, ["model", "get"])
    assert res_model.exit_code == 0
    assert "Global Model" in res_model.output

    res_model_json = runner.invoke(main, ["model", "get", "--json-out"])
    assert res_model_json.exit_code == 0
    assert "dense.weight" in res_model_json.output

    # Privacy audit
    res_priv = runner.invoke(main, ["privacy", "audit"])
    assert res_priv.exit_code == 0
    assert "Swarm Privacy Audit" in res_priv.output
