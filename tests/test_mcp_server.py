"""Unit tests for FastMCP server tools and JSON-RPC stdio protocol."""

import io
import json
import sys

from agent_federated_learning.mcp_server import (
    TOOLS,
    fl_aggregate_round,
    fl_create_round,
    fl_evaluate_contributions,
    fl_get_global_model,
    fl_get_swarm_status,
    fl_submit_update,
    fl_verify_privacy_budget,
    get_coordinator,
    run_stdio_server,
)


def test_mcp_tools_direct_invocation():
    coord = get_coordinator()
    coord.rounds.clear()
    coord.pending_updates.clear()
    coord.completed_rounds.clear()
    coord.current_round_id = 0

    # 1. fl_create_round
    r1 = fl_create_round(defense="krum", min_required_updates=2)
    assert r1["status"] == "success"
    assert r1["round_id"] == 1

    # 2. fl_submit_update
    params = {"dense.weight": [0.5] * 10, "dense.bias": [0.0] * 10}
    s1 = fl_submit_update(round_id=1, agent_id="agent-1", parameters=params)
    assert s1["accepted"] is True

    s2 = fl_submit_update(round_id=1, agent_id="agent-2", parameters=params)
    assert s2["accepted"] is True

    # 3. fl_evaluate_contributions
    eval_res = fl_evaluate_contributions(round_id=1)
    assert eval_res["round_id"] == 1
    assert eval_res["evaluated_count"] == 2

    # 4. fl_aggregate_round
    agg_res = fl_aggregate_round(round_id=1)
    assert agg_res["status"] == "success"
    assert agg_res["round_id"] == 1

    # 5. fl_get_global_model
    model = fl_get_global_model()
    assert "parameters" in model
    assert "layer_names" in model

    # 6. fl_verify_privacy_budget
    priv = fl_verify_privacy_budget()
    assert priv["completed_rounds"] == 1
    assert priv["dp_guarantee_valid"] is True

    # 7. fl_get_swarm_status
    status = fl_get_swarm_status()
    assert status["completed_rounds"] == 1


def test_mcp_evaluate_contributions_empty():
    eval_res = fl_evaluate_contributions(round_id=999)
    assert eval_res["status"] == "error"


def test_mcp_aggregate_round_error():
    res = fl_aggregate_round(round_id=9999)
    assert res["status"] == "error"


def test_mcp_stdio_jsonrpc_protocol():
    requests = [
        {"jsonrpc": "2.0", "method": "tools/list", "id": 1},
        {
            "jsonrpc": "2.0",
            "method": "fl_get_swarm_status",
            "params": {},
            "id": 2,
        },
        {"jsonrpc": "2.0", "method": "unknown_tool", "params": {}, "id": 3},
    ]

    stdin_data = "\n".join(json.dumps(r) for r in requests) + "\n"

    old_stdin = sys.stdin
    old_stdout = sys.stdout

    sys.stdin = io.StringIO(stdin_data)
    sys.stdout = io.StringIO()

    try:
        run_stdio_server()
        output = sys.stdout.getvalue()
        lines = [line for line in output.strip().split("\n") if line]
        assert len(lines) == 3

        # Response 1: tools/list
        resp1 = json.loads(lines[0])
        assert resp1["id"] == 1
        assert "tools" in resp1["result"]
        assert len(resp1["result"]["tools"]) == len(TOOLS)

        # Response 2: fl_get_swarm_status
        resp2 = json.loads(lines[1])
        assert resp2["id"] == 2
        assert "model_id" in resp2["result"]

        # Response 3: Method not found
        resp3 = json.loads(lines[2])
        assert resp3["id"] == 3
        assert "error" in resp3
        assert resp3["error"]["code"] == -32601
    finally:
        sys.stdin = old_stdin
        sys.stdout = old_stdout
