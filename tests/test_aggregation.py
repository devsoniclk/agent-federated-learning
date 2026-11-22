"""Unit tests for federated learning aggregation algorithms and defenses."""

import numpy as np
import pytest

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
from agent_federated_learning.models import ByzantineDefenseType, ModelUpdate


def test_flatten_and_unflatten_params():
    params = {
        "layer1.weight": [[1.0, 2.0], [3.0, 4.0]],
        "layer1.bias": [0.5, -0.5],
    }
    vec, shapes = flatten_params(params)
    assert len(vec) == 6
    assert "layer1.weight" in shapes
    assert "layer1.bias" in shapes

    restored = unflatten_params(vec, shapes)
    assert np.allclose(restored["layer1.weight"], params["layer1.weight"])
    assert np.allclose(restored["layer1.bias"], params["layer1.bias"])


def test_flatten_empty():
    vec, shapes = flatten_params({})
    assert len(vec) == 0
    assert shapes == {}


def test_fedavg_basic():
    u1 = ModelUpdate(agent_id="a1", round_id=1, parameters={"w": [1.0, 2.0]}, sample_count=10)
    u2 = ModelUpdate(agent_id="a2", round_id=1, parameters={"w": [3.0, 6.0]}, sample_count=10)
    res = fedavg([u1, u2])
    assert np.allclose(res["w"], [2.0, 4.0])


def test_fedavg_weighted():
    u1 = ModelUpdate(agent_id="a1", round_id=1, parameters={"w": [1.0, 2.0]}, sample_count=10)
    u2 = ModelUpdate(agent_id="a2", round_id=1, parameters={"w": [3.0, 6.0]}, sample_count=30)
    # total samples 40; u1 weight 0.25, u2 weight 0.75
    # w[0] = 0.25*1 + 0.75*3 = 2.5
    # w[1] = 0.25*2 + 0.75*6 = 5.0
    res = fedavg([u1, u2])
    assert np.allclose(res["w"], [2.5, 5.0])


def test_fedavg_empty():
    with pytest.raises(ValueError):
        fedavg([])


def test_compute_pairwise_distances():
    v1 = np.array([0.0, 0.0])
    v2 = np.array([3.0, 4.0])
    v3 = np.array([1.0, 1.0])
    dists = compute_pairwise_distances([v1, v2, v3])
    assert dists.shape == (3, 3)
    assert dists[0, 0] == 0.0
    assert dists[0, 1] == 25.0  # 3^2 + 4^2 = 25
    assert dists[1, 0] == 25.0


def test_krum_single_and_defense():
    # 4 honest agents around [1.0, 1.0], 1 adversary at [100.0, 100.0]
    updates = [
        ModelUpdate(agent_id="h1", round_id=1, parameters={"w": [1.0, 1.0]}),
        ModelUpdate(agent_id="h2", round_id=1, parameters={"w": [1.1, 0.9]}),
        ModelUpdate(agent_id="h3", round_id=1, parameters={"w": [0.9, 1.1]}),
        ModelUpdate(agent_id="h4", round_id=1, parameters={"w": [1.0, 0.95]}),
        ModelUpdate(agent_id="adv", round_id=1, parameters={"w": [100.0, 100.0]}),
    ]
    # Single update edge case
    res_single, acc_s, rej_s = krum([updates[0]])
    assert acc_s == "h1"

    # Multi update krum
    params, acc, rej = krum(updates, num_byzantine=1)
    assert acc in ["h1", "h2", "h3", "h4"]
    assert "adv" in rej
    assert np.allclose(params["w"], [1.0, 1.0], atol=0.2)


def test_multi_krum():
    updates = [
        ModelUpdate(agent_id="h1", round_id=1, parameters={"w": [2.0, 2.0]}),
        ModelUpdate(agent_id="h2", round_id=1, parameters={"w": [2.1, 1.9]}),
        ModelUpdate(agent_id="h3", round_id=1, parameters={"w": [1.9, 2.1]}),
        ModelUpdate(agent_id="h4", round_id=1, parameters={"w": [2.0, 2.0]}),
        ModelUpdate(agent_id="adv", round_id=1, parameters={"w": [-50.0, -50.0]}),
    ]
    params, acc, rej = multi_krum(updates, num_byzantine=1)
    assert "adv" in rej
    assert len(acc) > 1
    assert np.allclose(params["w"], [2.0, 2.0], atol=0.15)


def test_trimmed_mean():
    # 5 updates for 1 scalar: -10, 2, 3, 4, 100
    updates = [
        ModelUpdate(agent_id="a1", round_id=1, parameters={"w": [-10.0]}),
        ModelUpdate(agent_id="a2", round_id=1, parameters={"w": [2.0]}),
        ModelUpdate(agent_id="a3", round_id=1, parameters={"w": [3.0]}),
        ModelUpdate(agent_id="a4", round_id=1, parameters={"w": [4.0]}),
        ModelUpdate(agent_id="a5", round_id=1, parameters={"w": [100.0]}),
    ]
    # Trim with beta=0.2 (trims 1 from each end) -> mean of [2.0, 3.0, 4.0] = 3.0
    params, acc, rej = trimmed_mean(updates, beta=0.2)
    assert np.allclose(params["w"], [3.0])
    assert len(acc) == 5

    # Direct aggregate with TRIMMED_MEAN
    p_agg, acc, rej = aggregate(updates, defense=ByzantineDefenseType.TRIMMED_MEAN, num_byzantine=1)
    assert "w" in p_agg
    assert len(acc) == 5

    # Edge cases
    with pytest.raises(ValueError):
        trimmed_mean([])

    # Single update
    p_single, acc_s, rej_s = trimmed_mean([updates[1]])
    assert p_single["w"] == [2.0]


def test_coordinate_median():
    updates = [
        ModelUpdate(agent_id="a1", round_id=1, parameters={"w": [1.0, 100.0]}),
        ModelUpdate(agent_id="a2", round_id=1, parameters={"w": [2.0, 5.0]}),
        ModelUpdate(agent_id="a3", round_id=1, parameters={"w": [3.0, 6.0]}),
    ]
    params, acc, rej = coordinate_median(updates)
    assert np.allclose(params["w"], [2.0, 6.0])

    # Via aggregate
    p_agg, acc, rej = aggregate(updates, defense=ByzantineDefenseType.COORDINATE_MEDIAN)
    assert np.allclose(p_agg["w"], [2.0, 6.0])
    assert len(acc) == 3

    with pytest.raises(ValueError):
        coordinate_median([])


def test_bulyan():
    # Bulyan combines Krum selection with coordinate-wise trimmed mean
    updates = [
        ModelUpdate(agent_id="h1", round_id=1, parameters={"w": [5.0, 5.0]}),
        ModelUpdate(agent_id="h2", round_id=1, parameters={"w": [5.1, 4.9]}),
        ModelUpdate(agent_id="h3", round_id=1, parameters={"w": [4.9, 5.1]}),
        ModelUpdate(agent_id="h4", round_id=1, parameters={"w": [5.0, 5.2]}),
        ModelUpdate(agent_id="h5", round_id=1, parameters={"w": [5.2, 4.8]}),
        ModelUpdate(agent_id="adv1", round_id=1, parameters={"w": [999.0, 999.0]}),
    ]
    params, acc, rej = bulyan(updates, num_byzantine=1)
    assert "adv1" in rej
    assert np.allclose(params["w"], [5.0, 5.0], atol=0.2)

    # Edge cases
    with pytest.raises(ValueError):
        bulyan([])

    # Fallback when updates <= 2
    p_fallback, acc_f, _ = bulyan([updates[0], updates[1]], num_byzantine=1)
    assert len(acc_f) == 2


def test_aggregate_dispatcher():
    updates = [
        ModelUpdate(agent_id="h1", round_id=1, parameters={"w": [1.0]}),
        ModelUpdate(agent_id="h2", round_id=1, parameters={"w": [1.2]}),
        ModelUpdate(agent_id="h3", round_id=1, parameters={"w": [0.8]}),
    ]
    # Check all enum types
    for defense in ByzantineDefenseType:
        params, acc, rej = aggregate(updates, defense=defense, num_byzantine=1)
        assert "w" in params
        assert len(acc) >= 1
