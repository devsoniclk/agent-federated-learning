"""Benchmark comparing execution throughput of Byzantine defense aggregations."""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import numpy as np

from agent_federated_learning.aggregation import aggregate
from agent_federated_learning.models import ByzantineDefenseType, ModelUpdate


def run_benchmark(num_agents: int = 25, param_dim: int = 1000, iterations: int = 20) -> None:
    print(f"Benchmarking with {num_agents} agents, {param_dim} parameters over {iterations} iterations:")

    # Generate synthetic updates
    updates = []
    for i in range(num_agents):
        arr = np.random.normal(0, 1.0, size=param_dim).tolist()
        updates.append(
            ModelUpdate(
                agent_id=f"agent-{i}",
                round_id=1,
                parameters={"layer1": arr},
                sample_count=10,
            )
        )

    defenses = [
        ByzantineDefenseType.FEDAVG,
        ByzantineDefenseType.KRUM,
        ByzantineDefenseType.MULTI_KRUM,
        ByzantineDefenseType.TRIMMED_MEAN,
        ByzantineDefenseType.COORDINATE_MEDIAN,
        ByzantineDefenseType.BULYAN,
    ]

    print(f"{'Defense':<20} | {'Total Time (s)':<15} | {'Avg Latency (ms)':<18} | {'Throughput (ops/s)':<18}")
    print("-" * 75)

    for defense in defenses:
        t0 = time.perf_counter()
        for _ in range(iterations):
            _ = aggregate(updates, defense=defense, num_byzantine=2)
        total_time = time.perf_counter() - t0
        avg_ms = (total_time / iterations) * 1000.0
        ops_per_sec = iterations / total_time
        print(f"{defense.value:<20} | {total_time:<15.4f} | {avg_ms:<18.2f} | {ops_per_sec:<18.2f}")


if __name__ == "__main__":
    run_benchmark()
