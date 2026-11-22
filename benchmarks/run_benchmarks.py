"""Benchmark aggregation throughput across different swarm sizes and defense rules."""

from __future__ import annotations

import time
import numpy as np
from agent_federated_learning import (
    ByzantineDefenseType,
    ModelUpdate,
    aggregate,
)


def run_benchmark() -> None:
    print("=== Agent Federated Learning Aggregation Benchmarks ===")

    tensor_dim = 1000
    param_dict = {"weights": [0.1] * tensor_dim}

    swarm_sizes = [5, 10, 25, 50]
    defenses = [
        ByzantineDefenseType.FEDAVG,
        ByzantineDefenseType.COORDINATE_MEDIAN,
        ByzantineDefenseType.TRIMMED_MEAN,
        ByzantineDefenseType.KRUM,
        ByzantineDefenseType.MULTI_KRUM,
        ByzantineDefenseType.BULYAN,
    ]

    for n in swarm_sizes:
        print(f"\n--- Swarm Size: {n} Agents (Dimension: {tensor_dim}) ---")
        updates = [
            ModelUpdate(
                agent_id=f"agent-{i}",
                round_id=1,
                parameters={"weights": [float(x) for x in (np.ones(tensor_dim) * 0.1 + np.random.normal(0, 0.01, tensor_dim))]},
                sample_count=10,
            )
            for i in range(n)
        ]

        for defense in defenses:
            m = max(1, n // 5)
            # Warmup
            try:
                aggregate(updates, defense=defense, num_byzantine=m)
            except Exception:
                continue

            iterations = 10
            t0 = time.perf_counter()
            for _ in range(iterations):
                aggregate(updates, defense=defense, num_byzantine=m)
            t1 = time.perf_counter()

            avg_ms = ((t1 - t0) / iterations) * 1000.0
            throughput = n / ((t1 - t0) / iterations)
            print(f"  {defense.value:20s}: {avg_ms:8.2f} ms | {throughput:8.0f} updates/sec")


if __name__ == "__main__":
    run_benchmark()
