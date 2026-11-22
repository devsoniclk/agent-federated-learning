"""Byzantine-robust and classical federated learning aggregation algorithms."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np

from agent_federated_learning.models import ByzantineDefenseType, ModelUpdate


def flatten_params(
    params: Dict[str, List[float]]
) -> Tuple[np.ndarray, Dict[str, Tuple[int, int, Tuple[int, ...]]]]:
    """Flatten a dictionary of parameter tensors into a single 1D numpy array."""
    vectors: List[np.ndarray] = []
    shapes: Dict[str, Tuple[int, int, Tuple[int, ...]]] = {}
    curr_idx = 0

    for name in sorted(params.keys()):
        arr = np.asarray(params[name], dtype=np.float64)
        orig_shape = arr.shape
        flat = arr.ravel()
        length = len(flat)
        shapes[name] = (curr_idx, curr_idx + length, orig_shape)
        curr_idx += length
        vectors.append(flat)

    if not vectors:
        return np.array([], dtype=np.float64), {}

    return np.concatenate(vectors), shapes


def unflatten_params(
    flat_vector: np.ndarray,
    shapes: Dict[str, Tuple[int, int, Tuple[int, ...]]]
) -> Dict[str, List[float]]:
    """Reconstruct dictionary of parameter tensors from a flattened 1D array."""
    restored: Dict[str, List[float]] = {}
    for name, (start, end, orig_shape) in shapes.items():
        slice_arr = flat_vector[start:end]
        if orig_shape and len(orig_shape) > 1:
            restored[name] = slice_arr.reshape(orig_shape).tolist()
        else:
            restored[name] = slice_arr.tolist()
    return restored


def compute_pairwise_distances(vectors: List[np.ndarray]) -> np.ndarray:
    """Compute NxN Euclidean distance matrix between all flattened parameter vectors."""
    n = len(vectors)
    matrix = np.stack(vectors, axis=0)  # Shape (N, D)
    diff = matrix[:, np.newaxis, :] - matrix[np.newaxis, :, :]  # (N, N, D)
    dist_matrix = np.sum(diff ** 2, axis=-1)  # Squared Euclidean distances
    return dist_matrix


def fedavg(updates: List[ModelUpdate]) -> Dict[str, List[float]]:
    """Standard Federated Averaging (McMahan et al., 2017) weighted by sample counts."""
    if not updates:
        raise ValueError("Cannot aggregate empty list of updates.")

    total_samples = sum(u.sample_count for u in updates)
    if total_samples <= 0:
        total_samples = len(updates)
        weights = [1.0 / len(updates)] * len(updates)
    else:
        weights = [u.sample_count / total_samples for u in updates]

    flat_vectors = []
    shapes = None

    for u in updates:
        vec, shp = flatten_params(u.parameters)
        flat_vectors.append(vec)
        if shapes is None:
            shapes = shp

    accum = np.zeros_like(flat_vectors[0])
    for w, vec in zip(weights, flat_vectors):
        accum += w * vec

    assert shapes is not None
    return unflatten_params(accum, shapes)


def krum_select_indices(
    vectors: List[np.ndarray],
    num_byzantine: int,
    num_to_select: int = 1
) -> List[int]:
    """Select indices of vectors minimizing distance to their (n - m - 2) nearest neighbors."""
    n = len(vectors)
    if n <= 2:
        return list(range(min(num_to_select, n)))

    k_neighbors = n - num_byzantine - 2
    if k_neighbors <= 0:
        k_neighbors = 1
    k_neighbors = min(k_neighbors, n - 1)

    dist_matrix = compute_pairwise_distances(vectors)

    scores = []
    for i in range(n):
        row_dists = np.sort(dist_matrix[i])
        closest_dists = row_dists[1:k_neighbors + 1]
        score = float(np.sum(closest_dists))
        scores.append((score, i))

    scores.sort(key=lambda x: x[0])
    selected = [idx for _, idx in scores[:num_to_select]]
    return selected


def krum(
    updates: List[ModelUpdate],
    num_byzantine: int = 1
) -> Tuple[Dict[str, List[float]], str, List[str]]:
    """Single-choice Krum aggregation picking the vector minimizing neighbor Euclidean variance."""
    if not updates:
        raise ValueError("Cannot aggregate empty list of updates.")

    if len(updates) == 1:
        return updates[0].parameters, updates[0].agent_id, []

    flat_vectors = [flatten_params(u.parameters)[0] for u in updates]
    selected_idx = krum_select_indices(flat_vectors, num_byzantine, num_to_select=1)[0]

    chosen = updates[selected_idx]
    accepted_agent = chosen.agent_id
    rejected_agents = [u.agent_id for i, u in enumerate(updates) if i != selected_idx]

    return chosen.parameters, accepted_agent, rejected_agents


def multi_krum(
    updates: List[ModelUpdate],
    num_byzantine: int = 1,
    top_k: int = 2
) -> Tuple[Dict[str, List[float]], List[str], List[str]]:
    """Multi-Krum averaging the top-k candidate vectors selected by Krum."""
    if not updates:
        raise ValueError("Cannot aggregate empty list of updates.")

    if len(updates) <= 2:
        return fedavg(updates), [u.agent_id for u in updates], []

    top_k = max(1, min(top_k, len(updates) - num_byzantine))
    flat_vectors = [flatten_params(u.parameters)[0] for u in updates]
    selected_indices = krum_select_indices(flat_vectors, num_byzantine, num_to_select=top_k)

    chosen_updates = [updates[i] for i in selected_indices]
    accepted_agents = [u.agent_id for u in chosen_updates]
    rejected_agents = [u.agent_id for i, u in enumerate(updates) if i not in selected_indices]

    aggregated = fedavg(chosen_updates)
    return aggregated, accepted_agents, rejected_agents


def trimmed_mean(
    updates: List[ModelUpdate],
    beta: float = 0.1,
    num_byzantine: Optional[int] = None,
) -> Tuple[Dict[str, List[float]], List[str], List[str]]:
    """Coordinate-wise Trimmed Mean removing the top and bottom proportion of values."""
    if not updates:
        raise ValueError("Cannot aggregate empty list of updates.")

    n = len(updates)
    all_agents = [u.agent_id for u in updates]
    if n == 1:
        return updates[0].parameters, all_agents, []

    if num_byzantine is not None:
        trim_count = num_byzantine
    else:
        trim_count = int(np.floor(beta * n))

    trim_count = min(trim_count, (n - 1) // 2)

    flat_vectors, shapes = [], None
    for u in updates:
        vec, shp = flatten_params(u.parameters)
        flat_vectors.append(vec)
        if shapes is None:
            shapes = shp

    matrix = np.stack(flat_vectors, axis=0)  # Shape (N, D)
    matrix_sorted = np.sort(matrix, axis=0)

    if trim_count > 0 and n > 2 * trim_count:
        trimmed = matrix_sorted[trim_count:n - trim_count, :]
    else:
        trimmed = matrix_sorted

    mean_vec = np.mean(trimmed, axis=0)
    assert shapes is not None
    result = unflatten_params(mean_vec, shapes)
    return result, all_agents, []


def coordinate_median(
    updates: List[ModelUpdate]
) -> Tuple[Dict[str, List[float]], List[str], List[str]]:
    """Coordinate-wise Median aggregation robust against extreme gradient outliers."""
    if not updates:
        raise ValueError("Cannot aggregate empty list of updates.")

    all_agents = [u.agent_id for u in updates]
    flat_vectors, shapes = [], None
    for u in updates:
        vec, shp = flatten_params(u.parameters)
        flat_vectors.append(vec)
        if shapes is None:
            shapes = shp

    matrix = np.stack(flat_vectors, axis=0)
    median_vec = np.median(matrix, axis=0)
    assert shapes is not None
    result = unflatten_params(median_vec, shapes)
    return result, all_agents, []


def bulyan(
    updates: List[ModelUpdate],
    num_byzantine: int = 1
) -> Tuple[Dict[str, List[float]], List[str], List[str]]:
    """Bulyan aggregation combining recursive Krum selection with coordinate-wise trimmed mean.
    
    Guarantees optimal Byzantine robustness when n >= 4m + 3 (El Mhamdi et al.).
    """
    n = len(updates)
    if n <= 3:
        return fedavg(updates), [u.agent_id for u in updates], []

    flat_vectors, shapes = [], None
    for u in updates:
        vec, shp = flatten_params(u.parameters)
        flat_vectors.append(vec)
        if shapes is None:
            shapes = shp

    theta = max(1, n - 2 * num_byzantine)
    selected_indices: List[int] = []
    remaining_indices = list(range(n))

    for _ in range(min(theta, len(remaining_indices))):
        sub_vectors = [flat_vectors[i] for i in remaining_indices]
        chosen_sub_idx = krum_select_indices(sub_vectors, num_byzantine=num_byzantine, num_to_select=1)[0]
        actual_idx = remaining_indices[chosen_sub_idx]
        selected_indices.append(actual_idx)
        remaining_indices.remove(actual_idx)

    candidate_vectors = np.stack([flat_vectors[i] for i in selected_indices], axis=0)
    gamma = min(num_byzantine, (len(selected_indices) - 1) // 2)
    sorted_candidates = np.sort(candidate_vectors, axis=0)

    if gamma > 0 and len(selected_indices) > 2 * gamma:
        trimmed = sorted_candidates[gamma:len(selected_indices) - gamma, :]
    else:
        trimmed = sorted_candidates

    aggregated_vec = np.mean(trimmed, axis=0)
    assert shapes is not None
    result = unflatten_params(aggregated_vec, shapes)

    accepted_agents = [updates[i].agent_id for i in selected_indices]
    rejected_agents = [u.agent_id for i, u in enumerate(updates) if i not in selected_indices]

    return result, accepted_agents, rejected_agents


def aggregate(
    updates: List[ModelUpdate],
    defense: ByzantineDefenseType = ByzantineDefenseType.KRUM,
    num_byzantine: int = 1
) -> Tuple[Dict[str, List[float]], List[str], List[str]]:
    """Dispatch aggregation according to the specified Byzantine defense strategy."""
    all_agents = [u.agent_id for u in updates]

    if defense == ByzantineDefenseType.FEDAVG:
        return fedavg(updates), all_agents, []
    elif defense == ByzantineDefenseType.KRUM:
        res, acc, rej = krum(updates, num_byzantine=num_byzantine)
        return res, [acc], rej
    elif defense == ByzantineDefenseType.MULTI_KRUM:
        return multi_krum(updates, num_byzantine=num_byzantine)
    elif defense == ByzantineDefenseType.TRIMMED_MEAN:
        return trimmed_mean(updates, num_byzantine=num_byzantine)
    elif defense == ByzantineDefenseType.COORDINATE_MEDIAN:
        return coordinate_median(updates)
    elif defense == ByzantineDefenseType.BULYAN:
        return bulyan(updates, num_byzantine=num_byzantine)
    else:
        return fedavg(updates), all_agents, []
