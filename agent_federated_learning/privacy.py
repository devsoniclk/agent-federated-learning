"""Differential privacy and secure multiparty additive secret sharing engine."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from agent_federated_learning.models import DifferentialPrivacyConfig
from agent_federated_learning.aggregation import flatten_params, unflatten_params


class DifferentialPrivacyEngine:
    """Manages gradient L2 clipping, calibrated Gaussian perturbation, and RDP accounting."""

    @staticmethod
    def clip_l2_norm(
        target: Union[np.ndarray, Dict[str, List[float]]],
        clip_norm: Optional[float] = None,
        max_norm: Optional[float] = None,
    ) -> Tuple[Any, float]:
        """Clip vector or parameter dictionary to maximum L2 norm bound and return factor."""
        limit = max_norm if max_norm is not None else clip_norm
        if limit is None or limit <= 0:
            raise ValueError(f"Clipping norm must be strictly positive, got {limit}")

        if isinstance(target, np.ndarray):
            orig_norm = float(np.linalg.norm(target))
            if orig_norm > limit:
                factor = limit / (orig_norm + 1e-12)
                return target * factor, factor
            return target.copy(), 1.0
        else:
            flat_vec, shapes = flatten_params(target)
            orig_norm = float(np.linalg.norm(flat_vec))
            if orig_norm > limit:
                factor = limit / (orig_norm + 1e-12)
                clipped_flat = flat_vec * factor
                return unflatten_params(clipped_flat, shapes), factor
            return target, 1.0

    @staticmethod
    def add_gaussian_noise(
        target: Union[np.ndarray, Dict[str, List[float]]],
        clip_norm: float = 1.0,
        noise_multiplier: float = 0.5,
        seed: Optional[int] = None,
    ) -> Any:
        """Add calibrated Gaussian noise scaled to the sensitivity bound (sigma = noise_multiplier * clip_norm)."""
        if noise_multiplier <= 0.0:
            if isinstance(target, np.ndarray):
                return target.copy()
            return target

        rng = np.random.default_rng(seed)
        sigma = float(noise_multiplier * clip_norm)

        if isinstance(target, np.ndarray):
            noise = rng.normal(loc=0.0, scale=sigma, size=target.shape)
            return target + noise
        else:
            flat_vec, shapes = flatten_params(target)
            noise = rng.normal(loc=0.0, scale=sigma, size=flat_vec.shape)
            perturbed = flat_vec + noise
            return unflatten_params(perturbed, shapes)

    @classmethod
    def apply_dp(
        cls,
        params: Dict[str, List[float]],
        config: DifferentialPrivacyConfig,
        seed: Optional[int] = None,
    ) -> Tuple[Dict[str, List[float]], float]:
        """Apply full Local DP pipeline: clipping followed by Gaussian noise injection."""
        if not config.enabled:
            return params, 1.0

        clipped, factor = cls.clip_l2_norm(params, clip_norm=config.clip_norm)
        perturbed = cls.add_gaussian_noise(
            clipped,
            clip_norm=config.clip_norm,
            noise_multiplier=config.noise_multiplier,
            seed=seed,
        )
        return perturbed, factor

    @staticmethod
    def calculate_epsilon(
        num_rounds: int,
        noise_multiplier: float,
        target_delta: float = 1e-5,
    ) -> float:
        """Estimate cumulative privacy budget epsilon using standard Gaussian mechanism composition."""
        if num_rounds <= 0:
            return 0.0
        if noise_multiplier <= 0.0:
            return float("inf")

        log_term = math.log(1.25 / max(1e-12, target_delta))
        eps = (math.sqrt(2.0 * num_rounds * log_term)) / noise_multiplier
        return float(eps)


class AdditiveSecretSharing:
    """Zero-sum additive secret sharing for privacy-preserving federated aggregation."""

    @staticmethod
    def split_secret(
        params: Dict[str, List[float]],
        num_parties: int,
        seed: Optional[int] = None,
    ) -> List[Dict[str, List[float]]]:
        """Split a parameter dictionary into num_parties additive secret shares."""
        if num_parties < 2:
            raise ValueError(f"num_parties must be >= 2, got {num_parties}")

        flat_vec, shapes = flatten_params(params)
        dim = len(flat_vec)
        rng = np.random.default_rng(seed)

        shares_flat: List[np.ndarray] = []
        for _ in range(num_parties - 1):
            share = rng.normal(loc=0.0, scale=1.0, size=dim)
            shares_flat.append(share)

        last_share = flat_vec - sum(shares_flat)
        shares_flat.append(last_share)

        return [unflatten_params(s, shapes) for s in shares_flat]

    @staticmethod
    def reconstruct_secret(
        shares: List[Dict[str, List[float]]]
    ) -> Dict[str, List[float]]:
        """Reconstruct the original secret parameter dictionary by summing shares."""
        if not shares:
            raise ValueError("No shares provided for reconstruction.")

        flat_vectors = []
        shapes = None
        for s in shares:
            vec, shp = flatten_params(s)
            flat_vectors.append(vec)
            if shapes is None:
                shapes = shp

        summed = np.sum(flat_vectors, axis=0)
        assert shapes is not None
        return unflatten_params(summed, shapes)

    @staticmethod
    def generate_pairwise_masks(
        agent_ids: List[str],
        param_shapes: Dict[str, Tuple[int, int, Tuple[int, ...]]],
        total_dim: int,
        seed_base: int = 1000,
    ) -> Dict[str, np.ndarray]:
        """Generate pairwise zero-sum masks for all participating agents."""
        n = len(agent_ids)
        masks: Dict[str, np.ndarray] = {aid: np.zeros(total_dim, dtype=np.float64) for aid in agent_ids}

        for i in range(n):
            for j in range(i + 1, n):
                aid_i = agent_ids[i]
                aid_j = agent_ids[j]
                pair_seed = seed_base + (hash(f"{min(aid_i, aid_j)}:{max(aid_i, aid_j)}") % 1000000)
                rng = np.random.default_rng(abs(pair_seed))
                r_ij = rng.normal(loc=0.0, scale=1.0, size=total_dim)

                masks[aid_i] += r_ij
                masks[aid_j] -= r_ij

        return masks

    @classmethod
    def mask_update(
        cls,
        params: Dict[str, List[float]],
        mask_vector: np.ndarray,
    ) -> Dict[str, List[float]]:
        """Apply the generated mask to an agent's parameter update."""
        flat_vec, shapes = flatten_params(params)
        if len(flat_vec) != len(mask_vector):
            raise ValueError(f"Mask length {len(mask_vector)} mismatch with parameters {len(flat_vec)}")
        masked_vec = flat_vec + mask_vector
        return unflatten_params(masked_vec, shapes)

    @classmethod
    def aggregate_masked_updates(
        cls,
        masked_updates: List[Dict[str, List[float]]],
    ) -> Dict[str, List[float]]:
        """Sum masked updates; zero-sum masks cancel out exactly, recovering true average."""
        if not masked_updates:
            raise ValueError("No masked updates to aggregate.")

        n = len(masked_updates)
        flat_vectors = []
        shapes = None

        for upd in masked_updates:
            vec, shp = flatten_params(upd)
            flat_vectors.append(vec)
            if shapes is None:
                shapes = shp

        summed = np.sum(flat_vectors, axis=0)
        mean_vec = summed / float(n)
        assert shapes is not None
        return unflatten_params(mean_vec, shapes)
