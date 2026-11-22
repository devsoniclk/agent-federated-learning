"""Quality verification, anomaly detection, and Shapley valuation for swarm contributions."""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np

from agent_federated_learning.models import AgentContribution, ModelUpdate
from agent_federated_learning.aggregation import flatten_params


class ContributionEvaluator:
    """Evaluates gradient alignment, norm consistency, and marginal value of agent contributions."""

    @staticmethod
    def compute_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Compute cosine similarity between two parameter vectors."""
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a < 1e-12 or norm_b < 1e-12:
            return 0.0
        dot_prod = np.dot(vec_a, vec_b)
        return float(np.clip(dot_prod / (norm_a * norm_b), -1.0, 1.0))

    @classmethod
    def evaluate_updates(
        cls,
        updates: List[ModelUpdate],
        reference_params: Optional[Dict[str, List[float]]] = None,
        min_cosine_threshold: float = -0.1
    ) -> List[AgentContribution]:
        """Audit agent updates for adversarial inversion and abnormal gradient magnitude."""
        if not updates:
            return []

        flat_vectors = [flatten_params(u.parameters)[0] for u in updates]

        # Use reference parameters if provided; otherwise use coordinate median as pseudo-ground-truth
        if reference_params is not None:
            ref_vec, _ = flatten_params(reference_params)
        else:
            ref_vec = np.median(np.stack(flat_vectors, axis=0), axis=0)

        # Norm statistics across updates
        norms = [float(np.linalg.norm(v)) for v in flat_vectors]
        median_norm = float(np.median(norms)) if norms else 1.0
        mad_norm = float(np.median([abs(n - median_norm) for n in norms])) + 1e-6

        contributions: List[AgentContribution] = []

        for u, vec, norm in zip(updates, flat_vectors, norms):
            cos_sim = cls.compute_cosine_similarity(vec, ref_vec)
            is_byzantine = False
            reason: Optional[str] = None

            # Check 1: Gradient inversion / negative directional correlation
            if cos_sim < min_cosine_threshold:
                is_byzantine = True
                reason = f"Negative directional alignment: cos_sim={cos_sim:.3f} < {min_cosine_threshold}"

            # Check 2: Extreme norm explosion (e.g. > 5 * MAD above median)
            if not is_byzantine and (norm > median_norm + 5.0 * mad_norm):
                is_byzantine = True
                reason = f"Gradient norm explosion: norm={norm:.2f} > threshold={median_norm + 5.0 * mad_norm:.2f}"

            # Baseline reputation scaled by directional alignment
            reputation = max(0.0, min(1.0, (cos_sim + 1.0) / 2.0)) if not is_byzantine else 0.0

            contributions.append(
                AgentContribution(
                    agent_id=u.agent_id,
                    round_id=u.round_id,
                    sample_count=u.sample_count,
                    cosine_similarity=cos_sim,
                    shapley_value=0.0,
                    reputation_score=reputation,
                    is_byzantine=is_byzantine,
                    rejection_reason=reason
                )
            )

        return contributions

    @classmethod
    def calculate_shapley_values(
        cls,
        updates: List[ModelUpdate],
        target_gradient: Dict[str, List[float]],
        num_monte_carlo_samples: int = 100,
        seed: Optional[int] = 42
    ) -> Dict[str, float]:
        """Estimate marginal Data Shapley value of each agent update via Monte Carlo sampling.
        
        Utility is defined as the cosine alignment of the coalition's aggregate with the
        true target gradient direction.
        """
        n = len(updates)
        if n == 0:
            return {}
        if n == 1:
            return {updates[0].agent_id: 1.0}

        rng = np.random.default_rng(seed)
        target_vec, _ = flatten_params(target_gradient)
        flat_vectors = {u.agent_id: flatten_params(u.parameters)[0] for u in updates}
        agent_ids = [u.agent_id for u in updates]

        def utility(coalition: List[str]) -> float:
            if not coalition:
                return 0.0
            coalition_vec = np.mean([flat_vectors[aid] for aid in coalition], axis=0)
            return cls.compute_cosine_similarity(coalition_vec, target_vec)

        shapley_sums: Dict[str, float] = {aid: 0.0 for aid in agent_ids}

        for _ in range(num_monte_carlo_samples):
            permutation = list(rng.permutation(agent_ids))
            current_coalition: List[str] = []
            prev_util = 0.0

            for aid in permutation:
                current_coalition.append(aid)
                curr_util = utility(current_coalition)
                marginal = curr_util - prev_util
                shapley_sums[aid] += marginal
                prev_util = curr_util

        # Average over permutations
        shapley_values = {aid: shapley_sums[aid] / num_monte_carlo_samples for aid in agent_ids}
        return shapley_values
