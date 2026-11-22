"""Unit tests for differential privacy and additive secret sharing."""

import numpy as np
import pytest

from agent_federated_learning.models import DifferentialPrivacyConfig
from agent_federated_learning.privacy import (
    AdditiveSecretSharing,
    DifferentialPrivacyEngine,
)


def test_clip_l2_norm():
    params = {"w": [3.0, 4.0]}  # norm = 5.0
    clipped, factor = DifferentialPrivacyEngine.clip_l2_norm(params, max_norm=2.5)
    # factor is 2.5 / 5.0 = 0.5
    assert np.isclose(factor, 0.5)
    flat_clipped = np.array(clipped["w"])
    assert np.isclose(np.linalg.norm(flat_clipped), 2.5)

    # Sub-threshold norm (no clipping)
    params_small = {"w": [1.0, 1.0]}
    clipped_s, factor_s = DifferentialPrivacyEngine.clip_l2_norm(params_small, max_norm=5.0)
    assert np.isclose(factor_s, 1.0)
    assert np.allclose(clipped_s["w"], params_small["w"])


def test_add_gaussian_noise():
    params = {"w": [0.0] * 1000}
    noisy = DifferentialPrivacyEngine.add_gaussian_noise(
        params, clip_norm=1.0, noise_multiplier=0.5, seed=42
    )
    # std should be around 0.5
    vals = np.array(noisy["w"])
    assert np.isclose(np.std(vals), 0.5, atol=0.1)

    # Zero noise
    zero_noisy = DifferentialPrivacyEngine.add_gaussian_noise(
        params, clip_norm=1.0, noise_multiplier=0.0
    )
    assert np.allclose(zero_noisy["w"], params["w"])


def test_apply_dp_pipeline():
    params = {"w": [3.0, 4.0], "b": [1.0]}
    norm = np.sqrt(3**2 + 4**2 + 1**2)
    cfg = DifferentialPrivacyConfig(
        enabled=True, clip_norm=2.0, noise_multiplier=0.2, epsilon=1.0, delta=1e-5
    )
    perturbed, factor = DifferentialPrivacyEngine.apply_dp(params, cfg, seed=123)
    # factor is 2.0 / norm
    assert np.isclose(factor, 2.0 / norm)
    assert "w" in perturbed
    assert "b" in perturbed

    # DP disabled
    cfg_off = DifferentialPrivacyConfig(enabled=False)
    unperturbed, factor_off = DifferentialPrivacyEngine.apply_dp(params, cfg_off)
    assert factor_off == 1.0
    assert unperturbed == params


def test_calculate_epsilon():
    # Monotonicity test
    eps_r1 = DifferentialPrivacyEngine.calculate_epsilon(
        num_rounds=1, noise_multiplier=1.0, target_delta=1e-5
    )
    eps_r10 = DifferentialPrivacyEngine.calculate_epsilon(
        num_rounds=10, noise_multiplier=1.0, target_delta=1e-5
    )
    assert eps_r10 > eps_r1

    # Zero rounds returns 0.0
    assert (
        DifferentialPrivacyEngine.calculate_epsilon(
            num_rounds=0, noise_multiplier=1.0, target_delta=1e-5
        )
        == 0.0
    )

    # Zero noise multiplier returns inf
    assert (
        DifferentialPrivacyEngine.calculate_epsilon(
            num_rounds=5, noise_multiplier=0.0, target_delta=1e-5
        )
        == float("inf")
    )


def test_additive_secret_sharing_zero_sum_masks():
    agent_ids = ["agent-1", "agent-2", "agent-3"]
    params = {"layer1": [1.5, -2.5, 3.0], "layer2": [0.05]}
    from agent_federated_learning.aggregation import flatten_params

    _, shapes = flatten_params(params)
    total_dim = 4

    masks = AdditiveSecretSharing.generate_pairwise_masks(
        agent_ids=agent_ids,
        param_shapes=shapes,
        total_dim=total_dim,
        seed_base=123,
    )

    # Sum of all masks must be identically zero
    total_mask = sum(masks.values())
    assert np.allclose(total_mask, np.zeros(total_dim))

    # Mask each agent update
    masked_updates = []
    for aid in agent_ids:
        masked = AdditiveSecretSharing.mask_update(params, masks[aid])
        masked_updates.append(masked)

    # Aggregate masked updates
    aggregated = AdditiveSecretSharing.aggregate_masked_updates(masked_updates)
    assert np.allclose(aggregated["layer1"], params["layer1"])
    assert np.allclose(aggregated["layer2"], params["layer2"])


def test_secret_sharing_validation():
    params = {"w": [1.0, 2.0]}
    with pytest.raises(ValueError, match="Mask length"):
        AdditiveSecretSharing.mask_update(params, np.array([0.0]))

    with pytest.raises(ValueError, match="No masked updates"):
        AdditiveSecretSharing.aggregate_masked_updates([])


def test_additive_secret_sharing_split_and_reconstruct():
    params = {"w": [2.5, -1.0, 4.0], "b": [0.5]}
    shares = AdditiveSecretSharing.split_secret(params, num_parties=4, seed=42)
    assert len(shares) == 4

    # Reconstruct
    reconstructed = AdditiveSecretSharing.reconstruct_secret(shares)
    assert np.allclose(reconstructed["w"], params["w"])
    assert np.allclose(reconstructed["b"], params["b"])

    # Error cases
    with pytest.raises(ValueError, match="num_parties must be >= 2"):
        AdditiveSecretSharing.split_secret(params, num_parties=1)

    with pytest.raises(ValueError, match="No shares provided"):
        AdditiveSecretSharing.reconstruct_secret([])


def test_numpy_array_dp_methods():
    arr = np.array([3.0, 4.0])
    clipped_arr, factor = DifferentialPrivacyEngine.clip_l2_norm(arr, max_norm=2.5)
    assert np.isclose(factor, 0.5)
    assert np.isclose(np.linalg.norm(clipped_arr), 2.5)

    # Sub-threshold
    clipped_s, factor_s = DifferentialPrivacyEngine.clip_l2_norm(np.array([1.0, 1.0]), max_norm=5.0)
    assert factor_s == 1.0

    # Negative norm error
    with pytest.raises(ValueError, match="strictly positive"):
        DifferentialPrivacyEngine.clip_l2_norm(arr, max_norm=-1.0)

    # Add noise to numpy array
    noisy_arr = DifferentialPrivacyEngine.add_gaussian_noise(np.zeros(100), clip_norm=1.0, noise_multiplier=0.5, seed=12)
    assert len(noisy_arr) == 100
    zero_arr = DifferentialPrivacyEngine.add_gaussian_noise(arr, clip_norm=1.0, noise_multiplier=0.0)
    assert np.allclose(zero_arr, arr)

