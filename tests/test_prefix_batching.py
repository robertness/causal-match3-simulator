from __future__ import annotations

import numpy as np
import pytest
import torch

from match3_simulator import LEVELS
from match3_simulator.learned_model.action_policy import ActionPolicyConfig
from match3_simulator.learned_model.batching import build_prefix_target_batch
from match3_simulator.learned_model.data import split_player_trajectories
from match3_simulator.learned_model.encoder import PrefixEncoderConfig
from match3_simulator.learned_model.model import ContinuousCausalVAE
from match3_simulator.retention import WinPropensityModel
from match3_simulator.scm import TIER_NAMES
from match3_simulator.simulate import simulate_players
from match3_simulator.learned_model.train import (
    VAETrainConfig,
    evaluate_continuous_vae,
    train_continuous_vae,
)


def _propensity_model() -> WinPropensityModel:
    return WinPropensityModel(
        tuple(level.name for level in LEVELS),
        TIER_NAMES,
        tuple((0.0, 0.0, 0.0) for _ in LEVELS),
        tuple((0.3, 0.3, 0.3, 0.3) for _ in LEVELS),
        tuple(1.0 for _ in LEVELS),
    )


def _trajectories():
    return simulate_players(2, _propensity_model(), seed=331, max_attempts=2)


def test_batch_contains_only_completed_prefix_episodes() -> None:
    trajectories = _trajectories()
    prefix, target = build_prefix_target_batch(
        trajectories, target_attempt=2
    )
    assert prefix["episode_mask"].shape == (2, 1)
    assert prefix["episode_mask"].all()
    for player_index, trajectory in enumerate(trajectories):
        first = trajectory.attempts[0].episode
        second = trajectory.attempts[1].episode
        assert prefix["served_difficulty"][player_index, 0] == first.E
        assert target.served_difficulty[player_index] == second.E
        np.testing.assert_allclose(
            target.evidence[player_index].numpy(), second.proxy, rtol=1e-6
        )
        assert target.mastery_before[player_index] == trajectory.attempts[1].mastery_before
    assert target.churn_mask.all()


def test_trajectory_split_is_player_disjoint_and_deterministic() -> None:
    trajectories = simulate_players(
        6, _propensity_model(), seed=333, max_attempts=1
    )
    first = split_player_trajectories(trajectories, seed=17)
    second = split_player_trajectories(trajectories, seed=17)
    first_ids = [{item.player_id for item in split} for split in first]
    second_ids = [{item.player_id for item in split} for split in second]
    assert first_ids == second_ids
    assert all(first_ids[index].isdisjoint(first_ids[other]) for index in range(3) for other in range(index))
    assert set.union(*first_ids) == {item.player_id for item in trajectories}


def test_every_target_action_maps_to_its_player_and_is_legal() -> None:
    trajectories = _trajectories()
    _, target = build_prefix_target_batch(trajectories, target_attempt=2)
    assert len(target.actions) == sum(
        len(trajectory.attempts[1].episode.actions)
        for trajectory in trajectories
    )
    assert target.legal_actions[
        torch.arange(len(target.actions)), target.actions
    ].all()
    assert set(target.action_player.tolist()) <= {0, 1}


def test_real_trajectory_batch_runs_through_continuous_vae() -> None:
    prefix, target = build_prefix_target_batch(
        _trajectories(), target_attempt=2
    )
    torch.manual_seed(337)
    model = ContinuousCausalVAE(
        PrefixEncoderConfig(hidden_size=16),
        ActionPolicyConfig(d_model=16, n_layers=1, n_heads=2),
    )
    result = model.predictive_objective(prefix, target, kl_weight=0.1)
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert model.encoder.posterior.weight.grad is not None


def test_continuous_vae_training_runs_on_cpu_and_logs_kl_warmup() -> None:
    batch = build_prefix_target_batch(_trajectories(), target_attempt=2)
    torch.manual_seed(347)
    model = ContinuousCausalVAE(
        PrefixEncoderConfig(hidden_size=8),
        ActionPolicyConfig(d_model=8, n_layers=1, n_heads=2),
    )
    initial = model.encoder.posterior.weight.detach().clone()
    history = train_continuous_vae(
        model,
        [batch],
        [batch],
        config=VAETrainConfig(
            epochs=2,
            kl_warmup_epochs=2,
            learning_rate=1e-3,
            seed=349,
        ),
    )
    assert [row["kl_weight"] for row in history] == [0.5, 1.0]
    assert all(np.isfinite(row["train_loss"]) for row in history)
    assert all(np.isfinite(row["validation_loss"]) for row in history)
    assert not torch.equal(initial, model.encoder.posterior.weight.detach())


def test_vae_config_requires_complete_kl_warmup() -> None:
    with pytest.raises(ValueError, match="must not exceed epochs"):
        VAETrainConfig(epochs=2, kl_warmup_epochs=5)


def test_vae_evaluation_weights_partial_batches_by_actions() -> None:
    trajectories = simulate_players(
        3, _propensity_model(), seed=351, max_attempts=2
    )
    batches = [
        build_prefix_target_batch(trajectories[:2], target_attempt=2),
        build_prefix_target_batch(trajectories[2:], target_attempt=2),
    ]
    model = ContinuousCausalVAE(
        PrefixEncoderConfig(hidden_size=8),
        ActionPolicyConfig(d_model=8, n_layers=1, n_heads=2),
    )
    metrics = evaluate_continuous_vae(
        model, batches, device=torch.device("cpu")
    )
    weighted = 0.0
    count = 0
    for prefix, target in batches:
        value = model.predictive_objective(
            prefix, target, sample_posterior=False
        )["action_nll"]
        weighted += float(value.detach()) * target.actions.numel()
        count += target.actions.numel()
    assert np.isclose(metrics["action_nll"], weighted / count)