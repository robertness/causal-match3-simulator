from __future__ import annotations

import numpy as np
import pyro
import torch

from match3_simulator import LEVELS, ground_truth_model
from match3_simulator.learned_model.action_policy import (
    ActionPolicyConfig,
    ContinuousActionPolicy,
)
from match3_simulator.learned_model.data import (
    action_dataset_from_episodes,
    split_action_dataset_by_episode,
    split_action_dataset_by_player,
)
from match3_simulator.learned_model.train import (
    ActionTrainConfig,
    evaluate_action_policy,
    train_action_policy,
)


def _episodes():
    episodes = []
    for index in range(8):
        pyro.set_rng_seed(201 + index)
        episodes.append(
            ground_truth_model(level=LEVELS[index % len(LEVELS)], max_steps=2)
        )
    return episodes


def test_action_dataset_uses_current_states_and_legal_targets() -> None:
    episodes = _episodes()
    dataset = action_dataset_from_episodes(episodes)
    assert len(dataset) == sum(len(episode.actions) for episode in episodes)
    assert np.all(dataset.legal_actions[np.arange(len(dataset)), dataset.actions])
    first_episode = episodes[int(dataset.episode_indices[0])]
    np.testing.assert_array_equal(dataset.boards[0], first_episode.states[0].board.ravel())


def test_action_split_keeps_episodes_disjoint() -> None:
    dataset = action_dataset_from_episodes(_episodes())
    train, validation = split_action_dataset_by_episode(dataset, seed=11)
    assert set(train.episode_indices).isdisjoint(set(validation.episode_indices))
    assert len(train) + len(validation) == len(dataset)


def test_action_split_keeps_players_disjoint() -> None:
    episodes = _episodes()
    dataset = action_dataset_from_episodes(
        episodes,
        player_indices=[index // 2 for index in range(len(episodes))],
    )
    train, validation = split_action_dataset_by_player(dataset, seed=11)
    assert set(train.player_indices).isdisjoint(set(validation.player_indices))
    assert len(train) + len(validation) == len(dataset)


def test_oracle_skill_action_training_runs_on_cpu() -> None:
    torch.manual_seed(29)
    dataset = action_dataset_from_episodes(_episodes())
    train, validation = split_action_dataset_by_episode(dataset, seed=13)
    policy = ContinuousActionPolicy(
        ActionPolicyConfig(d_model=16, n_layers=1, n_heads=2)
    )
    initial = policy.skill.weight.detach().clone()
    history = train_action_policy(
        policy,
        train,
        validation,
        include_skill=True,
        config=ActionTrainConfig(epochs=2, batch_size=8, seed=17),
    )
    assert len(history) == 2
    assert all(np.isfinite(row["train_nll"]) for row in history)
    assert all(np.isfinite(row["validation_nll"]) for row in history)
    assert not torch.equal(initial, policy.skill.weight.detach())
    metrics = evaluate_action_policy(
        policy,
        validation,
        include_skill=True,
        device=torch.device("cpu"),
    )
    assert set(metrics) == {
        "nll",
        "top1_accuracy",
        "uniform_legal_nll",
        "n_actions",
    }
    assert all(np.isfinite(value) for value in metrics.values())
    assert 0.0 <= metrics["top1_accuracy"] <= 1.0