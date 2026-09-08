from __future__ import annotations

import inspect

import numpy as np
import pyro
import pytest
import torch

from match3_simulator import (
    LEVELS,
    TIER_MOVE_BUDGETS,
    ground_truth_model,
    legal_moves,
)
from match3_simulator.board import immediate_effect
from match3_simulator.learned_model.action_policy import (
    ActionPolicyConfig,
    ContinuousActionPolicy,
)
from match3_simulator.learned_model.encoder import PrefixEncoderConfig
from match3_simulator.learned_model.tokens import (
    ACTION_SLOTS,
    action_to_index,
    index_to_action,
    legal_mask,
)


def _episode_batch():
    pyro.set_rng_seed(151)
    episode = ground_truth_model(level=LEVELS[0], E=0.0, max_steps=1)
    state = episode.states[0]
    action = episode.actions[0]
    mask = legal_mask(state.board)
    return episode, state, action, mask


def _small_policy() -> ContinuousActionPolicy:
    torch.manual_seed(17)
    return ContinuousActionPolicy(
        ActionPolicyConfig(d_model=32, n_layers=1, n_heads=2)
    )


def test_action_index_round_trip_and_legal_mask() -> None:
    _, state, action, mask = _episode_batch()
    index = action_to_index(action)
    assert index_to_action(index) == action
    assert mask.shape == (ACTION_SLOTS,)
    assert mask[index]


def test_policy_normalizes_over_only_legal_actions() -> None:
    episode, state, _, mask = _episode_batch()
    policy = _small_policy()
    log_probabilities = policy(
        board=torch.as_tensor(state.board.reshape(1, -1), dtype=torch.long),
        goal_colour=torch.tensor([state.goal_colour]),
        moves_left=torch.tensor([state.moves_left]),
        goals_left=torch.tensor([state.goals_left]),
        skill=torch.as_tensor(episode.player.as_array()[None, :], dtype=torch.float32),
        legal_actions=torch.as_tensor(mask[None, :]),
    )
    probabilities = log_probabilities.exp().detach().numpy()[0]
    assert log_probabilities.shape == (1, ACTION_SLOTS)
    assert np.isclose(probabilities.sum(), 1.0)
    assert np.all(probabilities[~mask] == 0.0)


def test_counter_encodings_cover_every_tier_move_budget() -> None:
    maximum = max(TIER_MOVE_BUDGETS)
    assert ActionPolicyConfig().max_moves_left == maximum
    assert PrefixEncoderConfig().max_moves_left == maximum


def test_immediate_action_features_match_board_engine() -> None:
    _, state, _, _ = _episode_batch()
    policy = _small_policy()
    features = policy.immediate_action_features(
        torch.as_tensor(state.board.reshape(1, -1), dtype=torch.long),
        torch.tensor([state.goal_colour]),
    )
    for action in legal_moves(state.board):
        total_cleared, goal_cleared = immediate_effect(
            state.board, action, state.goal_colour
        )
        torch.testing.assert_close(
            features[0, action_to_index(action)],
            torch.tensor(
                [total_cleared, goal_cleared], dtype=torch.float32
            ),
        )


def test_immediate_features_standardize_over_legal_actions_only() -> None:
    _, state, _, mask = _episode_batch()
    policy = _small_policy()
    features = policy.immediate_action_features(
        torch.as_tensor(state.board.reshape(1, -1), dtype=torch.long),
        torch.tensor([state.goal_colour]),
    )
    legal = torch.as_tensor(mask[None, :])
    standardized = policy._standardize_action_features(features, legal)
    selected = standardized[0, legal[0]]
    torch.testing.assert_close(selected.mean(dim=0), torch.zeros(2), atol=1e-6, rtol=0)
    nonconstant = features[0, legal[0]].std(dim=0, correction=0) > 1e-6
    torch.testing.assert_close(
        selected.std(dim=0, correction=0)[nonconstant],
        torch.ones(int(nonconstant.sum())),
        atol=1e-6,
        rtol=0,
    )
    assert torch.count_nonzero(standardized[0, ~legal[0]]) == 0


def test_policy_has_no_direct_treatment_or_outcome_inputs() -> None:
    parameters = inspect.signature(ContinuousActionPolicy.forward).parameters
    for forbidden in ("E", "difficulty", "level", "outcome", "churn"):
        assert forbidden not in parameters


def test_action_loss_backpropagates_through_skill_path() -> None:
    episode, state, action, mask = _episode_batch()
    policy = _small_policy()
    skill = torch.as_tensor(
        episode.player.as_array()[None, :], dtype=torch.float32
    )
    loss = policy.negative_log_likelihood(
        action=torch.tensor([action_to_index(action)]),
        board=torch.as_tensor(state.board.reshape(1, -1), dtype=torch.long),
        goal_colour=torch.tensor([state.goal_colour]),
        moves_left=torch.tensor([state.moves_left]),
        goals_left=torch.tensor([state.goals_left]),
        skill=skill,
        legal_actions=torch.as_tensor(mask[None, :]),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert policy.skill.weight.grad is not None
    assert torch.linalg.vector_norm(policy.skill.weight.grad) > 0


def test_empty_legal_mask_is_rejected() -> None:
    episode, state, _, _ = _episode_batch()
    policy = _small_policy()
    with pytest.raises(ValueError, match="at least one legal action"):
        policy(
            board=torch.as_tensor(state.board.reshape(1, -1), dtype=torch.long),
            goal_colour=torch.tensor([state.goal_colour]),
            moves_left=torch.tensor([state.moves_left]),
            goals_left=torch.tensor([state.goals_left]),
            skill=torch.as_tensor(
                episode.player.as_array()[None, :], dtype=torch.float32
            ),
            legal_actions=torch.zeros((1, ACTION_SLOTS), dtype=torch.bool),
        )