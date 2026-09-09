from __future__ import annotations

import numpy as np
import pyro
import torch
from unittest.mock import patch

from match3_simulator import LEVELS, PlayerSkill, ground_truth_model, legal_moves
from match3_simulator.evidence import EVIDENCE_NAMES
from match3_simulator.learned_model.action_policy import (
    ActionPolicyConfig,
    ContinuousActionPolicy,
)
from match3_simulator.learned_model.rollout import NetworkActionPolicy
from match3_simulator.learned_model.arms import (
    GenerativeModelConfig,
    GenerativeWorldModel,
    ModelArm,
)
from match3_simulator.learned_model.encoder import PrefixEncoderConfig
from match3_simulator.learned_model.generative import GameplayRSSMConfig
from match3_simulator.learned_model.rollout import (
    rollout_learned_dynamics,
    sample_learned_initial_state,
)


def _network_policy(stochastic: bool = False) -> NetworkActionPolicy:
    model = ContinuousActionPolicy(
        ActionPolicyConfig(d_model=16, n_layers=1, n_heads=2)
    )
    return NetworkActionPolicy(model, stochastic=stochastic, seed=359)


def test_network_policy_rolls_out_only_legal_actions() -> None:
    pyro.set_rng_seed(353)
    episode = ground_truth_model(
        level=LEVELS[0],
        player=PlayerSkill((0.0, 0.0, 0.0, 0.0)),
        E=0.0,
        evidence=np.zeros(len(EVIDENCE_NAMES)),
        max_steps=3,
        action_policy=_network_policy(),
    )
    assert episode.actions
    assert episode.action_diagnostics == []
    for state, action in zip(episode.states, episode.actions):
        assert action in legal_moves(state.board)


def test_stochastic_network_policy_is_reproducible_from_its_own_seed() -> None:
    kwargs = {
        "level": LEVELS[1],
        "player": PlayerSkill((0.0, 0.0, 0.0, 0.0)),
        "E": 0.0,
        "evidence": np.zeros(len(EVIDENCE_NAMES)),
        "max_steps": 2,
    }
    pyro.set_rng_seed(367)
    first = ground_truth_model(
        **kwargs, action_policy=_network_policy(stochastic=True)
    )
    pyro.set_rng_seed(367)
    second = ground_truth_model(
        **kwargs, action_policy=_network_policy(stochastic=True)
    )
    assert first.actions == second.actions


def test_environment_rejects_illegal_policy_action() -> None:
    def illegal_policy(state, player):
        return type(legal_moves(state.board)[0])(0, 0, 0, 0)

    pyro.set_rng_seed(373)
    try:
        ground_truth_model(max_steps=1, action_policy=illegal_policy)
    except ValueError as error:
        assert "illegal move" in str(error)
    else:
        raise AssertionError("illegal learned action was accepted")


def test_learned_dynamics_rollout_filters_once_then_uses_priors() -> None:
    pyro.set_rng_seed(379)
    initial = ground_truth_model(
        level=LEVELS[0],
        player=PlayerSkill((0.0, 0.0, 0.0, 0.0)),
        E=0.0,
        max_steps=0,
    ).states[0]
    model = GenerativeWorldModel(
        ModelArm.CAUSAL,
        GenerativeModelConfig(
            dynamics=GameplayRSSMConfig(
                embedding_size=4,
                observation_size=8,
                task_context_size=4,
                hidden_size=8,
                stochastic_size=4,
            ),
            prefix=PrefixEncoderConfig(hidden_size=8),
            behavior=ActionPolicyConfig(d_model=8, n_layers=1, n_heads=2),
        ),
    ).eval()
    with torch.no_grad():
        board_layer = model.dynamics.rssm.board_decoder[-1]
        board_layer.weight.zero_()
        board_layer.bias.fill_(-10.0)
        for cell, colour in enumerate(initial.board.ravel()):
            board_layer.bias[cell * 6 + int(colour)] = 10.0
        counter_layer = model.dynamics.rssm.counter_decoder[-1]
        counter_layer.weight.zero_()
        counter_layer.bias.fill_(1.0)

    rssm = model.dynamics.rssm
    with (
        patch.object(rssm, "observe_step", wraps=rssm.observe_step) as observe,
        patch.object(rssm, "imagine_step", wraps=rssm.imagine_step) as imagine,
    ):
        rollout = rollout_learned_dynamics(
            model,
            initial,
            level_index=0,
            tier_index=1,
            served_difficulty=0.0,
            player_context=torch.zeros(4),
            max_steps=3,
        )

    assert len(rollout.actions) == 3
    assert observe.call_count == 1
    assert imagine.call_count == 2
    assert all(
        action in legal_moves(state.board)
        for state, action in zip(rollout.states, rollout.actions)
    )
    assert all(
        right.moves_left == left.moves_left - 1
        for left, right in zip(rollout.states, rollout.states[1:])
    )


def test_learned_initial_state_uses_only_task_conditioned_decoder() -> None:
    model = GenerativeWorldModel(
        ModelArm.POOLED,
        GenerativeModelConfig(
            dynamics=GameplayRSSMConfig(
                embedding_size=4,
                observation_size=8,
                task_context_size=4,
                hidden_size=8,
                stochastic_size=4,
            ),
            prefix=PrefixEncoderConfig(hidden_size=8),
            behavior=ActionPolicyConfig(d_model=8, n_layers=1, n_heads=2),
        ),
    ).eval()
    state = sample_learned_initial_state(
        model,
        level_index=1,
        tier_index=2,
        served_difficulty=0.5,
        stochastic=False,
    )

    assert state.board.shape == (8, 8)
    assert 0 <= int(state.board.min()) <= int(state.board.max()) < 6
    assert 1 <= state.moves_left <= model.config.dynamics.max_moves_left
    assert state.goals_left >= 1
    assert state.t == 0