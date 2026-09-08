from __future__ import annotations

import numpy as np
import pyro

from match3_simulator import LEVELS, PlayerSkill, ground_truth_model, legal_moves
from match3_simulator.evidence import EVIDENCE_NAMES
from match3_simulator.learned_model.action_policy import (
    ActionPolicyConfig,
    ContinuousActionPolicy,
)
from match3_simulator.learned_model.rollout import NetworkActionPolicy


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