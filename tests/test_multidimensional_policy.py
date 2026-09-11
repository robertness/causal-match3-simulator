from __future__ import annotations

import inspect

import numpy as np
import pyro
import pyro.poutine as poutine

from match3_simulator.policy import (
    distractor_fallback_probabilities,
    deterministic_rollout_value,
    MoveFeatures,
    generic_clear_weight,
    goal_weight,
    notice_probabilities,
    pattern_choice_sharpness,
    pattern_noise_scale,
    pattern_reliability,
    planning_weight,
    staged_action_probs,
)
from match3_simulator.scm import ground_truth_model, sample_A
from match3_simulator.board import legal_moves
from match3_simulator.spec import PlayerSkill


def _features() -> MoveFeatures:
    return MoveFeatures(
        total_cleared=np.asarray([3.0, 5.0, 4.0]),
        goal_cleared=np.asarray([2.0, 0.0, 1.0]),
        setup_value=np.asarray([0.0, 3.0, 1.0]),
    )


def test_each_skill_controls_its_declared_mechanism() -> None:
    features = _features()
    assert np.all(notice_probabilities(features, 1.0) > notice_probabilities(features, -1.0))
    assert pattern_noise_scale(1.0) < pattern_noise_scale(-1.0)
    assert pattern_choice_sharpness(1.0) > pattern_choice_sharpness(-1.0)
    assert pattern_reliability(1.0) > pattern_reliability(-1.0)
    assert planning_weight(1.0) > planning_weight(-1.0)
    assert goal_weight(1.0) > goal_weight(-1.0)
    assert generic_clear_weight(1.0) < generic_clear_weight(-1.0)


def test_strategy_increases_expected_goal_progress() -> None:
    features = _features()
    noticed = np.ones(features.n_moves, dtype=bool)
    noise = np.zeros(features.n_moves)
    low = PlayerSkill((0.0, 0.0, 0.0, -2.0))
    high = PlayerSkill((0.0, 0.0, 0.0, 2.0))

    low_probs = staged_action_probs(features, low, noticed, noise)
    high_probs = staged_action_probs(features, high, noticed, noise)
    assert high_probs @ features.goal_cleared > low_probs @ features.goal_cleared


def test_planning_increases_expected_setup_value() -> None:
    features = MoveFeatures(
        total_cleared=np.ones(2),
        goal_cleared=np.ones(2),
        setup_value=np.asarray([0.0, 3.0]),
    )
    noticed = np.ones(features.n_moves, dtype=bool)
    noise = np.zeros(features.n_moves)
    low = PlayerSkill((0.0, 0.0, -2.0, 0.0))
    high = PlayerSkill((0.0, 0.0, 2.0, 0.0))

    low_probs = staged_action_probs(features, low, noticed, noise)
    high_probs = staged_action_probs(features, high, noticed, noise)
    assert high_probs @ features.setup_value > low_probs @ features.setup_value


def test_staged_distribution_is_normalized_and_masks_unnoticed_moves() -> None:
    features = _features()
    noticed = np.asarray([True, False, True])
    probs = staged_action_probs(
        features,
        PlayerSkill((0.0, 0.0, 0.0, 0.0)),
        noticed,
        np.zeros(features.n_moves),
    )
    assert np.isclose(probs.sum(), 1.0)
    assert probs[1] == 0.0


def test_zero_notice_fallback_prefers_distractor_over_goal_progress() -> None:
    features = MoveFeatures(
        total_cleared=np.asarray([3.0, 6.0]),
        goal_cleared=np.asarray([3.0, 0.0]),
        setup_value=np.zeros(2),
    )
    probabilities = distractor_fallback_probabilities(features)
    assert probabilities[1] > probabilities[0]


def test_sample_a_has_only_state_and_skill_structural_parents() -> None:
    parameters = inspect.signature(sample_A).parameters
    assert "state" in parameters
    assert "player" in parameters
    assert "level" not in parameters
    assert "E" not in parameters


def test_episode_trace_contains_staged_policy_sites() -> None:
    pyro.set_rng_seed(31)
    trace = poutine.trace(ground_truth_model).get_trace(max_steps=1)
    assert "A/0/noticed" in trace.nodes
    assert "A/0/evaluation_noise" in trace.nodes
    assert "A/0/noticed_count" in trace.nodes
    assert "A/0/pattern_noise_scale" in trace.nodes
    assert "A/0" in trace.nodes


def test_episode_records_action_diagnostics() -> None:
    pyro.set_rng_seed(37)
    episode = ground_truth_model(max_steps=2)
    assert len(episode.action_diagnostics) == len(episode.actions)
    for diagnostic in episode.action_diagnostics:
        assert 0.0 < diagnostic.candidate_recall <= 1.0
        assert diagnostic.pattern_noise_scale > 0.0


def test_planning_rollout_value_is_deterministic_and_finite() -> None:
    pyro.set_rng_seed(39)
    episode = ground_truth_model(max_steps=0)
    state = episode.states[0]
    action = legal_moves(state.board)[0]
    first = deterministic_rollout_value(state, action)
    second = deterministic_rollout_value(state, action)
    assert np.isfinite(first)
    assert first == second