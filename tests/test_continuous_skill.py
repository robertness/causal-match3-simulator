from __future__ import annotations

import inspect

import numpy as np
import pyro

from match3_simulator.causal_queries import ASSIGNMENT_SCHEDULE
from match3_simulator.scm import (
    DDA_GAINS,
    E_SIGMAS,
    LEVELS,
    TIER_MOVE_BUDGETS,
    sample_D,
    sample_E,
    sample_K,
)
from match3_simulator.spec import (
    SKILL_COVARIANCE,
    SKILL_NAMES,
    Difficulty,
    PlayerSkill,
)
from match3_simulator.trajectory import episode_to_dict, summary_row
from match3_simulator import ground_truth_model


def test_skill_covariance_is_a_correlation_matrix() -> None:
    np.testing.assert_allclose(SKILL_COVARIANCE, SKILL_COVARIANCE.T)
    np.testing.assert_allclose(np.diag(SKILL_COVARIANCE), np.ones(4))
    np.linalg.cholesky(SKILL_COVARIANCE)


def test_effective_skill_has_unit_population_variance() -> None:
    for level in LEVELS:
        demand = level.demand_weights()
        scale = np.sqrt(demand @ SKILL_COVARIANCE @ demand)
        loading = demand / scale
        variance = loading @ SKILL_COVARIANCE @ loading
        assert np.isclose(variance, 1.0)


def test_sample_k_is_reproducible_and_four_dimensional() -> None:
    pyro.set_rng_seed(17)
    first = sample_K()
    pyro.set_rng_seed(17)
    second = sample_K()

    assert isinstance(first, PlayerSkill)
    assert len(first.values) == len(SKILL_NAMES)
    assert first == second


def test_baseline_tier_has_no_skill_input() -> None:
    assert "player" not in inspect.signature(sample_D).parameters
    assert "skill" not in inspect.signature(sample_D).parameters


def test_baseline_tiers_have_distinct_state_budgets() -> None:
    assert len(set(TIER_MOVE_BUDGETS)) == len(TIER_MOVE_BUDGETS)
    assert TIER_MOVE_BUDGETS[0] > TIER_MOVE_BUDGETS[1] > TIER_MOVE_BUDGETS[2]


def test_served_difficulty_responds_to_level_specific_skill() -> None:
    difficulty = Difficulty(baseline=0.0)
    low = PlayerSkill((-1.0, -1.0, -1.0, -1.0))
    high = PlayerSkill((1.0, 1.0, 1.0, 1.0))

    for level in LEVELS:
        low_E = sample_E(difficulty, level, low, sigma=0.0)
        high_E = sample_E(difficulty, level, high, sigma=0.0)
        assert low_E < high_E


def test_do_e_clamps_served_difficulty() -> None:
    clamped = sample_E(
        Difficulty(baseline=0.8),
        LEVELS[0],
        PlayerSkill((3.0, 3.0, 3.0, 3.0)),
        value=-0.375,
    )
    assert clamped == -0.375


def test_query_and_scm_assignment_defaults_match() -> None:
    assert ASSIGNMENT_SCHEDULE.skill_gains == DDA_GAINS
    assert ASSIGNMENT_SCHEDULE.sigmas == E_SIGMAS


def test_custom_level_uses_safe_assignment_fallback() -> None:
    from match3_simulator.spec import LevelContext

    value = sample_E(
        Difficulty(baseline=0.0),
        LevelContext(name="custom"),
        PlayerSkill((0.0, 0.0, 0.0, 0.0)),
        sigma=0.0,
    )
    assert value == 0.0


def test_scm_and_retention_import_in_either_order() -> None:
    import importlib

    import match3_simulator.retention as retention
    import match3_simulator.scm as scm

    assert importlib.reload(scm)
    assert importlib.reload(retention)


def test_schema_v2_serializes_named_skill_coordinates() -> None:
    pyro.set_rng_seed(9)
    episode = ground_truth_model(max_steps=1)
    document = episode_to_dict(episode)
    row = summary_row(episode)

    assert document["version"] == 2
    assert set(document["player"]["skill"]) == set(SKILL_NAMES)
    assert {f"k_{name}" for name in SKILL_NAMES} <= set(row)
    assert "segment" not in row
    assert "phi" not in row