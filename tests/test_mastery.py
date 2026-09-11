from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from match3_simulator.retention import (
    ChurnConfig,
    MasteryConfig,
    completion_margin,
    mastery_mismatch_hazard,
    sample_C,
    update_mastery,
)


def test_mastery_update_tracks_realized_outcomes() -> None:
    config = MasteryConfig(initial=0.55, update_rate=0.20)

    assert update_mastery(0.55, 1, config) == pytest.approx(0.64)
    assert update_mastery(0.55, 0, config) == pytest.approx(0.44)


def test_mastery_update_preserves_support_and_converges() -> None:
    config = MasteryConfig(initial=0.55, update_rate=0.35)
    mastery = config.initial
    for _ in range(100):
        mastery = update_mastery(mastery, 1, config)
        assert 0.0 <= mastery <= 1.0
    assert mastery == pytest.approx(1.0)

    for _ in range(100):
        mastery = update_mastery(mastery, 0, config)
        assert 0.0 <= mastery <= 1.0
    assert mastery == pytest.approx(0.0)


@pytest.mark.parametrize("mastery", [-0.01, 1.01])
def test_mastery_update_rejects_values_outside_support(mastery: float) -> None:
    with pytest.raises(ValueError, match="mastery"):
        update_mastery(mastery, 1)


def test_mastery_hazard_is_symmetric_around_target() -> None:
    config = ChurnConfig(
        intercept=-4.0,
        deviation_coefficient=48.0,
        mastery_target=0.55,
    )
    hazard = mastery_mismatch_hazard(np.asarray([0.35, 0.55, 0.75]), config)

    assert hazard[1] < hazard[0]
    assert hazard[1] < hazard[2]
    assert hazard[0] == pytest.approx(hazard[2])


def test_overchallenge_can_have_steeper_hazard_than_underchallenge() -> None:
    config = ChurnConfig(
        intercept=-4.0,
        deviation_coefficient=24.0,
        overchallenge_deviation_coefficient=72.0,
        mastery_target=0.55,
        margin_deviation_coefficient=20.0,
        margin_overchallenge_deviation_coefficient=60.0,
        margin_target=0.0,
    )

    mastery_hazard = mastery_mismatch_hazard(
        np.asarray([0.35, 0.75]),
        config,
        completion_margin=np.zeros(2),
    )
    margin_hazard = mastery_mismatch_hazard(
        np.full(2, 0.55),
        config,
        completion_margin=np.asarray([-0.2, 0.2]),
    )

    assert mastery_hazard[0] > mastery_hazard[1]
    assert margin_hazard[0] > margin_hazard[1]


def test_mastery_hazard_is_stable_at_extreme_logits() -> None:
    hazard = mastery_mismatch_hazard(
        np.asarray([0.0, 1.0]),
        ChurnConfig(
            intercept=-1e9,
            deviation_coefficient=1e10,
            mastery_target=0.5,
        ),
    )

    assert np.isfinite(hazard).all()
    assert np.all((hazard >= 0.0) & (hazard <= 1.0))


@pytest.mark.parametrize(
    ("outcome", "moves_left", "goals_left", "expected"),
    [(1, 4, 0, 0.2), (0, 0, 6, -0.25)],
)
def test_completion_margin_is_signed_distance_from_failure_boundary(
    outcome: int,
    moves_left: int,
    goals_left: int,
    expected: float,
) -> None:
    episode = SimpleNamespace(
        R=outcome,
        difficulty=SimpleNamespace(move_budget=20),
        served_goal_count=24,
        states=[SimpleNamespace(moves_left=moves_left, goals_left=goals_left)],
    )

    assert completion_margin(episode) == pytest.approx(expected)


def test_churn_hazard_can_use_current_completion_margin() -> None:
    config = ChurnConfig(
        intercept=-4.0,
        deviation_coefficient=1.0,
        mastery_target=0.5,
        margin_deviation_coefficient=80.0,
        margin_target=-0.1,
    )
    hazard = mastery_mismatch_hazard(
        np.full(3, 0.5),
        config,
        completion_margin=np.asarray([-0.3, -0.1, 0.1]),
    )

    assert hazard[1] < hazard[0]
    assert hazard[1] < hazard[2]
    assert hazard[0] == pytest.approx(hazard[2])


def test_sample_c_accepts_only_post_attempt_mastery() -> None:
    parameters = inspect.signature(sample_C).parameters

    assert "mastery_after" in parameters
    assert "win_probability" not in parameters
    assert "skill" not in parameters
    assert "player" not in parameters
    assert "E" not in parameters
    assert "outcome" not in parameters
    assert sample_C(0.55, active=False) == 1