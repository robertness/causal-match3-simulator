from __future__ import annotations

import inspect
from dataclasses import replace

import numpy as np
import pytest

from match3_simulator.causal_queries import (
    AssignmentConfig,
    LandmarkRiskSet,
    bootstrap_recommendation_contrasts,
    compare_oracle_curves,
    evaluate_landmark_benchmark,
    evaluate_validation_suite,
    generate_landmark_cohort,
    passes_initial_gates,
    randomized_assignment_control,
    select_grid_optimum,
)
from match3_simulator.retention import (
    CHURN_SCHEDULE,
    ChurnConfig,
    WinPropensityModel,
    challenge_mismatch_hazard,
    sample_C,
)
from match3_simulator.scm import LEVELS, TIER_LOGITS, TIER_NAMES, TIER_PROBS
from match3_simulator.spec import BENCHMARK_CONFIG, SKILL_COVARIANCE


@pytest.fixture(scope="module")
def synthetic_benchmark():
    rng = np.random.default_rng(13)
    n_players = 100_000
    skills = rng.multivariate_normal(
        np.zeros(SKILL_COVARIANCE.shape[0]),
        SKILL_COVARIANCE,
        size=n_players,
    )
    tiers = rng.choice(len(TIER_NAMES), size=n_players, p=TIER_PROBS)
    tier_logits = np.asarray(TIER_LOGITS)[tiers]

    coefficients = []
    risk_sets = []
    for level in LEVELS:
        demand = level.demand_weights()
        scale = np.sqrt(demand @ SKILL_COVARIANCE @ demand)
        coefficients.append(tuple(1.6 * demand / scale))
        effective_skill = skills @ demand / scale
        risk_sets.append(
            LandmarkRiskSet(
                level_name=level.name,
                skills=skills,
                tier_indices=tiers,
                assignment_locations=tier_logits + 0.8 * effective_skill,
                assignment_sigma=0.75,
            )
        )

    tier_intercepts = tuple(-0.35 * np.asarray(TIER_LOGITS))
    model = WinPropensityModel(
        level_names=tuple(level.name for level in LEVELS),
        tier_names=TIER_NAMES,
        intercepts=tuple(tier_intercepts for _ in LEVELS),
        skill_coefficients=tuple(coefficients),
        difficulty_coefficients=tuple(1.0 for _ in LEVELS),
    )
    return model, tuple(risk_sets)


def test_churn_hazard_is_symmetric_around_target() -> None:
    probabilities = np.asarray([0.35, 0.55, 0.75])
    hazard = challenge_mismatch_hazard(probabilities)

    assert hazard[1] < hazard[0]
    assert hazard[1] < hazard[2]
    assert np.isclose(hazard[0], hazard[2])


def test_sample_c_has_no_direct_skill_or_treatment_input() -> None:
    parameters = inspect.signature(sample_C).parameters
    assert "skill" not in parameters
    assert "player" not in parameters
    assert "E" not in parameters
    assert "outcome" not in parameters
    assert sample_C(0.55, active=False) == 1


def test_oracle_curves_have_separated_reversing_recommendations(
    synthetic_benchmark,
) -> None:
    model, risk_sets = synthetic_benchmark
    for risk_set in risk_sets:
        comparison = compare_oracle_curves(risk_set, model)
        assert passes_initial_gates(comparison)
        assert comparison.recommendation_gap >= 1.0
        assert comparison.causal_recommendation_contrast >= 0.02
        assert comparison.observational_recommendation_contrast <= -0.02
        assert comparison.causal_left_contrast >= 0.02
        assert comparison.causal_right_contrast >= 0.02
        assert min(item.effective_sample_fraction for item in comparison.overlap) >= 0.20


def test_causal_curve_rises_on_both_sides_of_interior_optimum(
    synthetic_benchmark,
) -> None:
    model, risk_sets = synthetic_benchmark
    grid = np.asarray(BENCHMARK_CONFIG.e_grid)
    for risk_set in risk_sets:
        comparison = compare_oracle_curves(risk_set, model)
        optimum = comparison.causal_optimum
        assert grid[0] < optimum < grid[-1]
        optimum_index = int(np.flatnonzero(grid == optimum)[0])
        left_index = int(np.flatnonzero(grid == optimum - 1.0)[0])
        right_index = int(np.flatnonzero(grid == optimum + 1.0)[0])
        assert comparison.causal[left_index] - comparison.causal[optimum_index] >= 0.02
        assert comparison.causal[right_index] - comparison.causal[optimum_index] >= 0.02


def test_randomized_assignment_collapses_observational_and_causal_curves(
    synthetic_benchmark,
) -> None:
    model, risk_sets = synthetic_benchmark
    for risk_set in risk_sets:
        randomized = LandmarkRiskSet(
            level_name=risk_set.level_name,
            skills=risk_set.skills,
            tier_indices=risk_set.tier_indices,
            assignment_locations=np.zeros(len(risk_set.skills)),
            assignment_sigma=risk_set.assignment_sigma,
        )
        comparison = compare_oracle_curves(randomized, model)
        np.testing.assert_allclose(comparison.observational, comparison.causal)
        assert comparison.observational_optimum == comparison.causal_optimum


def test_randomized_control_removes_both_skill_and_tier_assignment(
    synthetic_benchmark,
) -> None:
    model, _ = synthetic_benchmark
    natural = generate_landmark_cohort(model, n_players=1_000, seed=107)
    randomized = randomized_assignment_control(natural)
    for risk_set in randomized.risk_sets:
        np.testing.assert_array_equal(
            risk_set.assignment_locations, np.zeros(risk_set.skills.shape[0])
        )
        current_churn = CHURN_SCHEDULE.for_level(risk_set.level_name)
        comparison = compare_oracle_curves(risk_set, model, current_churn)
        np.testing.assert_allclose(comparison.observational, comparison.causal)
        assert all(
            item.effective_sample_fraction >= BENCHMARK_CONFIG.minimum_ess_fraction
            for item in comparison.overlap
        )


def test_observational_curve_uses_nonuniform_assignment_weights(
    synthetic_benchmark,
) -> None:
    model, risk_sets = synthetic_benchmark
    risk_set = risk_sets[0]
    comparison = compare_oracle_curves(risk_set, model)
    unweighted = compare_oracle_curves(
        LandmarkRiskSet(
            level_name=risk_set.level_name,
            skills=risk_set.skills,
            tier_indices=risk_set.tier_indices,
            assignment_locations=np.zeros(len(risk_set.skills)),
            assignment_sigma=risk_set.assignment_sigma,
        ),
        model,
    )
    assert not np.allclose(comparison.observational, unweighted.observational)


def test_grid_tie_breaking_prefers_smallest_absolute_then_lower_value() -> None:
    grid = np.asarray([-1.0, -0.5, 0.5, 1.0])
    values = np.asarray([1.0, 0.0, 0.0, 1.0])
    assert select_grid_optimum(grid, values) == -0.5


def test_landmark_cohort_is_deterministic_and_clones_survivors(
    synthetic_benchmark,
) -> None:
    model, _ = synthetic_benchmark
    first = generate_landmark_cohort(model, n_players=1_000, seed=101)
    second = generate_landmark_cohort(model, n_players=1_000, seed=101)

    assert first.n_active_players == first.n_initial_players
    assert first.survival_fraction == 1.0
    np.testing.assert_array_equal(first.active_player_ids, second.active_player_ids)
    for first_risk, second_risk in zip(first.risk_sets, second.risk_sets):
        np.testing.assert_array_equal(first_risk.skills, second_risk.skills)
        np.testing.assert_array_equal(first_risk.tier_indices, second_risk.tier_indices)
        np.testing.assert_allclose(
            first_risk.assignment_locations, second_risk.assignment_locations
        )
        np.testing.assert_array_equal(first_risk.skills, first.risk_sets[0].skills)


def test_landmark_assignment_locations_match_d_and_effective_skill(
    synthetic_benchmark,
) -> None:
    model, _ = synthetic_benchmark
    assignment = AssignmentConfig(skill_gain=0.8, sigma=0.75)
    cohort = generate_landmark_cohort(
        model,
        n_players=1_000,
        seed=103,
        assignment=assignment,
    )
    for level, risk_set in zip(LEVELS, cohort.risk_sets):
        demand = level.demand_weights()
        scale = np.sqrt(demand @ SKILL_COVARIANCE @ demand)
        effective_skill = risk_set.skills @ demand / scale
        expected = (
            np.asarray(TIER_LOGITS)[risk_set.tier_indices]
            + assignment.skill_gain * effective_skill
        )
        np.testing.assert_allclose(risk_set.assignment_locations, expected)


def test_landmark_report_exposes_all_level_gates(synthetic_benchmark) -> None:
    model, _ = synthetic_benchmark
    report = evaluate_landmark_benchmark(
        model,
        n_players=100_000,
        seed=109,
        assignment=AssignmentConfig(skill_gain=0.8, sigma=0.65),
    )
    assert report["passed"]
    assert report["schema_version"] == 1
    assert report["benchmark"]["validation_seeds"] == (
        BENCHMARK_CONFIG.validation_seeds
    )
    assert set(report["levels"]) == {level.name for level in LEVELS}
    for level_report in report["levels"].values():
        assert level_report["passed"]
        assert all(level_report["gates"].values())


def test_validation_suite_requires_every_preregistered_seed(
    synthetic_benchmark,
) -> None:
    model, _ = synthetic_benchmark
    benchmark = replace(BENCHMARK_CONFIG, validation_seeds=(109, 113))
    report = evaluate_validation_suite(
        model,
        n_players=100_000,
        assignment=AssignmentConfig(skill_gain=0.8, sigma=0.65),
        benchmark=benchmark,
    )
    assert report["passed"]
    assert report["validation_seeds"] == [109, 113]
    assert [item["seed"] for item in report["seed_reports"]] == [109, 113]
    assert all(item["passed"] for item in report["seed_reports"])


def test_bootstrap_contrast_intervals_are_deterministic_and_directional(
    synthetic_benchmark,
) -> None:
    model, risk_sets = synthetic_benchmark
    comparison = compare_oracle_curves(risk_sets[0], model)
    first = bootstrap_recommendation_contrasts(
        risk_sets[0], model, comparison, n_bootstrap=50, seed=127
    )
    second = bootstrap_recommendation_contrasts(
        risk_sets[0], model, comparison, n_bootstrap=50, seed=127
    )

    assert first == second
    assert first.causal.lower > 0.0
    assert first.observational.upper < 0.0