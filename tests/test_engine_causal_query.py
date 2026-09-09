from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from match3_simulator.calibrate import load_win_propensity_model
from match3_simulator.causal_queries import (
    bootstrap_engine_curves,
    compare_engine_curves,
    engine_outcome_surface,
    generate_engine_landmark_cohort,
    generate_landmark_cohort,
    load_engine_outcome_surface,
    load_landmark_risk_set,
    regrid_engine_outcome_surface,
    save_engine_outcome_surface,
    save_landmark_risk_set,
)
from match3_simulator.spec import BENCHMARK_CONFIG
from match3_simulator.engine_benchmark import (
    engine_level_report,
    evaluate_persisted_engine_risk_sets,
)
from match3_simulator.retention import ChurnConfig


def _tiny_risk_set():
    cohort = generate_landmark_cohort(
        load_win_propensity_model(),
        n_players=3,
        seed=101,
        benchmark=replace(BENCHMARK_CONFIG, landmark_attempt=1),
    )
    return cohort.risk_sets[0]


@pytest.fixture(scope="module")
def tiny_engine():
    risk_set = _tiny_risk_set()
    surface = engine_outcome_surface(
        risk_set,
        grid=np.asarray([-0.5, 0.0, 0.5]),
        rollouts_per_player=2,
        workers=1,
    )
    return risk_set, surface


def test_engine_surface_reuses_one_seed_per_player_replicate_across_e() -> None:
    risk_set = _tiny_risk_set()
    grid = np.asarray([-0.5, 0.5])
    first = engine_outcome_surface(
        risk_set,
        grid=grid,
        rollouts_per_player=2,
        workers=1,
    )
    second = engine_outcome_surface(
        risk_set,
        grid=grid,
        rollouts_per_player=2,
        workers=1,
    )

    assert first.outcomes.shape == (len(grid), len(risk_set.skills), 2)
    assert first.rollout_seeds.shape == (len(risk_set.skills), 2)
    assert np.all((first.outcomes == 0) | (first.outcomes == 1))
    np.testing.assert_array_equal(first.player_ids, risk_set.player_ids)
    np.testing.assert_array_equal(first.rollout_seeds, second.rollout_seeds)
    np.testing.assert_array_equal(first.outcomes, second.outcomes)


def test_goal_total_reuse_matches_direct_per_e_rollouts() -> None:
    risk_set = _tiny_risk_set()
    grid = np.asarray([-0.5, 0.0, 0.5])
    reused = engine_outcome_surface(
        risk_set,
        grid=grid,
        rollouts_per_player=2,
        workers=1,
        reuse_goal_totals=True,
    )
    direct = engine_outcome_surface(
        risk_set,
        grid=grid,
        rollouts_per_player=2,
        workers=1,
        reuse_goal_totals=False,
    )

    np.testing.assert_array_equal(reused.rollout_seeds, direct.rollout_seeds)
    np.testing.assert_array_equal(reused.outcomes, direct.outcomes)
    assert reused.outcome_method == "goal_total_threshold"
    assert direct.outcome_method == "direct_per_e"
    assert reused.goal_totals.shape == (len(risk_set.skills), 2)
    assert direct.goal_totals is None

    expanded_grid = np.asarray([-1.0, -0.5, 0.0, 0.5, 1.0])
    regridded = regrid_engine_outcome_surface(reused, expanded_grid)
    expanded_direct = engine_outcome_surface(
        risk_set,
        grid=expanded_grid,
        rollouts_per_player=2,
        workers=1,
        reuse_goal_totals=False,
    )
    np.testing.assert_array_equal(regridded.outcomes, expanded_direct.outcomes)


def test_engine_landmark_cohort_uses_realized_warmup_outcomes() -> None:
    benchmark = replace(
        BENCHMARK_CONFIG,
        landmark_attempt=2,
        warmup_churn_scale=0.0,
    )
    model = load_win_propensity_model()
    first = generate_engine_landmark_cohort(
        model,
        n_players=2,
        seed=107,
        benchmark=benchmark,
        workers=1,
    )
    second = generate_engine_landmark_cohort(
        model,
        n_players=2,
        seed=107,
        benchmark=benchmark,
        workers=1,
    )

    assert first.n_active_players == 2
    np.testing.assert_array_equal(first.active_player_ids, second.active_player_ids)
    for first_risk, second_risk in zip(first.risk_sets, second.risk_sets):
        np.testing.assert_array_equal(first_risk.skills, second_risk.skills)
        np.testing.assert_array_equal(
            first_risk.mastery_before, second_risk.mastery_before
        )
        np.testing.assert_array_equal(
            first_risk.warmup_outcomes, second_risk.warmup_outcomes
        )
        assert first_risk.warmup_outcomes.shape == (2, 1)
        assert np.all(
            np.isclose(first_risk.mastery_before[:, None], [0.245, 0.545]).any(
                axis=1
            )
        )


def test_engine_curve_uses_realized_outcomes_and_frozen_mastery(tiny_engine) -> None:
    risk_set, surface = tiny_engine
    comparison = compare_engine_curves(risk_set, surface)

    assert comparison.grid.tolist() == [-0.5, 0.0, 0.5]
    assert comparison.causal.shape == (3,)
    assert comparison.observational.shape == (3,)
    assert np.isfinite(comparison.causal).all()
    assert np.isfinite(comparison.observational).all()


def test_engine_surface_round_trip_preserves_pairing(tmp_path, tiny_engine) -> None:
    _, surface = tiny_engine
    path = save_engine_outcome_surface(surface, tmp_path / "surface.npz")
    restored = load_engine_outcome_surface(path)

    assert restored.level_name == surface.level_name
    np.testing.assert_array_equal(restored.grid, surface.grid)
    np.testing.assert_array_equal(restored.player_ids, surface.player_ids)
    np.testing.assert_array_equal(restored.rollout_seeds, surface.rollout_seeds)
    np.testing.assert_array_equal(restored.outcomes, surface.outcomes)
    assert restored.outcome_method == surface.outcome_method
    np.testing.assert_array_equal(restored.goal_totals, surface.goal_totals)


def test_landmark_risk_set_round_trip_preserves_causal_state(
    tmp_path, tiny_engine
) -> None:
    risk_set, _ = tiny_engine
    path = save_landmark_risk_set(risk_set, tmp_path / "risk-set.npz")
    restored = load_landmark_risk_set(path)

    assert restored.level_name == risk_set.level_name
    assert restored.assignment_sigma == risk_set.assignment_sigma
    for name in (
        "skills",
        "tier_indices",
        "mastery_before",
        "warmup_outcomes",
        "assignment_locations",
        "player_ids",
        "exogenous_seeds",
    ):
        np.testing.assert_array_equal(getattr(restored, name), getattr(risk_set, name))


def test_engine_bootstrap_resamples_whole_paired_players(tiny_engine) -> None:
    risk_set, surface = tiny_engine
    first = bootstrap_engine_curves(
        risk_set, surface, n_bootstrap=20, seed=103
    )
    second = bootstrap_engine_curves(
        risk_set, surface, n_bootstrap=20, seed=103
    )

    assert first == second
    assert first["n_bootstrap"] == 20
    assert len(first["causal_pointwise"]["lower"]) == len(surface.grid)
    assert len(first["observational_pointwise"]["upper"]) == len(surface.grid)
    assert sum(first["causal_optimum_frequencies"].values()) == pytest.approx(1.0)
    assert sum(first["observational_optimum_frequencies"].values()) == pytest.approx(
        1.0
    )


def test_engine_level_report_separates_engine_and_surrogate_evidence(
    tiny_engine,
) -> None:
    risk_set, surface = tiny_engine
    report = engine_level_report(
        risk_set,
        surface,
        load_win_propensity_model(),
        n_bootstrap=20,
        bootstrap_seed=109,
    )

    assert report["estimator"] == "board_engine"
    assert report["n_players"] == len(risk_set.skills)
    assert report["n_rollouts"] == surface.outcomes.size
    assert report["bootstrap"]["resampling_unit"] == "player"
    assert report["surrogate_check"]["role"] == "calibration_diagnostic_only"
    assert len(report["engine"]["causal"]) == len(surface.grid)


def test_engine_level_report_uses_explicit_churn_config(tiny_engine) -> None:
    risk_set, surface = tiny_engine
    model = load_win_propensity_model()
    first = engine_level_report(
        risk_set,
        surface,
        model,
        churn_config=ChurnConfig(
            intercept=-8.0,
            deviation_coefficient=50.0,
            mastery_target=0.35,
        ),
    )
    second = engine_level_report(
        risk_set,
        surface,
        model,
        churn_config=ChurnConfig(
            intercept=-2.0,
            deviation_coefficient=500.0,
            mastery_target=0.35,
        ),
    )

    assert first["engine"]["causal"] != second["engine"]["causal"]


def test_persisted_risk_sets_skip_warmup_and_generate_target_surfaces(
    tmp_path,
) -> None:
    benchmark = replace(BENCHMARK_CONFIG, landmark_attempt=1)
    model = load_win_propensity_model()
    cohort = generate_engine_landmark_cohort(
        model,
        n_players=2,
        seed=113,
        benchmark=benchmark,
        workers=1,
    )
    risk_dir = tmp_path / "risk"
    for risk_set in cohort.risk_sets:
        save_landmark_risk_set(
            risk_set, risk_dir / f"{risk_set.level_name}-risk-set.npz"
        )

    report = evaluate_persisted_engine_risk_sets(
        model,
        risk_set_dir=risk_dir,
        output_dir=tmp_path / "target",
        seed=113,
        e_grid=(-0.5, 0.0, 0.5),
        rollouts_per_player=1,
        workers=1,
        benchmark=benchmark,
    )

    assert report["risk_set_source"] == "persisted_board_engine"
    assert report["n_engine_players"] == 2
    for level in ("orchard", "harbour", "foundry"):
        assert (tmp_path / "target" / f"{level}-outcomes.npz").is_file()
        assert report["levels"][level]["outcome_method"] == "goal_total_threshold"