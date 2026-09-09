from __future__ import annotations

from dataclasses import replace
import sys

import numpy as np
import pytest

from match3_simulator import engine_benchmark as engine_benchmark_module
from match3_simulator.calibrate import load_win_propensity_model
from match3_simulator.causal_queries import (
    EngineWarmupPanel,
    bootstrap_engine_curves,
    compare_engine_curves,
    engine_outcome_surface,
    extend_engine_outcome_surface,
    generate_engine_warmup_panel,
    generate_engine_landmark_cohort,
    generate_landmark_cohort,
    load_engine_outcome_surface,
    load_engine_warmup_panel,
    load_landmark_risk_set,
    materialize_engine_landmark_cohort,
    regrid_engine_outcome_surface,
    save_engine_outcome_surface,
    save_engine_warmup_panel,
    save_landmark_risk_set,
)
from match3_simulator.spec import BENCHMARK_CONFIG
from match3_simulator.engine_benchmark import (
    engine_level_report,
    evaluate_persisted_engine_risk_sets,
    generate_engine_risk_set_artifacts,
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


def test_engine_surface_extension_preserves_existing_replicates() -> None:
    risk_set = _tiny_risk_set()
    initial = engine_outcome_surface(
        risk_set,
        grid=np.asarray([-0.5, 0.0, 0.5]),
        rollouts_per_player=1,
        workers=1,
    )
    extended = extend_engine_outcome_surface(
        risk_set,
        initial,
        rollouts_per_player=2,
        workers=1,
    )
    direct = engine_outcome_surface(
        risk_set,
        grid=initial.grid,
        rollouts_per_player=2,
        workers=1,
    )

    np.testing.assert_array_equal(extended.rollout_seeds[:, :1], initial.rollout_seeds)
    np.testing.assert_array_equal(extended.goal_totals[:, :1], initial.goal_totals)
    np.testing.assert_array_equal(extended.rollout_seeds, direct.rollout_seeds)
    np.testing.assert_array_equal(extended.goal_totals, direct.goal_totals)
    np.testing.assert_array_equal(extended.outcomes, direct.outcomes)


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


def test_engine_warmup_panel_round_trip_and_rescores_survival(tmp_path) -> None:
    benchmark = replace(
        BENCHMARK_CONFIG,
        landmark_attempt=3,
        warmup_churn_scale=1.0,
    )
    panel = generate_engine_warmup_panel(
        load_win_propensity_model(),
        n_players=4,
        seed=108,
        benchmark=benchmark,
        workers=1,
    )

    assert isinstance(panel, EngineWarmupPanel)
    assert panel.warmup_outcomes.shape == (4, 2)
    assert panel.warmup_completion_margins.shape == (4, 2)
    assert np.all(
        (panel.warmup_completion_margins >= -1.0)
        & (panel.warmup_completion_margins <= 1.0)
    )
    assert panel.level_indices.shape == (4, 2)
    assert panel.churn_uniforms.shape == (4, 2)
    assert panel.target_tier_indices.shape == (3, 4)
    assert np.all((panel.churn_uniforms >= 0.0) & (panel.churn_uniforms < 1.0))

    path = save_engine_warmup_panel(panel, tmp_path / "warmup-panel.npz")
    restored = load_engine_warmup_panel(path)
    for name in panel.__dict__:
        np.testing.assert_array_equal(getattr(restored, name), getattr(panel, name))

    legacy_path = tmp_path / "legacy-warmup-panel.npz"
    np.savez_compressed(
        legacy_path,
        schema_version=np.asarray([1], dtype=np.int16),
        seed=np.asarray([panel.seed], dtype=np.int64),
        landmark_attempt=np.asarray([panel.landmark_attempt], dtype=np.int16),
        player_ids=panel.player_ids,
        skills=panel.skills,
        warmup_outcomes=panel.warmup_outcomes,
        level_indices=panel.level_indices,
        churn_uniforms=panel.churn_uniforms,
        target_tier_indices=panel.target_tier_indices,
        exogenous_seeds=panel.exogenous_seeds,
        assignment_gains=panel.assignment_gains,
        assignment_sigmas=panel.assignment_sigmas,
    )
    legacy = load_engine_warmup_panel(legacy_path)
    assert legacy.warmup_completion_margins.shape == (4, 0)

    low_hazard = ChurnConfig(
        intercept=-100.0,
        deviation_coefficient=1.0,
        mastery_target=0.35,
    )
    high_hazard = ChurnConfig(
        intercept=100.0,
        deviation_coefficient=1.0,
        mastery_target=0.35,
    )
    all_active = materialize_engine_landmark_cohort(
        restored,
        benchmark=benchmark,
        churn_config=low_hazard,
    )
    none_active = materialize_engine_landmark_cohort(
        restored,
        benchmark=benchmark,
        churn_config=high_hazard,
    )

    assert all_active.n_active_players == 4
    assert none_active.n_active_players == 0


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


def test_engine_surface_records_exact_signed_completion_margins(tmp_path) -> None:
    risk_set = _tiny_risk_set()
    surface = engine_outcome_surface(
        risk_set,
        grid=np.asarray([-0.5, 0.0, 0.5]),
        rollouts_per_player=1,
        workers=1,
        include_completion_margins=True,
    )

    assert surface.completion_margins.shape == surface.outcomes.shape
    assert np.all(surface.completion_margins[surface.outcomes == 1] >= 0.0)
    assert np.all(surface.completion_margins[surface.outcomes == 0] <= 0.0)
    path = save_engine_outcome_surface(surface, tmp_path / "margins.npz")
    restored = load_engine_outcome_surface(path)
    np.testing.assert_array_equal(
        restored.completion_margins, surface.completion_margins
    )
    comparison = compare_engine_curves(
        risk_set,
        surface,
        churn_config=ChurnConfig(
            intercept=-4.0,
            deviation_coefficient=1.0,
            mastery_target=0.35,
            margin_deviation_coefficient=80.0,
            margin_target=-0.1,
        ),
    )
    assert np.isfinite(comparison.causal).all()
    report = engine_level_report(
        risk_set,
        surface,
        load_win_propensity_model(),
        churn_config=ChurnConfig(
            intercept=-4.0,
            deviation_coefficient=1.0,
            mastery_target=0.35,
            margin_deviation_coefficient=80.0,
            margin_target=-0.1,
        ),
    )
    assert report["surrogate_check"]["role"] == "unavailable"


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


def test_warmup_only_artifacts_include_all_player_panel(tmp_path) -> None:
    benchmark = replace(
        BENCHMARK_CONFIG,
        landmark_attempt=2,
        warmup_churn_scale=0.0,
    )
    report = generate_engine_risk_set_artifacts(
        load_win_propensity_model(),
        n_players=2,
        seed=127,
        output_dir=tmp_path,
        benchmark=benchmark,
        workers=1,
    )

    panel_path = tmp_path / report["warmup_panel"]["path"]
    panel = load_engine_warmup_panel(panel_path)
    assert report["warmup_panel"]["players"] == 2
    assert report["calibration_table"]["sha256"]
    assert len(panel.player_ids) == 2
    assert panel_path.is_file()


def test_engine_cli_keeps_initial_mastery_and_hazard_target_distinct(
    monkeypatch, tmp_path
) -> None:
    captured = {}

    def fake_generate(*args, **kwargs):
        captured.update(kwargs)
        return {"n_landmark_players": 1}

    monkeypatch.setattr(
        engine_benchmark_module,
        "generate_engine_risk_set_artifacts",
        fake_generate,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "engine-benchmark",
            "--warmup-only",
            "--players",
            "1",
            "--mastery-initial",
            "0.27",
            "--mastery-target",
            "0.33",
            "--churn-margin-curvatures",
            "10",
            "20",
            "30",
            "--churn-margin-targets",
            "-0.1",
            "-0.2",
            "-0.3",
            "--out",
            str(tmp_path),
        ],
    )

    engine_benchmark_module.main()

    assert captured["mastery_config"].initial == 0.27
    assert captured["churn_config"].mastery_target == 0.33
    assert captured["churn_config"].margin_deviation_coefficients == (
        10.0,
        20.0,
        30.0,
    )
    assert captured["churn_config"].margin_targets == (-0.1, -0.2, -0.3)