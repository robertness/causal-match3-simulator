from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from match3_simulator.calibrate import (
    WinPropensityData,
    collect_win_propensity_data,
    fit_win_propensity_model,
    goal_count_from_totals,
    load_win_propensity_model,
    load_win_propensity_data,
    propensity_brier_score,
    reference_goal_totals,
    save_win_propensity_model,
    save_win_propensity_data,
    split_win_propensity_data,
    validate_goal_count_table,
    win_propensity_design_summary,
    win_rate,
)
from match3_simulator.scm import LEVELS, TIER_NAMES


def _synthetic_data(seed: int = 41) -> WinPropensityData:
    rng = np.random.default_rng(seed)
    n_rows = 4_000
    skills = rng.normal(size=(n_rows, 4))
    levels = rng.integers(0, 2, size=n_rows)
    tiers = rng.integers(0, 3, size=n_rows)
    served = rng.uniform(-2.0, 2.0, size=n_rows)
    logits = -0.2 + 0.45 * skills.sum(axis=1) - 1.1 * served
    probability = 1.0 / (1.0 + np.exp(-logits))
    outcomes = rng.binomial(1, probability)
    return WinPropensityData(levels, tiers, skills, served, outcomes)


def test_constrained_fit_improves_over_intercept_only_prediction() -> None:
    data = _synthetic_data()
    model = fit_win_propensity_model(
        data,
        ("orchard", "harbour"),
        TIER_NAMES,
        max_iter=100,
    )
    score = propensity_brier_score(model, data)
    baseline = float(np.mean((data.outcomes - np.mean(data.outcomes)) ** 2))

    assert score < baseline - 0.05
    assert np.min(model.skill_coefficients) >= 0.0
    assert np.min(model.difficulty_coefficients) > 0.0


def test_aligned_fit_preserves_supplied_skill_directions() -> None:
    data = _synthetic_data()
    directions = np.asarray(
        [[0.6, 0.3, 0.1, 0.0], [0.2, 0.3, 0.4, 0.1]], dtype=np.float64
    )
    model = fit_win_propensity_model(
        data,
        ("orchard", "harbour"),
        TIER_NAMES,
        skill_directions=directions,
        max_iter=100,
    )
    coefficients = np.asarray(model.skill_coefficients)
    for row in range(len(directions)):
        nonzero = directions[row] > 0
        ratios = coefficients[row, nonzero] / directions[row, nonzero]
        np.testing.assert_allclose(ratios, np.repeat(ratios[0], len(ratios)))
        assert coefficients[row, ~nonzero].sum() == 0.0


def test_win_propensity_artifact_round_trip(tmp_path: Path) -> None:
    data = _synthetic_data()
    model = fit_win_propensity_model(
        data,
        ("orchard", "harbour"),
        TIER_NAMES,
        max_iter=50,
    )
    path = tmp_path / "win_propensity.json"
    save_win_propensity_model(model, str(path), metadata={"seed": 41})
    assert load_win_propensity_model(str(path)) == model


def test_win_propensity_data_round_trip(tmp_path: Path) -> None:
    data = _synthetic_data()
    path = save_win_propensity_data(data, tmp_path / "propensity_data.npz")
    restored = load_win_propensity_data(path)
    np.testing.assert_array_equal(restored.level_indices, data.level_indices)
    np.testing.assert_array_equal(restored.tier_indices, data.tier_indices)
    np.testing.assert_allclose(restored.skills, data.skills, rtol=1e-6)
    np.testing.assert_allclose(
        restored.served_difficulty, data.served_difficulty, rtol=1e-6
    )
    np.testing.assert_array_equal(restored.outcomes, data.outcomes)


def test_engine_collection_is_deterministic_and_aligned() -> None:
    kwargs = {
        "n_skill_draws": 1,
        "e_grid": (-0.5, 0.5),
        "n_replicates": 1,
        "seed": 73,
    }
    first = collect_win_propensity_data((LEVELS[0],), **kwargs)
    second = collect_win_propensity_data((LEVELS[0],), **kwargs)

    assert len(first.outcomes) == len(TIER_NAMES) * len(kwargs["e_grid"])
    np.testing.assert_array_equal(first.level_indices, second.level_indices)
    np.testing.assert_array_equal(first.tier_indices, second.tier_indices)
    np.testing.assert_array_equal(first.skills, second.skills)
    np.testing.assert_array_equal(first.served_difficulty, second.served_difficulty)
    np.testing.assert_array_equal(first.outcomes, second.outcomes)


def test_goal_count_probe_is_deterministic() -> None:
    first = win_rate(LEVELS[0], goal_count=31, n=5, seed=97)
    second = win_rate(LEVELS[0], goal_count=31, n=5, seed=97)
    assert first == second


def test_shared_goal_totals_calibrate_monotone_thresholds() -> None:
    totals = np.asarray([5, 10, 15, 20, 25, 30])
    easy_goal, easy_rate = goal_count_from_totals(totals, 0.80)
    hard_goal, hard_rate = goal_count_from_totals(totals, 0.20)
    assert easy_goal < hard_goal
    assert easy_rate >= hard_rate


def test_reference_goal_totals_are_deterministic() -> None:
    first = reference_goal_totals(LEVELS[0], n=3, seed=103)
    second = reference_goal_totals(LEVELS[0], n=3, seed=103)
    np.testing.assert_array_equal(first, second)
    assert np.all(first >= 0)


def test_parallel_goal_count_probe_matches_serial() -> None:
    serial = win_rate(LEVELS[1], goal_count=25, n=100, seed=100_003)
    with ProcessPoolExecutor(max_workers=2) as executor:
        parallel = win_rate(
            LEVELS[1],
            goal_count=25,
            n=100,
            seed=100_003,
            executor=executor,
        )
    assert parallel == serial


def test_propensity_split_is_skill_draw_disjoint_and_deterministic() -> None:
    data = _synthetic_data()
    train, validation = split_win_propensity_data(data, seed=19)
    train_again, validation_again = split_win_propensity_data(data, seed=19)

    train_skills = {tuple(row) for row in train.skills}
    validation_skills = {tuple(row) for row in validation.skills}
    assert train_skills.isdisjoint(validation_skills)
    assert len(train.outcomes) + len(validation.outcomes) == len(data.outcomes)
    np.testing.assert_array_equal(train.skills, train_again.skills)
    np.testing.assert_array_equal(validation.skills, validation_again.skills)


def test_propensity_design_summary_infers_factorial_dimensions() -> None:
    data = collect_win_propensity_data(
        (LEVELS[0],),
        n_skill_draws=2,
        e_grid=(-0.5, 0.5),
        n_replicates=1,
        seed=23,
    )
    summary = win_propensity_design_summary(data)
    assert summary == {
        "n_observations": 12,
        "n_skill_draws": 2,
        "n_levels": 1,
        "n_tiers": 3,
        "n_replicates": 1,
        "e_grid": [-0.5, 0.5],
    }


def test_goal_table_validation_uses_independent_seeded_rollouts() -> None:
    table = {
        LEVELS[0].name: {
            "move_budget": 20,
            "goal_colour": 1,
            "curve": [{"E": 0.0, "goal_count": 29}],
        }
    }
    first = validate_goal_count_table(
        table, (LEVELS[0],), n=2, seed=101
    )
    second = validate_goal_count_table(
        table, (LEVELS[0],), n=2, seed=101
    )
    assert first == second
    assert first["levels"][LEVELS[0].name][0]["target"] == 0.5