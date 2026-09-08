from __future__ import annotations

import csv
from dataclasses import replace
import hashlib
import json

import numpy as np

from match3_simulator import LEVELS, PlayerSkill
from match3_simulator.retention import (
    ChurnConfig,
    MasteryConfig,
    WinPropensityModel,
    simulate_player_trajectory,
    update_mastery,
)
from match3_simulator.scm import TIER_NAMES
from match3_simulator.spec import BENCHMARK_CONFIG
from match3_simulator.trajectory import attempt_summary_row, player_trajectory_to_dict
from match3_simulator.simulate import (
    simulate_players,
    write_attempt_table,
    write_player_dataset,
    write_oracle_attempt_table,
    write_transitions,
)


def _model() -> WinPropensityModel:
    return WinPropensityModel(
        level_names=tuple(level.name for level in LEVELS),
        tier_names=TIER_NAMES,
        intercepts=tuple((0.0, 0.0, 0.0) for _ in LEVELS),
        skill_coefficients=tuple((0.3, 0.3, 0.3, 0.3) for _ in LEVELS),
        difficulty_coefficients=tuple(1.0 for _ in LEVELS),
    )


def test_trajectory_reuses_one_skill_and_updates_mastery() -> None:
    player = PlayerSkill((0.2, -0.1, 0.4, 0.0), "fixed")
    mastery_config = MasteryConfig(initial=0.55, update_rate=0.20)
    trajectory = simulate_player_trajectory(
        _model(),
        player_id=7,
        seed=211,
        max_attempts=BENCHMARK_CONFIG.landmark_attempt,
        player=player,
        churn_config=ChurnConfig(intercept=-20.0),
        mastery_config=mastery_config,
    )
    assert len(trajectory.attempts) == BENCHMARK_CONFIG.landmark_attempt
    assert all(record.episode.player == player for record in trajectory.attempts)
    assert trajectory.attempts[0].mastery_before == mastery_config.initial
    for index, record in enumerate(trajectory.attempts):
        assert record.mastery_after == update_mastery(
            record.mastery_before, record.episode.R, mastery_config
        )
        if index:
            assert record.mastery_before == trajectory.attempts[index - 1].mastery_after
        assert 0.0 <= record.mastery_before <= 1.0
        assert 0.0 <= record.mastery_after <= 1.0


def test_trajectory_stops_at_first_churn() -> None:
    trajectory = simulate_player_trajectory(
        _model(),
        player_id=11,
        seed=223,
        max_attempts=30,
        benchmark=replace(BENCHMARK_CONFIG, landmark_attempt=1),
        churn_config=ChurnConfig(intercept=100.0),
    )
    assert trajectory.churned
    assert trajectory.churn_attempt == 1
    assert len(trajectory.attempts) == 1
    assert trajectory.attempts[-1].churn_after == 1


def test_trajectory_is_reproducible() -> None:
    first = simulate_player_trajectory(
        _model(), player_id=13, seed=227, max_attempts=3
    )
    second = simulate_player_trajectory(
        _model(), player_id=13, seed=227, max_attempts=3
    )
    np.testing.assert_allclose(first.player.as_array(), second.player.as_array())
    assert [record.episode.E for record in first.attempts] == [
        record.episode.E for record in second.attempts
    ]
    assert [record.episode.R for record in first.attempts] == [
        record.episode.R for record in second.attempts
    ]


def test_trajectory_schema_records_causal_and_churn_fields() -> None:
    trajectory = simulate_player_trajectory(
        _model(), player_id=17, seed=229, max_attempts=2
    )
    document = player_trajectory_to_dict(trajectory)
    row = attempt_summary_row(trajectory.attempts[0])

    assert document["version"] == 3
    assert document["player_id"] == 17
    assert len(document["attempts"]) == 2
    required = {
        "player_id",
        "attempt_id",
        "active_before",
        "mastery_before",
        "mastery_after",
        "oracle_win_probability",
        "churn_probability",
        "churn_after",
    }
    assert required <= set(row)


def test_repeated_player_simulation_and_attempt_table(tmp_path) -> None:
    trajectories = simulate_players(
        2, _model(), seed=233, max_attempts=2
    )
    path = write_attempt_table(trajectories, tmp_path / "episodes.csv")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(trajectories) == 2
    assert len(rows) == 4
    assert {row["player_id"] for row in rows} == {"0", "1"}
    assert {row["attempt_id"] for row in rows} == {"1", "2"}
    forbidden = {
        "mastery_before",
        "mastery_after",
        "oracle_win_probability",
        "churn_probability",
        "k_search",
        "k_pattern",
        "k_planning",
        "k_strategy",
    }
    assert forbidden.isdisjoint(rows[0])

    oracle_path = write_oracle_attempt_table(
        trajectories, tmp_path / "oracle" / "attempts.csv"
    )
    with oracle_path.open(newline="") as handle:
        oracle_rows = list(csv.DictReader(handle))
    assert len(oracle_rows) == len(rows)
    assert forbidden <= set(oracle_rows[0])
    assert {"E", "R", "churn_after"}.isdisjoint(oracle_rows[0])

    records = [record for trajectory in trajectories for record in trajectory.attempts]
    transition_path = write_transitions(
        [record.episode for record in records],
        tmp_path / "transitions.npz",
        player_ids=[record.player_id for record in records],
        attempt_ids=[record.attempt_id for record in records],
    )
    with np.load(transition_path) as arrays:
        assert "player_id" in arrays.files
        assert "attempt_id" in arrays.files
        assert set(arrays["player_id"]) == {0, 1}
        assert set(arrays["attempt_id"]) == {1, 2}


def test_player_dataset_manifest_hashes_logged_and_oracle_artifacts(tmp_path) -> None:
    trajectories = simulate_players(2, _model(), seed=239, max_attempts=2)
    manifest_path = write_player_dataset(
        trajectories,
        tmp_path,
        seed=239,
        max_attempts=2,
    )
    manifest = json.loads(manifest_path.read_text())

    assert manifest["schema_version"] == 1
    assert manifest["dataset_type"] == "longitudinal_player_histories"
    assert manifest["counts"]["players"] == 2
    assert manifest["counts"]["attempts"] == sum(
        len(trajectory.attempts) for trajectory in trajectories
    )
    assert set(manifest["logged_artifacts"]) == {"attempts", "transitions"}
    assert set(manifest["oracle_artifacts"]) == {"attempt_state"}
    for group in ("logged_artifacts", "oracle_artifacts"):
        for artifact in manifest[group].values():
            path = tmp_path / artifact["path"]
            assert path.is_file()
            assert artifact["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()