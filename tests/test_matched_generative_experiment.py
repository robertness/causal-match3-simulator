from __future__ import annotations

import json

from match3_simulator.learned_model.arms import ModelArm
from match3_simulator.learned_model.matched_experiment import (
    MatchedGenerativeSmokeConfig,
    MatchedGenerativeTrainConfig,
    run_matched_generative_smoke,
    run_matched_generative_experiment,
)


def test_all_arm_smoke_runs_one_real_step_and_saves_checkpoints(tmp_path) -> None:
    report = run_matched_generative_smoke(
        MatchedGenerativeSmokeConfig(
            n_players=3,
            target_attempt=2,
            embedding_size=4,
            observation_size=8,
            task_context_size=4,
            hidden_size=8,
            stochastic_size=4,
            behavior_size=8,
            prefix_hidden_size=8,
            seed=3901,
        ),
        output_dir=tmp_path,
    )

    assert report["schema_version"] == 1
    assert report["n_players"] == 3
    assert set(report["arms"]) == {arm.value for arm in ModelArm}
    for arm in ModelArm:
        metrics = report["arms"][arm.value]
        assert metrics["loss"] > 0
        assert metrics["parameters"]["total"] > 0
        assert metrics["runtime_seconds"] >= 0
        assert (tmp_path / metrics["checkpoint"]).is_file()
    assert json.loads((tmp_path / "metrics.json").read_text()) == report


def test_matched_experiment_trains_disjoint_splits_and_restores_best(tmp_path) -> None:
    report = run_matched_generative_experiment(
        MatchedGenerativeTrainConfig(
            n_players=3,
            max_attempts=2,
            target_attempts=(2,),
            batch_size=1,
            epochs=2,
            patience=1,
            min_delta=1e9,
            embedding_size=4,
            observation_size=8,
            task_context_size=4,
            hidden_size=8,
            stochastic_size=4,
            behavior_size=8,
            prefix_hidden_size=8,
            seed=3901,
        ),
        output_dir=tmp_path,
    )

    assert report["schema_version"] == 2
    split_ids = [set(values["player_ids"]) for values in report["splits"].values()]
    assert sum(map(len, split_ids)) == 3
    assert all(
        split_ids[left].isdisjoint(split_ids[right])
        for left in range(3)
        for right in range(left)
    )
    for arm in ModelArm:
        metrics = report["arms"][arm.value]
        assert metrics["epochs_ran"] == 2
        assert metrics["best_epoch"] == 1
        assert metrics["train_updates"] == 2
        assert metrics["validation"]["loss"] > 0
        assert metrics["test"]["loss"] > 0
        assert (tmp_path / metrics["checkpoint"]).is_file()
    assert json.loads((tmp_path / "metrics.json").read_text()) == report


def test_production_config_includes_every_attempt_by_default() -> None:
    config = MatchedGenerativeTrainConfig()
    assert config.target_attempts == tuple(range(1, config.max_attempts + 1))