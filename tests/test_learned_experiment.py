from __future__ import annotations

import json

from match3_simulator.learned_model.experiment import (
    LearnedExperimentConfig,
    run_learned_experiment,
)
from match3_simulator.learned_model.evaluate import (
    load_action_policy_checkpoint,
    load_continuous_vae_checkpoint,
)


def test_tiny_player_disjoint_experiment_runs_end_to_end(tmp_path) -> None:
    report = run_learned_experiment(
        LearnedExperimentConfig(
            n_players=3,
            max_attempts=2,
            target_attempt=2,
            win_head_first_attempt=2,
            action_epochs=1,
            vae_epochs=1,
            kl_warmup_epochs=1,
            win_head_epochs=2,
            win_head_patience=1,
            action_batch_size=32,
            players_per_batch=1,
            d_model=8,
            n_layers=1,
            n_heads=2,
            encoder_hidden_size=8,
            seed=401,
        ),
        output_dir=tmp_path,
    )
    assert report["partitions"]["train_players"] == 1
    assert report["partitions"]["validation_players"] == 1
    assert report["partitions"]["test_players"] == 1
    assert report["model_config"]["action"]["use_immediate_features"] is True
    assert set(report["action_models"]) == {
        "state_only",
        "oracle_skill",
        "inferred_skill",
    }
    assert report["continuous_vae"]["win_head_fit"]["epochs_run"] >= 1
    assert (tmp_path / "state_only_action.pt").is_file()
    assert (tmp_path / "oracle_skill_action.pt").is_file()
    assert (tmp_path / "continuous_vae.pt").is_file()
    assert (tmp_path / "evaluation_players.npz").is_file()
    assert load_action_policy_checkpoint(
        tmp_path / "state_only_action.pt"
    ).training is False
    assert load_action_policy_checkpoint(
        tmp_path / "oracle_skill_action.pt"
    ).training is False
    assert load_continuous_vae_checkpoint(
        tmp_path / "continuous_vae.pt"
    ).training is False
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert saved["schema_version"] == 1