from __future__ import annotations

import json

from match3_simulator.learned_model.arms import ModelArm
from match3_simulator.learned_model.matched_experiment import (
    MatchedGenerativeSmokeConfig,
    run_matched_generative_smoke,
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