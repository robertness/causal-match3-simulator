from __future__ import annotations

from dataclasses import asdict
import json
from unittest.mock import patch

import pytest
import torch

from match3_simulator.learned_model.action_policy import ActionPolicyConfig
from match3_simulator.learned_model.arms import (
    GenerativeModelConfig,
    GenerativeWorldModel,
    ModelArm,
    build_generative_arms,
)
from match3_simulator.learned_model.encoder import PrefixEncoderConfig
from match3_simulator.learned_model.generative import GameplayRSSMConfig
from match3_simulator.learned_model.matched_evaluate import (
    evaluate_matched_experiment_directory,
    evaluate_matched_imagination_curves,
    evaluate_matched_response_curves,
    imagine_response_curves,
    induced_response_curves,
    load_generative_world_model_checkpoint,
)
from match3_simulator.learned_model.matched_experiment import (
    MatchedGenerativeTrainConfig,
)
from match3_simulator.learned_model.rollout import (
    rollout_learned_dynamics as rollout_learned_dynamics_impl,
    sample_learned_initial_state as sample_learned_initial_state_impl,
)


def _config() -> GenerativeModelConfig:
    return GenerativeModelConfig(
        dynamics=GameplayRSSMConfig(
            embedding_size=4,
            observation_size=8,
            task_context_size=4,
            hidden_size=8,
            stochastic_size=4,
        ),
        prefix=PrefixEncoderConfig(hidden_size=8),
        behavior=ActionPolicyConfig(d_model=8, n_layers=1, n_heads=2),
    )


def _prefix() -> dict[str, torch.Tensor]:
    torch.manual_seed(4001)
    return {
        "boards": torch.randint(0, 6, (2, 1, 2, 64)),
        "actions": torch.randint(0, 128, (2, 1, 2)),
        "moves_left": torch.randint(0, 21, (2, 1, 2)),
        "goals_left": torch.randint(0, 40, (2, 1, 2)),
        "step_mask": torch.ones(2, 1, 2, dtype=torch.bool),
        "levels": torch.tensor([[0], [1]]),
        "tiers": torch.tensor([[0], [2]]),
        "served_difficulty": torch.tensor([[0.2], [-0.3]]),
        "outcomes": torch.tensor([[1], [0]]),
        "proxies": torch.rand(2, 1, 12),
        "episode_mask": torch.ones(2, 1, dtype=torch.bool),
        "baseline_evidence": torch.zeros(2, 12),
    }


def test_response_sweep_infers_causal_context_exactly_once() -> None:
    model = GenerativeWorldModel(ModelArm.CAUSAL, _config())
    with patch.object(
        model, "player_context", wraps=model.player_context
    ) as player_context:
        report = induced_response_curves(
            model,
            _prefix(),
            mastery_before=torch.tensor([0.4, 0.7]),
            e_grid=(-1.0, 0.0, 1.0),
        )

    assert player_context.call_count == 1
    assert report["fixed_player_context"] is True
    assert report["query"] == "structural_g_computation"
    assert all(len(level["rows"]) == 3 for level in report["levels"].values())


def test_matched_response_evaluation_keeps_oracle_skill_isolated() -> None:
    models = build_generative_arms(_config())
    report = evaluate_matched_response_curves(
        models,
        _prefix(),
        mastery_before=torch.tensor([0.4, 0.7]),
        oracle_skill=torch.tensor(
            [[-0.5, 0.2, 0.4, 0.1], [0.8, 0.3, -0.2, 0.5]]
        ),
        e_grid=(-1.0, 1.0),
    )

    assert set(report["arms"]) == {arm.value for arm in ModelArm}
    assert report["e_grid"] == [-1.0, 1.0]
    for arm in ModelArm:
        arm_report = report["arms"][arm.value]
        assert arm_report["arm"] == arm.value
        for level in arm_report["levels"].values():
            assert level["recommended_e"] in (-1.0, 1.0)
            assert all(0.0 <= row["win_probability"] <= 1.0 for row in level["rows"])
            assert all(0.0 <= row["churn_probability"] <= 1.0 for row in level["rows"])


def test_production_checkpoint_reloads_for_fresh_process_evaluation(tmp_path) -> None:
    model = GenerativeWorldModel(ModelArm.CAUSAL, _config())
    path = tmp_path / "causal-best.pt"
    torch.save(
        {
            "schema_version": 2,
            "arm": ModelArm.CAUSAL.value,
            "model_config": asdict(model.config),
            "model_state_dict": model.state_dict(),
        },
        path,
    )

    loaded = load_generative_world_model_checkpoint(path)

    assert loaded.arm is ModelArm.CAUSAL
    assert loaded.training is False
    for name, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], value)
    with pytest.raises(ValueError, match="checkpoint arm"):
        load_generative_world_model_checkpoint(
            path, expected_arm=ModelArm.ORACLE
        )


def test_free_running_curve_holds_context_fixed_across_interventions() -> None:
    model = GenerativeWorldModel(ModelArm.CAUSAL, _config())
    with (
        patch.object(
            model, "player_context", wraps=model.player_context
        ) as player_context,
        patch(
            "match3_simulator.learned_model.matched_evaluate."
            "sample_learned_initial_state",
            wraps=sample_learned_initial_state_impl,
        ) as initial_state,
        patch(
            "match3_simulator.learned_model.matched_evaluate."
            "rollout_learned_dynamics",
            wraps=rollout_learned_dynamics_impl,
        ) as dynamics,
    ):
        report = imagine_response_curves(
            model,
            _prefix(),
            mastery_before=torch.tensor([0.4, 0.7]),
            e_grid=(-1.0, 1.0),
            rollouts_per_player=1,
            max_steps=1,
            stochastic=False,
            seed=4021,
        )

    assert player_context.call_count == 1
    initial_seeds = [call.kwargs["seed"] for call in initial_state.call_args_list]
    dynamics_seeds = [call.kwargs["seed"] for call in dynamics.call_args_list]
    assert all(left != right for left, right in zip(initial_seeds, dynamics_seeds))
    assert initial_seeds[:6] == initial_seeds[6:12]
    assert dynamics_seeds[:6] == dynamics_seeds[6:12]
    assert report["query"] == "free_running_rssm_g_computation"
    assert report["initial_state"] == "learned_task_decoder"
    assert report["fixed_player_context"] is True
    for level in report["levels"].values():
        assert len(level["rows"]) == 2
        assert level["recommended_e"] in (-1.0, 1.0)
        assert all(row["n_rollouts"] == 6 for row in level["rows"])


def test_matched_free_running_curves_cover_all_arms() -> None:
    report = evaluate_matched_imagination_curves(
        build_generative_arms(_config()),
        _prefix(),
        mastery_before=torch.tensor([0.4, 0.7]),
        oracle_skill=torch.zeros(2, 4),
        e_grid=(0.0,),
        rollouts_per_player=1,
        max_steps=0,
        stochastic=False,
        seed=4027,
    )

    assert report["query"] == "matched_free_running_rssm_g_computation"
    assert set(report["arms"]) == {arm.value for arm in ModelArm}
    assert all(
        arm_report["query"] == "free_running_rssm_g_computation"
        for arm_report in report["arms"].values()
    )


def test_experiment_directory_reloads_best_arms_and_test_prefix(tmp_path) -> None:
    config = MatchedGenerativeTrainConfig(
        n_players=3,
        max_attempts=1,
        target_attempts=(1,),
        batch_size=1,
        epochs=1,
        patience=1,
        embedding_size=4,
        observation_size=8,
        task_context_size=4,
        hidden_size=8,
        stochastic_size=4,
        behavior_size=8,
        prefix_hidden_size=8,
        seed=4051,
    )
    models = build_generative_arms(_config())
    arms = {}
    for arm, model in models.items():
        checkpoint = f"{arm.value}-best.pt"
        torch.save(
            {
                "schema_version": 2,
                "arm": arm.value,
                "model_config": asdict(model.config),
                "model_state_dict": model.state_dict(),
            },
            tmp_path / checkpoint,
        )
        arms[arm.value] = {"checkpoint": checkpoint}
    (tmp_path / "metrics.json").write_text(
        json.dumps({"config": asdict(config), "arms": arms})
    )

    report = evaluate_matched_experiment_directory(
        tmp_path,
        target_attempt=1,
        e_grid=(0.0,),
        rollouts_per_player=1,
        max_steps=0,
        stochastic=False,
    )

    assert report["cohort"]["split"] == "test"
    assert report["cohort"]["target_attempt"] == 1
    assert set(report["structural"]["arms"]) == {arm.value for arm in ModelArm}
    assert set(report["imagination"]["arms"]) == {arm.value for arm in ModelArm}
    assert json.loads((tmp_path / "response-curves.json").read_text()) == report