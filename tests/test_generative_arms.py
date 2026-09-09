from __future__ import annotations

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
from match3_simulator.learned_model.model import PredictiveTarget
from match3_simulator.learned_model.train import train_generative_world_model_step


def _config() -> GenerativeModelConfig:
    return GenerativeModelConfig(
        dynamics=GameplayRSSMConfig(
            embedding_size=8,
            observation_size=16,
            task_context_size=8,
            hidden_size=16,
            stochastic_size=8,
        ),
        prefix=PrefixEncoderConfig(
            proxy_dimensions=12,
            hidden_size=16,
        ),
        behavior=ActionPolicyConfig(
            d_model=16,
            n_layers=1,
            n_heads=2,
        ),
    )


def _prefix() -> dict[str, torch.Tensor]:
    torch.manual_seed(3801)
    batch, episodes, steps, proxies = 2, 3, 2, 12
    episode_mask = torch.tensor(
        [[True, True, False], [True, False, False]]
    )
    return {
        "boards": torch.randint(0, 6, (batch, episodes, steps, 64)),
        "actions": torch.randint(0, 128, (batch, episodes, steps)),
        "moves_left": torch.randint(0, 23, (batch, episodes, steps)),
        "goals_left": torch.randint(0, 40, (batch, episodes, steps)),
        "step_mask": episode_mask.unsqueeze(-1).expand(-1, -1, steps).clone(),
        "levels": torch.randint(0, 3, (batch, episodes)),
        "tiers": torch.randint(0, 3, (batch, episodes)),
        "served_difficulty": torch.randn(batch, episodes),
        "outcomes": torch.randint(0, 2, (batch, episodes)),
        "proxies": torch.randn(batch, episodes, proxies),
        "episode_mask": episode_mask,
        "baseline_evidence": torch.randn(batch, proxies),
    }


def _transition_batch() -> dict[str, torch.Tensor]:
    torch.manual_seed(3802)
    batch, steps = 2, 3
    return {
        "boards": torch.randint(0, 6, (batch, steps, 64)),
        "next_boards": torch.randint(0, 6, (batch, steps, 64)),
        "actions": torch.randint(0, 128, (batch, steps)),
        "goal_colours": torch.randint(0, 6, (batch, steps)),
        "moves_left": torch.randint(1, 23, (batch, steps)),
        "goals_left": torch.randint(1, 40, (batch, steps)),
        "next_moves_left": torch.randint(0, 22, (batch, steps)),
        "next_goals_left": torch.randint(0, 40, (batch, steps)),
        "levels": torch.tensor([0, 2]),
        "tiers": torch.tensor([1, 0]),
        "served_difficulty": torch.tensor([-0.5, 0.75]),
        "step_mask": torch.ones(batch, steps, dtype=torch.bool),
    }


def _target() -> PredictiveTarget:
    torch.manual_seed(3804)
    legal_actions = torch.zeros(3, 128, dtype=torch.bool)
    legal_actions[:, :4] = True
    evidence = torch.tensor(
        [[2.0, 0.5, 1.0, 0.4, 0.6, 0.5, 0.5, 0.0, 0.4, 0.5, 1.2, 0.5]]
    ).expand(2, -1)
    return PredictiveTarget(
        episode_player=torch.tensor([0, 1]),
        served_difficulty=torch.tensor([-0.2, 0.4]),
        levels=torch.tensor([0, 1]),
        tiers=torch.tensor([1, 2]),
        evidence=evidence,
        outcomes=torch.tensor([1.0, 0.0]),
        mastery_before=torch.tensor([0.4, 0.3]),
        churn=torch.tensor([0.0, 1.0]),
        churn_mask=torch.ones(2, dtype=torch.bool),
        action_player=torch.tensor([0, 1, 0]),
        boards=torch.randint(0, 6, (3, 64)),
        goal_colours=torch.tensor([1, 2, 1]),
        moves_left=torch.tensor([20, 19, 18]),
        goals_left=torch.tensor([24, 20, 17]),
        legal_actions=legal_actions,
        actions=torch.tensor([0, 1, 2]),
    )


def test_matched_causal_and_oracle_arms_have_exact_parameter_parity() -> None:
    models = build_generative_arms(_config())

    assert set(models) == set(ModelArm)
    assert len(
        {
            models[arm].parameter_counts()["total"]
            for arm in (
                ModelArm.MATCHED_NO_K,
                ModelArm.CAUSAL,
                ModelArm.ORACLE,
            )
        }
    ) == 1
    assert len(
        {model.parameter_counts()["dynamics"] for model in models.values()}
    ) == 1
    assert len(
        {model.parameter_counts()["behavior"] for model in models.values()}
    ) == 1
    assert (
        models[ModelArm.POOLED].parameter_counts()["total"]
        < models[ModelArm.CAUSAL].parameter_counts()["total"]
    )


def test_arm_context_sources_are_strict_and_stable() -> None:
    prefix = _prefix()
    causal = GenerativeWorldModel(ModelArm.CAUSAL, _config()).eval()
    pooled = GenerativeWorldModel(ModelArm.POOLED, _config()).eval()
    matched = GenerativeWorldModel(ModelArm.MATCHED_NO_K, _config()).eval()
    oracle = GenerativeWorldModel(ModelArm.ORACLE, _config()).eval()

    causal_context = causal.player_context(prefix=prefix, sample=False)
    repeated = causal.player_context(prefix=prefix, sample=False)
    pooled_context = pooled.player_context(prefix=prefix, sample=False)
    matched_context = matched.player_context(prefix=prefix, sample=False)
    oracle_skill = torch.randn(2, 4)
    oracle_context = oracle.player_context(
        prefix=prefix,
        oracle_skill=oracle_skill,
        sample=False,
    )

    assert causal_context.skill.shape == (2, 4)
    assert causal_context.kl.shape == (2,)
    assert torch.isfinite(causal_context.skill).all()
    assert torch.isfinite(causal_context.kl).all()
    torch.testing.assert_close(causal_context.skill, repeated.skill)
    torch.testing.assert_close(pooled_context.skill, torch.zeros(2, 4))
    torch.testing.assert_close(matched_context.skill, torch.zeros(2, 4))
    torch.testing.assert_close(oracle_context.skill, oracle_skill)
    assert oracle_context.posterior_mean is None

    with pytest.raises(ValueError, match="oracle_skill is restricted"):
        causal.player_context(
            prefix=prefix,
            oracle_skill=oracle_skill,
            sample=False,
        )
    with pytest.raises(ValueError, match="requires oracle_skill"):
        oracle.player_context(prefix=prefix, sample=False)


def test_behavior_path_indexes_resolved_player_context() -> None:
    torch.manual_seed(3803)
    model = GenerativeWorldModel(ModelArm.CAUSAL, _config()).eval()
    context = model.player_context(prefix=_prefix(), sample=False)
    legal_actions = torch.zeros(3, 128, dtype=torch.bool)
    legal_actions[:, :4] = True
    log_probabilities = model.behavior_log_probabilities(
        boards=torch.randint(0, 6, (3, 64)),
        goal_colours=torch.tensor([1, 2, 3]),
        moves_left=torch.tensor([20, 19, 18]),
        goals_left=torch.tensor([24, 20, 17]),
        legal_actions=legal_actions,
        player_context=context.skill,
        action_players=torch.tensor([0, 1, 0]),
    )

    assert log_probabilities.shape == (3, 128)
    assert torch.isfinite(log_probabilities[:, :4]).all()
    torch.testing.assert_close(
        log_probabilities[:, :4].exp().sum(dim=1), torch.ones(3)
    )


def test_all_arms_share_skill_free_transition_mechanics() -> None:
    results = []
    for arm in ModelArm:
        torch.manual_seed(3805)
        model = GenerativeWorldModel(arm, _config()).eval()
        results.append(
            model.transition_objective(
                _transition_batch(), sample_posterior=False
            )
        )

    reference = results[0]
    for result in results[1:]:
        torch.testing.assert_close(result["loss"], reference["loss"])
        torch.testing.assert_close(
            result["board_logits"], reference["board_logits"]
        )
        torch.testing.assert_close(
            result["counter_mean"], reference["counter_mean"]
        )


@pytest.mark.parametrize("arm", list(ModelArm))
def test_every_arm_backpropagates_one_full_generative_objective(
    arm: ModelArm,
) -> None:
    torch.manual_seed(3807)
    model = GenerativeWorldModel(arm, _config())
    oracle_skill = torch.randn(2, 4) if arm is ModelArm.ORACLE else None
    result = model.objective(
        prefix=_prefix(),
        target=_target(),
        transitions=_transition_batch(),
        oracle_skill=oracle_skill,
        context_kl_weight=0.2,
        dynamics_kl_weight=0.2,
    )

    assert {
        "loss",
        "assignment_nll",
        "evidence_nll",
        "win_nll",
        "churn_nll",
        "action_nll",
        "context_kl",
        "board_nll",
        "counter_mse",
        "dynamics_kl",
    } <= set(result)
    assert all(torch.isfinite(result[name]) for name in result if name != "context")
    result["loss"].backward()
    assert model.dynamics.rssm.recurrent.weight_hh.grad is not None
    assert model.behavior.action_head[0].weight.grad is not None
    assert model.assignment_head.raw_skill.grad is not None
    assert model.win_head.raw_skill.grad is not None
    assert model.churn_head.raw_deviation.grad is not None
    if arm is ModelArm.CAUSAL:
        assert model.prefix_encoder is not None
        assert model.prefix_encoder.posterior.weight.grad is not None


@pytest.mark.parametrize("arm", list(ModelArm))
def test_every_arm_completes_one_optimizer_step(arm: ModelArm) -> None:
    torch.manual_seed(3809)
    model = GenerativeWorldModel(arm, _config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = model.dynamics.rssm.recurrent.weight_hh.detach().clone()
    metrics = train_generative_world_model_step(
        model,
        prefix=_prefix(),
        target=_target(),
        transitions=_transition_batch(),
        optimizer=optimizer,
        oracle_skill=torch.randn(2, 4) if arm is ModelArm.ORACLE else None,
        context_kl_weight=0.2,
        dynamics_kl_weight=0.2,
    )

    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert not torch.equal(before, model.dynamics.rssm.recurrent.weight_hh.detach())