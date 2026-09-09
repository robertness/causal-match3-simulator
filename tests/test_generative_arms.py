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
            proxy_dimensions=4,
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
    batch, episodes, steps, proxies = 2, 3, 2, 4
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