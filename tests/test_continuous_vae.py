from __future__ import annotations

import torch

from match3_simulator.learned_model.action_policy import ActionPolicyConfig
from match3_simulator.learned_model.encoder import PrefixEncoderConfig
from match3_simulator.learned_model.model import ContinuousCausalVAE, PredictiveTarget
from match3_simulator.retention import MasteryConfig


def _prefix():
    torch.manual_seed(281)
    batch, episodes, steps, proxies = 2, 2, 2, 12
    return {
        "boards": torch.randint(0, 6, (batch, episodes, steps, 64)),
        "actions": torch.randint(0, 128, (batch, episodes, steps)),
        "moves_left": torch.randint(0, 21, (batch, episodes, steps)),
        "goals_left": torch.randint(0, 40, (batch, episodes, steps)),
        "step_mask": torch.tensor([[[True, True], [False, False]]] * batch),
        "levels": torch.randint(0, 3, (batch, episodes)),
        "tiers": torch.randint(0, 3, (batch, episodes)),
        "served_difficulty": torch.randn(batch, episodes),
        "outcomes": torch.randint(0, 2, (batch, episodes)),
        "proxies": torch.randn(batch, episodes, proxies),
        "episode_mask": torch.tensor([[True, False]] * batch),
        "baseline_evidence": torch.randn(batch, proxies),
    }


def _target():
    torch.manual_seed(283)
    legal = torch.zeros(4, 128, dtype=torch.bool)
    legal[:, :3] = True
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
        mastery_before=torch.tensor([0.55, 0.55]),
        churn=torch.tensor([0.0, 1.0]),
        churn_mask=torch.ones(2, dtype=torch.bool),
        action_player=torch.tensor([0, 0, 1, 1]),
        boards=torch.randint(0, 6, (4, 64)),
        goal_colours=torch.tensor([1, 1, 2, 2]),
        moves_left=torch.tensor([20, 19, 20, 19]),
        goals_left=torch.tensor([25, 23, 28, 27]),
        legal_actions=legal,
        actions=torch.tensor([0, 1, 2, 0]),
    )


def _model():
    torch.manual_seed(293)
    return ContinuousCausalVAE(
        PrefixEncoderConfig(proxy_dimensions=12, hidden_size=16),
        ActionPolicyConfig(d_model=16, n_layers=1, n_heads=2),
    )


def test_predictive_objective_logs_finite_structural_terms() -> None:
    model = _model()
    target = _target()
    result = model.predictive_objective(_prefix(), target, kl_weight=0.5)
    expected = {
        "loss",
        "assignment_nll",
        "evidence_nll",
        "win_nll",
        "churn_nll",
        "action_nll",
        "kl",
    }
    assert expected <= set(result)
    assert all(torch.isfinite(result[name]) for name in expected)
    assert result["skill_sample"].shape == (2, 4)
    rate = MasteryConfig().update_rate
    torch.testing.assert_close(
        result["mastery_after"],
        target.mastery_before
        + rate * (target.outcomes - target.mastery_before),
    )


def test_predictive_objective_backpropagates_through_encoder_and_decoders() -> None:
    model = _model()
    result = model.predictive_objective(_prefix(), _target())
    result["loss"].backward()
    assert model.encoder.posterior.weight.grad is not None
    assert model.action_policy.skill.weight.grad is not None
    assert model.win_head.raw_skill.grad is not None
    assert model.churn_head.raw_deviation.grad is not None


def test_target_mutation_does_not_change_prefix_posterior() -> None:
    model = _model().eval()
    prefix = _prefix()
    first = model.posterior(prefix)
    target = _target()
    target.outcomes.fill_(1.0)
    target.churn.fill_(1.0)
    target.served_difficulty.fill_(100.0)
    second = model.posterior(prefix)
    torch.testing.assert_close(first[0], second[0])
    torch.testing.assert_close(first[1], second[1])


def test_target_outcome_changes_its_decoder_loss() -> None:
    model = _model().eval()
    prefix = _prefix()
    target_zero = _target()
    target_zero.outcomes.zero_()
    target_one = _target()
    target_one.outcomes.fill_(1.0)
    torch.manual_seed(299)
    zero_loss = model.predictive_objective(prefix, target_zero)["win_nll"]
    torch.manual_seed(299)
    one_loss = model.predictive_objective(prefix, target_one)["win_nll"]
    assert not torch.isclose(zero_loss, one_loss)


def test_churn_loss_uses_mastery_not_win_head_output() -> None:
    model = _model().eval()
    prefix = _prefix()
    target = _target()
    baseline = model.predictive_objective(
        prefix, target, sample_posterior=False
    )["churn_nll"]
    with torch.no_grad():
        model.win_head.intercept.add_(100.0)
        model.win_head.raw_skill.add_(100.0)
    changed = model.predictive_objective(
        prefix, target, sample_posterior=False
    )["churn_nll"]
    torch.testing.assert_close(baseline, changed)


def test_masked_pre_landmark_churn_contributes_no_loss() -> None:
    model = _model().eval()
    target = _target()
    target.churn_mask.zero_()
    result = model.predictive_objective(_prefix(), target)
    assert result["churn_nll"] == 0.0