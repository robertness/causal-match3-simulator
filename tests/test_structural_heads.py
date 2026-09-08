from __future__ import annotations

import inspect

import numpy as np
import torch

from match3_simulator.learned_model.heads import (
    AssignmentHead,
    ChurnHead,
    CorrelatedSkillTransform,
    EvidenceHead,
    WinHead,
)
from match3_simulator.evidence import EVIDENCE_Q_MATRIX
from match3_simulator.spec import SKILL_COVARIANCE


def test_correlated_transform_recovers_configured_covariance() -> None:
    torch.manual_seed(271)
    transform = CorrelatedSkillTransform()
    whitened = torch.randn(100_000, 4)
    skill = transform(whitened).detach().numpy()
    np.testing.assert_allclose(np.cov(skill, rowvar=False), SKILL_COVARIANCE, atol=0.02)


def test_assignment_head_has_no_d_generation_or_skill_to_d_path() -> None:
    head = AssignmentHead()
    assert "skill" in inspect.signature(head.mean).parameters
    assert not hasattr(head, "difficulty_head")
    log_prob = head.log_prob(
        torch.zeros(3),
        torch.zeros(3, 4),
        torch.tensor([0, 1, 2]),
        torch.tensor([0, 1, 2]),
    )
    assert torch.isfinite(log_prob).all()
    assert head.skill_coefficients.shape == (3, 4)
    assert head.scale.shape == (3,)
    assert torch.all(head.skill_coefficients > 0)


def test_win_head_is_monotone_in_served_difficulty_and_skill() -> None:
    head = WinHead()
    level = torch.zeros(1, dtype=torch.long)
    tier = torch.zeros(1, dtype=torch.long)
    low_e = head.probabilities(torch.tensor([-1.0]), torch.zeros(1, 4), level, tier)
    high_e = head.probabilities(torch.tensor([1.0]), torch.zeros(1, 4), level, tier)
    low_skill = head.probabilities(torch.zeros(1), -torch.ones(1, 4), level, tier)
    high_skill = head.probabilities(torch.zeros(1), torch.ones(1, 4), level, tier)
    assert low_e > high_e
    assert high_skill > low_skill


def test_churn_head_is_minimized_at_target_and_symmetric() -> None:
    head = ChurnHead()
    probabilities = head.probabilities(
        torch.tensor([0.35, 0.55, 0.75]), torch.zeros(3, dtype=torch.long)
    )
    assert probabilities[1] < probabilities[0]
    assert probabilities[1] < probabilities[2]
    torch.testing.assert_close(probabilities[0], probabilities[2])


def test_churn_head_has_level_specific_positive_curvature() -> None:
    head = ChurnHead()
    probability = torch.tensor([0.25, 0.25, 0.25])
    output = head.probabilities(probability, torch.tensor([0, 1, 2]))
    assert torch.all(head.deviation_coefficient > 0)
    assert output[0] > output[1]
    torch.testing.assert_close(output[1], output[2])


def test_evidence_head_returns_one_log_density_per_row() -> None:
    head = EvidenceHead()
    evidence = torch.tensor(
        [[2.0, 0.5, 1.0, 0.4, 0.6, 0.5, 0.5, 0.0, 0.4, 0.5, 1.2, 0.5]]
    ).expand(7, -1)
    log_prob = head.log_prob(
        evidence, torch.zeros(7, 4), torch.zeros(7, dtype=torch.long)
    )
    assert log_prob.shape == (7,)
    assert torch.isfinite(log_prob).all()


def test_evidence_head_preserves_q_matrix_sparsity_and_signs() -> None:
    head = EvidenceHead()
    loadings = head.loadings.detach().numpy()
    np.testing.assert_array_equal(loadings == 0, EVIDENCE_Q_MATRIX == 0)
    np.testing.assert_array_equal(
        np.sign(loadings), np.sign(EVIDENCE_Q_MATRIX)
    )