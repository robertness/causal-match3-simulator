from __future__ import annotations

import numpy as np
import pyro
import pyro.poutine as poutine

from match3_simulator import LEVELS, PlayerSkill, ground_truth_model
from match3_simulator.evidence import (
    EVIDENCE_NAMES,
    EVIDENCE_Q_MATRIX,
    EVIDENCE_SPECS,
    evidence_expectations,
    evidence_metadata,
    sample_evidence,
)


def test_q_matrix_is_sparse_full_rank_and_covers_each_skill() -> None:
    assert EVIDENCE_Q_MATRIX.shape == (12, 4)
    assert np.linalg.matrix_rank(EVIDENCE_Q_MATRIX) == 4
    assert np.all(np.count_nonzero(EVIDENCE_Q_MATRIX, axis=0) >= 3)
    assert np.count_nonzero(EVIDENCE_Q_MATRIX) <= 13
    np.testing.assert_array_equal(
        EVIDENCE_Q_MATRIX,
        np.asarray([spec.loading for spec in EVIDENCE_SPECS]),
    )


def test_sampled_evidence_respects_declared_support() -> None:
    pyro.set_rng_seed(307)
    values = sample_evidence(PlayerSkill((0.0, 0.0, 0.0, 0.0)), LEVELS[0])
    assert values.shape == (len(EVIDENCE_NAMES),)
    assert np.isfinite(values).all()
    for value, spec in zip(values, EVIDENCE_SPECS):
        if spec.distribution == "beta":
            assert 0.0 < value < 1.0
        elif spec.distribution in {"log_normal", "poisson"}:
            assert value >= 0.0
        if spec.distribution == "poisson":
            assert value == int(value)


def test_each_skill_moves_its_primary_expectations_in_declared_direction() -> None:
    level = LEVELS[0]
    for skill_index in range(4):
        low = [0.0] * 4
        high = [0.0] * 4
        low[skill_index] = -1.0
        high[skill_index] = 1.0
        low_mean = evidence_expectations(PlayerSkill(tuple(low)), level)
        high_mean = evidence_expectations(PlayerSkill(tuple(high)), level)
        rows = np.flatnonzero(EVIDENCE_Q_MATRIX[:, skill_index])
        assert len(rows) >= 3
        for row in rows:
            loading = EVIDENCE_Q_MATRIX[row, skill_index]
            assert np.sign(high_mean[row] - low_mean[row]) == np.sign(loading)


def test_metadata_records_distribution_timing_direction_and_loadings() -> None:
    metadata = evidence_metadata()
    assert len(metadata) == len(EVIDENCE_NAMES)
    for row in metadata:
        assert row["distribution"] in {"beta", "log_normal", "poisson", "normal"}
        assert row["timing"] == "post_episode"
        assert row["direction"] in {"higher", "lower"}
        assert len(row["loading"]) == 4


def test_distribution_specific_dispersion_scales_are_plausible() -> None:
    log_normal_scales = [
        spec.dispersion
        for spec in EVIDENCE_SPECS
        if spec.distribution == "log_normal"
    ]
    beta_concentrations = [
        spec.dispersion
        for spec in EVIDENCE_SPECS
        if spec.distribution == "beta"
    ]
    assert all(0.0 < value < 1.0 for value in log_normal_scales)
    assert all(value > 2.0 for value in beta_concentrations)


def test_evidence_is_emitted_after_episode_outcome() -> None:
    pyro.set_rng_seed(311)
    trace = poutine.trace(ground_truth_model).get_trace(max_steps=1)
    names = list(trace.nodes)
    assert names.index("R") < names.index("X/search_latency")
    assert names.index("X/search_latency") < names.index("X")


def test_clamped_evidence_does_not_change_gameplay() -> None:
    zeros = np.zeros(len(EVIDENCE_NAMES))
    pyro.set_rng_seed(313)
    sampled = ground_truth_model(max_steps=2)
    pyro.set_rng_seed(313)
    clamped = ground_truth_model(max_steps=2, evidence=zeros)
    assert sampled.R == clamped.R
    assert sampled.actions == clamped.actions
    np.testing.assert_array_equal(clamped.proxy, zeros)