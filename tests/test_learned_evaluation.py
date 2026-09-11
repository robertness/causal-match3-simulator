from __future__ import annotations

import numpy as np
import torch

from match3_simulator.calibrate import load_win_propensity_model
from match3_simulator.learned_model.action_policy import ActionPolicyConfig
from match3_simulator.learned_model.encoder import PrefixEncoderConfig
from match3_simulator.learned_model.evaluate import (
    evaluate_causal_query,
    evaluate_closed_loop,
    oracle_propensity_curves,
)
from match3_simulator.learned_model.model import ContinuousCausalVAE


def test_closed_loop_evaluation_returns_every_level_and_legal_rollouts() -> None:
    torch.manual_seed(431)
    model = ContinuousCausalVAE(
        PrefixEncoderConfig(hidden_size=8),
        ActionPolicyConfig(d_model=8, n_layers=1, n_heads=2),
    )
    report = evaluate_closed_loop(
        model,
        np.zeros((1, 4), dtype=np.float32),
        e_grid=(0.0,),
        rollouts_per_player=1,
        seed=433,
    )
    assert report["schema_version"] == 1
    assert report["seed"] == 433
    assert set(report["levels"]) == {"orchard", "harbour", "foundry"}
    assert all(rows[0]["n_rollouts"] == 1 for rows in report["levels"].values())
    assert np.isfinite(report["mean_absolute_win_gap"])


def test_oracle_propensity_curves_cover_every_level() -> None:
    model = load_win_propensity_model()
    curves = oracle_propensity_curves(
        model, np.zeros((2, 4)), e_grid=(-1.0, 1.0)
    )
    assert set(curves) == {"orchard", "harbour", "foundry"}
    for rows in curves.values():
        assert rows[0]["win_probability"] > rows[1]["win_probability"]


def test_causal_query_reports_optima_and_oracle_regret() -> None:
    torch.manual_seed(437)
    model = ContinuousCausalVAE(
        PrefixEncoderConfig(hidden_size=8),
        ActionPolicyConfig(d_model=8, n_layers=1, n_heads=2),
    )
    report = evaluate_causal_query(
        model,
        np.zeros((2, 4), dtype=np.float32),
        np.zeros((2, 4), dtype=np.float32),
        load_win_propensity_model(),
        e_grid=(-1.0, 0.0, 1.0),
    )
    assert set(report["levels"]) == {"orchard", "harbour", "foundry"}
    for level in report["levels"].values():
        assert level["oracle_optimum"] in (-1.0, 0.0, 1.0)
        assert level["learned_optimum"] in (-1.0, 0.0, 1.0)
        assert level["oracle_regret"] >= 0.0