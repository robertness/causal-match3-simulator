from __future__ import annotations

import math

import numpy as np
import torch

from match3_simulator.learned_model.data import load_gameplay_transition_dataset
from match3_simulator.learned_model.generative import (
    GameplayRSSM,
    GameplayRSSMConfig,
)
from match3_simulator.learned_model.train import train_gameplay_rssm_step
from match3_simulator.simulate import simulate, write_transitions


def test_logged_transitions_drive_one_cpu_rssm_training_step(tmp_path) -> None:
    episodes = simulate(3, seed=3701)
    path = write_transitions(episodes, tmp_path / "transitions.npz")
    dataset = load_gameplay_transition_dataset(path)
    selected_episodes = dataset.episodes[:2]
    batch = dataset.batch(selected_episodes, device=torch.device("cpu"))

    assert set(batch) == {
        "boards",
        "next_boards",
        "actions",
        "goal_colours",
        "moves_left",
        "goals_left",
        "next_moves_left",
        "next_goals_left",
        "levels",
        "tiers",
        "served_difficulty",
        "step_mask",
    }
    assert batch["boards"].shape[:2] == batch["step_mask"].shape
    assert batch["boards"].shape[-1] == 64
    assert int(batch["step_mask"].sum()) == int(
        np.isin(dataset.episode_ids, selected_episodes).sum()
    )
    assert batch["levels"].shape == (2,)
    assert batch["tiers"].shape == (2,)
    assert batch["served_difficulty"].shape == (2,)
    assert int(batch["actions"][batch["step_mask"]].max()) < 128

    torch.manual_seed(3703)
    model = GameplayRSSM(
        GameplayRSSMConfig(
            embedding_size=8,
            observation_size=16,
            task_context_size=8,
            hidden_size=16,
            stochastic_size=8,
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = model.rssm.recurrent.weight_hh.detach().clone()
    metrics = train_gameplay_rssm_step(
        model,
        batch,
        optimizer,
        kl_weight=0.2,
        gradient_clip=1.0,
    )

    assert {"loss", "board_nll", "counter_mse", "kl", "gradient_norm"} == set(
        metrics
    )
    assert all(math.isfinite(value) for value in metrics.values())
    assert not torch.equal(before, model.rssm.recurrent.weight_hh.detach())