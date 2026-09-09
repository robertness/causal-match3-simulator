from __future__ import annotations

import math

import numpy as np
import torch

from match3_simulator.calibrate import load_win_propensity_model
from match3_simulator.learned_model.batching import build_generative_training_batch
from match3_simulator.learned_model.data import (
    gameplay_transition_dataset_from_episodes,
    load_gameplay_transition_dataset,
)
from match3_simulator.learned_model.generative import (
    GameplayRSSM,
    GameplayRSSMConfig,
)
from match3_simulator.learned_model.train import train_gameplay_rssm_step
from match3_simulator.simulate import (
    simulate,
    simulate_players,
    write_transitions,
)


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


def test_episode_converter_matches_logged_transition_contract() -> None:
    episodes = simulate(3, seed=3707)
    dataset = gameplay_transition_dataset_from_episodes(
        episodes,
        player_ids=[10, 11, 12],
        attempt_ids=[2, 3, 4],
    )

    assert len(dataset.actions) == sum(len(episode.actions) for episode in episodes)
    assert dataset.episodes.tolist() == [0, 1, 2]
    assert set(dataset.player_ids) == {10, 11, 12}
    assert set(dataset.attempt_ids) == {2, 3, 4}
    batch = dataset.batch(dataset.episodes, device=torch.device("cpu"))
    assert int(batch["step_mask"].sum()) == len(dataset.actions)
    assert batch["boards"].shape[-1] == 64


def test_generative_training_batch_aligns_prefix_target_and_oracle() -> None:
    trajectories = simulate_players(
        2,
        load_win_propensity_model(),
        seed=3711,
        max_attempts=2,
    )
    prefix, target, transitions, oracle_skill = build_generative_training_batch(
        trajectories,
        target_attempt=2,
        device=torch.device("cpu"),
    )

    assert prefix["episode_mask"].shape[0] == 2
    assert target.episode_player.tolist() == [0, 1]
    assert transitions["boards"].shape[0] == 2
    assert transitions["step_mask"].sum() == target.actions.numel()
    assert oracle_skill.shape == (2, 4)
    assert not any(name.startswith("oracle") for name in prefix)
    assert not hasattr(target, "oracle_skill")