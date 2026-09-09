from __future__ import annotations

import torch

from match3_simulator.learned_model.generative import (
    GameplayRSSM,
    GameplayRSSMConfig,
)


def _model() -> GameplayRSSM:
    torch.manual_seed(3101)
    return GameplayRSSM(
        GameplayRSSMConfig(
            embedding_size=8,
            observation_size=16,
            task_context_size=8,
            hidden_size=16,
            stochastic_size=8,
        )
    )


def _batch() -> dict[str, torch.Tensor]:
    torch.manual_seed(3103)
    batch, steps = 2, 4
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
        "step_mask": torch.tensor(
            [[True, True, True, True], [True, True, False, False]]
        ),
    }


def test_gameplay_rssm_encodes_state_and_task_context() -> None:
    model = _model()
    batch = _batch()
    observation = model.encode_observation(
        batch["boards"],
        batch["goal_colours"],
        batch["moves_left"],
        batch["goals_left"],
    )
    context = model.encode_task(
        batch["levels"], batch["tiers"], batch["served_difficulty"]
    )

    assert observation.shape == (2, 4, 16)
    assert context.shape == (2, 8)
    assert torch.isfinite(observation).all()
    assert torch.isfinite(context).all()


def test_gameplay_rssm_objective_is_masked_finite_and_differentiable() -> None:
    model = _model()
    batch = _batch()
    result = model.objective(**batch, kl_weight=0.2)

    assert {
        "loss",
        "board_nll",
        "counter_mse",
        "kl",
        "board_logits",
    } <= set(result)
    assert result["board_logits"].shape == (2, 4, 64, 6)
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert model.observation_projection[0].weight.grad is not None
    assert model.rssm.recurrent.weight_hh.grad is not None


def test_masked_future_targets_do_not_change_generative_loss() -> None:
    model = _model().eval()
    batch = _batch()
    first = model.objective(**batch, sample_posterior=False)
    mutated = {name: value.clone() for name, value in batch.items()}
    masked = ~batch["step_mask"]
    mutated["next_boards"][masked] = 5
    mutated["next_moves_left"][masked] = 99
    mutated["next_goals_left"][masked] = 99
    second = model.objective(**mutated, sample_posterior=False)

    torch.testing.assert_close(first["loss"], second["loss"])