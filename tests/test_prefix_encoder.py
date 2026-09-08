from __future__ import annotations

import torch

from match3_simulator.learned_model.encoder import (
    CausalPrefixEncoder,
    PrefixEncoderConfig,
)


def _inputs():
    torch.manual_seed(251)
    batch, episodes, steps, proxies = 2, 3, 2, 4
    episode_mask = torch.tensor([[True, True, False], [False, False, False]])
    step_mask = episode_mask.unsqueeze(-1).expand(-1, -1, steps).clone()
    return {
        "boards": torch.randint(0, 6, (batch, episodes, steps, 64)),
        "actions": torch.randint(0, 128, (batch, episodes, steps)),
        "moves_left": torch.randint(0, 21, (batch, episodes, steps)),
        "goals_left": torch.randint(0, 40, (batch, episodes, steps)),
        "step_mask": step_mask,
        "levels": torch.randint(0, 3, (batch, episodes)),
        "tiers": torch.randint(0, 3, (batch, episodes)),
        "served_difficulty": torch.randn(batch, episodes),
        "outcomes": torch.randint(0, 2, (batch, episodes)),
        "proxies": torch.randn(batch, episodes, proxies),
        "episode_mask": episode_mask,
        "baseline_evidence": torch.randn(batch, proxies),
    }


def _encoder():
    torch.manual_seed(257)
    return CausalPrefixEncoder(
        PrefixEncoderConfig(proxy_dimensions=4, hidden_size=16)
    ).eval()


def test_encoder_returns_finite_whitened_gaussian_parameters() -> None:
    encoder = _encoder()
    mean, log_scale = encoder(**_inputs())
    assert mean.shape == (2, 4)
    assert log_scale.shape == (2, 4)
    assert torch.isfinite(mean).all()
    assert torch.isfinite(log_scale).all()
    assert torch.isfinite(encoder.kl_standard_normal(mean, log_scale)).all()


def test_masked_current_and_future_episodes_cannot_change_posterior() -> None:
    encoder = _encoder()
    inputs = _inputs()
    first_mean, first_scale = encoder(**inputs)

    mutated = {name: value.clone() for name, value in inputs.items()}
    masked = ~inputs["episode_mask"]
    mutated["boards"][masked] = 5
    mutated["actions"][masked] = 127
    mutated["moves_left"][masked] = 20
    mutated["goals_left"][masked] = 99
    mutated["levels"][masked] = 2
    mutated["tiers"][masked] = 2
    mutated["served_difficulty"][masked] = 100.0
    mutated["outcomes"][masked] = 1
    mutated["proxies"][masked] = 100.0
    second_mean, second_scale = encoder(**mutated)

    torch.testing.assert_close(first_mean, second_mean)
    torch.testing.assert_close(first_scale, second_scale)


def test_noncontiguous_episode_mask_is_rejected() -> None:
    encoder = _encoder()
    inputs = _inputs()
    inputs["episode_mask"][0] = torch.tensor([True, False, True])
    try:
        encoder(**inputs)
    except ValueError as error:
        assert "contiguous prefix" in str(error)
    else:
        raise AssertionError("noncontiguous prefix mask was accepted")


def test_reparameterized_sample_backpropagates_to_posterior() -> None:
    encoder = _encoder().train()
    mean, log_scale = encoder(**_inputs())
    sample = encoder.rsample(mean, log_scale)
    sample.square().mean().backward()
    assert encoder.posterior.weight.grad is not None
    assert torch.linalg.vector_norm(encoder.posterior.weight.grad) > 0