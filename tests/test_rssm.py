from __future__ import annotations

import torch

from match3_simulator.learned_model.rssm import FastRSSM, RSSMConfig


def _model() -> FastRSSM:
    torch.manual_seed(3001)
    return FastRSSM(
        RSSMConfig(
            hidden_size=16,
            stochastic_size=8,
            observation_size=12,
            context_size=4,
            action_size=128,
        )
    )


def test_rssm_observe_returns_aligned_prior_posterior_and_decoders() -> None:
    model = _model()
    batch, steps = 3, 5
    result = model.observe(
        observations=torch.randn(batch, steps, 12),
        actions=torch.randint(0, 128, (batch, steps)),
        context=torch.randn(batch, 4),
    )

    assert result["hidden"].shape == (batch, steps, 16)
    assert result["prior_mean"].shape == (batch, steps, 8)
    assert result["posterior_mean"].shape == (batch, steps, 8)
    assert result["board_logits"].shape == (batch, steps, 64, 6)
    assert result["counter_mean"].shape == (batch, steps, 2)
    assert torch.isfinite(result["kl"]).all()


def test_rssm_context_changes_dynamics_and_decoder_predictions() -> None:
    model = _model().eval()
    observations = torch.randn(2, 3, 12)
    actions = torch.randint(0, 128, (2, 3))
    zero = model.observe(
        observations=observations,
        actions=actions,
        context=torch.zeros(2, 4),
        sample_posterior=False,
    )
    shifted = model.observe(
        observations=observations,
        actions=actions,
        context=torch.ones(2, 4),
        sample_posterior=False,
    )

    assert not torch.allclose(zero["prior_mean"], shifted["prior_mean"])
    assert not torch.allclose(zero["board_logits"], shifted["board_logits"])


def test_rssm_imagine_is_free_running_and_backpropagates() -> None:
    model = _model()
    actions = torch.randint(0, 128, (2, 4))
    result = model.imagine(
        actions=actions,
        context=torch.randn(2, 4),
        sample_prior=False,
    )

    assert result["board_logits"].shape == (2, 4, 64, 6)
    assert result["counter_mean"].shape == (2, 4, 2)
    loss = result["board_logits"].square().mean() + result["counter_mean"].square().mean()
    loss.backward()
    assert model.recurrent.weight_hh.grad is not None
    assert model.board_decoder[-1].weight.grad is not None


def test_repeated_imagine_steps_match_deterministic_batched_imagination() -> None:
    model = _model().eval()
    actions = torch.randint(0, 128, (2, 4))
    context = torch.randn(2, 4)
    expected = model.imagine(
        actions=actions,
        context=context,
        sample_prior=False,
    )
    hidden, stochastic = model.initial_state(2, context)
    steps = []
    for step_index in range(actions.shape[1]):
        result = model.imagine_step(
            hidden=hidden,
            stochastic=stochastic,
            action=actions[:, step_index],
            context=context,
            sample_prior=False,
        )
        hidden = result["hidden"]
        stochastic = result["stochastic"]
        steps.append(result)

    for name in expected:
        actual = torch.stack([step[name] for step in steps], dim=1)
        torch.testing.assert_close(actual, expected[name])


def test_repeated_observe_steps_match_deterministic_filtering() -> None:
    model = _model().eval()
    observations = torch.randn(2, 4, 12)
    actions = torch.randint(0, 128, (2, 4))
    context = torch.randn(2, 4)
    expected = model.observe(
        observations=observations,
        actions=actions,
        context=context,
        sample_posterior=False,
    )
    hidden, stochastic = model.initial_state(2, context)
    steps = []
    for step_index in range(actions.shape[1]):
        result = model.observe_step(
            hidden=hidden,
            stochastic=stochastic,
            action=actions[:, step_index],
            context=context,
            observation=observations[:, step_index],
            sample_posterior=False,
        )
        hidden = result["hidden"]
        stochastic = result["stochastic"]
        steps.append(result)

    for name in expected:
        actual = torch.stack([step[name] for step in steps], dim=1)
        torch.testing.assert_close(actual, expected[name])


def test_stochastic_imagine_step_uses_explicit_generator() -> None:
    model = _model().eval()
    context = torch.randn(2, 4)
    hidden, stochastic = model.initial_state(2, context)
    action = torch.tensor([3, 7])
    first = model.imagine_step(
        hidden=hidden,
        stochastic=stochastic,
        action=action,
        context=context,
        sample_prior=True,
        generator=torch.Generator().manual_seed(4041),
    )
    torch.randn(100)
    second = model.imagine_step(
        hidden=hidden,
        stochastic=stochastic,
        action=action,
        context=context,
        sample_prior=True,
        generator=torch.Generator().manual_seed(4041),
    )

    torch.testing.assert_close(first["stochastic"], second["stochastic"])
    torch.testing.assert_close(first["board_logits"], second["board_logits"])


def test_rssm_parameter_counts_cover_every_trainable_parameter() -> None:
    model = _model()
    counts = model.parameter_counts()

    assert set(counts) == {
        "action_embedding",
        "recurrent",
        "prior",
        "posterior",
        "board_decoder",
        "counter_decoder",
        "total",
    }
    assert counts["total"] == sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )