"""Context-conditioned recurrent state-space model for gameplay dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .tokens import ACTION_SLOTS, N_CELLS


@dataclass(frozen=True)
class RSSMConfig:
    n_colours: int = 6
    hidden_size: int = 128
    stochastic_size: int = 32
    observation_size: int = 128
    context_size: int = 4
    action_size: int = ACTION_SLOTS

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if value < 1:
                raise ValueError(f"{name} must be positive")


class FastRSSM(nn.Module):
    """Gaussian RSSM with deterministic recurrence and typed state decoders."""

    def __init__(self, config: RSSMConfig = RSSMConfig()):
        super().__init__()
        self.config = config
        hidden = config.hidden_size
        stochastic = config.stochastic_size
        context = config.context_size
        self.action = nn.Embedding(config.action_size, hidden)
        self.recurrent = nn.GRUCell(hidden + stochastic + context, hidden)
        self.prior = nn.Sequential(
            nn.Linear(hidden + context, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * stochastic),
        )
        self.posterior = nn.Sequential(
            nn.Linear(hidden + context + config.observation_size, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * stochastic),
        )
        decoder_input = hidden + stochastic + context
        self.board_decoder = nn.Sequential(
            nn.Linear(decoder_input, hidden),
            nn.GELU(),
            nn.Linear(hidden, N_CELLS * config.n_colours),
        )
        self.counter_decoder = nn.Sequential(
            nn.Linear(decoder_input, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )

    @staticmethod
    def _distribution_parameters(
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, raw_scale = values.chunk(2, dim=-1)
        log_scale = raw_scale.clamp(-6.0, 3.0)
        return mean, log_scale

    @staticmethod
    def _sample(
        mean: torch.Tensor,
        log_scale: torch.Tensor,
        sample: bool,
    ) -> torch.Tensor:
        if not sample:
            return mean
        return mean + torch.randn_like(mean) * log_scale.exp()

    @staticmethod
    def _kl(
        posterior_mean: torch.Tensor,
        posterior_log_scale: torch.Tensor,
        prior_mean: torch.Tensor,
        prior_log_scale: torch.Tensor,
    ) -> torch.Tensor:
        posterior_variance = torch.exp(2.0 * posterior_log_scale)
        prior_variance = torch.exp(2.0 * prior_log_scale)
        return 0.5 * torch.sum(
            (
                posterior_variance
                + (posterior_mean - prior_mean).square()
            )
            / prior_variance
            - 1.0
            + 2.0 * (prior_log_scale - posterior_log_scale),
            dim=-1,
        )

    def _decode(
        self,
        hidden: torch.Tensor,
        stochastic: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.cat((hidden, stochastic, context), dim=-1)
        board_logits = self.board_decoder(features).view(
            *features.shape[:-1], N_CELLS, self.config.n_colours
        )
        counter_mean = self.counter_decoder(features)
        return board_logits, counter_mean

    def _initial_state(
        self, batch_size: int, reference: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = reference.new_zeros(batch_size, self.config.hidden_size)
        stochastic = reference.new_zeros(batch_size, self.config.stochastic_size)
        return hidden, stochastic

    def observe(
        self,
        *,
        observations: torch.Tensor,
        actions: torch.Tensor,
        context: torch.Tensor,
        sample_posterior: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Filter an observed sequence and return aligned latent predictions."""
        if observations.ndim != 3:
            raise ValueError("observations must have shape (batch, steps, features)")
        batch_size, steps, observation_size = observations.shape
        if observation_size != self.config.observation_size:
            raise ValueError("observation feature size does not match config")
        if actions.shape != (batch_size, steps):
            raise ValueError("actions must align with observations")
        if context.shape != (batch_size, self.config.context_size):
            raise ValueError("context has the wrong shape")
        hidden, stochastic = self._initial_state(batch_size, observations)
        outputs: dict[str, list[torch.Tensor]] = {
            name: []
            for name in (
                "hidden",
                "prior_mean",
                "prior_log_scale",
                "posterior_mean",
                "posterior_log_scale",
                "stochastic",
                "board_logits",
                "counter_mean",
                "kl",
            )
        }
        for step in range(steps):
            hidden = self.recurrent(
                torch.cat(
                    (self.action(actions[:, step]), stochastic, context), dim=-1
                ),
                hidden,
            )
            prior_mean, prior_log_scale = self._distribution_parameters(
                self.prior(torch.cat((hidden, context), dim=-1))
            )
            posterior_mean, posterior_log_scale = self._distribution_parameters(
                self.posterior(
                    torch.cat(
                        (hidden, context, observations[:, step]), dim=-1
                    )
                )
            )
            stochastic = self._sample(
                posterior_mean, posterior_log_scale, sample_posterior
            )
            board_logits, counter_mean = self._decode(
                hidden, stochastic, context
            )
            values = {
                "hidden": hidden,
                "prior_mean": prior_mean,
                "prior_log_scale": prior_log_scale,
                "posterior_mean": posterior_mean,
                "posterior_log_scale": posterior_log_scale,
                "stochastic": stochastic,
                "board_logits": board_logits,
                "counter_mean": counter_mean,
                "kl": self._kl(
                    posterior_mean,
                    posterior_log_scale,
                    prior_mean,
                    prior_log_scale,
                ),
            }
            for name, value in values.items():
                outputs[name].append(value)
        return {
            name: torch.stack(values, dim=1) for name, values in outputs.items()
        }

    def imagine(
        self,
        *,
        actions: torch.Tensor,
        context: torch.Tensor,
        sample_prior: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Generate latent states and decoded observations without inputs."""
        if actions.ndim != 2:
            raise ValueError("actions must have shape (batch, steps)")
        batch_size, steps = actions.shape
        if context.shape != (batch_size, self.config.context_size):
            raise ValueError("context has the wrong shape")
        hidden, stochastic = self._initial_state(batch_size, context)
        hidden_values = []
        prior_means = []
        prior_log_scales = []
        stochastic_values = []
        board_values = []
        counter_values = []
        for step in range(steps):
            hidden = self.recurrent(
                torch.cat(
                    (self.action(actions[:, step]), stochastic, context), dim=-1
                ),
                hidden,
            )
            prior_mean, prior_log_scale = self._distribution_parameters(
                self.prior(torch.cat((hidden, context), dim=-1))
            )
            stochastic = self._sample(prior_mean, prior_log_scale, sample_prior)
            board_logits, counter_mean = self._decode(
                hidden, stochastic, context
            )
            hidden_values.append(hidden)
            prior_means.append(prior_mean)
            prior_log_scales.append(prior_log_scale)
            stochastic_values.append(stochastic)
            board_values.append(board_logits)
            counter_values.append(counter_mean)
        return {
            "hidden": torch.stack(hidden_values, dim=1),
            "prior_mean": torch.stack(prior_means, dim=1),
            "prior_log_scale": torch.stack(prior_log_scales, dim=1),
            "stochastic": torch.stack(stochastic_values, dim=1),
            "board_logits": torch.stack(board_values, dim=1),
            "counter_mean": torch.stack(counter_values, dim=1),
        }

    def parameter_counts(self) -> dict[str, int]:
        """Return trainable parameters by component and in total."""
        components = {
            "action_embedding": self.action,
            "recurrent": self.recurrent,
            "prior": self.prior,
            "posterior": self.posterior,
            "board_decoder": self.board_decoder,
            "counter_decoder": self.counter_decoder,
        }
        counts = {
            name: sum(
                parameter.numel()
                for parameter in module.parameters()
                if parameter.requires_grad
            )
            for name, module in components.items()
        }
        counts["total"] = sum(counts.values())
        return counts


__all__ = ["FastRSSM", "RSSMConfig"]