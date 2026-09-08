"""Closed-loop adapters from learned action logits to simulator actions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..spec import Action, PlayerSkill, State
from .action_policy import ContinuousActionPolicy
from .tokens import index_to_action, legal_mask


@dataclass
class NetworkActionPolicy:
    """Feed each evolving state into a learned legal-action distribution."""

    model: ContinuousActionPolicy
    device: torch.device = torch.device("cpu")
    stochastic: bool = False
    seed: int = 0
    skill_override: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.model.to(self.device).eval()
        self.generator = torch.Generator(device=self.device).manual_seed(self.seed)
        if self.skill_override is not None:
            values = np.asarray(self.skill_override, dtype=np.float32)
            if values.shape != (self.model.config.skill_dimensions,):
                raise ValueError("skill_override has the wrong shape")
            self.skill_override = values

    @torch.no_grad()
    def __call__(self, state: State, player: PlayerSkill) -> Action | None:
        mask = legal_mask(state.board)
        if not np.any(mask):
            return None
        skill = (
            self.skill_override
            if self.skill_override is not None
            else player.as_array().astype(np.float32)
        )
        log_probabilities = self.model(
            board=torch.as_tensor(
                state.board.reshape(1, -1), dtype=torch.long, device=self.device
            ),
            goal_colour=torch.tensor(
                [state.goal_colour], dtype=torch.long, device=self.device
            ),
            moves_left=torch.tensor(
                [state.moves_left], dtype=torch.long, device=self.device
            ),
            goals_left=torch.tensor(
                [state.goals_left], dtype=torch.long, device=self.device
            ),
            skill=torch.as_tensor(
                skill[None, :], dtype=torch.float32, device=self.device
            ),
            legal_actions=torch.as_tensor(
                mask[None, :], dtype=torch.bool, device=self.device
            ),
        )[0]
        if self.stochastic:
            index = int(
                torch.multinomial(
                    log_probabilities.exp(),
                    num_samples=1,
                    generator=self.generator,
                )
            )
        else:
            index = int(torch.argmax(log_probabilities))
        return index_to_action(index)


__all__ = ["NetworkActionPolicy"]