"""The action mechanism: how ``K`` and ``S_t`` produce ``A_t``.

The player is a softmax over legal swaps. Move quality is scored by the tiles a
swap clears, weighted towards the goal colour; the *sharpness* of the softmax is
the channel through which skill enters behaviour.

This is the edge that makes ``K`` a confounder of ``E -> R`` rather than merely
a cause of the treatment. If skill reached the outcome only through the served
difficulty, no adjustment would be needed.
"""

from __future__ import annotations

import numpy as np

from .board import immediate_effect, legal_moves
from .spec import Action

#: Near-uniform over legal moves: the player is effectively flailing.
BETA_MIN = 0.15
#: Near-greedy: the player reliably takes the best available swap. Measured to be
#: the saturation point -- sharper selection buys no further win rate.
BETA_MAX = 3.0

#: Weight on total tiles cleared, relative to goal tiles cleared.
W_TOTAL = 0.25

#: How sharply skill maps onto selection sharpness. Raising it widens the spread
#: in win rate across segments at fixed difficulty, which is what makes the
#: confounding statistically resolvable. It leaves the phi = 0 reference player
#: unchanged, so the difficulty calibration remains valid.
BETA_SLOPE = 1.5


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def beta_from_skill(
    phi: float,
    beta_min: float = BETA_MIN,
    beta_max: float = BETA_MAX,
    slope: float = BETA_SLOPE,
) -> float:
    """Map player skill to softmax sharpness.

    A weak player is close to uniform over legal swaps; a strong one is close to
    greedy.
    """
    return beta_min + (beta_max - beta_min) * sigmoid(slope * phi)


def move_scores(
    board: np.ndarray,
    moves: list[Action],
    goal_colour: int,
    w_total: float = W_TOTAL,
) -> np.ndarray:
    """Score each legal swap by its immediate effect.

    Deliberately one-ply: a player sees the match in front of them, not the
    cascade it sets off.
    """
    scores = np.empty(len(moves), dtype=np.float64)
    for i, action in enumerate(moves):
        cleared, goal = immediate_effect(board, action, goal_colour)
        scores[i] = goal + w_total * cleared
    return scores


def action_probs(
    board: np.ndarray,
    moves: list[Action],
    goal_colour: int,
    beta: float,
) -> np.ndarray:
    """Softmax over legal swaps at inverse temperature ``beta``."""
    if not moves:
        return np.zeros(0, dtype=np.float64)
    scores = move_scores(board, moves, goal_colour)
    shifted = beta * (scores - scores.max())
    weights = np.exp(shifted)
    return weights / weights.sum()


def policy_distribution(
    board: np.ndarray, goal_colour: int, phi: float
) -> tuple[list[Action], np.ndarray]:
    """Convenience wrapper returning the support and the probabilities."""
    moves = legal_moves(board)
    return moves, action_probs(board, moves, goal_colour, beta_from_skill(phi))


__all__ = [
    "BETA_MAX",
    "BETA_MIN",
    "BETA_SLOPE",
    "action_probs",
    "beta_from_skill",
    "move_scores",
    "policy_distribution",
    "sigmoid",
]
