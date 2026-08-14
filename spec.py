"""Variable definitions for the match-3 structural causal model.

Each dataclass here corresponds to one node (or node group) in the DAG:

    L  LevelContext   non-difficulty context; parameterises the transition kernel
    D  Difficulty     baseline tier; sets the move budget and the goal colour
    K  PlayerType     latent player skill
    E  float          served difficulty, in logits
    S  State          board plus the two counters that make D and E front-loaded
    A  Action         a swap of two orthogonally adjacent cells
    R  int            1 if the quota was cleared within the move budget
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

EMPTY = -1


@dataclass(frozen=True)
class LevelContext:
    """``L`` -- everything about a level that is not a difficulty setting.

    These fields parameterise the *transition kernel*: the colour alphabet and
    the refill distribution are consulted at every step, not just at ``t=0``.
    """

    height: int = 8
    width: int = 8
    n_colours: int = 5
    #: Unnormalised spawn weights per colour; ``None`` means uniform.
    spawn_weights: tuple[float, ...] | None = None
    name: str = "default"

    def weights(self) -> np.ndarray:
        if self.spawn_weights is None:
            w = np.ones(self.n_colours, dtype=np.float64)
        else:
            w = np.asarray(self.spawn_weights, dtype=np.float64)
            if w.shape != (self.n_colours,):
                raise ValueError(
                    f"spawn_weights has length {w.shape[0]}, expected {self.n_colours}"
                )
        return w / w.sum()

    def colour_entropy(self) -> float:
        """Entropy of the spawn distribution, the scalar Lily's Garden logs."""
        w = self.weights()
        return float(-(w * np.log(w)).sum())


@dataclass(frozen=True)
class Difficulty:
    """``D`` -- the baseline tier the studio assigns to a level.

    ``goal_count`` here is the *nominal* ask for the tier. What a given player is
    actually served is set by ``E``, and lands in ``S_0.goals_left``.
    """

    move_budget: int = 20
    goal_colour: int = 1
    goal_count: int = 30
    #: The tier's difficulty in logits, on the same scale as ``E``.
    baseline: float = 0.0

    def baseline_logit(self, level: "LevelContext" | None = None) -> float:
        return self.baseline


@dataclass(frozen=True)
class PlayerType:
    """``K`` -- a latent player segment, carrying skill on the Rasch logit scale."""

    segment: int
    label: str
    phi: float


#: The population of player types. ``phi`` spans roughly +/- 2.25 logits, in
#: line with the attempts-per-clear spread reported for commercial puzzle games,
#: where individual players range from one attempt to thirty on the same level.
SEGMENTS: tuple[PlayerType, ...] = (
    PlayerType(0, "novice", phi=-2.25),
    PlayerType(1, "casual", phi=-0.75),
    PlayerType(2, "regular", phi=0.75),
    PlayerType(3, "expert", phi=2.25),
)

SEGMENT_PROBS: tuple[float, ...] = (0.30, 0.35, 0.25, 0.10)


@dataclass(frozen=True)
class Action:
    """``A_t`` -- swap ``(row, col)`` with its neighbour one step down or right."""

    row: int
    col: int
    drow: int
    dcol: int

    @property
    def cells(self) -> tuple[tuple[int, int], tuple[int, int]]:
        return (self.row, self.col), (self.row + self.drow, self.col + self.dcol)

    def __str__(self) -> str:  # pragma: no cover - display only
        (r, c), (r2, c2) = self.cells
        return f"({r},{c})<->({r2},{c2})"


@dataclass
class State:
    """``S_t`` -- the board, goal colour, and counters carried through play."""

    board: np.ndarray
    moves_left: int
    goals_left: int
    goal_colour: int
    t: int = 0

    def copy(self) -> "State":
        return State(
            self.board.copy(),
            self.moves_left,
            self.goals_left,
            self.goal_colour,
            self.t,
        )

    @property
    def won(self) -> bool:
        return self.goals_left <= 0

    @property
    def lost(self) -> bool:
        return self.goals_left > 0 and self.moves_left <= 0

    @property
    def terminal(self) -> bool:
        return self.won or self.lost

    def as_lists(self) -> list[list[int]]:
        return self.board.astype(int).tolist()


@dataclass
class CascadeStep:
    """One clear-fall-refill round inside a single move."""

    matched: list[tuple[int, int]]
    board_before: np.ndarray
    board_after: np.ndarray
    fall: list[tuple[int, int, int, int]]
    spawned: list[tuple[int, int, int]]
    goal_cleared: int


@dataclass
class Transition:
    """Everything that happened between ``S_t`` and ``S_{t+1}``.

    Retained in full because the renderer animates the phases, and because the
    cascade depth is a useful difficulty diagnostic.
    """

    action: Action | None
    board_before: np.ndarray
    board_swapped: np.ndarray
    steps: list[CascadeStep] = field(default_factory=list)
    reshuffled: bool = False

    @property
    def goal_cleared(self) -> int:
        return sum(s.goal_cleared for s in self.steps)

    @property
    def tiles_cleared(self) -> int:
        return sum(len(s.matched) for s in self.steps)

    @property
    def cascade_depth(self) -> int:
        return len(self.steps)


def uniform_board(rng: np.random.Generator, level: LevelContext) -> np.ndarray:
    return rng.choice(
        level.n_colours, size=(level.height, level.width), p=level.weights()
    ).astype(np.int8)


def state_from(
    level: LevelContext, difficulty: Difficulty, board: np.ndarray, goals: int
) -> State:
    return State(
        board=board,
        moves_left=difficulty.move_budget,
        goals_left=goals,
        goal_colour=difficulty.goal_colour,
        t=0,
    )


__all__ = [
    "EMPTY",
    "Action",
    "CascadeStep",
    "Difficulty",
    "LevelContext",
    "PlayerType",
    "SEGMENTS",
    "SEGMENT_PROBS",
    "State",
    "Transition",
    "state_from",
    "uniform_board",
]
