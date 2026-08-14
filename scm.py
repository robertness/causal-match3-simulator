"""The structural causal model, one function per node of the DAG.

Every node gets its own function taking its parents as arguments, so the graph

    L -> D,  D -> E,  K -> E,  D -> S_0,  E -> S_0,  L -> S_t,
    K -> A_t,  S_T -> R,  K -> X

can be read straight off the call sites in :func:`ground_truth_model`.

Two conventions hold throughout:

* Every node function accepts ``value=``. Passing it clamps the node, which is
    how interventions are expressed: ``sample_E(D, K, value=1.2)`` is
  ``do(E = 1.2)``.
* ``pyro.sample`` appears at stochastic SCM nodes and gameplay events: the root
    variables, noisy treatment and proxy, opening deal, refill colours, and
    player's choice of swap. Match detection, gravity and cascade resolution are
    deterministic given those draws and are recorded with ``pyro.deterministic``.

The load-bearing structural choice is that ``E`` is the difficulty the policy
*served*, not how hard the level *felt*. Skill reaches the outcome through
``K -> A_t``, independently of ``E``. Were skill instead absorbed into ``E``, the
two would be d-separated from ``R`` given ``E`` and there would be no confounding
to correct.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyro
import pyro.distributions as dist
import torch

from .board import deal, legal_moves, resolve_move
from .policy import action_probs, beta_from_skill
from .spec import (
    SEGMENT_PROBS,
    SEGMENTS,
    Action,
    Difficulty,
    LevelContext,
    PlayerType,
    State,
    Transition,
    state_from,
)

# --------------------------------------------------------------------- setup --

#: Level catalogue. ``L`` is a root node, so this is its support.
LEVELS: tuple[LevelContext, ...] = (
    LevelContext(height=8, width=8, n_colours=5, name="orchard"),
    LevelContext(height=8, width=8, n_colours=5, name="harbour",
                 spawn_weights=(1.0, 0.8, 1.0, 1.1, 1.1)),
    LevelContext(height=8, width=8, n_colours=6, name="foundry"),
)
LEVEL_PROBS: tuple[float, ...] = (0.5, 0.3, 0.2)

#: Baseline tiers, on the same logit scale as ``E``.
TIER_LOGITS: tuple[float, ...] = (-0.8, 0.0, 0.8)
TIER_NAMES: tuple[str, ...] = ("easy", "medium", "hard")
TIER_PROBS: tuple[float, ...] = (0.3, 0.45, 0.25)
DEFAULT_MOVE_BUDGET = 20
DEFAULT_GOAL_COLOUR = 1

#: How hard the difficulty policy chases skill. At 0 the treatment is
#: unconfounded; as it rises the backdoor grows. Sweeping it gives a controlled
#: family of problems of increasing difficulty for an estimator.
DDA_GAIN = 1.0

#: Residual noise in the difficulty-assignment policy. This is what makes the
#: effect of E identifiable at all: with sigma = 0 the policy is deterministic
#: given (D, K) and there is no variation left to exploit.
E_SIGMA = 0.35

#: Names of the proxy coordinates, in the order :func:`sample_X` emits them.
PROXY_NAMES: tuple[str, ...] = (
    "moves_left_frac_on_win",
    "attempts_per_clear",
    "goals_left_on_loss",
    "cascade_rate",
    "tiles_per_move",
    "booster_ingame",
    "booster_pregame",
    "sessions_per_day",
    "session_length",
    "retry_latency",
    "levels_per_session",
    "reshuffles_seen",
)

#: Rows are proxy coordinates, columns are (phi, intercept).
PROXY_LOADINGS = np.array(
    [
        [0.40, 0.35],
        [-0.55, 3.20],
        [-0.60, 8.00],
        [0.30, 0.55],
        [0.45, 4.00],
        [-0.30, 0.80],
        [-0.22, 0.40],
        [0.35, 2.20],
        [0.30, 9.00],
        [-0.40, 2.50],
        [0.38, 6.00],
        [-0.15, 1.20],
    ],
    dtype=np.float64,
)
PROXY_NOISE = 0.35


@dataclass
class Episode:
    """One level attempt, with every node of the DAG recorded."""

    level: LevelContext
    difficulty: Difficulty
    player: PlayerType
    E: float
    served_goal_count: int
    states: list[State]
    actions: list[Action]
    transitions: list[Transition]
    proxy: np.ndarray
    R: int
    tier: str = ""

    @property
    def moves_used(self) -> int:
        return len(self.actions)

    @property
    def goals_cleared(self) -> int:
        return self.served_goal_count - self.states[-1].goals_left

    @property
    def reshuffles(self) -> int:
        return sum(1 for t in self.transitions if t.reshuffled)


def _tensor(x, dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x), dtype=dtype)


def _make_draw(level: LevelContext, prefix: str):
    """Build the refill sampler handed to the deterministic board code."""
    probs = _tensor(level.weights())

    def draw(n: int, tag: str) -> np.ndarray:
        site = dist.Categorical(probs=probs).expand([n]).to_event(1)
        drawn = pyro.sample(f"{prefix}/{tag}", site)
        return drawn.detach().cpu().numpy().astype(np.int8)

    return draw


# ------------------------------------------------------------------- nodes ----


def sample_L(name: str = "L", value: LevelContext | None = None) -> LevelContext:
    """``L`` -- level context. A root node; parameterises the transition kernel."""
    if value is not None:
        return value
    idx = pyro.sample(f"{name}/index", dist.Categorical(probs=_tensor(LEVEL_PROBS)))
    return LEVELS[int(idx)]


def sample_K(name: str = "K", value: PlayerType | None = None) -> PlayerType:
    """``K`` -- latent player skill. Root node, observed only through ``X``."""
    if value is not None:
        return value
    idx = pyro.sample(f"{name}/segment", dist.Categorical(probs=_tensor(SEGMENT_PROBS)))
    return SEGMENTS[int(idx)]


def sample_D(
    level: LevelContext,
    name: str = "D",
    value: Difficulty | None = None,
    move_budget: int = DEFAULT_MOVE_BUDGET,
    goal_colour: int = DEFAULT_GOAL_COLOUR,
) -> tuple[Difficulty, str]:
    """``D`` -- the baseline tier the studio assigns, given ``L``.

    The nominal goal count is read off the calibration curve, so a tier means the
    same thing across levels with different palettes. That is the ``L -> D``
    edge: the same ask is a different proposition at five colours than at six.
    """
    if value is not None:
        return value, "given"

    idx = int(pyro.sample(f"{name}/tier", dist.Categorical(probs=_tensor(TIER_PROBS))))
    baseline = TIER_LOGITS[idx]

    from .calibrate import goal_count_for_E

    nominal = goal_count_for_E(level.name, baseline)
    difficulty = Difficulty(
        move_budget=move_budget,
        goal_colour=goal_colour,
        goal_count=nominal,
        baseline=baseline,
    )
    return difficulty, TIER_NAMES[idx]


def sample_E(
    difficulty: Difficulty,
    player: PlayerType,
    name: str = "E",
    value: float | None = None,
    gain: float = DDA_GAIN,
    sigma: float = E_SIGMA,
) -> float:
    """``E`` -- the difficulty served, given ``D`` and ``K``.

    Stronger players are served harder levels, which is what every published DDA
    does. The plus sign is the source of the confounding: it correlates the
    treatment with a variable that independently raises the outcome.
    """
    if value is not None:
        return float(pyro.deterministic(name, _tensor(value)))
    location = difficulty.baseline + gain * player.phi
    if sigma == 0:
        return float(pyro.deterministic(name, _tensor(location)))
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    return float(pyro.sample(name, dist.Normal(_tensor(location), sigma)))


def sample_S0(
    level: LevelContext,
    difficulty: Difficulty,
    E: float,
    name: str = "S0",
    value: State | None = None,
    served: int | None = None,
) -> State:
    """``S_0`` -- opening state, given ``L``, ``D`` and ``E``.

    ``E`` sets the quota by inverting the calibration curve; ``D`` sets the move
    budget and goal colour. Both enter only here, so ``S_0`` is a sufficient
    statistic for the level configuration.
    """
    if value is not None:
        return value.copy()

    if served is None:
        from .calibrate import goal_count_for_E

        served = goal_count_for_E(level.name, E, fallback=difficulty.goal_count)

    board = deal(level, _make_draw(level, name), tag="deal")
    pyro.deterministic(f"{name}/goals", _tensor(served))
    return state_from(level, difficulty, board, served)


def sample_A(
    state: State,
    player: PlayerType,
    name: str | None = None,
    value: Action | None = None,
) -> Action | None:
    """``A_t`` -- the swap, given ``S_t`` and ``K``.

    Skill enters through the sharpness of the softmax over legal moves. Returns
    ``None`` when the board offers no legal move.
    """
    site = name or f"A/{state.t}"
    if value is not None:
        return value

    moves = legal_moves(state.board)
    if not moves:
        return None

    probs = action_probs(
        state.board, moves, state.goal_colour, beta_from_skill(player.phi)
    )
    idx = pyro.sample(site, dist.Categorical(probs=_tensor(probs)))
    return moves[int(idx)]


def sample_S_next(
    state: State,
    action: Action,
    level: LevelContext,
    name: str | None = None,
    value: tuple[State, Transition] | None = None,
) -> tuple[State, Transition]:
    """``S_{t+1}`` -- next state, given ``S_t``, ``A_t`` and ``L``.

    The only randomness is the refill colours, drawn from ``L``'s spawn
    distribution. Matching, gravity and the cascade fixpoint are deterministic.
    """
    if value is not None:
        return value

    prefix = name or f"S/{state.t + 1}"
    nxt, transition = resolve_move(
        state, action, level, _make_draw(level, prefix)
    )
    pyro.deterministic(f"{prefix}/goals_left", _tensor(nxt.goals_left))
    return nxt, transition


def sample_X(
    player: PlayerType, name: str = "X", value: np.ndarray | None = None
) -> np.ndarray:
    """``X`` -- the telemetry proxy, given ``K``.

    A noisy linear emission of skill: higher dimensional than what it proxies,
    and no single coordinate identifies the segment on its own.
    """
    if value is not None:
        return np.asarray(value, dtype=np.float64)
    location = PROXY_LOADINGS @ np.array([player.phi, 1.0])
    drawn = pyro.sample(
        name, dist.Normal(_tensor(location), PROXY_NOISE).to_event(1)
    )
    return drawn.detach().cpu().numpy().astype(np.float64)


def sample_R(
    terminal_state: State, name: str = "R", value: int | None = None
) -> int:
    """``R`` -- success: was the quota cleared within the move budget?

    Deterministic. Every source of randomness that produces ``R`` has already
    acted upstream, in the deal, the refills and the swaps.
    """
    outcome = int(value) if value is not None else int(terminal_state.goals_left <= 0)
    pyro.deterministic(name, _tensor(outcome))
    return outcome


# ------------------------------------------------------------------- model ----


def ground_truth_model(
    level: LevelContext | None = None,
    player: PlayerType | None = None,
    difficulty: Difficulty | None = None,
    E: float | None = None,
    served_goal_count: int | None = None,
    dda_gain: float = DDA_GAIN,
    e_sigma: float = E_SIGMA,
    max_steps: int | None = None,
) -> Episode:
    """The full data-generating process for one level attempt.

    Any node can be pinned by passing it, which is how interventional
    distributions are drawn: pass ``E=`` for ``do(E = e)``, ``player=`` to
    condition on a segment, and so on.
    """
    L = sample_L(value=level)
    K = sample_K(value=player)
    D, tier = sample_D(L, value=difficulty)
    eff = sample_E(D, K, value=E, gain=dda_gain, sigma=e_sigma)
    X = sample_X(K)

    state = sample_S0(L, D, eff, served=served_goal_count)
    served = state.goals_left
    states: list[State] = [state]
    actions: list[Action] = []
    transitions: list[Transition] = []

    budget = D.move_budget if max_steps is None else min(D.move_budget, max_steps)
    for _ in range(budget):
        if state.terminal:
            break
        action = sample_A(state, K)
        if action is None:
            break
        state, transition = sample_S_next(state, action, L)
        actions.append(action)
        transitions.append(transition)
        states.append(state)

    return Episode(
        level=L,
        difficulty=D,
        player=K,
        E=eff,
        served_goal_count=served,
        states=states,
        actions=actions,
        transitions=transitions,
        proxy=X,
        R=sample_R(states[-1]),
        tier=tier,
    )


__all__ = [
    "DDA_GAIN",
    "Episode",
    "E_SIGMA",
    "LEVELS",
    "LEVEL_PROBS",
    "PROXY_LOADINGS",
    "PROXY_NAMES",
    "TIER_LOGITS",
    "TIER_NAMES",
    "ground_truth_model",
    "sample_A",
    "sample_D",
    "sample_E",
    "sample_K",
    "sample_L",
    "sample_R",
    "sample_S0",
    "sample_S_next",
    "sample_X",
]
