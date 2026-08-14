"""Calibrate the served difficulty scale by simulation.

``E`` is *defined* by the win rate it induces for a fixed reference player:

    E  ==  -logit Pr(R = 1 | reference player)

so ``E = 0`` is a configuration the reference clears half the time. Calibration
bisects the goal count against that reference over a grid of ``E``, producing an
invertible curve per level.

Doing this by simulation rather than arithmetic is not fastidiousness. In a
production difficulty model, features derived from a play-testing agent were five
to six times more predictive than the static level attributes: realised
difficulty is not a function of the knobs.

The reference player is deliberately mid-skill. An agent that is too strong
passes everything and the easy end of the scale stops separating -- a failure
mode reported directly in the published work.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .spec import Difficulty, LevelContext, PlayerType

REFERENCE_PLAYER = PlayerType(-1, "reference", phi=0.0)

#: The difficulty scale the curve is solved over. Spans the range the DDA
#: actually reaches: baseline tier plus gain times skill, plus policy noise.
E_GRID: tuple[float, ...] = (
    -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0
)

TABLE_PATH = Path(__file__).with_name("calibration.json")


@dataclass
class CurvePoint:
    E: float
    goal_count: int
    pass_rate: float


def target_rate(E: float) -> float:
    return float(1.0 / (1.0 + np.exp(E)))


def win_rate(
    level: LevelContext,
    goal_count: int,
    move_budget: int = 20,
    goal_colour: int = 1,
    n: int = 200,
) -> float:
    """Reference player's clear rate on a configuration, bypassing the curve."""
    from .scm import ground_truth_model

    difficulty = Difficulty(move_budget, goal_colour, goal_count, baseline=0.0)
    wins = 0
    for _ in range(n):
        wins += ground_truth_model(
            level=level,
            player=REFERENCE_PLAYER,
            difficulty=difficulty,
            E=0.0,
            served_goal_count=goal_count,
        ).R
    return wins / n


def solve_goal_count(
    level: LevelContext,
    target: float,
    move_budget: int = 20,
    goal_colour: int = 1,
    n: int = 200,
    lo: int = 3,
    hi: int = 140,
) -> tuple[int, float]:
    """Bisect on goal count for a target win rate.

    Win rate is monotone decreasing in the goal count, which is what makes
    bisection valid and the resulting curve invertible.
    """
    best: tuple[int, float] | None = None
    while lo <= hi:
        mid = (lo + hi) // 2
        rate = win_rate(level, mid, move_budget, goal_colour, n)
        if best is None or abs(rate - target) < abs(best[1] - target):
            best = (mid, rate)
        if rate > target:
            lo = mid + 1
        else:
            hi = mid - 1
    assert best is not None
    return best


def calibrate(
    levels: list[LevelContext],
    grid: tuple[float, ...] = E_GRID,
    n: int = 200,
    move_budget: int = 20,
    goal_colour: int = 1,
) -> dict[str, dict]:
    table: dict[str, dict] = {}
    for level in levels:
        points: list[CurvePoint] = []
        # A harder E means a bigger quota, so goal count rises along the grid and
        # each solution bounds the next from below.
        lo = 3
        for E in grid:
            goal_count, rate = solve_goal_count(
                level, target_rate(E), move_budget, goal_colour, n, lo=lo, hi=200
            )
            points.append(CurvePoint(float(E), int(goal_count), float(rate)))
            lo = max(3, goal_count)
        table[level.name] = {
            "move_budget": move_budget,
            "goal_colour": goal_colour,
            "n": n,
            "curve": [asdict(p) for p in points],
        }
    return table


def save(table: dict[str, dict], path: Path = TABLE_PATH) -> None:
    path.write_text(json.dumps(table, indent=2) + "\n")
    global _CACHE
    _CACHE = table


_CACHE: dict[str, dict] | None = None


def load(path: Path = TABLE_PATH) -> dict[str, dict]:
    global _CACHE
    if _CACHE is None:
        _CACHE = json.loads(path.read_text()) if path.exists() else {}
    return _CACHE


def goal_count_for_E(level_name: str, E: float, fallback: int = 30) -> int:
    """Invert the calibration curve. Linear interpolation, clamped at the ends."""
    entry = load().get(level_name)
    if not entry:
        return fallback
    curve = entry["curve"]
    xs = [p["E"] for p in curve]
    ys = [p["goal_count"] for p in curve]
    return int(round(float(np.interp(E, xs, ys))))


def main() -> None:  # pragma: no cover - CLI
    import argparse

    from .scm import LEVELS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=200, help="rollouts per probe")
    args = parser.parse_args()

    table = calibrate(list(LEVELS), n=args.n)
    save(table)

    for name, entry in table.items():
        print(f"\n{name}")
        for point in entry["curve"]:
            print(
                f"  E={point['E']:+.1f}  goal={point['goal_count']:3d}  "
                f"win={point['pass_rate']:.2f}  (target {target_rate(point['E']):.2f})"
            )
    print(f"\nwrote {TABLE_PATH}")


if __name__ == "__main__":  # pragma: no cover
    main()
