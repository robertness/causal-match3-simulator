"""Bulk simulation: turn the generative model into training data.

Two artefacts come out, because two different things want to consume them:

``episodes.csv``
    One row per attempt, with every node of the DAG that is constant within an
    episode. This is the table for the causal estimation work -- it carries the
    treatment, the outcome, the observed context, the latent segment and the
    proxy coordinates side by side.

``transitions.npz``
    Flat ``(S_t, A_t, S_{t+1})`` triples for fitting a world model.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pyro

from .scm import Episode, ground_truth_model
from .trajectory import summary_row


def simulate(n: int, seed: int = 0, progress_every: int = 0) -> list[Episode]:
    pyro.set_rng_seed(seed)
    episodes: list[Episode] = []
    for index in range(n):
        episodes.append(ground_truth_model())
        if progress_every and (index + 1) % progress_every == 0:
            print(f"  {index + 1}/{n}", flush=True)
    return episodes


def write_episode_table(episodes: list[Episode], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [summary_row(e) for e in episodes]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_transitions(episodes: list[Episode], path: str | Path) -> Path:
    """Flatten every move into a supervised transition triple."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    before, after, actions = [], [], []
    moves_left, goals_left, goals_left_next, goal_colour = [], [], [], []
    episode_id, step_id = [], []

    for index, episode in enumerate(episodes):
        for t, transition in enumerate(episode.transitions):
            state, nxt = episode.states[t], episode.states[t + 1]
            action = transition.action
            assert action is not None
            before.append(state.board)
            after.append(nxt.board)
            actions.append([action.row, action.col, action.drow, action.dcol])
            moves_left.append(state.moves_left)
            goals_left.append(state.goals_left)
            goals_left_next.append(nxt.goals_left)
            goal_colour.append(state.goal_colour)
            episode_id.append(index)
            step_id.append(t)

    np.savez_compressed(
        path,
        board_before=np.asarray(before, dtype=np.int8),
        board_after=np.asarray(after, dtype=np.int8),
        action=np.asarray(actions, dtype=np.int8),
        moves_left=np.asarray(moves_left, dtype=np.int16),
        goals_left=np.asarray(goals_left, dtype=np.int16),
        goals_left_next=np.asarray(goals_left_next, dtype=np.int16),
        goal_colour=np.asarray(goal_colour, dtype=np.int8),
        episode_id=np.asarray(episode_id, dtype=np.int32),
        step_id=np.asarray(step_id, dtype=np.int16),
    )
    return path


def main() -> None:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", type=int, default=500, help="episodes to simulate")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="data", help="output directory")
    parser.add_argument("--no-transitions", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    print(f"simulating {args.n} episodes (seed {args.seed})")
    episodes = simulate(args.n, args.seed, progress_every=max(1, args.n // 10))

    table = write_episode_table(episodes, out / "episodes.csv")
    print(f"wrote {table}  ({len(episodes)} rows)")

    if not args.no_transitions:
        tensors = write_transitions(episodes, out / "transitions.npz")
        total = sum(len(e.transitions) for e in episodes)
        print(f"wrote {tensors}  ({total} transitions)")

    wins = np.mean([e.R for e in episodes])
    print(f"win rate {wins:.2f}")


if __name__ == "__main__":  # pragma: no cover
    main()
