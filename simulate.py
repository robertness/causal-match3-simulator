"""Bulk simulation: turn the generative model into training data.

Two artefacts come out, because two different things want to consume them:

``episodes.csv``
    One row per attempt, with every node of the DAG that is constant within an
    episode. This is the table for the causal estimation work -- it carries the
    treatment, the outcome, the observed context, the latent skill and the
    proxy coordinates side by side.

``transitions.npz``
    Flat ``(S_t, A_t, S_{t+1})`` triples for fitting a world model.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pyro

from .scm import DDA_GAINS, E_SIGMAS, Episode, ground_truth_model
from .retention import (
    CHURN_SCHEDULE,
    MasteryConfig,
    PlayerTrajectory,
    WinPropensityModel,
    simulate_player_trajectory,
)
from .spec import BENCHMARK_CONFIG
from .trajectory import (
    logged_attempt_summary_row,
    oracle_attempt_summary_row,
    summary_row,
)


def simulate(n: int, seed: int = 0, progress_every: int = 0) -> list[Episode]:
    pyro.set_rng_seed(seed)
    episodes: list[Episode] = []
    for index in range(n):
        episodes.append(ground_truth_model())
        if progress_every and (index + 1) % progress_every == 0:
            print(f"  {index + 1}/{n}", flush=True)
    return episodes


def simulate_players(
    n_players: int,
    propensity_model: WinPropensityModel,
    *,
    seed: int = 0,
    max_attempts: int = 30,
    progress_every: int = 0,
) -> list[PlayerTrajectory]:
    trajectories: list[PlayerTrajectory] = []
    for player_id in range(n_players):
        trajectories.append(
            simulate_player_trajectory(
                propensity_model,
                player_id=player_id,
                seed=seed,
                max_attempts=max_attempts,
            )
        )
        if progress_every and (player_id + 1) % progress_every == 0:
            print(f"  {player_id + 1}/{n_players}", flush=True)
    return trajectories


def write_episode_table(episodes: list[Episode], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [summary_row(e) for e in episodes]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_attempt_table(
    trajectories: list[PlayerTrajectory], path: str | Path
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        logged_attempt_summary_row(record)
        for trajectory in trajectories
        for record in trajectory.attempts
    ]
    if not rows:
        raise ValueError("cannot write an empty attempt table")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_oracle_attempt_table(
    trajectories: list[PlayerTrajectory], path: str | Path
) -> Path:
    """Write simulator-only player state outside the deployable table."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        oracle_attempt_summary_row(record)
        for trajectory in trajectories
        for record in trajectory.attempts
    ]
    if not rows:
        raise ValueError("cannot write an empty oracle attempt table")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parent,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def write_player_dataset(
    trajectories: list[PlayerTrajectory],
    output_dir: str | Path,
    *,
    seed: int,
    max_attempts: int,
    include_transitions: bool = True,
) -> Path:
    """Write physically separated logged/oracle artifacts plus a manifest."""
    if not trajectories:
        raise ValueError("cannot write an empty player dataset")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = [
        record for trajectory in trajectories for record in trajectory.attempts
    ]
    attempts_path = write_attempt_table(trajectories, output / "episodes.csv")
    oracle_path = write_oracle_attempt_table(
        trajectories, output / "oracle" / "attempts.csv"
    )
    logged_artifacts: dict[str, dict[str, object]] = {
        "attempts": {
            "path": attempts_path.relative_to(output).as_posix(),
            "rows": len(records),
            "sha256": _sha256(attempts_path),
        }
    }
    transition_count = sum(len(record.episode.transitions) for record in records)
    if include_transitions:
        transitions_path = write_transitions(
            [record.episode for record in records],
            output / "transitions.npz",
            player_ids=[record.player_id for record in records],
            attempt_ids=[record.attempt_id for record in records],
        )
        logged_artifacts["transitions"] = {
            "path": transitions_path.relative_to(output).as_posix(),
            "rows": transition_count,
            "sha256": _sha256(transitions_path),
        }
    configuration = {
        "seed": seed,
        "max_attempts": max_attempts,
        "benchmark": asdict(BENCHMARK_CONFIG),
        "mastery": asdict(MasteryConfig()),
        "churn": asdict(CHURN_SCHEDULE),
        "assignment": {
            "skill_gains": list(DDA_GAINS),
            "sigmas": list(E_SIGMAS),
        },
    }
    configuration_sha256 = hashlib.sha256(
        json.dumps(configuration, sort_keys=True).encode()
    ).hexdigest()
    manifest = {
        "schema_version": 1,
        "dataset_type": "longitudinal_player_histories",
        "code_sha": _git_revision(),
        "configuration_sha256": configuration_sha256,
        "configuration": configuration,
        "counts": {
            "players": len(trajectories),
            "attempts": len(records),
            "transitions": transition_count,
            "churned_players": sum(
                int(trajectory.churned) for trajectory in trajectories
            ),
        },
        "logged_artifacts": logged_artifacts,
        "oracle_artifacts": {
            "attempt_state": {
                "path": oracle_path.relative_to(output).as_posix(),
                "rows": len(records),
                "sha256": _sha256(oracle_path),
            }
        },
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    return manifest_path


def write_transitions(
    episodes: list[Episode],
    path: str | Path,
    *,
    player_ids: list[int] | None = None,
    attempt_ids: list[int] | None = None,
) -> Path:
    """Flatten every move into a supervised transition triple."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    before, after, actions = [], [], []
    moves_left, goals_left, goals_left_next, goal_colour = [], [], [], []
    episode_id, step_id = [], []
    transition_player_id, transition_attempt_id = [], []

    if (player_ids is None) != (attempt_ids is None):
        raise ValueError("player_ids and attempt_ids must be supplied together")
    if player_ids is not None and (
        len(player_ids) != len(episodes) or len(attempt_ids) != len(episodes)
    ):
        raise ValueError("trajectory identifiers must align with episodes")

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
            if player_ids is not None and attempt_ids is not None:
                transition_player_id.append(player_ids[index])
                transition_attempt_id.append(attempt_ids[index])

    arrays = {
        "board_before": np.asarray(before, dtype=np.int8),
        "board_after": np.asarray(after, dtype=np.int8),
        "action": np.asarray(actions, dtype=np.int8),
        "moves_left": np.asarray(moves_left, dtype=np.int16),
        "goals_left": np.asarray(goals_left, dtype=np.int16),
        "goals_left_next": np.asarray(goals_left_next, dtype=np.int16),
        "goal_colour": np.asarray(goal_colour, dtype=np.int8),
        "episode_id": np.asarray(episode_id, dtype=np.int32),
        "step_id": np.asarray(step_id, dtype=np.int16),
    }
    if player_ids is not None:
        arrays["player_id"] = np.asarray(transition_player_id, dtype=np.int32)
        arrays["attempt_id"] = np.asarray(transition_attempt_id, dtype=np.int16)
    np.savez_compressed(path, **arrays)
    return path


def main() -> None:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", type=int, default=500, help="episodes or players")
    parser.add_argument("--mode", choices=("episodes", "players"), default="episodes")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="data", help="output directory")
    parser.add_argument("--max-attempts", type=int, default=30)
    parser.add_argument("--propensity", default=None)
    parser.add_argument("--no-transitions", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    if args.mode == "players":
        from .calibrate import WIN_PROPENSITY_PATH, load_win_propensity_model

        propensity_path = args.propensity or WIN_PROPENSITY_PATH
        model = load_win_propensity_model(propensity_path)
        print(f"simulating {args.n} players (seed {args.seed})")
        trajectories = simulate_players(
            args.n,
            model,
            seed=args.seed,
            max_attempts=args.max_attempts,
            progress_every=max(1, args.n // 10),
        )
        records = [
            record for trajectory in trajectories for record in trajectory.attempts
        ]
        manifest = write_player_dataset(
            trajectories,
            out,
            seed=args.seed,
            max_attempts=args.max_attempts,
            include_transitions=not args.no_transitions,
        )
        print(f"wrote {out / 'episodes.csv'}  ({len(records)} attempts)")
        print(f"wrote {out / 'oracle' / 'attempts.csv'}  (simulator-only)")
        if not args.no_transitions:
            print(
                f"wrote {out / 'transitions.npz'}  "
                f"({sum(len(record.episode.transitions) for record in records)} transitions)"
            )
        print(f"wrote {manifest}")
        print(
            f"churn rate {np.mean([trajectory.churned for trajectory in trajectories]):.2f}"
        )
        return

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
