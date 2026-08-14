"""Serialise an episode to plain JSON for replay, inspection, or export.

The pygame renderer currently consumes the in-memory episode directly. This
format provides a stable boundary for a future browser-based renderer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .scm import Episode


def _board(array: np.ndarray) -> list[list[int]]:
    return np.asarray(array).astype(int).tolist()


def episode_to_dict(episode: Episode) -> dict[str, Any]:
    return {
        "version": 1,
        "level": {
            "name": episode.level.name,
            "height": episode.level.height,
            "width": episode.level.width,
            "n_colours": episode.level.n_colours,
            "spawn_weights": list(episode.level.weights()),
            "colour_entropy": episode.level.colour_entropy(),
        },
        "difficulty": {
            "tier": episode.tier,
            "move_budget": episode.difficulty.move_budget,
            "goal_colour": episode.difficulty.goal_colour,
            "nominal_goal_count": episode.difficulty.goal_count,
            "served_goal_count": episode.served_goal_count,
            "baseline_logit": episode.difficulty.baseline,
        },
        "player": {
            "segment": episode.player.segment,
            "label": episode.player.label,
            "phi": episode.player.phi,
        },
        "E": episode.E,
        "proxy": list(map(float, episode.proxy)),
        "outcome": {
            "R": int(episode.R),
            "moves_used": episode.moves_used,
            "goals_cleared": episode.goals_cleared,
            "reshuffles": episode.reshuffles,
        },
        "states": [
            {
                "t": s.t,
                "board": _board(s.board),
                "moves_left": s.moves_left,
                "goals_left": s.goals_left,
                "goal_colour": s.goal_colour,
            }
            for s in episode.states
        ],
        "transitions": [
            {
                "t": t_index,
                "action": {
                    "row": tr.action.row,
                    "col": tr.action.col,
                    "drow": tr.action.drow,
                    "dcol": tr.action.dcol,
                }
                if tr.action
                else None,
                "board_before": _board(tr.board_before),
                "board_swapped": _board(tr.board_swapped),
                "reshuffled": bool(tr.reshuffled),
                "steps": [
                    {
                        "matched": [list(c) for c in step.matched],
                        "board_before": _board(step.board_before),
                        "board_after": _board(step.board_after),
                        "fall": [list(f) for f in step.fall],
                        "spawned": [list(s) for s in step.spawned],
                        "goal_cleared": step.goal_cleared,
                    }
                    for step in tr.steps
                ],
            }
            for t_index, tr in enumerate(episode.transitions)
        ],
    }


def save_episode(episode: Episode, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(episode_to_dict(episode), indent=1) + "\n")
    return path


def load_trajectory(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def summary_row(episode: Episode) -> dict[str, Any]:
    """One flat record per episode, for building training tables."""
    row: dict[str, Any] = {
        "level": episode.level.name,
        "n_colours": episode.level.n_colours,
        "colour_entropy": episode.level.colour_entropy(),
        "tier": episode.tier,
        "move_budget": episode.difficulty.move_budget,
        "nominal_goal_count": episode.difficulty.goal_count,
        "baseline_logit": episode.difficulty.baseline,
        "segment": episode.player.segment,
        "segment_label": episode.player.label,
        "phi": episode.player.phi,
        "E": episode.E,
        "served_goal_count": episode.served_goal_count,
        "R": int(episode.R),
        "moves_used": episode.moves_used,
        "goals_cleared": episode.goals_cleared,
        "reshuffles": episode.reshuffles,
        "cascade_depth_mean": float(
            np.mean([t.cascade_depth for t in episode.transitions])
        )
        if episode.transitions
        else 0.0,
        "tiles_per_move": float(
            np.mean([t.tiles_cleared for t in episode.transitions])
        )
        if episode.transitions
        else 0.0,
    }
    from .scm import PROXY_NAMES

    for name, value in zip(PROXY_NAMES, episode.proxy):
        row[f"x_{name}"] = float(value)
    return row


__all__ = [
    "episode_to_dict",
    "load_trajectory",
    "save_episode",
    "summary_row",
]
