from __future__ import annotations

import numpy as np
import re

from match3_simulator.board import collapse, resolve_move
from match3_simulator.spec import (
    HORIZONTAL_STRIPE,
    NO_SPECIAL,
    VERTICAL_STRIPE,
    Action,
    LevelContext,
    State,
)


def _draw(n: int, tag: str) -> np.ndarray:
    column = re.search(r"_c(\d+)$", tag)
    offset = 0 if column is None else 2 * int(column.group(1))
    return (np.arange(n, dtype=np.int8) + offset) % 5


def test_horizontal_four_match_creates_stripe_at_moved_tile() -> None:
    board = np.asarray(
        [
            [1, 2, 3, 4, 0],
            [2, 3, 1, 0, 2],
            [1, 1, 2, 1, 3],
            [3, 4, 0, 2, 4],
            [4, 0, 2, 3, 0],
        ],
        dtype=np.int8,
    )
    state = State(
        board=board,
        moves_left=10,
        goals_left=20,
        goal_colour=1,
    )

    next_state, transition = resolve_move(
        state,
        Action(1, 2, 1, 0),
        LevelContext(height=5, width=5),
        _draw,
    )

    assert transition.created_specials == [(2, 2, HORIZONTAL_STRIPE)]
    assert np.count_nonzero(next_state.specials == HORIZONTAL_STRIPE) == 1
    assert transition.goal_cleared == 3


def test_existing_stripe_in_four_match_activates_instead_of_being_replaced() -> None:
    board = np.asarray(
        [
            [1, 2, 3, 4, 0],
            [2, 3, 1, 0, 1],
            [1, 1, 2, 1, 3],
            [3, 4, 0, 2, 4],
            [4, 0, 2, 3, 0],
        ],
        dtype=np.int8,
    )
    specials = np.full(board.shape, NO_SPECIAL, dtype=np.int8)
    specials[1, 2] = HORIZONTAL_STRIPE
    state = State(
        board=board,
        specials=specials,
        moves_left=10,
        goals_left=20,
        goal_colour=1,
    )

    _, transition = resolve_move(
        state,
        Action(1, 2, 1, 0),
        LevelContext(height=5, width=5),
        _draw,
    )

    assert transition.created_specials == []
    assert (2, 2) in transition.activated_specials
    assert {(2, col) for col in range(5)} <= set(
        transition.steps[0].matched
    )


def test_vertical_four_match_creates_vertical_stripe() -> None:
    board = np.asarray(
        [
            [0, 1, 2, 3, 4],
            [1, 2, 3, 1, 0],
            [2, 3, 4, 1, 0],
            [3, 4, 0, 2, 1],
            [4, 0, 2, 1, 3],
        ],
        dtype=np.int8,
    )
    state = State(
        board=board,
        moves_left=10,
        goals_left=20,
        goal_colour=1,
    )

    next_state, transition = resolve_move(
        state,
        Action(3, 3, 0, 1),
        LevelContext(height=5, width=5),
        _draw,
    )

    assert transition.created_specials == [(3, 3, VERTICAL_STRIPE)]
    assert np.count_nonzero(next_state.specials == VERTICAL_STRIPE) == 1
    assert transition.goal_cleared == 3


def test_matching_stripe_clears_row_and_chains_vertical_stripe() -> None:
    board = np.asarray(
        [
            [0, 2, 3, 4, 1],
            [2, 3, 1, 0, 2],
            [1, 1, 2, 3, 4],
            [3, 4, 0, 2, 1],
            [4, 0, 1, 3, 0],
        ],
        dtype=np.int8,
    )
    specials = np.full(board.shape, NO_SPECIAL, dtype=np.int8)
    specials[2, 1] = HORIZONTAL_STRIPE
    specials[2, 4] = VERTICAL_STRIPE
    state = State(
        board=board,
        specials=specials,
        moves_left=10,
        goals_left=20,
        goal_colour=1,
    )

    next_state, transition = resolve_move(
        state,
        Action(1, 2, 1, 0),
        LevelContext(height=5, width=5),
        _draw,
    )

    first_step = transition.steps[0]
    assert set(first_step.activated_specials) == {(2, 1), (2, 4)}
    assert {(2, col) for col in range(5)} <= set(first_step.matched)
    assert {(row, 4) for row in range(5)} <= set(first_step.matched)
    assert first_step.goal_cleared == 5
    assert not np.any(next_state.specials == HORIZONTAL_STRIPE)
    assert not np.any(next_state.specials == VERTICAL_STRIPE)


def test_gravity_moves_special_kind_with_its_colour() -> None:
    board = np.asarray(
        [
            [-1, 0],
            [2, 1],
            [-1, 2],
            [3, 3],
        ],
        dtype=np.int8,
    )
    specials = np.full(board.shape, NO_SPECIAL, dtype=np.int8)
    specials[1, 0] = HORIZONTAL_STRIPE

    collapse(board, _draw, "gravity", specials=specials)

    assert board[2, 0] == 2
    assert specials[2, 0] == HORIZONTAL_STRIPE
    assert np.count_nonzero(specials) == 1
