"""Fuzz cross-check: drives IDENTICAL random move sequences through the
existing pure-Python engine (colourwars.game) and the new Rust engine
(colourwars_rs, via PyO3), asserting the two produce identical board states,
captures, and outcomes after EVERY single ply - not just matching hand-picked
test scenarios. This is the required gate before any MCTS port or PyO3
integration proceeds (see the project goal): thousands of random games with
zero divergence must pass here first.

Move selection: at each ply, the Python engine's legal-move list is computed,
one is chosen uniformly at random via Python's `random` module (the single
source of randomness both engines follow), and that same (row, col) is
applied to both engines. This avoids any dependence on the two engines
agreeing on internal legal-move ordering or having compatible RNG streams -
only the actual chosen move needs to match, and it's applied identically to
both.

Run with: python -m colourwars.tests.compare_rust_engine --games 3000
"""

from __future__ import annotations

import argparse
import random
import sys

import numpy as np

import colourwars_rs as rs

from colourwars.env import ColourWarsEnv
from colourwars.game import Cell, create_game, is_valid_move, play_move


def _py_board_owners_counts(board):
    owners = []
    counts = []
    for row in board:
        for cell in row:
            owners.append(-1 if cell.owner is None else cell.owner)
            counts.append(cell.count)
    return owners, counts


def _assert_states_match(py_state, rs_state, context: str):
    py_owners, py_counts = _py_board_owners_counts(py_state.board)
    rs_owners = rs_state.board_owners()
    rs_counts = rs_state.board_counts()

    if py_owners != rs_owners:
        diffs = [(i, po, ro) for i, (po, ro) in enumerate(zip(py_owners, rs_owners)) if po != ro]
        raise AssertionError(f"{context}: board OWNER mismatch at cells {diffs[:10]} (showing up to 10)")
    if py_counts != rs_counts:
        diffs = [(i, pc, rc) for i, (pc, rc) in enumerate(zip(py_counts, rs_counts)) if pc != rc]
        raise AssertionError(f"{context}: board COUNT mismatch at cells {diffs[:10]} (showing up to 10)")

    if py_state.current_player_index != rs_state.current_player_index:
        raise AssertionError(
            f"{context}: current_player_index mismatch: py={py_state.current_player_index} "
            f"rs={rs_state.current_player_index}"
        )
    if py_state.total_moves != rs_state.total_moves:
        raise AssertionError(f"{context}: total_moves mismatch: py={py_state.total_moves} rs={rs_state.total_moves}")
    if py_state.game_over != rs_state.game_over:
        raise AssertionError(f"{context}: game_over mismatch: py={py_state.game_over} rs={rs_state.game_over}")

    py_winner = py_state.winner if py_state.winner is not None else -1
    rs_winner = rs_state.winner if rs_state.winner is not None else -1
    if py_winner != rs_winner:
        raise AssertionError(f"{context}: winner mismatch: py={py_winner} rs={rs_winner}")

    py_active = [p.active for p in py_state.players]
    rs_active = rs_state.players_active()
    if py_active != rs_active:
        raise AssertionError(f"{context}: players_active mismatch: py={py_active} rs={rs_active}")

    py_has_moved = [p.has_moved for p in py_state.players]
    rs_has_moved = rs_state.players_has_moved()
    if py_has_moved != rs_has_moved:
        raise AssertionError(f"{context}: players_has_moved mismatch: py={py_has_moved} rs={rs_has_moved}")

    py_encoded = ColourWarsEnv.from_state(py_state).encode_state().flatten()
    rs_encoded = np.array(rs_state.encode_state(), dtype=np.float32)
    if not np.allclose(py_encoded, rs_encoded, atol=1e-6):
        n_diff = int(np.sum(~np.isclose(py_encoded, rs_encoded, atol=1e-6)))
        raise AssertionError(f"{context}: encode_state mismatch at {n_diff}/{py_encoded.size} values")


def play_one_fuzz_game(num_players: int, max_moves: int, game_index: int) -> int:
    py_state = create_game(num_players, 7, 7)
    rs_state = rs.create_game(num_players, 7, 7)

    _assert_states_match(py_state, rs_state, f"game {game_index}, initial state")

    move_count = 0
    while not py_state.game_over and move_count < max_moves:
        mover = py_state.current_player_index
        has_moved = py_state.players[mover].has_moved
        legal = [
            (r, c)
            for r in range(py_state.rows)
            for c in range(py_state.cols)
            if is_valid_move(py_state.board, r, c, mover, has_moved)
        ]
        if not legal:
            break
        row, col = random.choice(legal)

        py_result = play_move(py_state, row, col)
        rs_new_state, rs_waves = rs.play_move(rs_state, row, col)

        py_state = py_result.state
        rs_state = rs_new_state

        context = f"game {game_index}, ply {move_count} (move=({row},{col}), n_players={num_players})"
        _assert_states_match(py_state, rs_state, context)

        py_waves = len(py_result.steps)
        if py_waves != rs_waves:
            raise AssertionError(f"{context}: explosion-wave count mismatch: py={py_waves} rs={rs_waves}")

        move_count += 1

    return move_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=3000)
    parser.add_argument("--max-moves", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    total_moves = 0
    for i in range(args.games):
        num_players = random.choice([2, 3, 4])
        try:
            n_moves = play_one_fuzz_game(num_players, args.max_moves, i)
        except AssertionError as e:
            print(f"\nDIVERGENCE FOUND at game {i}: {e}", file=sys.stderr)
            raise SystemExit(1)
        total_moves += n_moves
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{args.games} games checked, {total_moves} total plies, zero divergence so far")

    print(f"\nOK: {args.games} random games, {total_moves} total plies, ZERO divergence between "
          f"Python and Rust engines.")


if __name__ == "__main__":
    main()
