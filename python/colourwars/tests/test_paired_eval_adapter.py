"""Tests the Rust-eval-backend glue in evaluate.py in isolation - no GPU/
network/colourwars_rs needed. board_from_flat is pure; _play_paired_2p_games_
batch_rust is tested against a hand-built fake Rust result (a plain object
with the same attributes colourwars_rs.RustEvalGameRecord exposes), not a
real Rust call, so this stays fast and doesn't require the extension to be
built. The real Rust boundary itself is covered by rust/src/paired_eval.rs's
own unit tests plus the (separate, manual) compare_rust_eval.py.

Run with: pytest python/colourwars/tests/test_paired_eval_adapter.py -v
"""

import os
import sys
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from colourwars.evaluate import _play_paired_2p_games_batch_rust, board_from_flat  # noqa: E402
from colourwars.game import COLS, ROWS  # noqa: E402


def test_board_from_flat_round_trips_owners_and_counts():
    owners = [-1, 0, 1] + [-1] * (ROWS * COLS - 3)
    counts = [0, 2, 3] + [0] * (ROWS * COLS - 3)
    board = board_from_flat(owners, counts, ROWS, COLS)
    assert len(board) == ROWS
    assert all(len(row) == COLS for row in board)
    assert board[0][0].owner is None
    assert board[0][0].count == 0
    assert board[0][1].owner == 0
    assert board[0][1].count == 2
    assert board[0][2].owner == 1
    assert board[0][2].count == 3


def test_board_from_flat_minus_one_always_becomes_none_not_the_int_minus_one():
    owners = [-1] * (ROWS * COLS)
    counts = [0] * (ROWS * COLS)
    board = board_from_flat(owners, counts, ROWS, COLS)
    for row in board:
        for cell in row:
            assert cell.owner is None  # not -1


@dataclass
class _FakeRustEvalGameRecord:
    """Stands in for colourwars_rs.RustEvalGameRecord's attribute shape,
    without needing the actual extension built."""
    opening_index: int
    candidate_seat: int
    decided: bool
    candidate_score: float
    move_count: int
    mid_owners: Optional[list]
    mid_counts: Optional[list]
    final_owners: list
    final_counts: list


def _fake_board(fill_owner):
    owners = [fill_owner] * (ROWS * COLS)
    counts = [1 if fill_owner != -1 else 0] * (ROWS * COLS)
    return owners, counts


def test_batch_rust_reshapes_into_per_opening_seat_pairs(monkeypatch):
    final_owners, final_counts = _fake_board(0)
    mid_owners, mid_counts = _fake_board(1)

    records = [
        _FakeRustEvalGameRecord(0, 0, True, 1.0, 15, None, None, final_owners, final_counts),
        _FakeRustEvalGameRecord(0, 1, True, 0.0, 15, None, None, final_owners, final_counts),
        _FakeRustEvalGameRecord(1, 0, False, 0.5, 40, mid_owners, mid_counts, final_owners, final_counts),
        _FakeRustEvalGameRecord(1, 1, False, 0.5, 40, mid_owners, mid_counts, final_owners, final_counts),
    ]

    class _FakeRs:
        @staticmethod
        def run_batched_paired_eval_rust(candidate_fn, opponent_fn, opening_actions, **kwargs):
            assert len(opening_actions) == 2
            return records

    monkeypatch.setitem(sys.modules, "colourwars_rs", _FakeRs())
    monkeypatch.setattr(
        "colourwars.rust_selfplay._make_numpy_forward_fn",
        lambda net, device: (lambda states: None),
    )

    openings = [
        {"opening_actions": [24, 17], "canonical_key": "k0"},
        {"opening_actions": [10, 30], "canonical_key": "k1"},
    ]

    per_opening = _play_paired_2p_games_batch_rust(
        candidate_net=None, opponent_net=None, device=None,
        num_simulations=4, openings=openings, max_moves=300, batch_size=8,
    )

    assert len(per_opening) == 2

    seat0, seat1 = per_opening[0]
    assert seat0["candidate_seat"] == 0 and seat0["decided"] is True
    assert seat0["candidate_score"] == 1.0
    assert seat0["reason"] == "decided"
    assert seat0["mid20_key"] is None  # mid_owners was None
    assert seat0["final_key"] is not None
    assert seat1["candidate_score"] == 0.0

    draw0, draw1 = per_opening[1]
    assert draw0["decided"] is False
    assert draw0["candidate_score"] == 0.5
    assert draw0["reason"] == "max_moves_reached (scored as draw)"
    assert draw0["mid20_key"] is not None  # mid_owners was present
    assert draw0["move_count"] == 40


def test_batch_rust_reshapes_correctly_even_when_seat1_finishes_first(monkeypatch):
    # The real bug found 2026-08-31: the two seat-games for one opening are
    # separate games that can finish (and so appear in `records`) in EITHER
    # order in the real concurrent slot-pool execution - a naive .append()
    # would silently produce [seat1, seat0] here, and compare_rust_eval.py's
    # zip()-based comparison treated that as "divergent play" against
    # python's always-[seat0, seat1] order, when the two games were never
    # actually paired at all. Records deliberately arrive seat-1-first here.
    final_owners, final_counts = _fake_board(0)

    records = [
        _FakeRustEvalGameRecord(0, 1, True, 0.0, 12, None, None, final_owners, final_counts),
        _FakeRustEvalGameRecord(0, 0, True, 1.0, 20, None, None, final_owners, final_counts),
    ]

    class _FakeRs:
        @staticmethod
        def run_batched_paired_eval_rust(candidate_fn, opponent_fn, opening_actions, **kwargs):
            return records

    monkeypatch.setitem(sys.modules, "colourwars_rs", _FakeRs())
    monkeypatch.setattr(
        "colourwars.rust_selfplay._make_numpy_forward_fn",
        lambda net, device: (lambda states: None),
    )

    openings = [{"opening_actions": [24, 17], "canonical_key": "k0"}]

    per_opening = _play_paired_2p_games_batch_rust(
        candidate_net=None, opponent_net=None, device=None,
        num_simulations=4, openings=openings, max_moves=300, batch_size=8,
    )

    seat0, seat1 = per_opening[0]
    assert seat0["candidate_seat"] == 0 and seat0["move_count"] == 20 and seat0["candidate_score"] == 1.0
    assert seat1["candidate_seat"] == 1 and seat1["move_count"] == 12 and seat1["candidate_score"] == 0.0
