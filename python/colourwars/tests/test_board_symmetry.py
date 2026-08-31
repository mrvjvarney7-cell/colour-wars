"""Tests board_symmetry.py directly against hand-built positions - not just
"read the code and conclude it's right", the same standard the rest of this
investigation has been held to.

Run with: pytest python/colourwars/tests/test_board_symmetry.py -v
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from colourwars.board_symmetry import canonical_key, count_distinct  # noqa: E402
from colourwars.game import Cell, create_empty_board  # noqa: E402


def board_from_owners(owners, rows, cols):
    """owners: dict {(r, c): (owner, count)} - everything else stays empty."""
    board = create_empty_board(rows, cols)
    for (r, c), (owner, count) in owners.items():
        board[r][c] = Cell(owner=owner, count=count)
    return board


def test_identical_boards_have_equal_keys():
    b1 = board_from_owners({(0, 0): (0, 2)}, 7, 7)
    b2 = board_from_owners({(0, 0): (0, 2)}, 7, 7)
    assert canonical_key(b1) == canonical_key(b2)


def test_90_degree_rotation_is_recognised_as_the_same_position():
    # A single owned cell at the top-left corner...
    original = board_from_owners({(0, 0): (0, 3)}, 7, 7)
    # ...rotated 90 degrees clockwise lands at (c, rows-1-r) = (0, 6).
    rotated = board_from_owners({(0, 6): (0, 3)}, 7, 7)
    assert canonical_key(original) == canonical_key(rotated)


def test_horizontal_flip_is_recognised_as_the_same_position():
    original = board_from_owners({(2, 1): (1, 2)}, 7, 7)
    flipped = board_from_owners({(2, 5): (1, 2)}, 7, 7)  # cols-1-c = 6-1 = 5
    assert canonical_key(original) == canonical_key(flipped)


def test_genuinely_different_positions_have_different_keys():
    a = board_from_owners({(0, 0): (0, 3)}, 7, 7)
    b = board_from_owners({(3, 3): (0, 3)}, 7, 7)  # center cell has no symmetric equivalent to a corner
    assert canonical_key(a) != canonical_key(b)


def test_owner_identity_is_preserved_not_just_occupancy():
    # Same cell, same coordinates, different owner - must NOT compare equal.
    a = board_from_owners({(0, 0): (0, 2)}, 7, 7)
    b = board_from_owners({(0, 0): (1, 2)}, 7, 7)
    assert canonical_key(a) != canonical_key(b)


def test_non_square_board_only_uses_dimension_preserving_symmetries():
    # canonical_key() restricts a non-square board to symmetries (0,2,4,5) -
    # the ones that don't swap rows and columns. Regression guard: if that
    # restriction were ever removed, rotate90/rotate270/transpose/
    # anti-transpose would compute a column index as if it were a row index
    # (and vice versa) on a board where rows != cols, which either raises
    # IndexError outright or silently misplaces cells depending on the
    # exact shape - this just needs to run cleanly on a non-square board
    # without either happening.
    rows, cols = 3, 5
    board = board_from_owners({(1, 4): (0, 3)}, rows, cols)
    key = canonical_key(board)
    assert key is not None
    assert len(key) == rows
    assert all(len(row) == cols for row in key)


def test_count_distinct_collapses_symmetric_duplicates():
    corner = board_from_owners({(0, 0): (0, 3)}, 7, 7)
    same_corner_rotated = board_from_owners({(0, 6): (0, 3)}, 7, 7)  # 90-degree rotation of `corner`
    center = board_from_owners({(3, 3): (0, 3)}, 7, 7)
    assert count_distinct([corner, same_corner_rotated, center]) == 2
