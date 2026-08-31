"""Canonicalises a board under D4 (the dihedral group of a square: 4
rotations x optional reflection) so two positions that are the same game
state up to rotation/reflection compare equal. Used by the opening sampler
to measure and enforce real diversity - "100 openings" should mean 100
genuinely distinct positions, not 100 labels covering a handful of
positions repeated under different rotations, which is exactly what the
2026-08-31 eval-harness investigation found (12 distinct positions out of
100 policy-sampled openings by move 20, one group alone covering 54).

Only the 4 dimension-preserving symmetries (identity, 180-degree rotation,
horizontal flip, vertical flip) apply to a non-square board; the other 4
(90/270-degree rotation, the two diagonal reflections) swap rows and
columns and are skipped unless rows == cols. The board is currently always
7x7, but this module doesn't assume that.
"""

from __future__ import annotations

from colourwars.game import Board


def _transform_cell(r: int, c: int, sym: int, rows: int, cols: int) -> tuple[int, int]:
    if sym == 0:
        return (r, c)
    if sym == 1:  # rotate 90 clockwise - only valid when rows == cols
        return (c, rows - 1 - r)
    if sym == 2:  # rotate 180
        return (rows - 1 - r, cols - 1 - c)
    if sym == 3:  # rotate 270 clockwise - only valid when rows == cols
        return (cols - 1 - c, r)
    if sym == 4:  # flip horizontal (mirror left-right)
        return (r, cols - 1 - c)
    if sym == 5:  # flip vertical (mirror top-bottom)
        return (rows - 1 - r, c)
    if sym == 6:  # transpose - only valid when rows == cols
        return (c, r)
    if sym == 7:  # anti-transpose - only valid when rows == cols
        return (cols - 1 - c, rows - 1 - r)
    raise ValueError(f"unknown symmetry index {sym}")


def canonical_key(board: Board) -> tuple:
    """A hashable, comparable key that's identical for any two boards that
    are the same position up to D4 symmetry, and different otherwise
    (assuming no false-positive hash collisions, which a plain tuple
    comparison doesn't have). Picks the lexicographically smallest of the
    valid transforms' (owner, count) grids as the canonical representative -
    an arbitrary but consistent choice, so any two symmetric boards land on
    the same key regardless of which one happens to be "first"."""
    rows = len(board)
    cols = len(board[0]) if rows else 0
    symmetries = range(8) if rows == cols else (0, 2, 4, 5)

    best = None
    for sym in symmetries:
        grid = [[None] * cols for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                nr, nc = _transform_cell(r, c, sym, rows, cols)
                cell = board[r][c]
                # None isn't orderable against int in Python 3 - -1 is a
                # safe stand-in since real owners are always >= 0.
                owner = cell.owner if cell.owner is not None else -1
                grid[nr][nc] = (owner, cell.count)
        key = tuple(tuple(row) for row in grid)
        if best is None or key < best:
            best = key
    return best


def count_distinct(boards: list[Board]) -> int:
    """How many genuinely distinct positions are in `boards`, under D4."""
    return len({canonical_key(b) for b in boards})
