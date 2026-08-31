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

from functools import lru_cache

import numpy as np

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
    symmetries = valid_symmetries(rows, cols)

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


def valid_symmetries(rows: int, cols: int) -> tuple[int, ...]:
    """Which of the 8 symmetry indices are valid for a board of this shape -
    all 8 for a square board, only the 4 dimension-preserving ones (see
    _transform_cell) otherwise. Shared by canonical_key and the training-time
    augmentation below so both use exactly the same restriction."""
    return tuple(range(8)) if rows == cols else (0, 2, 4, 5)


@lru_cache(maxsize=None)
def _index_map(sym: int, rows: int, cols: int) -> tuple:
    """Flat (rows*cols,) tuple where index_map[old_flat_index] = new_flat_index
    under symmetry `sym`, built from the SAME _transform_cell used by
    canonical_key - single source of truth for the coordinate transform, so
    board-object canonicalisation (used for eval-harness distinctness) and
    array-based training augmentation (used by transform_state/
    transform_policy below) can never silently disagree about what "symmetry
    3" means, which would otherwise pair a transformed state with the wrong
    policy target and silently corrupt training data. Cached: there are only
    8 symmetries x however many distinct (rows, cols) shapes ever occur (in
    practice, one), and this is called on the hot path of every training
    example access."""
    index_map = [0] * (rows * cols)
    for r in range(rows):
        for c in range(cols):
            nr, nc = _transform_cell(r, c, sym, rows, cols)
            index_map[r * cols + c] = nr * cols + nc
    return tuple(index_map)


def transform_state(state: np.ndarray, sym: int) -> np.ndarray:
    """Applies D4 symmetry `sym` to every plane of a (NUM_PLANES, rows, cols)
    state tensor - the same geometric transform on every plane's own
    (rows, cols) grid, since a symmetry of the BOARD acts identically on
    every plane regardless of what that plane encodes. Training-time
    augmentation (see ReplayDataset in train.py) - always call
    transform_policy with the SAME sym on the corresponding policy target,
    or the two will describe different positions."""
    planes, rows, cols = state.shape
    index_map = np.asarray(_index_map(sym, rows, cols))
    flat = state.reshape(planes, rows * cols)
    new_flat = np.empty_like(flat)
    new_flat[:, index_map] = flat
    return new_flat.reshape(planes, rows, cols)


def transform_policy(policy: np.ndarray, sym: int, rows: int, cols: int) -> np.ndarray:
    """Applies the same D4 symmetry to a flat (rows*cols,) policy vector -
    see transform_state's docstring; must be called with the same sym."""
    index_map = np.asarray(_index_map(sym, rows, cols))
    new_policy = np.empty_like(policy)
    new_policy[index_map] = policy
    return new_policy
