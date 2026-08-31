"""Tests transform_state/transform_policy - the numpy-array symmetry
transforms used for iteration-41 training-time augmentation (see
ReplayDataset in train.py). Both are built from the SAME _transform_cell
canonical_key already uses (via board_symmetry._index_map), but that shared
plumbing is exactly why a mistake here would be silent and severe: if the
state transform and policy transform ever disagreed about what a given
symmetry index means, every augmented training example would pair a
transformed board with the WRONG move probabilities - corrupted training
data that trains fine and produces a weaker network with no error anywhere.

Run with: pytest python/colourwars/tests/test_symmetry_augmentation.py -v
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from colourwars.board_symmetry import (  # noqa: E402
    transform_policy,
    transform_state,
    valid_symmetries,
)


def test_identity_leaves_state_and_policy_unchanged():
    state = np.arange(2 * 5 * 5).reshape(2, 5, 5).astype(np.float32)
    policy = np.arange(25).astype(np.float32)
    assert np.array_equal(transform_state(state, 0), state)
    assert np.array_equal(transform_policy(policy, 0, 5, 5), policy)


def test_180_degree_rotation_reverses_a_single_plane():
    # Hand-verified: a 180 rotation of a 3x3 grid reverses reading order.
    state = np.arange(3 * 3).reshape(1, 3, 3).astype(np.float32)
    rotated = transform_state(state, 2)[0]
    expected = np.array([[8, 7, 6], [5, 4, 3], [2, 1, 0]], dtype=np.float32)
    assert np.array_equal(rotated, expected)


def test_state_and_policy_transforms_stay_paired_under_every_symmetry():
    # The critical correctness property: for EVERY symmetry, moving a piece
    # from (r, c) in the state must correspond to the SAME (r, c) -> (nr, nc)
    # move in the policy - i.e. a policy that is entirely concentrated on
    # one cell in the original must be entirely concentrated on that cell's
    # transformed position afterward. This is what would silently break if
    # transform_state and transform_policy ever used different index maps.
    rows, cols = 7, 7
    for r in range(rows):
        for c in range(cols):
            state = np.zeros((1, rows, cols), dtype=np.float32)
            state[0, r, c] = 1.0
            policy = np.zeros(rows * cols, dtype=np.float32)
            policy[r * cols + c] = 1.0
            for sym in valid_symmetries(rows, cols):
                t_state = transform_state(state, sym)
                t_policy = transform_policy(policy, sym, rows, cols)
                state_peak = np.argwhere(t_state[0] == 1.0)[0]
                policy_peak_flat = int(np.argmax(t_policy))
                policy_peak = divmod(policy_peak_flat, cols)
                assert tuple(state_peak) == policy_peak, (
                    f"sym={sym}: state peak moved to {tuple(state_peak)} but "
                    f"policy peak moved to {policy_peak} - state/policy disagree"
                )


def test_every_symmetry_is_a_bijection_preserving_the_multiset_of_values():
    # A genuine board symmetry must be a permutation - every cell's value
    # appears exactly once afterward too, nothing duplicated or dropped.
    rows, cols = 7, 7
    policy = np.arange(rows * cols).astype(np.float32)
    for sym in valid_symmetries(rows, cols):
        transformed = transform_policy(policy, sym, rows, cols)
        assert sorted(transformed.tolist()) == sorted(policy.tolist())


def test_non_square_board_only_offers_dimension_preserving_symmetries():
    rows, cols = 3, 5
    state = np.zeros((1, rows, cols), dtype=np.float32)
    state[0, 1, 4] = 1.0
    policy = np.zeros(rows * cols, dtype=np.float32)
    policy[1 * cols + 4] = 1.0
    for sym in valid_symmetries(rows, cols):
        # Must run cleanly without raising/misplacing on a non-square shape.
        t_state = transform_state(state, sym)
        t_policy = transform_policy(policy, sym, rows, cols)
        assert t_state.shape == state.shape
        assert t_policy.shape == policy.shape
