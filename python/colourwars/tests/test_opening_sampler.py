"""Tests the uniform-random opening generator and its generation-time
distinctness loop directly - no GPU/network needed, since opening
generation no longer uses either (that's the whole point of this change).
evaluate_vs_checkpoint_2p_paired's own end-to-end behaviour (including
this generator) is still covered by test_eval_breakdown_persistence.py,
which runs it for real against real checkpoints.

Run with: pytest python/colourwars/tests/test_opening_sampler.py -v
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest  # noqa: E402

from colourwars.evaluate import (  # noqa: E402
    _generate_distinct_random_openings,
    _generate_random_opening,
)


def test_generate_random_opening_returns_actions_and_a_canonical_key():
    actions, key = _generate_random_opening(opening_plies=8)
    assert len(actions) == 8
    assert all(isinstance(a, int) for a in actions)
    assert key is not None


def test_distinct_openings_are_actually_distinct():
    openings = _generate_distinct_random_openings(num_openings=20, opening_plies=8)
    assert len(openings) == 20
    keys = [o["canonical_key"] for o in openings]
    assert len(set(keys)) == 20, "dedup loop let a duplicate canonical position through"
    # Each opening_actions list is AT MOST opening_plies long - a random
    # sequence can legitimately end the game early (elimination via
    # cascade), which _generate_random_opening correctly stops for rather
    # than playing moves into an already-finished game.
    assert all(len(o["opening_actions"]) <= 8 for o in openings)


def test_raises_rather_than_silently_returning_fewer_than_requested():
    # opening_plies=0 means every "opening" is the untouched starting
    # board - always the same canonical position, no matter how many times
    # it's regenerated. Asking for 2 distinct openings from a space that
    # only ever contains 1 must raise, not silently return 1.
    with pytest.raises(RuntimeError, match="Could not generate"):
        _generate_distinct_random_openings(num_openings=2, opening_plies=0)


def test_a_single_opening_never_needs_to_raise():
    # Sanity check the raise path isn't over-eager: requesting exactly 1
    # opening from the degenerate opening_plies=0 space must succeed, since
    # 1 distinct position is trivially available.
    openings = _generate_distinct_random_openings(num_openings=1, opening_plies=0)
    assert len(openings) == 1
    assert openings[0]["opening_actions"] == []
