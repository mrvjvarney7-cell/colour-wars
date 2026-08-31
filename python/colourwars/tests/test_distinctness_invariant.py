"""Tests _check_distinctness in isolation - no GPU/network/real games needed,
same rationale as test_opening_sampler.py. This is Correction 2's hard
requirement: the played-game distinctness check must RAISE on a collapsed
sample rather than silently returning a win rate built on a handful of
repeated scenarios, and must NOT fire on an ordinary healthy run.

Run with: pytest python/colourwars/tests/test_distinctness_invariant.py -v
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest  # noqa: E402

from colourwars.evaluate import _check_distinctness  # noqa: E402


def test_raises_on_a_collapsed_mid20_sample():
    # 100 games, but only 8 distinct positions at move 20 (12% distinct) -
    # exactly the shape of the real collapse this investigation found.
    mid20_keys = [f"pos{i % 8}" for i in range(100)]
    final_keys = [f"final{i}" for i in range(100)]  # final positions fine on their own
    with pytest.raises(RuntimeError, match="collapsed"):
        _check_distinctness(mid20_keys, final_keys, attempted=100)


def test_raises_on_a_collapsed_final_sample():
    # mid20 is healthy, but final positions collapsed - both checks must be
    # independently enforced, not just the first one that happens to run.
    mid20_keys = [f"pos{i}" for i in range(100)]
    final_keys = [f"final{i % 5}" for i in range(100)]
    with pytest.raises(RuntimeError, match="collapsed"):
        _check_distinctness(mid20_keys, final_keys, attempted=100)


def test_does_not_raise_on_a_healthy_sample():
    # Matches the validated real result: ~99% distinct at move 20 and at
    # final position. Must return the summary, not raise.
    mid20_keys = [f"pos{i}" for i in range(89)] + ["pos0"]  # 89/90 distinct
    final_keys = [f"final{i}" for i in range(100)]
    distinct_at_mid20, games_reaching_mid20, distinct_final = _check_distinctness(
        mid20_keys, final_keys, attempted=100
    )
    assert distinct_at_mid20 == 89
    assert games_reaching_mid20 == 90
    assert distinct_final == 100


def test_exactly_at_the_threshold_does_not_raise():
    # 50 distinct out of 100 is exactly MIN_DISTINCTNESS_RATIO - the check
    # uses a strict "<", so this must pass, not raise.
    mid20_keys = [f"pos{i}" for i in range(50)] * 2  # 50 distinct among 100
    final_keys = [f"final{i}" for i in range(50)] * 2
    distinct_at_mid20, games_reaching_mid20, distinct_final = _check_distinctness(
        mid20_keys, final_keys, attempted=100
    )
    assert distinct_at_mid20 == 50
    assert distinct_final == 50


def test_no_games_reaching_mid20_is_not_treated_as_a_collapse():
    # An opening_plies/max_moves combination where no game reaches move 20
    # yet (e.g. very short games) must not divide by zero or falsely raise -
    # there's simply nothing to check at that checkpoint.
    final_keys = [f"final{i}" for i in range(10)]
    distinct_at_mid20, games_reaching_mid20, distinct_final = _check_distinctness(
        [], final_keys, attempted=10
    )
    assert games_reaching_mid20 == 0
    assert distinct_at_mid20 == 0
    assert distinct_final == 10
