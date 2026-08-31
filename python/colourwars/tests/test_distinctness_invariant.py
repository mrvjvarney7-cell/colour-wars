"""Tests the distinctness invariant in isolation - no GPU/network/real games
needed, same rationale as test_opening_sampler.py. This is Correction 2's
hard requirement: the played-game distinctness check must cause
evaluate_vs_checkpoint_2p_paired to raise rather than silently returning a
win rate built on a handful of repeated scenarios, and must NOT fire on an
ordinary healthy run - and per the 2026-08-31 hardening pass, the raised
exception must carry the full partial result so a caller can still persist
it for audit.

Run with: pytest python/colourwars/tests/test_distinctness_invariant.py -v
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from colourwars.evaluate import (  # noqa: E402
    MIN_DISTINCTNESS_RATIO,
    EvalDistinctnessError,
    _check_distinctness,
    _distinctness_failure_reason,
)


def test_check_distinctness_is_pure_computation_and_never_raises():
    # Even wildly collapsed input must not raise here - _check_distinctness
    # is just counting, the pass/fail judgment is a separate function.
    mid20_keys = [f"pos{i % 8}" for i in range(100)]
    final_keys = [f"final{i % 3}" for i in range(100)]
    distinct_at_mid20, games_reaching_mid20, distinct_final = _check_distinctness(
        mid20_keys, final_keys, attempted=100
    )
    assert distinct_at_mid20 == 8
    assert games_reaching_mid20 == 100
    assert distinct_final == 3


def test_no_games_reaching_mid20_counts_as_zero_not_an_error():
    distinct_at_mid20, games_reaching_mid20, distinct_final = _check_distinctness(
        [], [f"final{i}" for i in range(10)], attempted=10
    )
    assert games_reaching_mid20 == 0
    assert distinct_at_mid20 == 0
    assert distinct_final == 10


def test_failure_reason_none_on_a_healthy_sample():
    # Matches the validated real result: ~99% distinct at move 20 and 100%
    # at final position.
    assert _distinctness_failure_reason(
        distinct_at_mid20=89, games_reaching_mid20=90, distinct_final=100, attempted=100
    ) is None


def test_failure_reason_set_on_a_collapsed_mid20_sample():
    reason = _distinctness_failure_reason(
        distinct_at_mid20=12, games_reaching_mid20=100, distinct_final=100, attempted=100
    )
    assert reason is not None
    assert "12" in reason and "100" in reason
    assert f"{MIN_DISTINCTNESS_RATIO:.0%}" in reason


def test_failure_reason_set_on_a_collapsed_final_sample():
    # mid20 healthy, final collapsed - both checks must be independently
    # enforced, not just the first one that happens to run.
    reason = _distinctness_failure_reason(
        distinct_at_mid20=90, games_reaching_mid20=90, distinct_final=5, attempted=100
    )
    assert reason is not None
    assert "5" in reason and "100" in reason


def test_exactly_at_the_threshold_is_not_a_failure():
    # 50/100 is exactly MIN_DISTINCTNESS_RATIO - the check uses a strict
    # "<", so this must not be judged a failure.
    assert _distinctness_failure_reason(
        distinct_at_mid20=50, games_reaching_mid20=100, distinct_final=50, attempted=100
    ) is None


def test_eval_distinctness_error_carries_the_partial_result():
    fake_result = {"win_rate": 0.5, "wins": 1, "openings": [{"opening_index": 0}]}
    err = EvalDistinctnessError("collapsed for testing", result=fake_result)
    assert err.result is fake_result
    assert "collapsed for testing" in str(err)
    assert isinstance(err, RuntimeError)
