"""Tests lr_for_iteration - a pure function of the global iteration number,
not a stateful scheduler, specifically so LR decay survives this training
run's frequent process restarts (see the function's own docstring). No
GPU/network needed.

Run with: pytest python/colourwars/tests/test_lr_decay.py -v
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from colourwars.train import lr_for_iteration  # noqa: E402


def test_gamma_1_is_always_a_no_op_regardless_of_iteration():
    for iteration in (0, 1, 40, 41, 1000):
        assert lr_for_iteration(1e-3, iteration, decay_start_iteration=0,
                                 decay_every=10, decay_gamma=1.0) == 1e-3


def test_before_decay_start_iteration_lr_is_unchanged():
    lr = lr_for_iteration(1e-3, iteration=40, decay_start_iteration=41,
                           decay_every=10, decay_gamma=0.5)
    assert lr == 1e-3


def test_at_decay_start_iteration_lr_is_still_unchanged():
    # Zero full decay-steps have elapsed exactly at the boundary itself.
    lr = lr_for_iteration(1e-3, iteration=41, decay_start_iteration=41,
                           decay_every=10, decay_gamma=0.5)
    assert lr == 1e-3


def test_one_full_decay_step_after_start():
    lr = lr_for_iteration(1e-3, iteration=51, decay_start_iteration=41,
                           decay_every=10, decay_gamma=0.5)
    assert abs(lr - 5e-4) < 1e-12


def test_two_full_decay_steps_after_start():
    lr = lr_for_iteration(1e-3, iteration=61, decay_start_iteration=41,
                           decay_every=10, decay_gamma=0.5)
    assert abs(lr - 2.5e-4) < 1e-12


def test_result_is_identical_regardless_of_process_restarts():
    # The whole point: the same global iteration number must produce the
    # same LR whether this is the first process to ever run iteration 55
    # or the fifth process restarted since - there is no hidden state.
    lr_a = lr_for_iteration(1e-3, iteration=55, decay_start_iteration=41,
                             decay_every=10, decay_gamma=0.5)
    lr_b = lr_for_iteration(1e-3, iteration=55, decay_start_iteration=41,
                             decay_every=10, decay_gamma=0.5)
    assert lr_a == lr_b
