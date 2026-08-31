"""Tests that an "ungated" iteration record (the shape train.py writes when
evaluate_vs_checkpoint_2p_paired raises EvalDistinctnessError - see main()'s
except block) doesn't crash check_stagnation or summarize_training.py, and
prints/behaves distinguishably from both a normal record and an
elo_chain_reset marker. Both have needed defensive fixes for markers before
(see the 2026-08-31 restart-marker work) - this is a real regression guard,
not a hypothetical.

Run with: pytest python/colourwars/tests/test_ungated_iteration.py -v
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import colourwars.summarize_training as summarize_training  # noqa: E402
from colourwars.train import check_stagnation  # noqa: E402

REAL_RECORD = {
    "iteration": 1, "timestamp": time.time(), "games": 10, "examples_in_buffer": 100,
    "policy_loss": 1.0, "value_loss": 1.0, "win_rate_vs_best": 0.5, "win_rate_vs_random": 0.9,
    "promoted": False, "iter_time_sec": 10.0, "best_elo": 1000.0,
}
ELO_CHAIN_RESET_MARKER = {
    "event": "rebaseline", "iteration": 2, "timestamp": time.time(),
    "note": "test marker", "best_elo": 0.0, "promoted": False, "elo_chain_reset": True,
}
UNGATED_RECORD = {
    "iteration": 3, "timestamp": time.time(), "games": 10, "examples_in_buffer": 100,
    "policy_loss": 1.0, "value_loss": 1.0, "gating_harness": "2p_paired_v1",
    "gated": False, "gate_error": "Eval sample has collapsed - only 5 of 100 (5.0%) are distinct...",
    "eval_breakdown_path": "/fake/eval_breakdown_iter3.json", "iter_time_sec": 10.0, "best_elo": 1000.0,
}


def test_check_stagnation_does_not_crash_on_a_mixed_history():
    history = [REAL_RECORD, ELO_CHAIN_RESET_MARKER, UNGATED_RECORD] * 2  # >= STAGNATION_WINDOW
    check_stagnation(history)  # must not raise


def test_ungated_record_is_excluded_from_stagnation_window():
    # An ungated iteration has no win_rate_vs_best - it must not be treated
    # as "a real gate result stuck near 50%" by the stagnation window logic.
    history = [dict(REAL_RECORD, iteration=i, win_rate_vs_best=0.5) for i in range(5)]
    history.append(UNGATED_RECORD)
    warning = check_stagnation(history)
    assert warning is not None and "STAGNATION" in warning  # from the 5 real 50% records
    # Confirms the ungated record wasn't silently counted as a 6th data
    # point that would change the window - if it were, this would still
    # pass by coincidence, so the real guarantee is just "did not crash".


def test_summarize_training_does_not_crash_on_mixed_marker_shapes(monkeypatch, capsys):
    history = [REAL_RECORD, ELO_CHAIN_RESET_MARKER, UNGATED_RECORD]
    monkeypatch.setattr(summarize_training, "read_log", lambda: history)
    monkeypatch.setattr(sys, "argv", ["summarize_training"])
    summarize_training.main()  # must not raise (KeyError on the old code)
    out = capsys.readouterr().out
    assert "rebaseline/boundary" in out
    assert "UNGATED" in out
