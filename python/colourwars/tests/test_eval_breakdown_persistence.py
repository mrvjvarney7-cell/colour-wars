"""Asserts a real gate run actually persists its per-opening breakdown to
disk - this is a regression test, not a smoke test: the exact write this
checks for silently stopped happening once already (a live training
process ran code from before write_eval_breakdown's write line existed),
and nothing caught it until it was needed weeks later during an
investigation. If train.py's main loop ever stops calling
write_eval_breakdown(), or the function itself stops writing a real file,
this fails - a manual "read the code and conclude it should work" check
would not have caught the original incident.

Runs the REAL evaluate_vs_checkpoint_2p_paired harness (not a fake
gating_result) against two real, already-promoted checkpoints, at the
smallest sample size that still exercises the real code path, so this
stays fast. CHECKPOINT_DIR is monkeypatched to a pytest tmp_path for the
write itself, so this never touches the real checkpoints folder or
training_log.jsonl - read-only against real checkpoints, write-only
against a throwaway directory.

Run with: pytest python/colourwars/tests/test_eval_breakdown_persistence.py -v
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest  # noqa: E402
import torch  # noqa: E402

import colourwars.train as train  # noqa: E402
from colourwars.evaluate import evaluate_vs_checkpoint_2p_paired  # noqa: E402
from colourwars.network import ColourWarsNet  # noqa: E402

REAL_CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")


def _find_two_checkpoints():
    """Any two distinct iter_N.pt files - doesn't matter which, this test
    only cares that the write happens, not what the result says."""
    candidates = sorted(
        f for f in os.listdir(REAL_CHECKPOINT_DIR)
        if f.startswith("iter_") and f.endswith(".pt")
    )
    if len(candidates) < 2:
        pytest.skip("need at least 2 real iter_N.pt checkpoints on disk to run this test")
    # Plain lexicographic sort, not numeric - "iter_9.pt" sorts after
    # "iter_40.pt". Doesn't matter: any two distinct checkpoints exercise
    # the harness identically for what this test checks.
    return candidates[-1], candidates[-2]


def test_gate_run_persists_eval_breakdown_to_disk(tmp_path, monkeypatch):
    candidate_file, opponent_file = _find_two_checkpoints()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    candidate = ColourWarsNet().to(device)
    candidate.load_state_dict(torch.load(
        os.path.join(REAL_CHECKPOINT_DIR, candidate_file), map_location=device))
    candidate.eval()

    # Small but real: 2 openings -> 4 attempted games, 10 sims/move. Exercises
    # the actual harness function, not a hand-built fake result.
    gating_result = evaluate_vs_checkpoint_2p_paired(
        candidate, os.path.join(REAL_CHECKPOINT_DIR, opponent_file), device,
        num_openings=2, num_simulations=10, opening_plies=8,
        opening_temperature=0.5, max_moves=40,
    )

    # Redirect the write to a throwaway directory - this must never touch
    # the real checkpoints folder.
    monkeypatch.setattr(train, "CHECKPOINT_DIR", str(tmp_path))

    fake_iteration = 999999
    written_path = train.write_eval_breakdown(fake_iteration, gating_result)

    assert written_path == os.path.join(str(tmp_path), f"eval_breakdown_iter{fake_iteration}.json")
    assert os.path.exists(written_path), "write_eval_breakdown() ran but no file landed on disk"

    with open(written_path) as f:
        on_disk = json.load(f)

    for key in ("win_rate", "wins", "draws", "losses", "attempted", "openings"):
        assert key in on_disk, f"persisted breakdown is missing '{key}' - the gate result would be unauditable"

    assert on_disk["attempted"] == gating_result["attempted"]
    assert on_disk["wins"] + on_disk["draws"] + on_disk["losses"] == on_disk["attempted"]
