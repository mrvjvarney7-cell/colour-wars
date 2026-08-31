"""Tests how compute_promoted_elo_chain/find_elo_chain_reset_iteration/
derive_version_info handle a "harness boundary" marker that sets
elo_chain_reset but does NOT change which checkpoint file is best.pt - the
2026-08-31 restart marker's exact shape, distinct from the iteration-26
elo_chain_reset marker (which also carries new_best_checkpoint because that
one really did rebaseline best.pt to a different file).

Without the new_best_checkpoint distinction, derive_version_info's best.pt
branch would misattribute best.pt's identity to a marker that never touched
best.pt, and its per-checkpoint branch could pick up the marker's own
(mostly-null) fields instead of that iteration's real eval record when the
two share an iteration number and the marker was written first - both are
real bugs this file guards against, not hypotheticals.

Run with: pytest python/colourwars/tests/test_elo_chain_markers.py -v
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from colourwars.export_weights import (  # noqa: E402
    compute_promoted_elo_chain,
    derive_version_info,
    find_elo_chain_reset_iteration,
)

REAL_RECORD_ITER36 = {
    "iteration": 36, "timestamp": 1788073177.0, "games": 1500, "examples_in_buffer": 100,
    "policy_loss": 2.2, "value_loss": 0.17, "gating_harness": "2p_paired_v1",
    "win_rate_vs_best": 0.97, "win_rate_vs_random": 1.0, "promoted": True,
    "iter_time_sec": 100.0, "elo": 1973.86, "best_elo": 1973.86,
}
REAL_RECORD_ITER37_NOT_PROMOTED = {
    "iteration": 37, "timestamp": 1788080369.0, "games": 1500, "examples_in_buffer": 100,
    "policy_loss": 2.2, "value_loss": 0.17, "gating_harness": "2p_paired_v1",
    "win_rate_vs_best": 0.48, "win_rate_vs_random": 0.98, "promoted": False,
    "iter_time_sec": 100.0, "elo": 1964.0, "best_elo": 1973.86,
}
# Shape of the 2026-08-31 harness-boundary marker: elo_chain_reset without
# new_best_checkpoint, since best.pt itself does not change at this boundary.
HARNESS_BOUNDARY_MARKER_ITER41 = {
    "event": "harness_boundary", "iteration": 41, "timestamp": 1788110000.0,
    "note": "eval_simulations 20->100, opening sampler policy-guided->uniform random D4 dedup, "
            "iteration 41 training changes (symmetry augmentation + LR decay)",
    "eval_simulations": 100, "opening_sampler": "uniform_random_d4_dedup",
    "training_changes": ["symmetry_augmentation", "lr_decay"],
    "checkpoint_at_boundary": "iter_36.pt", "best_elo": 0.0, "promoted": False,
    "elo_chain_reset": True,
}
REAL_RECORD_ITER41_PROMOTED = {
    "iteration": 41, "timestamp": 1788111000.0, "games": 1500, "examples_in_buffer": 100,
    "policy_loss": 2.1, "value_loss": 0.16, "gating_harness": "2p_paired_v1",
    "win_rate_vs_best": 0.62, "win_rate_vs_random": 1.0, "promoted": True,
    "iter_time_sec": 100.0, "elo": 0.0, "best_elo": 0.0,
}


def _write_log(tmp_path, records):
    path = os.path.join(str(tmp_path), "training_log.jsonl")
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return path


def test_find_elo_chain_reset_iteration_sees_the_boundary_marker(tmp_path):
    log = _write_log(tmp_path, [
        REAL_RECORD_ITER36, REAL_RECORD_ITER37_NOT_PROMOTED, HARNESS_BOUNDARY_MARKER_ITER41,
    ])
    assert find_elo_chain_reset_iteration(log) == 41


def test_compute_promoted_elo_chain_drops_pre_boundary_records(tmp_path):
    log = _write_log(tmp_path, [
        REAL_RECORD_ITER36, REAL_RECORD_ITER37_NOT_PROMOTED, HARNESS_BOUNDARY_MARKER_ITER41,
    ])
    chain = compute_promoted_elo_chain(log)
    assert 36 not in chain, "iteration 36's old-harness Elo must not carry across the boundary"
    assert chain[41] == 0.0, "the marker's best_elo becomes the new chain anchor"


def test_compute_promoted_elo_chain_combines_boundary_and_a_real_same_iteration_promotion(tmp_path):
    # If iteration 41 ALSO promotes for real (its own record appears after
    # the marker, same iteration number), the chain's value at key 41 must
    # end up as the anchor plus that real promotion's own diff - not just
    # the bare anchor, and not crash on the duplicate key.
    log = _write_log(tmp_path, [
        REAL_RECORD_ITER36, REAL_RECORD_ITER37_NOT_PROMOTED,
        HARNESS_BOUNDARY_MARKER_ITER41, REAL_RECORD_ITER41_PROMOTED,
    ])
    chain = compute_promoted_elo_chain(log)
    assert chain[41] != 0.0, "a real 62% promotion at iteration 41 must move the anchor, not leave it bare"


def test_derive_version_info_for_the_boundary_iteration_checkpoint_uses_the_real_record(tmp_path):
    # iter_41.pt's own info must come from its real eval record, not the
    # marker that happens to share its iteration number and was written
    # first - the marker has no winRateVsBest/promoted of its own.
    log = _write_log(tmp_path, [
        REAL_RECORD_ITER36, REAL_RECORD_ITER37_NOT_PROMOTED,
        HARNESS_BOUNDARY_MARKER_ITER41, REAL_RECORD_ITER41_PROMOTED,
    ])
    info = derive_version_info("iter_41.pt", log)
    assert info["winRateVsBest"] == 0.62
    assert info["promoted"] is True


def test_derive_version_info_for_best_pt_is_not_misattributed_to_the_boundary_marker(tmp_path):
    # Before iteration 41's real record lands, best.pt is still iter_36.pt -
    # the boundary marker must NOT be picked up as "best.pt changed" just
    # because it sets elo_chain_reset; only new_best_checkpoint means that.
    log = _write_log(tmp_path, [
        REAL_RECORD_ITER36, REAL_RECORD_ITER37_NOT_PROMOTED, HARNESS_BOUNDARY_MARKER_ITER41,
    ])
    info = derive_version_info("best.pt", log)
    assert info["iteration"] == 36
    assert info.get("rebaselined") is not True


def test_derive_version_info_for_best_pt_does_follow_a_real_post_boundary_promotion(tmp_path):
    log = _write_log(tmp_path, [
        REAL_RECORD_ITER36, REAL_RECORD_ITER37_NOT_PROMOTED,
        HARNESS_BOUNDARY_MARKER_ITER41, REAL_RECORD_ITER41_PROMOTED,
    ])
    info = derive_version_info("best.pt", log)
    assert info["iteration"] == 41
    assert info["promoted"] is True
