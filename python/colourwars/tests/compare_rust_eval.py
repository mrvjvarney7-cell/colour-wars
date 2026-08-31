"""Correctness/parity evidence for the Rust paired-eval backend
(--eval-backend rust, see evaluate.py's _play_paired_2p_games_batch_rust)
against the trusted, production Python backend (_play_paired_2p_games).

Unlike compare_rust_engine.py - which drives ONE pre-chosen move through
both engines and diffs deterministic output - this compares TWO INDEPENDENT
MCTS implementations each choosing their own moves from the same real
checkpoints. Bit-identical play is not the right bar here: a single
near-tied PUCT score can legitimately flip on floating-point/summation-order
differences between the two implementations and cascade into a different
game from that point on. So this file draws an explicit line between hard
requirements (any failure is a real bug, script exits 1) and soft/
statistical parity (reported plainly, investigate if it looks bad, but not
auto-failed).

Neither backend logs its full per-ply move sequence (not needed for
production gating, and adding it would be new instrumentation solely for
this one-off script) - final-canonical-position equality is used as the
practical proxy for "the two backends played the same game": two different
move sequences landing on the identical D4-canonical final position on a
49-cell board under this ruleset is astronomically unlikely, so treating it
as a proxy for "picked the same moves" is sound.

This script produces EVIDENCE FOR A HUMAN DECISION, not a CI gate - whether
to flip --eval-backend's default is the user's call after reading its
output, exactly like the opening-sampler fix's own validation test before
it was approved.

Run with:
    python -m colourwars.tests.compare_rust_eval \
        --candidate-checkpoint colourwars/checkpoints/iter_36.pt \
        --opponent-checkpoint colourwars/checkpoints/iter_29.pt \
        --openings 200 --eval-simulations 100

Use two checkpoints that predate whatever iteration is currently training
(or run only once the live process is between subprocess launches) - this
only ever READS checkpoints, never touches training_log.jsonl or writes
into CHECKPOINT_DIR, but it does hold the GPU for its own forward passes.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

import colourwars_rs as rs

from colourwars.evaluate import (
    _generate_distinct_random_openings,
    _play_paired_2p_games,
    _play_paired_2p_games_batch_rust,
    board_from_flat,
    canonical_key,
)
from colourwars.game import COLS, ROWS, CRITICAL_MASS
from colourwars.network import ColourWarsNet
from colourwars.rust_selfplay import _make_numpy_forward_fn


def _load_net(checkpoint_path: str, device) -> ColourWarsNet:
    net = ColourWarsNet().to(device)
    net.load_state_dict(torch.load(checkpoint_path, map_location=device))
    net.eval()
    return net


def _run_python(candidate, opponent, device, openings, num_simulations, max_moves) -> list:
    return [
        _play_paired_2p_games(candidate, opponent, device, num_simulations, o["opening_actions"], max_moves=max_moves)
        for o in openings
    ]


def _run_rust(candidate, opponent, device, openings, num_simulations, max_moves, batch_size) -> list:
    return _play_paired_2p_games_batch_rust(
        candidate, opponent, device, num_simulations, openings, max_moves, batch_size
    )


def _run_rust_raw(candidate, opponent, device, openings, num_simulations, max_moves, batch_size):
    """Bypasses the Python adapter to get the raw flat owner/count arrays
    (not just their canonicalised keys) - needed to check board legality
    directly, which _play_paired_2p_games_batch_rust's reshaped output
    doesn't retain."""
    candidate_fn = _make_numpy_forward_fn(candidate, device)
    opponent_fn = _make_numpy_forward_fn(opponent, device)
    return rs.run_batched_paired_eval_rust(
        candidate_fn, opponent_fn, [o["opening_actions"] for o in openings],
        num_simulations=num_simulations, batch_size=batch_size, max_moves=max_moves,
    )


def _flatten(per_opening: list) -> list:
    """[[seat0, seat1], [seat0, seat1], ...] -> flat list of game dicts, one
    per (opening, seat) pair, in the same opening-major/seat-major order
    both backends already produce."""
    return [g for games in per_opening for g in games]


def _check_hard_requirements(label: str, per_opening: list, num_openings: int) -> None:
    flat = _flatten(per_opening)
    if len(per_opening) != num_openings:
        raise SystemExit(f"[{label}] FAIL: expected {num_openings} openings, got {len(per_opening)}")
    if len(flat) != 2 * num_openings:
        raise SystemExit(f"[{label}] FAIL: expected {2 * num_openings} games, got {len(flat)}")
    for opening_idx, games in enumerate(per_opening):
        seats = sorted(g["candidate_seat"] for g in games)
        if seats != [0, 1]:
            raise SystemExit(f"[{label}] FAIL: opening {opening_idx} has seats {seats}, expected [0, 1]")
    for g in flat:
        if g["candidate_score"] not in (0.0, 0.5, 1.0):
            raise SystemExit(f"[{label}] FAIL: candidate_score {g['candidate_score']} not in {{0, 0.5, 1}}")
        if g["decided"] and g["candidate_score"] == 0.5:
            raise SystemExit(f"[{label}] FAIL: decided=True but candidate_score=0.5 ({g})")
        if not g["decided"] and g["candidate_score"] != 0.5:
            raise SystemExit(f"[{label}] FAIL: decided=False but candidate_score!=0.5 ({g})")
    print(f"[{label}] hard requirements OK: {len(flat)} games, correct pairing, scores consistent with decided")


def _check_legality(label: str, raw_records) -> None:
    for r in raw_records:
        for owners, counts, name in (
            (r.mid_owners, r.mid_counts, "mid"),
            (r.final_owners, r.final_counts, "final"),
        ):
            if owners is None:
                continue
            if len(owners) != ROWS * COLS or len(counts) != ROWS * COLS:
                raise SystemExit(f"[{label}] FAIL: {name} board wrong size for opening {r.opening_index}")
            for c in counts:
                if not (0 <= c < CRITICAL_MASS):
                    raise SystemExit(f"[{label}] FAIL: illegal {name} board count {c} for opening {r.opening_index}")
    print(f"[{label}] board legality OK: every returned board is a legal position")


def _check_determinism(label: str, run_a: list, run_b: list) -> None:
    flat_a, flat_b = _flatten(run_a), _flatten(run_b)
    if len(flat_a) != len(flat_b):
        raise SystemExit(f"[{label}] FAIL: determinism check got different game counts ({len(flat_a)} vs {len(flat_b)})")
    for i, (ga, gb) in enumerate(zip(flat_a, flat_b)):
        for key in ("candidate_seat", "decided", "candidate_score", "move_count", "mid20_key", "final_key"):
            if ga[key] != gb[key]:
                raise SystemExit(
                    f"[{label}] FAIL: NOT DETERMINISTIC - game {i} field {key!r} differs between two runs "
                    f"with identical inputs: {ga[key]!r} vs {gb[key]!r}"
                )
    print(f"[{label}] determinism OK: two runs with identical inputs produced identical results")


def _report_statistical_parity(python_games: list, rust_games: list) -> None:
    flat_py, flat_rs = _flatten(python_games), _flatten(rust_games)
    n = len(flat_py)
    assert n == len(flat_rs)

    final_match = sum(1 for a, b in zip(flat_py, flat_rs) if a["final_key"] == b["final_key"])
    move_count_match = sum(1 for a, b in zip(flat_py, flat_rs) if a["move_count"] == b["move_count"])
    decided_match = sum(1 for a, b in zip(flat_py, flat_rs) if a["decided"] == b["decided"])

    py_win_rate = sum(g["candidate_score"] for g in flat_py) / n
    rs_win_rate = sum(g["candidate_score"] for g in flat_rs) / n

    print("\n=== Statistical parity (soft - investigate if these look bad, not an automatic failure) ===")
    print(f"Final-position match rate (proxy for \"played the same game\" - neither backend logs "
          f"per-ply moves): {final_match}/{n} ({final_match / n:.1%})")
    print(f"Move-count match rate: {move_count_match}/{n} ({move_count_match / n:.1%})")
    print(f"decided-vs-capped agreement rate: {decided_match}/{n} ({decided_match / n:.1%}) - a systematic "
          f"difference here (not just occasional tie-breaking noise) would suggest a real strength/looping "
          f"difference, not floating-point divergence")
    print(f"Aggregate win_rate: python={py_win_rate:.1%} rust={rs_win_rate:.1%} "
          f"(delta={abs(py_win_rate - rs_win_rate):.1%}) - a few points is expected given the diverging-game "
          f"rate above; a double-digit delta is a real signal worth investigating before trusting this backend")
    if final_match / n < 0.90:
        print(f"\n*** NOTE: final-position match rate ({final_match / n:.1%}) is below the ~90% rough "
              f"investigate-if-below threshold - worth understanding why before proposing the default flip. ***")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--opponent-checkpoint", required=True)
    parser.add_argument("--openings", type=int, default=20,
                         help="default is a quick sanity size - pass e.g. --openings 200 for the real "
                              "verification run before proposing the --eval-backend default flip; the "
                              "python side is the slow, never-Rust-ported path, so scale deliberately")
    parser.add_argument("--opening-plies", type=int, default=8)
    parser.add_argument("--eval-simulations", type=int, default=100, help="production value - see train.py")
    parser.add_argument("--eval-max-moves", type=int, default=300)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check-python-determinism", action="store_true",
                         help="also runs the (slow) python backend twice to confirm it's internally "
                              "deterministic too - off by default purely for time; python's own "
                              "determinism isn't new/at-risk the way the new rust backend's is")
    args = parser.parse_args()

    if not os.path.exists(args.candidate_checkpoint):
        raise SystemExit(f"candidate checkpoint not found: {args.candidate_checkpoint}")
    if not os.path.exists(args.opponent_checkpoint):
        raise SystemExit(f"opponent checkpoint not found: {args.opponent_checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    candidate = _load_net(args.candidate_checkpoint, device)
    opponent = _load_net(args.opponent_checkpoint, device)

    import random
    random.seed(args.seed)
    openings = _generate_distinct_random_openings(args.openings, args.opening_plies)
    print(f"Generated {len(openings)} distinct openings ({args.opening_plies} plies each).\n")

    print(f"=== Rust backend: run A ({args.eval_simulations} sims/move) ===")
    t0 = time.time()
    rust_a = _run_rust(candidate, opponent, device, openings, args.eval_simulations, args.eval_max_moves, args.eval_batch_size)
    print(f"  done in {time.time() - t0:.1f}s")
    _check_hard_requirements("rust A", rust_a, len(openings))

    print(f"\n=== Rust backend: run B (determinism check) ===")
    t0 = time.time()
    rust_b = _run_rust(candidate, opponent, device, openings, args.eval_simulations, args.eval_max_moves, args.eval_batch_size)
    print(f"  done in {time.time() - t0:.1f}s")
    _check_determinism("rust A vs B", rust_a, rust_b)

    print(f"\n=== Rust backend: raw run (board legality check) ===")
    t0 = time.time()
    raw = _run_rust_raw(candidate, opponent, device, openings, args.eval_simulations, args.eval_max_moves, args.eval_batch_size)
    print(f"  done in {time.time() - t0:.1f}s")
    _check_legality("rust raw", raw)

    print(f"\n=== Python backend (this WILL be slow - {args.eval_simulations} sims/move, "
          f"single-game pure-Python MCTS) ===")
    t0 = time.time()
    python_a = _run_python(candidate, opponent, device, openings, args.eval_simulations, args.eval_max_moves)
    print(f"  done in {time.time() - t0:.1f}s")
    _check_hard_requirements("python A", python_a, len(openings))

    if args.check_python_determinism:
        print(f"\n=== Python backend: run B (determinism check - --check-python-determinism) ===")
        t0 = time.time()
        python_b = _run_python(candidate, opponent, device, openings, args.eval_simulations, args.eval_max_moves)
        print(f"  done in {time.time() - t0:.1f}s")
        _check_determinism("python A vs B", python_a, python_b)

    _report_statistical_parity(python_a, rust_a)

    print("\nOK: all hard requirements passed. Review the statistical parity section above before "
          "deciding whether to propose changing --eval-backend's default.")


if __name__ == "__main__":
    main()
