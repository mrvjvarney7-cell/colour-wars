"""Training loop: alternates self-play data generation with network training,
periodically evaluating the new network against the previous best and only
promoting it if it actually wins more often. Checkpoints every iteration so
runs can be resumed.

Usage:
    python -m colourwars.train --iterations 3 --games-per-iter 20 \
        --simulations 100 --epochs 4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from colourwars.evaluate import evaluate_vs_random, evaluate_vs_checkpoint_2p_paired
from colourwars.network import ColourWarsNet
from colourwars.selfplay import TrainingExample, generate_selfplay_games, generate_selfplay_games_batched, play_one_game

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
TRAINING_LOG_PATH = os.path.join(CHECKPOINT_DIR, "training_log.jsonl")

# If win rate vs previous-best stays in this "no clear improvement" band for
# this many consecutive iterations, flag it - see check_stagnation().
STAGNATION_BAND = (0.40, 0.60)
STAGNATION_WINDOW = 5

# Elo is tracked purely as a reporting/diagnostic metric (see win_rate_to_elo_diff
# below) - it does NOT drive promotion, which still uses the win-rate-vs-best
# threshold below. It's only ever meaningful relative to this run's own
# checkpoint lineage, not an absolute/external scale: the first best.pt this
# run's log has Elo history for is anchored at INITIAL_ELO, and every later
# rating is a relative estimate of strength gained or lost from there.
INITIAL_ELO = 1000.0


def win_rate_to_elo_diff(win_rate: float) -> float:
    """Maximum-likelihood Elo difference implied by a win rate against a
    fixed-strength opponent, under the standard logistic Elo model:
    win_rate = 1 / (1 + 10**(-diff/400)), solved for diff. Clamped away from
    0/1 (which imply an infinite gap) since a real match result - even a
    100/100 sweep - never actually proves infinite skill difference."""
    s = min(max(win_rate, 0.02), 0.98)
    return 400.0 * math.log10(s / (1.0 - s))


def write_eval_breakdown(iteration: int, gating_result: dict) -> str:
    """Persists the full per-opening gating breakdown (wins/draws/losses,
    every opening's own candidate_seat/decided/candidate_score/move_count/
    reason - see evaluate_vs_checkpoint_2p_paired's docstring) so a gate
    result can be audited after the fact instead of trusting a single
    win-rate number. Pulled out as its own function specifically so it's a
    unit the test suite can call and assert against directly
    (test_eval_breakdown_persistence.py) - this exact write silently
    stopped happening once before (a live process ran code from before this
    line existed) with nothing catching it until it was needed weeks later."""
    breakdown_path = os.path.join(CHECKPOINT_DIR, f"eval_breakdown_iter{iteration}.json")
    with open(breakdown_path, "w") as f:
        json.dump(gating_result, f)
    return breakdown_path


def append_log(record: dict):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    with open(TRAINING_LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def read_log() -> list:
    if not os.path.exists(TRAINING_LOG_PATH):
        return []
    records = []
    with open(TRAINING_LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def check_stagnation(history: list) -> str | None:
    """Returns a human-readable warning string if the recent history shows no
    clear improvement (win rate vs previous-best stuck near 50%) or the
    losses look like they've plateaued/diverged, else None."""
    # Non-eval marker records (e.g. an elo_chain_reset rebaseline) have no
    # win_rate_vs_best/policy_loss/value_loss - exclude them so they can't
    # crash the window logic below or silently widen it.
    history = [r for r in history if "win_rate_vs_best" in r]

    if len(history) < STAGNATION_WINDOW:
        return None

    recent = history[-STAGNATION_WINDOW:]
    win_rates = [r["win_rate_vs_best"] for r in recent]
    lo, hi = STAGNATION_BAND
    if all(lo <= w <= hi for w in win_rates):
        return (
            f"STAGNATION WARNING: win rate vs previous-best has stayed within "
            f"[{lo:.0%}, {hi:.0%}] for the last {STAGNATION_WINDOW} iterations "
            f"({[f'{w:.0%}' for w in win_rates]}) - no clear sign of improvement."
        )

    for r in recent:
        pl, vl = r.get("policy_loss"), r.get("value_loss")
        if pl is not None and (math.isnan(pl) or math.isinf(pl)):
            return f"DIVERGENCE WARNING: policy_loss is {pl} at iteration {r['iteration']}."
        if vl is not None and (math.isnan(vl) or math.isinf(vl)):
            return f"DIVERGENCE WARNING: value_loss is {vl} at iteration {r['iteration']}."

    value_losses = [r["value_loss"] for r in recent]
    if value_losses[-1] > 2.0 * min(value_losses):
        return (
            f"DIVERGENCE WARNING: value_loss climbed from a recent low of "
            f"{min(value_losses):.4f} to {value_losses[-1]:.4f} within the last "
            f"{STAGNATION_WINDOW} iterations."
        )

    return None


class ReplayDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex: TrainingExample = self.examples[idx]
        return (
            torch.from_numpy(ex.state),
            torch.from_numpy(ex.policy),
            torch.from_numpy(ex.value),
        )


def train_epoch(net: ColourWarsNet, loader: DataLoader, optimizer, device) -> dict:
    net.train()
    total_policy_loss = 0.0
    total_value_loss = 0.0
    n_batches = 0

    for states, policies, values in loader:
        states = states.to(device)
        policies = policies.to(device)
        values = values.to(device)

        policy_logits, pred_values = net(states)
        log_probs = F.log_softmax(policy_logits, dim=1)
        policy_loss = -(policies * log_probs).sum(dim=1).mean()
        value_loss = F.mse_loss(pred_values, values)
        loss = policy_loss + value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        n_batches += 1

    return {
        "policy_loss": total_policy_loss / max(n_batches, 1),
        "value_loss": total_value_loss / max(n_batches, 1),
    }


def save_checkpoint(net: ColourWarsNet, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(net.state_dict(), path)


def load_checkpoint(net: ColourWarsNet, path: str, device):
    net.load_state_dict(torch.load(path, map_location=device))


def generate_selfplay(net: ColourWarsNet, device, args, num_games: int, seed: int):
    """Shared self-play call used both for a normal iteration and for the
    optional buffer prefill below - same backend selection either way."""
    if args.selfplay_backend == "rust":
        from colourwars.rust_selfplay import generate_selfplay_games_rust
        return generate_selfplay_games_rust(
            net, device, num_games,
            num_simulations=args.simulations, batch_size=args.selfplay_batch_size,
            seed=seed,
        )
    elif args.selfplay_batch_size > 1:
        return generate_selfplay_games_batched(
            net, device, num_games,
            num_simulations=args.simulations, batch_size=args.selfplay_batch_size,
        )
    else:
        games_examples = []
        for g in range(num_games):
            n_players = random.choice([2, 3, 4])
            examples = play_one_game(net, device, n_players, num_simulations=args.simulations)
            games_examples.append(examples)
        return games_examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--games-per-iter", type=int, default=20)
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--selfplay-batch-size", type=int, default=32,
                         help="number of self-play games run concurrently, batching their MCTS "
                              "leaf evaluations into one network call per round; 1 disables "
                              "batching and uses the original single-game path (python backend only)")
    parser.add_argument("--selfplay-backend", choices=["rust", "python"], default="rust",
                         help="rust: colourwars_rs engine+MCTS (5.5x faster, correctness-verified "
                              "against the python engine across 700K+ plies). python: original "
                              "pure-Python path, kept for fallback/debugging.")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--replay-buffer-games", type=int, default=200,
                         help="max self-play GAMES worth of examples kept in the replay buffer")
    parser.add_argument("--eval-games", type=int, default=100,
                         help="game count for the DIAGNOSTIC-ONLY vs-random check "
                              "(win_rate_vs_random) - logged but does not control promotion")
    parser.add_argument("--eval-openings", type=int, default=100,
                         help="the actual promotion-gating metric: this many distinct 2-player "
                              "openings, each played twice with the candidate in both seats "
                              "(so up to 2x this many games total) - see evaluate_vs_checkpoint_2p_paired. "
                              "2p-only and paired because free-for-all parity isn't 50% and "
                              "seat/first-player advantage otherwise dominates the variance")
    parser.add_argument("--opening-plies", type=int, default=8,
                         help="how many opening moves (per paired-eval game) are sampled at "
                              "--opening-temperature before play goes fully greedy")
    parser.add_argument("--opening-temperature", type=float, default=0.5,
                         help="move-selection temperature for the first --opening-plies moves of "
                              "each paired-eval opening (KataGo's gating protocol: temperature for "
                              "position diversity, root Dirichlet noise OFF since that would degrade "
                              "search quality itself rather than just diversify positions)")
    parser.add_argument("--eval-simulations", type=int, default=100,
                         help="MCTS sims/move for evaluation games. Was 20 by default - too far "
                              "below --simulations (100, what self-play and thus promotion is "
                              "actually meant to measure): a network can look flat at 20 sims and "
                              "strong at 100, so grading at 20 measures a different thing than what "
                              "gets deployed. Raised to match, even though eval still uses the "
                              "slower Python single-game MCTS (evaluate.py wasn't part of the Rust "
                              "port) - 200 games/iteration at 100 sims is still small next to 1500 "
                              "self-play games/iteration.")
    parser.add_argument("--eval-max-moves", type=int, default=300,
                         help="ply cap per paired-eval game before it's scored as a 0.5/0.5 draw "
                              "(engine-tournament adjudication convention - never discarded, so the "
                              "denominator can't silently shrink with however many games grind out)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start-iteration", type=int, default=1,
                         help="iteration number to label this run's first iteration as - use when "
                              "resuming into an existing training_log.jsonl/checkpoints dir so "
                              "numbering (and the stagnation-detection window, which reads "
                              "training_log.jsonl directly) stays continuous instead of restarting "
                              "at 1 and colliding with existing iter_N.pt files")
    parser.add_argument("--prefill-games", type=int, default=0,
                         help="generate this many extra self-play games from best.pt (mixed 2/3/4p, "
                              "same as any other self-play round) and seed the replay buffer with "
                              "them before the first iteration, instead of waiting several "
                              "iterations for it to fill back up on its own after a fresh process "
                              "start or a revert-to-best. Typically pass "
                              "replay-buffer-games minus games-per-iter, so combined with the "
                              "first iteration's own games the buffer is already at capacity.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    net = ColourWarsNet().to(device)
    best_path = os.path.join(CHECKPOINT_DIR, "best.pt")
    if args.resume and os.path.exists(best_path):
        print(f"Resuming from {best_path}")
        load_checkpoint(net, best_path, device)
    else:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        save_checkpoint(net, best_path)

    def make_optimizer():
        return torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)

    optimizer = make_optimizer()

    replay_buffer = []  # list of games, each a list[TrainingExample]

    if args.prefill_games > 0:
        print(f"\nPrefilling the replay buffer with {args.prefill_games} self-play games "
              f"from {'best.pt' if (args.resume and os.path.exists(best_path)) else 'the current network'} "
              f"(mixed 2/3/4p, {args.simulations} sims/move) before the first iteration...")
        t_prefill = time.time()
        # Distinct from any real iteration's seed (args.seed * 1_000_003 + iteration)
        # without ever going negative - the Rust seed parameter is unsigned.
        prefill_examples = generate_selfplay(
            net, device, args, args.prefill_games, seed=args.seed * 1_000_003 + 999_983
        )
        replay_buffer.extend(prefill_examples)
        replay_buffer = replay_buffer[-args.replay_buffer_games:]
        flat_prefill = sum(len(g) for g in replay_buffer)
        print(f"Prefill done in {time.time() - t_prefill:.1f}s. "
              f"Replay buffer: {len(replay_buffer)} games, {flat_prefill} examples.\n")

    # best_elo is the Elo rating of whatever is currently best.pt. Seed from
    # the existing log if it already has Elo history (e.g. resuming after
    # this feature was added); otherwise anchor best.pt at INITIAL_ELO.
    best_elo = INITIAL_ELO
    for record in reversed(read_log()):
        if "best_elo" in record:
            best_elo = record["best_elo"]
            break

    # Self-play is always generated from the LATEST trained net, never reverted
    # to best.pt. This is deliberate policy iteration (AlphaZero-style), not
    # best-player gating: reverting on every non-promoted iteration (the old
    # behaviour) meant self-play after iteration 11 was generated exclusively
    # by the frozen iteration-11 network for 15 straight iterations - fitting
    # a student to a fixed teacher instead of improving. win_rate_vs_best below
    # is kept purely as a logged benchmark / what auto_deploy ships from
    # best.pt; it no longer feeds back into which weights generate data.

    for i in range(args.iterations):
        iteration = args.start_iteration + i
        t0 = time.time()

        print(f"\n=== Iteration {iteration} ({i + 1}/{args.iterations} this run) ===")

        print(f"Generating {args.games_per_iter} self-play games "
              f"({args.simulations} MCTS sims/move, batch size {args.selfplay_batch_size}, "
              f"backend={args.selfplay_backend})...")
        games_examples = generate_selfplay(
            net, device, args, args.games_per_iter, seed=args.seed * 1_000_003 + iteration
        )
        replay_buffer.extend(games_examples)
        replay_buffer = replay_buffer[-args.replay_buffer_games:]

        flat_examples = [ex for game in replay_buffer for ex in game]
        print(f"Self-play done in {time.time() - t0:.1f}s. "
              f"Replay buffer: {len(replay_buffer)} games, {len(flat_examples)} examples.")

        dataset = ReplayDataset(flat_examples)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        t1 = time.time()
        for epoch in range(args.epochs):
            stats = train_epoch(net, loader, optimizer, device)
            print(f"  epoch {epoch + 1}/{args.epochs}: "
                  f"policy_loss={stats['policy_loss']:.4f} value_loss={stats['value_loss']:.4f}")
        print(f"Training done in {time.time() - t1:.1f}s.")

        candidate_path = os.path.join(CHECKPOINT_DIR, f"iter_{iteration}.pt")
        save_checkpoint(net, candidate_path)

        t2 = time.time()
        print(f"Evaluating candidate vs previous best - GATING metric "
              f"({args.eval_openings} paired 2p openings, up to {2 * args.eval_openings} games, "
              f"{args.opening_plies} plies @ T={args.opening_temperature} then greedy, "
              f"{args.eval_simulations} sims/move)...")
        gating_result = evaluate_vs_checkpoint_2p_paired(
            net, best_path, device, num_openings=args.eval_openings, num_simulations=args.eval_simulations,
            opening_plies=args.opening_plies, opening_temperature=args.opening_temperature,
            max_moves=args.eval_max_moves,
        )
        win_rate_vs_best = gating_result["win_rate"]
        print(f"Candidate win rate vs previous best (2p paired, gating): {win_rate_vs_best:.1%} "
              f"({gating_result['wins']}W/{gating_result['draws']}D/{gating_result['losses']}L "
              f"of {gating_result['attempted']} attempted, draws scored 0.5 each)")

        write_eval_breakdown(iteration, gating_result)

        win_rate_vs_random = evaluate_vs_random(
            net, device, num_games=args.eval_games, num_simulations=args.eval_simulations
        )
        print(f"Candidate win rate vs random baseline: {win_rate_vs_random:.1%}")
        print(f"Eval done in {time.time() - t2:.1f}s.")

        # Elo is purely a reporting metric here - it does not affect promoted
        # below, which is still decided by the win-rate threshold alone. Uses
        # the 2p-paired rate, matching the symmetric-match assumption the
        # win_rate_to_elo_diff formula is derived from - a free-for-all rate
        # would give a meaningless Elo gap (see the removed
        # evaluate_vs_checkpoint's own docstring: at exact parity a
        # 2/3/4-blended rate's baseline is 1/num_players, not 50%).
        candidate_elo = best_elo + win_rate_to_elo_diff(win_rate_vs_best)
        print(f"Candidate Elo estimate: {candidate_elo:.0f} (best.pt is {best_elo:.0f})")

        promoted = win_rate_vs_best > 0.55
        if promoted:
            print("Candidate is stronger (2p paired gating) -> promoting to best.pt "
                  "(benchmark/deploy checkpoint only - self-play already continues from "
                  "the latest net either way)")
            save_checkpoint(net, best_path)
            best_elo = candidate_elo
        else:
            print("Candidate did not beat previous best by 55% margin on the 2p paired gating "
                  "metric (logged only - self-play continues from this net next iteration, no revert).")

        iter_time = time.time() - t0
        print(f"Iteration total time: {iter_time:.1f}s")

        record = {
            "iteration": iteration,
            "timestamp": time.time(),
            "games": args.games_per_iter,
            "examples_in_buffer": len(flat_examples),
            "policy_loss": stats["policy_loss"],
            "value_loss": stats["value_loss"],
            # Explicit, unambiguous marker for export_weights.py's
            # is_measured_on_fixed_harness() - added instead of relying on
            # incidental field presence (the old check inferred this from
            # win_rate_vs_best_multiplayer existing, which broke once that
            # field stopped being written here; see this commit).
            "gating_harness": "2p_paired_v1",
            "win_rate_vs_best": win_rate_vs_best,
            "win_rate_vs_best_wins": gating_result["wins"],
            "win_rate_vs_best_draws": gating_result["draws"],
            "win_rate_vs_best_losses": gating_result["losses"],
            "win_rate_vs_best_attempted": gating_result["attempted"],
            "win_rate_vs_random": win_rate_vs_random,
            "promoted": promoted,
            "iter_time_sec": iter_time,
            "elo": candidate_elo,
            "best_elo": best_elo,
        }
        append_log(record)

        history = read_log()
        warning = check_stagnation(history)
        if warning:
            print(f"\n*** {warning} ***\n")


if __name__ == "__main__":
    main()
