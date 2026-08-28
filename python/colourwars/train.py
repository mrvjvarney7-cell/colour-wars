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

from colourwars.evaluate import evaluate_vs_random, evaluate_vs_checkpoint
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
                         help="raised from 20->100 default: 20-game promotion checks proved too "
                              "noisy to trust near the 55% bar (see iter 5-9 of the first real run)")
    parser.add_argument("--eval-simulations", type=int, default=20,
                         help="MCTS sims/move for evaluation games (kept independent of "
                              "--simulations since evaluation still uses the slower Python "
                              "single-game MCTS - evaluate.py wasn't part of the Rust port)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start-iteration", type=int, default=1,
                         help="iteration number to label this run's first iteration as - use when "
                              "resuming into an existing training_log.jsonl/checkpoints dir so "
                              "numbering (and the stagnation-detection window, which reads "
                              "training_log.jsonl directly) stays continuous instead of restarting "
                              "at 1 and colliding with existing iter_N.pt files")
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

    # best_elo is the Elo rating of whatever is currently best.pt. Seed from
    # the existing log if it already has Elo history (e.g. resuming after
    # this feature was added); otherwise anchor best.pt at INITIAL_ELO.
    best_elo = INITIAL_ELO
    for record in reversed(read_log()):
        if "best_elo" in record:
            best_elo = record["best_elo"]
            break

    # Whether `net`'s current weights are exactly best.pt (true to start: net was
    # just loaded from - or saved as - best_path above). Self-play must always
    # be generated by a network that's at least as strong as the best verified
    # one; without this, a candidate that LOSES its promotion eval would still
    # be kept as the base for the next iteration's self-play and training, with
    # nothing ever pulling a drifting network back toward the best-known point.
    # That's exactly what happened for 8 straight non-promoted iterations
    # (11-19 of the run this was diagnosed from) before this fix.
    net_is_best = True

    for i in range(args.iterations):
        iteration = args.start_iteration + i
        t0 = time.time()

        if not net_is_best:
            print(f"Previous candidate was not promoted - reverting to best.pt "
                  f"before this iteration (replay buffer kept: those games are still "
                  f"valid training data - each one's own MCTS visit counts/outcome are "
                  f"correct regardless of which network generated them - and self-play "
                  f"from here on is generated by the restored best network again, so "
                  f"there's no drift risk left to guard against by discarding them).")
            load_checkpoint(net, best_path, device)
            optimizer = make_optimizer()
            net_is_best = True

        print(f"\n=== Iteration {iteration} ({i + 1}/{args.iterations} this run) ===")

        print(f"Generating {args.games_per_iter} self-play games "
              f"({args.simulations} MCTS sims/move, batch size {args.selfplay_batch_size}, "
              f"backend={args.selfplay_backend})...")
        if args.selfplay_backend == "rust":
            from colourwars.rust_selfplay import generate_selfplay_games_rust
            games_examples = generate_selfplay_games_rust(
                net, device, args.games_per_iter,
                num_simulations=args.simulations, batch_size=args.selfplay_batch_size,
                seed=args.seed * 1_000_003 + iteration,
            )
        elif args.selfplay_batch_size > 1:
            games_examples = generate_selfplay_games_batched(
                net, device, args.games_per_iter,
                num_simulations=args.simulations, batch_size=args.selfplay_batch_size,
            )
        else:
            games_examples = []
            for g in range(args.games_per_iter):
                n_players = random.choice([2, 3, 4])
                examples = play_one_game(net, device, n_players, num_simulations=args.simulations)
                games_examples.append(examples)
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
        print(f"Evaluating candidate vs previous best ({args.eval_games} games, "
              f"{args.eval_simulations} sims/move)...")
        win_rate_vs_best = evaluate_vs_checkpoint(
            net, best_path, device, num_games=args.eval_games, num_simulations=args.eval_simulations
        )
        print(f"Candidate win rate vs previous best: {win_rate_vs_best:.1%}")

        win_rate_vs_random = evaluate_vs_random(
            net, device, num_games=args.eval_games, num_simulations=args.eval_simulations
        )
        print(f"Candidate win rate vs random baseline: {win_rate_vs_random:.1%}")
        print(f"Eval done in {time.time() - t2:.1f}s.")

        # Elo is purely a reporting metric here - it does not affect promoted
        # below, which is still decided by the win-rate threshold alone.
        candidate_elo = best_elo + win_rate_to_elo_diff(win_rate_vs_best)
        print(f"Candidate Elo estimate: {candidate_elo:.0f} (best.pt is {best_elo:.0f})")

        promoted = win_rate_vs_best > 0.55
        net_is_best = promoted
        if promoted:
            print("Candidate is stronger -> promoting to best.pt")
            save_checkpoint(net, best_path)
            best_elo = candidate_elo
        else:
            print("Candidate did not beat previous best by enough margin -> reverting to best.pt "
                  "before the next iteration (see the net_is_best check above).")

        iter_time = time.time() - t0
        print(f"Iteration total time: {iter_time:.1f}s")

        record = {
            "iteration": iteration,
            "timestamp": time.time(),
            "games": args.games_per_iter,
            "examples_in_buffer": len(flat_examples),
            "policy_loss": stats["policy_loss"],
            "value_loss": stats["value_loss"],
            "win_rate_vs_best": win_rate_vs_best,
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
