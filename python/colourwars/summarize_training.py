"""Prints a summary of training progress from training_log.jsonl - safe to
run anytime, including while train.py is running in another process (it only
reads the log file, never touches the checkpoint/optimizer state).

Usage: python -m colourwars.summarize_training [--last N]
"""

from __future__ import annotations

import argparse
import datetime

from colourwars.train import CHECKPOINT_DIR, STAGNATION_BAND, STAGNATION_WINDOW, check_stagnation, read_log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--last", type=int, default=20, help="show only the last N iterations")
    args = parser.parse_args()

    history = read_log()
    if not history:
        print(f"No training log found yet at {CHECKPOINT_DIR}/training_log.jsonl")
        return

    print(f"{len(history)} iterations completed so far.\n")
    print(f"{'iter':>5} {'games':>6} {'buf.ex':>8} {'pol_loss':>9} {'val_loss':>9} "
          f"{'vs_best':>8} {'vs_rand':>8} {'promoted':>9} {'time(s)':>8}  when")
    for r in history[-args.last:]:
        when = datetime.datetime.fromtimestamp(r["timestamp"]).strftime("%Y-%m-%d %H:%M")
        if r.get("elo_chain_reset"):
            print(f"  -- rebaseline: best.pt -> iteration {r.get('iteration')}, "
                  f"Elo chain reset to {r.get('best_elo', 0):.0f} ({when}) --")
            continue
        print(f"{r['iteration']:>5} {r['games']:>6} {r['examples_in_buffer']:>8} "
              f"{r['policy_loss']:>9.4f} {r['value_loss']:>9.4f} "
              f"{r['win_rate_vs_best']:>7.1%} {r['win_rate_vs_random']:>7.1%} "
              f"{'YES' if r['promoted'] else 'no':>9} {r['iter_time_sec']:>8.0f}  {when}")

    n_promoted = sum(1 for r in history if r.get("promoted"))
    total_hours = sum(r.get("iter_time_sec", 0) for r in history) / 3600
    print(f"\n{n_promoted}/{len(history)} iterations promoted a new best checkpoint.")
    print(f"Total training wall-clock so far: {total_hours:.1f} hours.")

    warning = check_stagnation(history)
    if warning:
        print(f"\n*** {warning} ***")
    else:
        eval_history = [r for r in history if "win_rate_vs_best" in r]
        if len(eval_history) >= STAGNATION_WINDOW:
            recent = [r["win_rate_vs_best"] for r in eval_history[-STAGNATION_WINDOW:]]
            print(f"\nNo stagnation/divergence warning. Last {STAGNATION_WINDOW} win-rates-vs-best: "
                  f"{[f'{w:.0%}' for w in recent]} (band checked: {STAGNATION_BAND[0]:.0%}-{STAGNATION_BAND[1]:.0%})")


if __name__ == "__main__":
    main()
