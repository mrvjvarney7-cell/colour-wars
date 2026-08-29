"""Exports every PROMOTED checkpoint (not just the current best.pt) as a
separate, fetch()-on-demand JSON file, plus an index.json the browser can
read to populate a version-picker dropdown - so a player can choose which
trained iteration to play against, not just always the latest.

Unlike weights.js (a `var AI_WEIGHTS = {...}` assignment, loaded eagerly via
a <script> tag so the default AI is ready with zero extra latency), these
are plain JSON: fetched only when a player actually picks that version, so
adding more promoted iterations over time doesn't grow the page's default
load weight at all.

Usage:
    python -m colourwars.export_all_versions [--out-dir ../js/ai/versions]
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from colourwars.export_weights import (
    CHECKPOINT_DIR,
    TRAINING_LOG_PATH,
    _read_training_log,
    compute_promoted_elo_chain,
    export_weights,
)
from colourwars.network import ColourWarsNet

DEFAULT_OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "js", "ai", "versions")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--training-log", default=TRAINING_LOG_PATH)
    args = parser.parse_args()

    promoted = [r for r in _read_training_log(args.training_log) if r.get("promoted")]
    if not promoted:
        print("No promoted iterations found in the training log - nothing to export.")
        return

    elo_chain = compute_promoted_elo_chain(args.training_log)
    os.makedirs(args.out_dir, exist_ok=True)

    index = []
    for record in promoted:
        iteration = record["iteration"]
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"iter_{iteration}.pt")
        if not os.path.exists(checkpoint_path):
            print(f"Skipping iteration {iteration}: {checkpoint_path} not found.")
            continue

        net = ColourWarsNet()
        net.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        data = export_weights(net)

        filename = f"iter{iteration}.json"
        out_path = os.path.join(args.out_dir, filename)
        with open(out_path, "w") as f:
            json.dump(data, f)
        size_kb = os.path.getsize(out_path) / 1024
        print(f"Wrote {out_path} ({size_kb:.0f} KB) from {checkpoint_path}")

        index.append({
            "iteration": iteration,
            "file": filename,
            "elo": round(elo_chain.get(iteration, 0.0)),
            "winRateVsRandom": record.get("win_rate_vs_random"),
            # See derive_version_info in export_weights.py for what this means
            # and why it isn't shown as an equally-trustworthy Elo when false.
            "measuredOnFixedHarness": "win_rate_vs_best_draws" in record,
        })

    index_path = os.path.join(args.out_dir, "index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"Wrote {index_path} ({len(index)} versions)")


if __name__ == "__main__":
    main()
