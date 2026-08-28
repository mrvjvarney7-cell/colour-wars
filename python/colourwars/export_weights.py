"""Exports a trained checkpoint (e.g. best.pt) into a browser-loadable JS
file: js/ai/weights.js, defining `window.AI_WEIGHTS = {...}`.

Plain <script>-tag-loadable (no fetch(), no build step, no CORS concerns
when opened directly from disk or served statically) - matches how every
other file on this site is already loaded.

Usage:
    python -m colourwars.export_weights [--checkpoint checkpoints/best.pt] [--out ../js/ai/weights.js]
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re

import torch

from colourwars.env import MAX_PLAYERS, NUM_PLANES
from colourwars.game import COLS, ROWS
from colourwars.network import ColourWarsNet

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
TRAINING_LOG_PATH = os.path.join(CHECKPOINT_DIR, "training_log.jsonl")
DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "..", "..", "js", "ai", "weights.js")

_ITER_RE = re.compile(r"iter_(\d+)\.pt$")


def _read_training_log(path: str) -> list:
    if not os.path.exists(path):
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def derive_version_info(checkpoint_path: str, training_log_path: str = TRAINING_LOG_PATH) -> dict:
    """Best-effort metadata about which training iteration a checkpoint is
    from, so the browser can display it. Falls back to just the filename
    if the iteration number or its log record can't be found."""
    checkpoint_file = os.path.basename(checkpoint_path)
    info = {
        "checkpointFile": checkpoint_file,
        "iteration": None,
        "winRateVsRandom": None,
        "winRateVsBest": None,
        "promoted": None,
        "elo": None,
        "exportedAt": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    match = _ITER_RE.search(checkpoint_file)
    if match:
        iteration = int(match.group(1))
        info["iteration"] = iteration
        for record in _read_training_log(training_log_path):
            if record.get("iteration") == iteration:
                info["winRateVsRandom"] = record.get("win_rate_vs_random")
                info["winRateVsBest"] = record.get("win_rate_vs_best")
                info["promoted"] = record.get("promoted")
                info["elo"] = record.get("elo")
                break
    elif checkpoint_file == "best.pt":
        # best.pt is whichever candidate most recently beat the previous
        # best - that's the last record with promoted=True, if any.
        for record in reversed(_read_training_log(training_log_path)):
            if record.get("promoted"):
                info["iteration"] = record.get("iteration")
                info["winRateVsRandom"] = record.get("win_rate_vs_random")
                info["winRateVsBest"] = record.get("win_rate_vs_best")
                info["promoted"] = True
                info["elo"] = record.get("elo")
                break

    return info


# float32 only carries ~7 significant decimal digits; json.dump's default
# repr() emits up to 17. Rounding here shrinks weights.js by >2x with zero
# meaningful precision loss (well below the exported-vs-original numerical
# tolerance checked in cross_check_js_network.py).
_ROUND_DIGITS = 9


def _round(obj):
    if isinstance(obj, float):
        return round(obj, _ROUND_DIGITS)
    if isinstance(obj, list):
        return [_round(x) for x in obj]
    return obj


def _bn_dict(bn: torch.nn.BatchNorm2d) -> dict:
    return {
        "weight": _round(bn.weight.detach().cpu().tolist()),
        "bias": _round(bn.bias.detach().cpu().tolist()),
        "mean": _round(bn.running_mean.detach().cpu().tolist()),
        "var": _round(bn.running_var.detach().cpu().tolist()),
        "eps": bn.eps,
    }


def export_weights(net: ColourWarsNet) -> dict:
    net.eval()
    data = {
        "rows": ROWS,
        "cols": COLS,
        "numPlanes": NUM_PLANES,
        "maxPlayers": MAX_PLAYERS,
        "channels": net.stem[0].out_channels,
        "numResBlocks": len(net.res_blocks),
        "stem": {
            "convW": _round(net.stem[0].weight.detach().cpu().tolist()),
            "bn": _bn_dict(net.stem[1]),
        },
        "resBlocks": [],
        "policy": {
            "convW": _round(net.policy_conv.weight.detach().cpu().tolist()),
            "bn": _bn_dict(net.policy_bn),
            "fcW": _round(net.policy_fc.weight.detach().cpu().tolist()),
            "fcB": _round(net.policy_fc.bias.detach().cpu().tolist()),
        },
        "value": {
            "convW": _round(net.value_conv.weight.detach().cpu().tolist()),
            "bn": _bn_dict(net.value_bn),
            "fc1W": _round(net.value_fc1.weight.detach().cpu().tolist()),
            "fc1B": _round(net.value_fc1.bias.detach().cpu().tolist()),
            "fc2W": _round(net.value_fc2.weight.detach().cpu().tolist()),
            "fc2B": _round(net.value_fc2.bias.detach().cpu().tolist()),
        },
    }
    for block in net.res_blocks:
        data["resBlocks"].append({
            "conv1W": _round(block.conv1.weight.detach().cpu().tolist()),
            "bn1": _bn_dict(block.bn1),
            "conv2W": _round(block.conv2.weight.detach().cpu().tolist()),
            "bn2": _bn_dict(block.bn2),
        })
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=os.path.join(CHECKPOINT_DIR, "best.pt"))
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    net = ColourWarsNet()
    net.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))

    data = export_weights(net)
    version_info = derive_version_info(args.checkpoint)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("// Auto-generated by python -m colourwars.export_weights - do not hand-edit.\n")
        f.write(f"// Source checkpoint: {os.path.basename(args.checkpoint)}\n")
        # A plain top-level `var` becomes a global in both classic browser
        # scripts (window.AI_WEIGHTS) and non-browser JS engines like
        # MiniRacer/Node (used for the cross-check tests) - unlike an
        # explicit `window.AI_WEIGHTS =`, which only works where `window`
        # actually exists.
        f.write("var AI_WEIGHTS = ")
        json.dump(data, f)
        f.write(";\n")
        f.write("var AI_VERSION = ")
        json.dump(version_info, f)
        f.write(";\n")

    n_params = sum(1 for _ in _flatten(data))
    size_kb = os.path.getsize(args.out) / 1024
    print(f"Wrote {args.out} ({size_kb:.0f} KB, {n_params} scalar weights) from {args.checkpoint}")
    print(f"Version info: {version_info}")


def _flatten(obj):
    if isinstance(obj, (int, float)):
        yield obj
    elif isinstance(obj, list):
        for x in obj:
            yield from _flatten(x)
    elif isinstance(obj, dict):
        for x in obj.values():
            yield from _flatten(x)


if __name__ == "__main__":
    main()
