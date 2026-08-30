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


def compute_promoted_elo_chain(training_log_path: str = TRAINING_LOG_PATH, anchor: float = 1000.0) -> dict:
    """Elo estimate for every promoted iteration, anchored at the FIRST
    promotion = `anchor`, chained forward through each later promotion's own
    measured win_rate_vs_best. This is exact, not approximate: only a
    promotion ever changes best.pt, so each promoted record's
    win_rate_vs_best is literally "this iteration vs the previous promoted
    iteration" - exactly the pairwise comparison an Elo chain needs, already
    measured over 100 real games rather than estimated.

    Recomputed fresh from the full promotion history every time this is
    called, rather than trusted from whatever may be stored in past log
    records - training_log.jsonl may have entries written under an earlier,
    different anchor convention (e.g. before this function existed, or from
    a still-running process that hasn't picked up a later correction).
    """
    from colourwars.train import win_rate_to_elo_diff  # local import: keeps torch off this module's critical path until needed

    records = _read_training_log(training_log_path)

    # A rebaseline marker (elo_chain_reset=true) discards all Elo history
    # before it - see the 2026-08-29 iteration-26 rebaseline: iteration 11's
    # promotion was measured under a since-fixed, structurally-broken eval
    # (free-for-all win rate, fully deterministic games), so continuing its
    # Elo chain forward would just be extending a number with no real content.
    # If a reset marker exists, only it and later records feed the chain, and
    # it becomes the new anchor point (at its own "best_elo", not `anchor`).
    reset_index = None
    reset_elo = anchor
    for idx, r in enumerate(records):
        if r.get("elo_chain_reset"):
            reset_index = idx
            reset_elo = r.get("best_elo", anchor)
    if reset_index is not None:
        records = records[reset_index:]
        anchor = reset_elo

    promoted = [r for r in records if r.get("promoted") or r.get("elo_chain_reset")]
    chain = {}
    elo = anchor
    for i, record in enumerate(promoted):
        if i > 0:
            elo += win_rate_to_elo_diff(record["win_rate_vs_best"])
        chain[record["iteration"]] = elo
    return chain


def find_elo_chain_reset_iteration(training_log_path: str = TRAINING_LOG_PATH):
    """The iteration number of the most recent elo_chain_reset marker, or
    None if there hasn't been one. Iterations promoted BEFORE this one had
    their Elo chain discarded (see compute_promoted_elo_chain) - their old
    numbers are on a different, no-longer-trusted scale and shouldn't be
    shown next to the current chain's numbers as if directly comparable."""
    reset_iteration = None
    for r in _read_training_log(training_log_path):
        if r.get("elo_chain_reset"):
            reset_iteration = r.get("iteration")
    return reset_iteration


def is_measured_on_fixed_harness(record: dict) -> bool:
    """True if this training-log record's win rate came from the 2p-paired,
    draws-scored eval harness (see the 2026-08-29 eval-harness fixes).

    Checked via an explicit "gating_harness" marker first - train.py writes
    "gating_harness": "2p_paired_v1" on every record from that harness,
    added when the diagnostic-only mixed 2/3/4p eval was dropped from the
    training loop (its win_rate_vs_best_multiplayer field had been doing
    double duty as this detector's signal; removing the eval without adding
    an explicit marker would have silently misclassified every later
    genuinely-fixed-harness record as provisional).

    Falls back to the two field names for records written before that
    marker existed: a short window of iterations (28-31) ran against a live
    training process that had already switched to the new harness's gating
    logic before a later, purely-cosmetic edit added the detailed
    win/draw/loss breakdown fields - that edit landed on disk after the
    process had already started, so those iterations' log records never
    picked it up despite being measured correctly."""
    if record.get("gating_harness") == "2p_paired_v1":
        return True
    return "win_rate_vs_best_draws" in record or "win_rate_vs_best_multiplayer" in record


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
        # True only if the record that produced this checkpoint's numbers was
        # written by the 2p-paired, draw-scoring eval harness (identified by
        # the "win_rate_vs_best_draws" field, which only that harness writes -
        # see the 2026-08-29 eval-harness fixes). Earlier records' win rates
        # came from a harness later found to be structurally biased
        # (deterministic games, effectively ~9 distinct outcomes, discarded
        # unfinished games instead of scoring them as draws) - their Elo is
        # not comparable to a fixed-harness one and shouldn't be shown as
        # equally authoritative. None here (no record matched at all) is
        # display-equivalent to False in ui.js - both mean "don't trust this
        # Elo as-is" - the split only matters for future record-keeping.
        "measuredOnFixedHarness": None,
        # True if this checkpoint's iteration was promoted BEFORE the most
        # recent elo_chain_reset marker - its Elo (shown as 0 above, since
        # compute_promoted_elo_chain drops pre-reset records from the chain
        # entirely) is not on the same scale as post-reset iterations and
        # should read as "not comparable", not "roughly average".
        "preReset": False,
        "eloChainResetIteration": find_elo_chain_reset_iteration(training_log_path),
        "exportedAt": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    elo_chain = compute_promoted_elo_chain(training_log_path)

    match = _ITER_RE.search(checkpoint_file)
    if match:
        iteration = int(match.group(1))
        info["iteration"] = iteration
        info["preReset"] = (info["eloChainResetIteration"] is not None
                             and iteration < info["eloChainResetIteration"])
        for record in _read_training_log(training_log_path):
            if record.get("iteration") == iteration:
                info["winRateVsRandom"] = record.get("win_rate_vs_random")
                info["winRateVsBest"] = record.get("win_rate_vs_best")
                info["promoted"] = record.get("promoted")
                info["elo"] = elo_chain.get(iteration)
                info["measuredOnFixedHarness"] = is_measured_on_fixed_harness(record)
                break
    elif checkpoint_file == "best.pt":
        # best.pt is whichever candidate most recently became the reference -
        # normally the last record with promoted=True, but an elo_chain_reset
        # marker (see compute_promoted_elo_chain) explicitly overrides that
        # when best.pt was re-baselined outside the normal promotion path
        # (e.g. iteration 26, 2026-08-29 - promoted from bad eval data, not
        # a real win-rate check, so winRateVsBest/winRateVsRandom are left
        # null rather than showing a number that would mean nothing).
        for record in reversed(_read_training_log(training_log_path)):
            if record.get("promoted") or record.get("elo_chain_reset"):
                info["iteration"] = record.get("iteration")
                info["winRateVsRandom"] = record.get("win_rate_vs_random")
                info["winRateVsBest"] = record.get("win_rate_vs_best")
                info["promoted"] = bool(record.get("promoted"))
                info["elo"] = elo_chain.get(record.get("iteration"))
                info["measuredOnFixedHarness"] = is_measured_on_fixed_harness(record)
                info["preReset"] = (info["eloChainResetIteration"] is not None
                                     and record.get("iteration") < info["eloChainResetIteration"])
                if record.get("elo_chain_reset"):
                    info["rebaselined"] = True
                    info["rebaselineNote"] = record.get("note")
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
