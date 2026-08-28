"""Watches the training run for newly-PROMOTED checkpoints and automatically
deploys each one to the browser AI: exports js/ai/weights.js from best.pt
(including AI_VERSION metadata so the site can state which iteration is
live), bumps the site's cache-buster version, commits, and pushes to GitHub
- so the live site only ever shows a verified-stronger version, never a
candidate that lost its promotion eval, without a manual export/commit/push
each time.

A "new promotion" is defined by training_log.jsonl gaining a record with
promoted=true for a higher iteration than what's currently deployed - that
only happens after self-play, training AND evaluation have all finished and
the candidate actually beat the previous best, so best.pt is guaranteed to
already reflect it by the time we act.

Run with: python -m colourwars.auto_deploy
Stop with Ctrl+C, or just kill the process - it's a simple polling loop that
only acts once per newly-promoted iteration, so it's safe to interrupt at
any point and resume later (it picks up from whatever js/ai/weights.js
currently says was last deployed).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
PYTHON_DIR = os.path.join(REPO_ROOT, "python")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
TRAINING_LOG_PATH = os.path.join(CHECKPOINT_DIR, "training_log.jsonl")
WEIGHTS_JS_PATH = os.path.join(REPO_ROOT, "js", "ai", "weights.js")

# Every file that carries the site's `?v=N` cache buster / `build N` tag;
# kept in sync so the browser test fixtures never drift from index.html.
HTML_FILES_WITH_CACHE_BUSTER = [
    os.path.join(REPO_ROOT, "index.html"),
    os.path.join(REPO_ROOT, "browser_ai_e2e_test.html"),
    os.path.join(REPO_ROOT, "ai_move_timing_test.html"),
    os.path.join(REPO_ROOT, "browser_ai_seat0_test.html"),
]

POLL_SECONDS = 120


def _read_training_log() -> list:
    if not os.path.exists(TRAINING_LOG_PATH):
        return []
    records = []
    with open(TRAINING_LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _deployed_iteration() -> int | None:
    """Which iteration js/ai/weights.js currently reflects, read straight
    from its embedded AI_VERSION - the single source of truth for "what's
    live", so a restart of this script can't lose track or double-deploy."""
    if not os.path.exists(WEIGHTS_JS_PATH):
        return None
    with open(WEIGHTS_JS_PATH) as f:
        text = f.read()
    match = re.search(r'"iteration":\s*(\d+)', text)
    return int(match.group(1)) if match else None


def _current_cache_buster_version() -> int:
    with open(HTML_FILES_WITH_CACHE_BUSTER[0]) as f:
        text = f.read()
    match = re.search(r"\?v=(\d+)", text)
    if not match:
        raise RuntimeError(f"No ?v=N cache buster found in {HTML_FILES_WITH_CACHE_BUSTER[0]}")
    return int(match.group(1))


def _bump_cache_buster(old_version: int, new_version: int) -> None:
    for path in HTML_FILES_WITH_CACHE_BUSTER:
        with open(path) as f:
            text = f.read()
        text = text.replace(f"?v={old_version}", f"?v={new_version}")
        text = text.replace(f"build {old_version}", f"build {new_version}")
        with open(path, "w") as f:
            f.write(text)


def _run(cmd: list, **kwargs) -> None:
    print("+ " + " ".join(cmd))
    subprocess.run(cmd, cwd=kwargs.pop("cwd", REPO_ROOT), check=True, **kwargs)


def _latest_promoted_record(records: list) -> dict | None:
    promoted = [r for r in records if r.get("promoted")]
    if not promoted:
        return None
    return max(promoted, key=lambda r: r["iteration"])


def deploy_promotion(record: dict) -> bool:
    iteration = record["iteration"]
    # best.pt is exactly the weights this promotion just saved - using it
    # (rather than iter_N.pt) means we always ship whatever is CURRENTLY
    # best even if this watcher fell behind by more than one promotion.
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "best.pt")
    if not os.path.exists(checkpoint_path):
        print(f"best.pt not found yet ({checkpoint_path}); will retry.")
        return False

    print(f"\n=== Deploying newly-promoted iteration {iteration} ===")
    _run(
        [sys.executable, "-m", "colourwars.export_weights",
         "--checkpoint", checkpoint_path, "--out", os.path.join("..", "js", "ai", "weights.js")],
        cwd=PYTHON_DIR,
    )

    old_version = _current_cache_buster_version()
    new_version = old_version + 1
    _bump_cache_buster(old_version, new_version)

    _run(["git", "add", "js/ai/weights.js"] + [
        os.path.relpath(p, REPO_ROOT).replace(os.sep, "/") for p in HTML_FILES_WITH_CACHE_BUSTER
    ])

    win_rate = record.get("win_rate_vs_random")
    win_rate_str = f"{win_rate:.0%}" if isinstance(win_rate, (int, float)) else "n/a"
    win_rate_vs_best = record.get("win_rate_vs_best")
    win_rate_vs_best_str = f"{win_rate_vs_best:.0%}" if isinstance(win_rate_vs_best, (int, float)) else "n/a"
    message = (
        f"Auto-deploy newly-promoted AI iteration {iteration} to the website (build {new_version})\n\n"
        f"Win rate vs random: {win_rate_str}; beat the previous best {win_rate_vs_best_str} of games.\n"
        f"Deployed automatically by colourwars.auto_deploy (promotions only)."
    )
    _run(["git", "commit", "-m", message])
    _run(["git", "push", "origin", "main"])
    print(f"Deployed and pushed promoted iteration {iteration} (build {new_version}).")
    return True


def main():
    print(f"Watching {TRAINING_LOG_PATH} for newly-promoted checkpoints to auto-deploy "
          f"(polling every {POLL_SECONDS}s)...")
    while True:
        try:
            records = _read_training_log()
            latest_promoted = _latest_promoted_record(records)
            if latest_promoted:
                deployed = _deployed_iteration()
                # != rather than > : corrects the case where what's currently
                # live isn't the promoted checkpoint at all (e.g. deployed
                # under the old "every iteration" policy), not just the case
                # where a newer promotion has since landed.
                if deployed != latest_promoted["iteration"]:
                    deploy_promotion(latest_promoted)
        except Exception as exc:
            print(f"auto_deploy: error during check/deploy: {exc}", file=sys.stderr)
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
