"""Real wall-clock timing of an AI move (js/ai/mcts.js + network.js) in an
actual browser engine (headless Microsoft Edge), WITHOUT --virtual-time-
budget (which freezes performance.now() during synchronous execution and
would silently report ~0ms regardless of true cost - see
run_browser_e2e.py's docstring).

Run with: python -m colourwars.tests.run_ai_move_timing
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time

from colourwars.tests.run_browser_e2e import find_edge

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
TEST_PAGE = os.path.join(REPO_ROOT, "ai_move_timing_test.html")


def main():
    edge = find_edge()
    file_url = "file:///" + os.path.abspath(TEST_PAGE).replace("\\", "/")

    t0 = time.time()
    result = subprocess.run(
        [edge, "--headless=new", "--disable-gpu", "--no-sandbox", "--dump-dom", file_url],
        capture_output=True, text=True, timeout=120,
    )
    wall_clock = time.time() - t0

    match = re.search(r"<title>TIMING:(.*?)</title>", result.stdout, re.DOTALL)
    if not match:
        print("Could not find TIMING: in dumped DOM. Raw stdout:")
        print(result.stdout[:3000])
        return 1

    data = json.loads(match.group(1))
    print(f"In-page performance.now() measurement: {data}")
    print(f"External wall-clock for the whole headless run "
          f"(includes browser startup + {data['repeats']} AI moves): {wall_clock:.1f}s")
    print(f"\n-> ~{data['perMoveMs']:.0f}ms per AI move at 60 MCTS simulations "
          f"(the default configured in js/ui.js's AI_SIMULATIONS).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
