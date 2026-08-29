"""Covers T7 (local game history): a game that finishes for real gets
recorded to localStorage with the right summary fields, the history
screen's stats reflect it, and clicking the row replays the same game via
T6's replay path.

Loads the page with a hand-crafted ?cwn= position one explosion away from
eliminating player 2, so a real game-over is reached with a single click
instead of playing a full game out move by move.

Run with: python -m colourwars.tests.run_browser_history_test
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import urllib.parse

from colourwars.tests.run_browser_e2e import find_edge

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
TEST_PAGE = os.path.join(REPO_ROOT, "browser_history_test.html")

# Player 0 at (0,1)=b7 with 3 dots (one placement from exploding), player 1
# with exactly one cell, (0,0)=a7, 1 dot. See browser_history_test.html for
# the full reasoning.
START_CWN = "dc5/7/7/7/7/7/7 0 - 2"


def main():
    edge = find_edge()
    cwn_encoded = urllib.parse.quote(START_CWN, safe="")
    file_url = "file:///" + os.path.abspath(TEST_PAGE).replace("\\", "/") + "?cwn=" + cwn_encoded

    result = subprocess.run(
        [edge, "--headless=new", "--disable-gpu", "--no-sandbox", "--allow-file-access-from-files",
         "--virtual-time-budget=10000", "--dump-dom", file_url],
        capture_output=True, text=True, timeout=60,
    )

    match = re.search(r"<title>RESULT:(.*?)</title>", result.stdout, re.DOTALL)
    if not match:
        print("Could not find RESULT: in dumped DOM. Raw stdout:")
        print(result.stdout[:3000])
        return 1

    data = json.loads(match.group(1))
    print(json.dumps(data, indent=2))

    ok = data.get("ok", False)
    print(f"\n{'PASS' if ok else 'FAIL'}: local history test (errors={data.get('errors')}).")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
