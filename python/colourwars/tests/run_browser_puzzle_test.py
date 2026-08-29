"""Covers T10 (daily puzzle): the button appears once js/puzzles.json has
content, opening it decodes the puzzle's CWN onto the board with a "find
the winning move" prompt, a wrong guess gives "try again" feedback without
mutating the board, and the actual solution move is accepted and reports
success.

js/puzzles.json is the REAL, production file mine_puzzles.py writes to -
this test backs up whatever's there (if anything), writes a known synthetic
puzzle, runs, then restores the original state (deleting the file again if
it didn't exist before). Self-contained and safe to run at any time,
independent of whether a real mining run has produced anything yet.

Run with: python -m colourwars.tests.run_browser_puzzle_test
"""

from __future__ import annotations

import json
import os
import re
import subprocess

from colourwars.tests.run_browser_e2e import find_edge

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
TEST_PAGE = os.path.join(REPO_ROOT, "browser_puzzle_test.html")
PUZZLES_PATH = os.path.join(REPO_ROOT, "js", "puzzles.json")

# Matches the position browser_puzzle_test.html's script drives: player 0
# owns two cells, (0,1)=b7 (the solution, 3 dots - explodes into (0,0)
# eliminating player 1) and (6,6)=g1 (a legal but wrong guess, 1 dot).
TEST_PUZZLE = [{"cwn": "dc5/7/7/7/7/7/6a 0 - 2", "solution": "b7", "before": 0.1, "after": 0.99}]


def main():
    edge = find_edge()
    file_url = "file:///" + os.path.abspath(TEST_PAGE).replace("\\", "/")

    had_original = os.path.exists(PUZZLES_PATH)
    original_content = None
    if had_original:
        with open(PUZZLES_PATH, encoding="utf-8") as f:
            original_content = f.read()

    try:
        with open(PUZZLES_PATH, "w", encoding="utf-8") as f:
            json.dump(TEST_PUZZLE, f)

        result = subprocess.run(
            [edge, "--headless=new", "--disable-gpu", "--no-sandbox", "--allow-file-access-from-files",
             "--virtual-time-budget=10000", "--dump-dom", file_url],
            capture_output=True, text=True, timeout=60,
        )
    finally:
        if had_original:
            with open(PUZZLES_PATH, "w", encoding="utf-8") as f:
                f.write(original_content)
        else:
            os.remove(PUZZLES_PATH)

    match = re.search(r"<title>RESULT:(.*?)</title>", result.stdout, re.DOTALL)
    if not match:
        print("Could not find RESULT: in dumped DOM. Raw stdout:")
        print(result.stdout[:3000])
        return 1

    data = json.loads(match.group(1))
    print(json.dumps(data, indent=2))

    ok = data.get("ok", False)
    print(f"\n{'PASS' if ok else 'FAIL'}: daily puzzle test (errors={data.get('errors')}).")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
