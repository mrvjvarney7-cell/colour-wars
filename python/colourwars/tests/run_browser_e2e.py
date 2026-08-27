"""Drives the real, unmodified game page end-to-end in a real browser
(headless Microsoft Edge) via browser_ai_e2e_test.html - a copy of
index.html with one automation script appended at the very end. Verifies:
the page loads with no JS errors, the Human-vs-AI mode toggle works, a
human move applies, and the AI then automatically computes and plays a
legal move (with a direct wall-clock timing measurement of the AI's move
computation, since --virtual-time-budget freezes performance.now() during
synchronous execution and can't be used for that).

Run with: python -m colourwars.tests.run_browser_e2e
"""

from __future__ import annotations

import json
import os
import re
import subprocess

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
TEST_PAGE = os.path.join(REPO_ROOT, "browser_ai_e2e_test.html")

EDGE_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]


def find_edge():
    for path in EDGE_CANDIDATES:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Microsoft Edge not found in any of: {EDGE_CANDIDATES}. "
        "Install Edge, or adjust EDGE_CANDIDATES for this machine."
    )


def main():
    edge = find_edge()
    file_url = "file:///" + os.path.abspath(TEST_PAGE).replace("\\", "/")

    result = subprocess.run(
        [edge, "--headless=new", "--disable-gpu", "--no-sandbox",
         "--virtual-time-budget=40000", "--dump-dom", file_url],
        capture_output=True, text=True, timeout=90,
    )

    match = re.search(r"<title>RESULT:(.*?)</title>", result.stdout, re.DOTALL)
    if not match:
        print("Could not find RESULT: in dumped DOM. Raw stdout:")
        print(result.stdout[:3000])
        return 1

    data = json.loads(match.group(1))
    print(json.dumps(data, indent=2))

    ok = data.get("ok", False)
    print(f"\n{'PASS' if ok else 'FAIL'}: browser end-to-end test "
          f"(errors={data.get('errors')}, ownedCellCount={data.get('ownedCellCount')}). "
          f"See run_ai_move_timing.py for a real wall-clock AI-move timing measurement.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
