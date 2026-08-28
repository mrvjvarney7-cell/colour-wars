"""Regression test for a real reported bug: an AI seated at seat 0 (Player 1)
never played its opening move, leaving the game stuck (not the human's turn,
and the AI never triggered). Root cause: maybePlayAiTurn() in js/ui.js was
only ever called reactively from inside commitMove(), after a preceding
move - the very first turn of a new game has no preceding move to react to.
Fixed by also calling it once at the end of startGame().

Drives browser_ai_seat0_test.html (a copy of index.html with one automation
script appended) in a real browser (headless Microsoft Edge): toggles seat 0
to AI, starts the game, and asserts the AI's opening move actually lands and
the turn correctly passes to the human at seat 1.

Run with: python -m colourwars.tests.run_browser_ai_seat0_test
"""

from __future__ import annotations

import json
import os
import re
import subprocess

from colourwars.tests.run_browser_e2e import find_edge

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
TEST_PAGE = os.path.join(REPO_ROOT, "browser_ai_seat0_test.html")


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
    print(f"\n{'PASS' if ok else 'FAIL'}: AI-at-seat-0 regression test "
          f"(errors={data.get('errors')}, ownedAfterAiOpening={data.get('ownedAfterAiOpening')}, "
          f"turnLabel={data.get('turnLabel')!r}).")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
