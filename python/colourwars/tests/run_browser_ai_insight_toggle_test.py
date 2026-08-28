"""Covers the "Show AI's win chance & considered moves" checkbox: unchecking
it must suppress the insight display entirely (no "AI: NN% chance" text, no
candidate-move badges) while the AI still computes and plays a legal move.

Run with: python -m colourwars.tests.run_browser_ai_insight_toggle_test
"""

from __future__ import annotations

import json
import os
import re
import subprocess

from colourwars.tests.run_browser_e2e import find_edge

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
TEST_PAGE = os.path.join(REPO_ROOT, "browser_ai_insight_toggle_test.html")


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
    print(f"\n{'PASS' if ok else 'FAIL'}: AI insight-toggle test "
          f"(errors={data.get('errors')}, insightWasShownWhileOff={data.get('insightWasShownWhileOff')}).")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
