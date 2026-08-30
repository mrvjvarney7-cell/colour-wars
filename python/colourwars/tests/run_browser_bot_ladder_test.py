"""Covers T8 (bot ladder): promoted checkpoints show as named/avatared
opponents, every one immediately selectable (the progressive-unlock gate
that used to sit here was removed on request) - and the actual feature that
removal enables: two different iterations picked for the two AI seats
actually play each other unattended, start to finish, from a single Start
Game click.

Run with: python -m colourwars.tests.run_browser_bot_ladder_test
"""

from __future__ import annotations

import json
import os
import re
import subprocess

from colourwars.tests.run_browser_e2e import find_edge

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
TEST_PAGE = os.path.join(REPO_ROOT, "browser_bot_ladder_test.html")


def main():
    edge = find_edge()
    file_url = "file:///" + os.path.abspath(TEST_PAGE).replace("\\", "/")

    result = subprocess.run(
        [edge, "--headless=new", "--disable-gpu", "--no-sandbox", "--allow-file-access-from-files",
         "--virtual-time-budget=180000", "--dump-dom", file_url],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=240,
    )

    match = re.search(r"<title>RESULT:(.*?)</title>", result.stdout, re.DOTALL)
    if not match:
        print("Could not find RESULT: in dumped DOM. Raw stdout:")
        print(result.stdout[:3000])
        return 1

    data = json.loads(match.group(1))
    print(json.dumps(data, indent=2))

    ok = data.get("ok", False)
    print(f"\n{'PASS' if ok else 'FAIL'}: bot ladder test (errors={data.get('errors')}).")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
