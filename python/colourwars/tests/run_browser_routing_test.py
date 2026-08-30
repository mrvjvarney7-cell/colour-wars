"""Covers hash routing, added on top of showScreen(): nav-equivalent actions
(open/close History and Rules) keep location.hash in sync, a bare hash write
(simulating the browser's own back/forward, since headless Edge with
--virtual-time-budget can't drive real navigation) moves the screen the
other way, an unrecognized route leaves the current screen alone, a
bookmarked hash-only link lands directly on that screen, and - the one that
actually matters - a ?cwn= link always wins over any hash also present in
the URL.

browser_routing_test.html is one fixture shared by four separate loads (see
SCENARIOS below), each hitting a different URL shape; its driver script
picks which scenario to run from location.search/location.hash at load
time. All four must pass for this to report success.

Run with: python -m colourwars.tests.run_browser_routing_test
"""

from __future__ import annotations

import json
import os
import re
import subprocess

from colourwars.tests.run_browser_e2e import find_edge

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
TEST_PAGE = os.path.join(REPO_ROOT, "browser_routing_test.html")

# A minimal, hand-built 2p position - only used by the cwn scenarios, which
# don't care what's on the board, only that the game screen ends up showing.
TEST_CWN = "dc5/7/7/7/7/7/7 0 - 2"

SCENARIOS = [
    ("mid-session", ""),
    ("hash-only", "#/rules"),
    ("hash-bots", "#/bots"),
    ("cwn-no-hash", "?cwn=" + TEST_CWN),
    ("cwn-with-hash", "?cwn=" + TEST_CWN + "#/games"),
]


def main():
    edge = find_edge()
    base_url = "file:///" + os.path.abspath(TEST_PAGE).replace("\\", "/")

    results = {}
    overall_ok = True
    for name, suffix in SCENARIOS:
        result = subprocess.run(
            [edge, "--headless=new", "--disable-gpu", "--no-sandbox", "--allow-file-access-from-files",
             "--virtual-time-budget=10000", "--dump-dom", base_url + suffix],
            capture_output=True, text=True, timeout=60, encoding="utf-8", errors="replace",
        )
        match = re.search(r"<title>RESULT:(.*?)</title>", result.stdout, re.DOTALL)
        if not match:
            results[name] = {"ok": False, "reason": "no RESULT in dumped DOM"}
            overall_ok = False
            continue
        data = json.loads(match.group(1))
        results[name] = data
        if not data.get("ok", False):
            overall_ok = False

    print(json.dumps(results, indent=2))
    print(f"\n{'PASS' if overall_ok else 'FAIL'}: routing test ({sum(1 for r in results.values() if r.get('ok'))}/{len(SCENARIOS)} scenarios ok).")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
