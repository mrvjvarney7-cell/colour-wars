"""Covers the drawer/sidebar shell itself: opening via the hamburger sets
dialog semantics and moves focus in, the scrim physically blocks taps to
the board underneath it, Tab wraps at the ends of the focusable list while
open, Escape and a scrim tap both close it and return focus to the
hamburger, a leftward swipe closes it (a mostly-vertical drag does not),
disabled ("Soon") items render inert, and the mid-game confirm gate
actually blocks navigation when declined and allows it when accepted.

Run with: python -m colourwars.tests.run_browser_nav_shell_test
"""

from __future__ import annotations

import json
import os
import re
import subprocess

from colourwars.tests.run_browser_e2e import find_edge

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
TEST_PAGE = os.path.join(REPO_ROOT, "browser_nav_shell_test.html")


def main():
    edge = find_edge()
    file_url = "file:///" + os.path.abspath(TEST_PAGE).replace("\\", "/")

    result = subprocess.run(
        [edge, "--headless=new", "--disable-gpu", "--no-sandbox", "--allow-file-access-from-files",
         "--virtual-time-budget=10000", "--dump-dom", file_url],
        capture_output=True, text=True, timeout=60, encoding="utf-8", errors="replace",
    )

    match = re.search(r"<title>RESULT:(.*?)</title>", result.stdout, re.DOTALL)
    if not match:
        print("Could not find RESULT: in dumped DOM. Raw stdout:")
        print(result.stdout[:3000])
        return 1

    data = json.loads(match.group(1))
    print(json.dumps(data, indent=2))

    ok = data.get("ok", False)
    print(f"\n{'PASS' if ok else 'FAIL'}: nav shell test (errors={data.get('errors')}).")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
