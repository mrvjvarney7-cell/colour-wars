"""Covers Bots, Engine and Settings - the three nav items that were still
build:false placeholders after the Analysis-mode work - plus Home, the
logo's new destination and the actual default landing page. See
browser_new_screens_test.html's own header comment for how it seeds ladder
history and why Settings' "Clear local data" is only exercised on the
decline path.

Run with: python -m colourwars.tests.run_browser_new_screens_test
"""

from __future__ import annotations

import json
import os
import re
import subprocess

from colourwars.tests.run_browser_e2e import find_edge

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
TEST_PAGE = os.path.join(REPO_ROOT, "browser_new_screens_test.html")


def main():
    edge = find_edge()
    file_url = "file:///" + os.path.abspath(TEST_PAGE).replace("\\", "/")

    result = subprocess.run(
        [edge, "--headless=new", "--disable-gpu", "--no-sandbox", "--allow-file-access-from-files",
         "--virtual-time-budget=15000", "--dump-dom", file_url],
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
    print(f"\n{'PASS' if ok else 'FAIL'}: Bots/Engine/Settings/Home test (errors={data.get('errors')}).")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
