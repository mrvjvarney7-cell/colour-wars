"""Covers the AI version picker: js/ai/versions/index.json (built by
python -m colourwars.export_all_versions) lists every PROMOTED iteration,
the setup screen's dropdown populates from it, and switching to a
non-default version actually changes which weights power the AI's real
moves - not just the displayed label.

Uses --allow-file-access-from-files, since Chromium blocks fetch() for
plain file:// pages otherwise (confirmed: without this flag, the fetch
fails with "TypeError: Failed to fetch", and the picker gracefully falls
back to just the default AI - the feature works unrestricted once actually
served over https, e.g. GitHub Pages, which is same-origin and needs no
such flag).

Run with: python -m colourwars.tests.run_browser_ai_version_select_test
"""

from __future__ import annotations

import json
import os
import re
import subprocess

from colourwars.tests.run_browser_e2e import find_edge

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
TEST_PAGE = os.path.join(REPO_ROOT, "browser_ai_version_select_test.html")


def main():
    edge = find_edge()
    file_url = "file:///" + os.path.abspath(TEST_PAGE).replace("\\", "/")

    result = subprocess.run(
        [edge, "--headless=new", "--disable-gpu", "--no-sandbox", "--allow-file-access-from-files",
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
    print(f"\n{'PASS' if ok else 'FAIL'}: AI version-select test "
          f"(errors={data.get('errors')}, ownedAfterSwitchedAiMove={data.get('ownedAfterSwitchedAiMove')}).")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
