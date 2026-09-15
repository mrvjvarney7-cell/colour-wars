"""Forces the <=479px phone rules in the real end-to-end fixture.

Headless Edge on this machine clamps its CSS viewport above phone width, so
window-size alone cannot exercise the narrow breakpoint. This mirrors the
desktop-layout runner: it creates throwaway copies, widens only the phone
media-query threshold, runs the complete game fixture, then removes both.
"""

from __future__ import annotations

import json
import os
import re
import subprocess

from colourwars.tests.run_browser_e2e import find_edge

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
FIXTURE_PATH = os.path.join(REPO_ROOT, "browser_ai_e2e_test.html")
REAL_STYLE_PATH = os.path.join(REPO_ROOT, "style.css")
SCRATCH_STYLE_PATH = os.path.join(REPO_ROOT, "_scratch_phone_layout_style.css")
SCRATCH_HTML_PATH = os.path.join(REPO_ROOT, "_scratch_phone_layout_test.html")

BREAKPOINT = "@media (max-width: 479px)"
FORCED = "@media (max-width: 9999px)"


def main():
    edge = find_edge()
    with open(REAL_STYLE_PATH, encoding="utf-8") as f:
        real_style = f.read()
    swapped_count = real_style.count(BREAKPOINT)
    if swapped_count == 0:
        print(f"ERROR: '{BREAKPOINT}' not found in style.css")
        return 1
    with open(FIXTURE_PATH, encoding="utf-8") as f:
        fixture = f.read()

    scratch_style = real_style.replace(BREAKPOINT, FORCED)
    scratch_html, substitutions = re.subn(
        r"style\.css\?v=\d+", "_scratch_phone_layout_style.css", fixture
    )
    if substitutions == 0:
        print("ERROR: fixture stylesheet reference was not found")
        return 1

    try:
        with open(SCRATCH_STYLE_PATH, "w", encoding="utf-8") as f:
            f.write(scratch_style)
        with open(SCRATCH_HTML_PATH, "w", encoding="utf-8") as f:
            f.write(scratch_html)
        file_url = "file:///" + os.path.abspath(SCRATCH_HTML_PATH).replace("\\", "/")
        result = subprocess.run(
            [edge, "--headless=new", "--disable-gpu", "--no-sandbox",
             "--allow-file-access-from-files", "--virtual-time-budget=40000",
             "--dump-dom", file_url],
            capture_output=True, text=True, timeout=90, encoding="utf-8", errors="replace",
        )
    finally:
        for path in (SCRATCH_STYLE_PATH, SCRATCH_HTML_PATH):
            if os.path.exists(path):
                os.remove(path)

    match = re.search(r"<title>RESULT:(.*?)</title>", result.stdout, re.DOTALL)
    if not match:
        print("Could not find RESULT in phone-layout browser output")
        return 1
    data = json.loads(match.group(1))
    layout = data.get("layoutCheck", {})
    ok = bool(data.get("ok") and layout.get("ok"))
    print(json.dumps(data, indent=2))
    print(f"\n{'PASS' if ok else 'FAIL'}: forced phone layout "
          f"({swapped_count} breakpoint(s) swapped, scrollWidth={layout.get('scrollWidth')}).")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
