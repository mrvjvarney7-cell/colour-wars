"""Covers the >=860px desktop layout - the one part of this site's CSS the
rest of the suite structurally cannot reach, because headless Edge clamps
window.innerWidth to ~492px regardless of --window-size (see
run_browser_e2e.py's long comment on the same quirk). Every other browser
test in this suite therefore only ever exercises mobile-mode CSS.

This is the sed-swap technique used ad hoc (and then deleted) while
debugging the body/.app-shell flex-direction bug that shipped and stayed
undetected until a real screenshot caught it - promoted into a permanent
fixture instead of a one-off scratch script, per the 2026-08-30 bug-hunt
finding that this whole viewport band had zero regression coverage.

At run time (nothing here is committed as a duplicate stylesheet, so
style.css can't drift out of sync with a forgotten copy):
  1. Read the real style.css, swap "@media (min-width: 860px)" to
     "@media (min-width: 0px)" so headless Edge's clamped ~492px width
     already satisfies the (now unconditional) desktop breakpoint.
  2. Read browser_desktop_layout_test.html (itself a normal, regenerated
     fixture - see regen_browser_fixtures.py), and rewrite its
     "style.css?v=N" reference to point at the scratch stylesheet.
  3. Write both to throwaway files, run headless Edge against the scratch
     HTML, then delete both scratch files whether the run passed or not.

Run with: python -m colourwars.tests.run_browser_desktop_layout_test
"""

from __future__ import annotations

import json
import os
import re
import subprocess

from colourwars.tests.run_browser_e2e import find_edge

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
FIXTURE_PATH = os.path.join(REPO_ROOT, "browser_desktop_layout_test.html")
REAL_STYLE_PATH = os.path.join(REPO_ROOT, "style.css")
SCRATCH_STYLE_PATH = os.path.join(REPO_ROOT, "_scratch_desktop_layout_style.css")
SCRATCH_HTML_PATH = os.path.join(REPO_ROOT, "_scratch_desktop_layout_test.html")

BREAKPOINT = "@media (min-width: 860px)"
FORCED = "@media (min-width: 0px)"


def main():
    edge = find_edge()

    with open(REAL_STYLE_PATH, encoding="utf-8") as f:
        real_style = f.read()
    swapped_count = real_style.count(BREAKPOINT)
    if swapped_count == 0:
        print(f"ERROR: '{BREAKPOINT}' not found in style.css - it may have been "
              "reworded/restructured. This test can't force desktop mode without it.")
        return 1
    scratch_style = real_style.replace(BREAKPOINT, FORCED)

    with open(FIXTURE_PATH, encoding="utf-8") as f:
        fixture_html = f.read()
    scratch_html, n = re.subn(r'style\.css\?v=\d+', '_scratch_desktop_layout_style.css', fixture_html)
    if n == 0:
        print("ERROR: no 'style.css?vN' reference found in the fixture to redirect.")
        return 1

    try:
        with open(SCRATCH_STYLE_PATH, "w", encoding="utf-8") as f:
            f.write(scratch_style)
        with open(SCRATCH_HTML_PATH, "w", encoding="utf-8") as f:
            f.write(scratch_html)

        file_url = "file:///" + os.path.abspath(SCRATCH_HTML_PATH).replace("\\", "/")
        result = subprocess.run(
            [edge, "--headless=new", "--disable-gpu", "--no-sandbox", "--allow-file-access-from-files",
             "--virtual-time-budget=10000", "--dump-dom", file_url],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
        )
    finally:
        for p in (SCRATCH_STYLE_PATH, SCRATCH_HTML_PATH):
            if os.path.exists(p):
                os.remove(p)

    match = re.search(r"<title>RESULT:(.*?)</title>", result.stdout, re.DOTALL)
    if not match:
        print("Could not find RESULT: in dumped DOM. Raw stdout:")
        print(result.stdout[:3000])
        return 1

    data = json.loads(match.group(1))
    print(json.dumps(data, indent=2))

    ok = data.get("ok", False)
    print(f"\n{'PASS' if ok else 'FAIL'}: desktop layout test ({swapped_count} breakpoint(s) "
          f"swapped, errors={data.get('errors')}).")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
