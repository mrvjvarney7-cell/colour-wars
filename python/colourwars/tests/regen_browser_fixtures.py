"""Regenerates every browser_*_test.html fixture from the CURRENT
index.html, preserving only each fixture's own appended test-driver script.

Why this exists: a fixture used to be a hand-maintained copy of markup
(some full-page, some a trimmed slice) that silently fell out of sync with
index.html as features were added - three separate times, a fixture missing
an element the live page now has caused either a false pass (the trimmed
DOM never triggered a code path that would have broken) or a crash once
some other change (see showScreen()) assumed every fixture had it. Guarding
every lookup with `if (el)` kept fixing the crash but not the root cause:
the fixture was testing a DOM that doesn't exist in production.

This makes every fixture provably identical to the live page except for its
own appended script, and turns "fixture is stale" from a silent, discovered-
by-accident problem into "re-run this after any index.html change" - a
90-line index.html has a single bare `<script>` tag as the very first line
of any fixture's driver (index.html's OWN script tags all carry a src=
attribute, so a bare one only ever marks where a fixture's test code
begins) - that's the split point: everything before it, in the fixture, is
disposable; everything from it onward is the actual test.

Usage:
    python -m colourwars.tests.regen_browser_fixtures                 # all fixtures
    python -m colourwars.tests.regen_browser_fixtures foo.html bar.html # just these
"""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
INDEX_HTML_PATH = os.path.join(REPO_ROOT, "index.html")

FIXTURES = [
    "browser_ai_e2e_test.html",
    "browser_ai_insight_toggle_test.html",
    "browser_ai_seat0_test.html",
    "browser_ai_version_select_test.html",
    "browser_bot_ladder_test.html",
    "browser_chain_preview_test.html",
    "browser_cwn_share_test.html",
    "browser_eval_bar_test.html",
    "browser_game_review_test.html",
    "browser_history_test.html",
    "browser_nav_shell_test.html",
    "browser_policy_heatmap_test.html",
    "browser_puzzle_test.html",
    "browser_rules_test.html",
    "browser_routing_test.html",
    "browser_theme_toggle_test.html",
]

DRIVER_MARKER = "<script>"  # bare - never how index.html writes its own script tags


def regenerate(fixture_name: str) -> None:
    fixture_path = os.path.join(REPO_ROOT, fixture_name)
    with open(fixture_path, encoding="utf-8") as f:
        old_content = f.read()
    idx = old_content.index(DRIVER_MARKER)
    driver_and_tail = old_content[idx:]

    with open(INDEX_HTML_PATH, encoding="utf-8") as f:
        index_html = f.read()
    # index.html's own closing tags aren't kept here - driver_and_tail
    # (extracted above) already ends with the fixture's own </body></html>.
    head = index_html.rsplit("</body>", 1)[0]

    with open(fixture_path, "w", encoding="utf-8") as f:
        f.write(head + driver_and_tail)
    print(f"Regenerated {fixture_name} from the current index.html.")


def main():
    targets = sys.argv[1:] or FIXTURES
    for name in targets:
        regenerate(name)


if __name__ == "__main__":
    main()
