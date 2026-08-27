"""Actually EXECUTES the existing, unmodified js/gameLogic.test.js suite (via
a real V8 engine - py_mini_racer) and asserts it still passes, rather than
just assuming it does because gameLogic.js wasn't touched by the browser-AI
work. Node isn't available in this environment, so this is the real
verification path for "the existing tests still pass".

The test file is loaded verbatim and unmodified; only `require` (Node's
module loader, which the file uses for `require('assert')` and
`require('./gameLogic.js')`) and `process.exit` are shimmed, since MiniRacer
is a bare JS engine with neither Node's module system nor its `process`
global.

Run with: python -m colourwars.tests.run_existing_js_tests
"""

from __future__ import annotations

import os

from py_mini_racer import MiniRacer

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
GAMELOGIC_PATH = os.path.join(REPO_ROOT, "js", "gameLogic.js")
TEST_PATH = os.path.join(REPO_ROOT, "js", "gameLogic.test.js")

SHIM = """
var __assertFailures = [];
function __assertStrictEqual(actual, expected, msg) {
  if (actual !== expected) {
    throw new Error((msg || '') + ' - expected ' + JSON.stringify(expected) +
                     ' but got ' + JSON.stringify(actual));
  }
}
function __assertOk(val, msg) {
  if (!val) throw new Error(msg || 'assertion failed');
}
var __gameLogicModule = null; // set after gameLogic.js loads below
function require(name) {
  if (name === 'assert') return { strictEqual: __assertStrictEqual, ok: __assertOk };
  if (name === './gameLogic.js') return __gameLogicModule;
  throw new Error('unexpected require(' + name + ')');
}
var process = { exit: function (code) { __processExitCode = code; } };
var __processExitCode = null;
var console = { log: function () { __consoleLines.push(Array.prototype.slice.call(arguments).join(' ')); } };
var __consoleLines = [];
"""


def main():
    mr = MiniRacer()
    mr.eval(SHIM)

    with open(GAMELOGIC_PATH) as f:
        gamelogic_src = f.read()
    # gameLogic.js checks `typeof module !== 'undefined' && module.exports`
    # to decide whether to export CommonJS-style or attach to a global -
    # since our shim doesn't define `module`, it takes the global-attach
    # branch (`root.GameLogic = GameLogic`), same as it would in a browser.
    mr.eval(gamelogic_src)
    mr.eval("__gameLogicModule = GameLogic;")

    with open(TEST_PATH) as f:
        test_src = f.read()
    # The last two lines of gameLogic.test.js are `console.log(...)` and
    # `if (failed > 0) process.exit(1);` - both handled by the shim above,
    # so the file runs completely unmodified otherwise.
    mr.eval(test_src)

    console_lines = mr.eval("__consoleLines.join('\\n')")
    exit_code = mr.eval("__processExitCode")

    print(console_lines)
    ok = exit_code is None or exit_code == 0
    print(f"\n{'PASS' if ok else 'FAIL'}: js/gameLogic.test.js executed via py_mini_racer "
          f"(process.exit code: {exit_code})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
