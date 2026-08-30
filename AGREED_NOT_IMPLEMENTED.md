# Agreed, not implemented

Things discussed and agreed in conversation with Claude Code that never
landed in code (or landed but never took effect on the live training
process). Started 2026-08-31 after the eval-harness investigation found the
same pattern at least four separate times - a recommendation gets agreed on,
the conversation moves on, and nothing distinguishes "decided against" from
"just forgotten" without a document like this one.

Update this file whenever an item here actually ships (move it out, or mark
it done with the commit) or whenever a new open item is agreed but not yet
built. Keep entries short - the surrounding commit/conversation history is
the record of *why*, this file is just the record of *what's outstanding*.

## Training pipeline

- **`--eval-max-moves` raise.** Recommended at some point before this
  document existed; never implemented. `git blame` shows exactly one commit
  (`93224eb`) ever touched that line, at `default=300`, and nothing since.
  Re-confirmed 2026-08-31 and **deliberately held at 300 for now** - draws
  are pinned at exactly the cap (100% of the sample checked), so raising it
  wouldn't change any observed outcome, and holding it constant keeps the
  before/after comparison clean once the opening-sampler fix lands. Not a
  live gap any more, but recording the history so "we decided to hold it"
  isn't later misread as "nobody ever thought about it."

- **`--eval-simulations` default changed but never took effect until the
  iteration-41 restart.** `93224eb` also changed the default from 20 to
  100, in the same commit as the harness rework. The live training process
  (running continuously since before that commit's session) kept using 20
  the entire time - confirmed directly from `train_run.log`'s own printed
  "20 sims/move" on every gate, iterations 28 through 40. The 2026-08-30
  restart (for the multiplayer-eval removal) didn't pass an explicit
  `--eval-simulations` override, so **iteration 41's gate, once training
  resumes, will silently run at 100 sims/move instead of 20** - a 5x
  fidelity jump with no marker anywhere in the log, landing at the same
  moment as the multiplayer-eval removal and the `gating_harness` tag. Not
  itself broken, just unmarked - flagging so a future "why did the numbers
  change again" investigation finds this instead of re-deriving it.

- **Round-robin sanity check (iter_36 vs iter_29, iter_36 vs iter_26).**
  Asked for early in the iterations-37-40 investigation, explicitly
  deprioritised in favour of the iter_40-vs-iter_39/iter_40-vs-iter_36
  comparisons ("the round-robin can wait"), never run. Likely superseded by
  the opening-sampler-collapse finding - re-running it now, on the OLD
  sampler, would just be measuring the same collapsed handful of scenarios
  again. Worth redoing once the sampler fix lands, not before.

- **Opening-sampler fix** (uniform-random opening plies, D4 canonicalise +
  dedupe at generation time, distinctness-of-played-games as the hard
  failure threshold, persisted `canonical_key`/`distinct_opening_count`).
  Design approved 2026-08-31, gated on a validation test (does random-opening
  diversity actually survive greedy continuation to move 20-50, the same way
  policy-sampled openings didn't). Not built yet.

- **No-progress draw rule** (a chess-fifty-move-style draw after N plies with
  no cell changing owner). Identified as the likely fix for the 100%-pinned-
  at-cap stalemate pattern, but explicitly NOT yet agreed to build - waiting
  on the random-opening draw-rate number to see whether diverse openings
  change the picture first. Pending a decision, not yet a "yes."

- **Elo chain marker / reset for the opening-sampler fix.** Once a real fix
  lands, gate results measure something structurally different (genuinely
  diverse trials, not 6-12 repeated scenarios) and the existing Elo chain
  (already reset once at iteration 26 for an analogous reason) won't be
  comparable across that boundary. Explicitly: flag when the fix lands,
  don't act unilaterally beforehand.

## Front end

- **A4: `?cwn=&mode=analysis`.** Approved design (2026-08-30): existing bare
  `?cwn=` links keep meaning exactly what they mean today (a live,
  continuable position); a new `&mode=analysis` query param opens the same
  position in Analysis mode instead; a new "Share for analysis" action
  alongside the existing Share button generates links with it. Not built -
  the bug-hunt pass moved on to A5/B1 and then the eval-harness
  investigation before reaching this.

- **A5: opening temperature for the browser AI.** Approved design
  (2026-08-30): `js/ai/mcts.js`'s `bestAction()` is deterministic (temperature
  0) by design for play, which is why two identical bots replay an identical
  game. Agreed fix: sample from the visit-count policy for the first several
  plies (same shape as self-play's `temperature_moves`), greedy after -
  explicitly NOT applied in Analysis mode or the eval bar, which must stay
  reproducible. Not built.

- **B1: console warnings on the four screen-render guard functions**
  (`renderBotsScreen`, `renderEngineScreen`, `renderSettingsThemeButtons`,
  `renderSettingsThinkTimeButtons`). Approved scope (2026-08-30): warn on
  these four specifically, skip the ~15 smaller per-button micro-guards
  elsewhere in `ui.js`. Not built.

- **Permanent regression test for the hashchange mid-game guard.** A3
  (2026-08-30) fixed the actual bug and verified it with a real, scripted
  browser check, but that check was a one-off scratch script, not a
  committed fixture - unlike every other guard behaviour in this codebase,
  there's no permanent test that would catch this specific regression
  recurring. Self-flagged in A3's own commit message, not something the user
  asked for directly, but real.
