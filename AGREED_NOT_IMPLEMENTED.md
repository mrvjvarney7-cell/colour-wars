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

- ~~**`--eval-simulations` default changed but never took effect.**~~ **DONE
  (2026-08-31).** Pinned explicitly to 100 on the iteration-41 restart
  command line (not left to the on-disk default, per the lesson of this
  exact item) - see the `harness_boundary` marker in `training_log.jsonl`
  and the restart command in this session's own transcript.

- **Round-robin sanity check (iter_36 vs iter_29, iter_36 vs iter_26).**
  Asked for early in the iterations-37-40 investigation, explicitly
  deprioritised in favour of the iter_40-vs-iter_39/iter_40-vs-iter_36
  comparisons ("the round-robin can wait"), never run. Likely superseded by
  the opening-sampler-collapse finding - re-running it now, on the OLD
  sampler, would just be measuring the same collapsed handful of scenarios
  again. Worth redoing once the sampler fix lands, not before.

- ~~**Opening-sampler fix**~~ **DONE (2026-08-31).** Validation test confirmed
  uniform-random openings hold their diversity through move 20-50 (89-93
  distinct of ~90-100, largest group never exceeding 2) where the old
  policy-sampled openings collapsed (8-26 distinct, groups up to 54). Built
  across four commits: `5d7880d` (`board_symmetry.py` D4 canonicalisation),
  `86d776e` (uniform-random generator + generation-time dedup, replacing the
  MCTS sampler), `f519881` (played-game distinctness invariant - raises
  rather than returning a win rate if distinctness at move 20 or at final
  position falls below 50%), `67019be` (logs the new metrics in `train.py`),
  `b451add` (asserts `canonical_key`/`distinct_opening_count` survive the
  `write_eval_breakdown` write). `opening_temperature` stays in
  `evaluate_vs_checkpoint_2p_paired`'s signature but is unused - kept so
  `train.py`'s call site didn't need touching. Side finding: this does NOT
  fix the stalemate/dropout problem - see the no-progress draw rule below.

- **No-progress draw rule** (a chess-fifty-move-style draw after N plies with
  no cell changing owner). Identified as the likely fix for the 100%-pinned-
  at-cap stalemate pattern. The pending data point has now come in: even with
  uniform-random openings, 34/34 (100%) of drawn games are still pinned at
  exactly the 300-move cap - the opening-sampler fix does not touch this,
  confirming it's a separate real problem, not an artifact of the collapsed
  sampler. Still explicitly NOT yet agreed to build - this result was reported
  and flagged "still open, don't build" per the 2026-08-31 approval message,
  awaiting an actual go-ahead. Pending a decision, not yet a "yes."

- ~~**Elo chain marker / reset for the opening-sampler fix.**~~ **DONE
  (2026-08-31).** One `harness_boundary` marker written to
  `training_log.jsonl` at iteration 41, naming all three simultaneous
  changes (eval sims, opening sampler, iteration-41 training changes) in a
  single record, using the same `elo_chain_reset: true` mechanism as the
  iteration-26 rebaseline - `compute_promoted_elo_chain` and
  `find_elo_chain_reset_iteration` already handle it generically.
  Deliberately does NOT set `new_best_checkpoint` (best.pt itself is
  unchanged, still `iter_36.pt` - only the measuring stick and training
  recipe change), which required fixing `derive_version_info` in
  `export_weights.py`: its `best.pt` branch previously treated any
  `elo_chain_reset` record as "this is the new best.pt", which would have
  misattributed best.pt's identity to a marker that never touched it; its
  per-checkpoint branch took the FIRST record matching an iteration number,
  which would have let this marker (written before iteration 41's real
  eval record, sharing its iteration number) shadow the real result once
  `iter_41.pt` exists. Both fixed and tested (`test_elo_chain_markers.py`).

- ~~**Iteration-41 training changes: symmetry augmentation + LR decay.**~~
  **DONE (2026-08-31).** Neither existed anywhere in the codebase when
  checked - built from scratch rather than assumed. `transform_state`/
  `transform_policy` in `board_symmetry.py` apply a D4 symmetry to a
  training example's state tensor and policy vector respectively, sharing
  the same underlying index map `canonical_key` uses (so the two transforms
  can't silently disagree about what a given symmetry means - verified
  against every cell x symmetry combination on a 7x7 board).
  `ReplayDataset(symmetry_augment=True)` applies one uniformly random
  symmetry per example per access; opt-in via `--symmetry-augment`, off by
  default. `lr_for_iteration` computes LR as a pure function of the global
  iteration number (not a stateful scheduler, which would silently reset
  to `base_lr` on every process restart - this run has restarted many
  times); wired via `--lr-decay-gamma`/`--lr-decay-every`/
  `--lr-decay-start-iteration`, gamma defaults to 1.0 (off) so it's opt-in
  too. The iteration-41 restart used gamma=0.7, every=10,
  start_iteration=41 - a 30% cut every 10 iterations from this boundary,
  a judgment call (not user-specified numbers) flagged as such when made.

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
