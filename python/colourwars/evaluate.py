"""Head-to-head evaluation: pits a candidate network's MCTS against either a
uniformly-random move picker or a previously-checkpointed network's MCTS.
Each game seats exactly one "candidate" player against (n-1) opponents of the
other kind, with the candidate's seat randomized, and reports the candidate's
win rate across games (2/3/4-player games mixed).
"""

from __future__ import annotations

import random

import numpy as np
import torch

from colourwars.board_symmetry import canonical_key
from colourwars.env import ColourWarsEnv
from colourwars.mcts import run_mcts, visit_count_policy
from colourwars.network import ColourWarsNet

# How many random-opening attempts to allow per opening actually needed,
# before giving up and raising rather than silently returning fewer than
# asked for. Generous: on a 49-cell board this should be astronomically
# unlikely to matter for any reasonable opening_plies/num_openings - see
# _generate_distinct_random_openings.
MAX_OPENING_ATTEMPTS_PER_OPENING = 50

# Ply at which PLAYED games' distinctness is checked - see
# evaluate_vs_checkpoint_2p_paired's distinctness invariant. 20 is where
# the 2026-08-31 investigation's collapse was most visible (the old MCTS
# sampler: 26 distinct openings at move 8 but only 12 by move 20).
MID_GAME_DISTINCTNESS_CHECKPOINT = 20

# Floor for played-game distinctness (at MID_GAME_DISTINCTNESS_CHECKPOINT
# and at final position) below which evaluate_vs_checkpoint_2p_paired
# refuses to report a win rate. Deliberately conservative, not tuned to a
# hair's width: the 2026-08-31 validation measured a healthy uniform-random
# run at ~99% distinct by move 20, and the broken MCTS-sampled run this
# replaces at 12% distinct. 50% sits far below the former and far above the
# latter, so it catches a real collapse immediately without needing
# precision, and won't fire on ordinary sampling variance in a healthy run.
MIN_DISTINCTNESS_RATIO = 0.5


class EvalDistinctnessError(RuntimeError):
    """Raised by evaluate_vs_checkpoint_2p_paired when played-game
    distinctness collapses below MIN_DISTINCTNESS_RATIO - see
    _distinctness_failure_reason for the check itself.

    Carries the full partial result as `.result` - everything that would
    have been returned on success (wins/draws/losses/attempted, the
    distinctness counts, and the real per-opening/per-game breakdown with
    each game's canonical_key), computed from games that were actually
    played before the check ran. A caller (train.py) can still persist this
    via write_eval_breakdown for audit even though no win rate is reported -
    refusing to report a number is only useful if someone can still see WHY,
    and the per-game breakdown is the only place that's visible. `.result`
    does include a "win_rate" key (computed the same way as the healthy
    path) - it's real arithmetic, just not a trustworthy measurement of
    anything, kept for audit ("what would the naive number even have
    shown") the same way this whole investigation started from noticing
    that number looked wrong."""

    def __init__(self, message: str, result: dict):
        super().__init__(message)
        self.result = result


def _check_distinctness(mid20_keys: list, final_keys: list, attempted: int) -> tuple:
    """Pure computation, no raising - so it's testable without a GPU/network/
    real games. Returns (distinct_at_mid20, games_reaching_mid20,
    distinct_final). See _distinctness_failure_reason for the actual
    pass/fail check against MIN_DISTINCTNESS_RATIO."""
    games_reaching_mid20 = len(mid20_keys)
    distinct_at_mid20 = len(set(mid20_keys))
    distinct_final = len(set(final_keys))
    return distinct_at_mid20, games_reaching_mid20, distinct_final


def _distinctness_failure_reason(
    distinct_at_mid20: int, games_reaching_mid20: int, distinct_final: int, attempted: int
) -> str | None:
    """None if both distinctness ratios clear MIN_DISTINCTNESS_RATIO, else a
    human-readable reason - self-contained (measured counts, ratio computed
    inline, threshold named) but deliberately does NOT know the iteration
    number or a breakdown file path, since this function has no access to
    either; the caller (evaluate_vs_checkpoint_2p_paired, then train.py)
    adds those, since only train.py actually has that context."""
    if games_reaching_mid20 > 0 and distinct_at_mid20 / games_reaching_mid20 < MIN_DISTINCTNESS_RATIO:
        return (
            f"only {distinct_at_mid20} of {games_reaching_mid20} played games "
            f"({distinct_at_mid20 / games_reaching_mid20:.1%}) are distinct (D4-canonicalised) "
            f"at move {MID_GAME_DISTINCTNESS_CHECKPOINT}, below the {MIN_DISTINCTNESS_RATIO:.0%} floor"
        )
    if attempted > 0 and distinct_final / attempted < MIN_DISTINCTNESS_RATIO:
        return (
            f"only {distinct_final} of {attempted} played games "
            f"({distinct_final / attempted:.1%}) are distinct (D4-canonicalised) at their final "
            f"position, below the {MIN_DISTINCTNESS_RATIO:.0%} floor"
        )
    return None


def _random_move(env: ColourWarsEnv) -> int:
    legal = env.legal_moves()
    return int(random.choice(legal))


def _mcts_move(env: ColourWarsEnv, net: ColourWarsNet, device, num_simulations: int) -> int:
    root = run_mcts(env, net, device, num_simulations=num_simulations, add_root_noise=False)
    pi = visit_count_policy(root, env.rows * env.cols, temperature=0.0)
    return int(np.argmax(pi))


def _play_game(candidate_net, opponent_net_or_none, device, num_players, num_simulations, max_moves=300):
    env = ColourWarsEnv(num_players)
    candidate_seat = random.randrange(num_players)

    move_count = 0
    while not env.done and move_count < max_moves:
        if env.current_player == candidate_seat:
            action = _mcts_move(env, candidate_net, device, num_simulations)
        elif opponent_net_or_none is None:
            action = _random_move(env)
        else:
            action = _mcts_move(env, opponent_net_or_none, device, num_simulations)
        env = env.step(action)
        move_count += 1

    return env.winner == candidate_seat if env.done else False


def evaluate_vs_random(net: ColourWarsNet, device, num_games: int = 20, num_simulations: int = 50) -> float:
    wins = 0
    for _ in range(num_games):
        n_players = random.choice([2, 3, 4])
        if _play_game(net, None, device, n_players, num_simulations):
            wins += 1
    return wins / num_games


def evaluate_vs_checkpoint(
    net: ColourWarsNet, checkpoint_path: str, device, num_games: int = 20, num_simulations: int = 50
) -> float:
    """Mixed 2/3/4-player win rate vs a checkpoint. Diagnostic only - NOT used
    for promotion (see evaluate_vs_checkpoint_2p_paired). At exact parity a
    free-for-all's expected win rate is 1/num_players, not 50%, so a single
    blended number here is meaningless as a gating threshold; it's kept only
    as a logged sanity-check metric."""
    opponent = ColourWarsNet().to(device)
    opponent.load_state_dict(torch.load(checkpoint_path, map_location=device))
    opponent.eval()

    wins = 0
    for _ in range(num_games):
        n_players = random.choice([2, 3, 4])
        if _play_game(net, opponent, device, n_players, num_simulations):
            wins += 1
    return wins / num_games


def _generate_random_opening(opening_plies: int) -> tuple:
    """Plays `opening_plies` moves of a fresh 2-player game using uniform-
    random legal moves - no network, no MCTS, no policy at all. Replaces
    the old MCTS-sampled-at-temperature approach: even at
    opening_temperature=0.5, a strong reference network's own policy was
    peaked enough that 100 nominally different openings collapsed to a
    handful of positions by move 20 (see the 2026-08-31 eval-harness
    investigation - 12 distinct out of 100, one group covering 54; the
    formula `visit_counts ** (1/T)` SHARPENS for T<1, so 0.5 was making
    this worse, not better). Uniform-random can't collapse the same way -
    there's no policy to be peaked - and this was validated directly before
    building it: 89 distinct out of 90 at move 20 for random openings
    continued the same way, vs. 12 out of 100 for the old sampler.

    Returns (actions, canonical_key) - the action list so the exact same
    opening can be replayed deterministically (via env.step, no re-search)
    for a seat-swapped pairing, and its D4-canonicalised position so the
    caller can dedupe without re-deriving it."""
    env = ColourWarsEnv(2)
    actions = []
    for _ in range(opening_plies):
        if env.done:
            break
        legal = env.legal_moves()
        action = int(random.choice(legal))
        env = env.step(action)
        actions.append(action)
    return actions, canonical_key(env.state.board)


def _generate_distinct_random_openings(num_openings: int, opening_plies: int) -> list:
    """Generates `num_openings` openings, deduped under D4 canonical
    position (board_symmetry.canonical_key) - "100 openings" should mean
    100 genuinely distinct starting positions, not 100 labels that might
    collapse onto a handful under symmetry. Retries a fresh random opening
    on a collision; RAISES if it can't find enough within a generous
    budget rather than silently returning fewer than asked for - on a
    49-cell board this should be astronomically unlikely for any
    reasonable opening_plies/num_openings, so hitting the budget means
    something is actually wrong (opening_plies too small for the requested
    count, or the random source isn't behaving), not routine variance.

    Returns a list of {"opening_actions": [...], "canonical_key": ...}."""
    seen_keys = set()
    result = []
    attempts = 0
    max_attempts = num_openings * MAX_OPENING_ATTEMPTS_PER_OPENING
    while len(result) < num_openings:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"Could not generate {num_openings} distinct (D4-canonicalised) random "
                f"openings after {attempts - 1} attempts - only found {len(result)}. This "
                f"should be astronomically unlikely for uniform-random openings; something "
                f"is almost certainly wrong (opening_plies too small for the requested "
                f"count, or the random source isn't actually random)."
            )
        actions, key = _generate_random_opening(opening_plies)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        result.append({"opening_actions": actions, "canonical_key": key})
    return result


def _play_paired_2p_games(candidate_net: ColourWarsNet, opponent_net: ColourWarsNet, device,
                           num_simulations: int, opening_actions: list, max_moves: int = 300) -> list:
    """Replays one pre-generated opening twice from the identical position -
    once with the candidate seated first, once seated second - then finishes
    each game greedily (temperature 0, no root noise, matching KataGo's
    gating protocol). Seat/first-player advantage, which the 3-dot opening
    rule amplifies, cancels out exactly this way instead of adding variance.

    A game hitting max_moves without a winner is scored as a DRAW (0.5 to
    each side), not dropped - engine-tournament adjudication convention.
    Discarding it instead would let the eval's effective sample size (and
    thus its denominator) silently shrink by however many games happened to
    grind out, which is exactly the mechanism that made an earlier run's
    win_rate_vs_best swing between a 26%- and 51%-game dropout rate across
    otherwise-similar matches. Returns one dict per attempted game (always
    length 2, one per seat assignment) - each also carries mid20_key (the
    D4-canonicalised board at the first move reaching
    MID_GAME_DISTINCTNESS_CHECKPOINT, or None if the game ended before
    then) and final_key (canonicalised board at whatever move the game
    actually ended, decided or capped) - see
    evaluate_vs_checkpoint_2p_paired's distinctness invariant, which is
    what these two exist for: generation-time dedup only guarantees the
    OPENINGS were distinct, not that they're still distinct once played
    out."""
    games = []
    for candidate_seat in (0, 1):
        env = ColourWarsEnv(2)
        for a in opening_actions:
            env = env.step(a)
        move_count = len(opening_actions)
        mid20_key = canonical_key(env.state.board) if move_count >= MID_GAME_DISTINCTNESS_CHECKPOINT else None
        while not env.done and move_count < max_moves:
            net = candidate_net if env.current_player == candidate_seat else opponent_net
            action = _mcts_move(env, net, device, num_simulations)
            env = env.step(action)
            move_count += 1
            if mid20_key is None and move_count >= MID_GAME_DISTINCTNESS_CHECKPOINT:
                mid20_key = canonical_key(env.state.board)
        if env.done:
            candidate_score = 1.0 if env.winner == candidate_seat else 0.0
            reason = "decided"
        else:
            candidate_score = 0.5
            reason = "max_moves_reached (scored as draw)"
        games.append({
            "candidate_seat": candidate_seat,
            "decided": env.done,
            "candidate_score": candidate_score,
            "move_count": move_count,
            "reason": reason,
            "mid20_key": mid20_key,
            "final_key": canonical_key(env.state.board),
        })
    return games


def evaluate_vs_checkpoint_2p_paired(
    net: ColourWarsNet, checkpoint_path: str, device,
    num_openings: int = 100, num_simulations: int = 50,
    opening_plies: int = 8, opening_temperature: float = 0.5, max_moves: int = 300,
) -> dict:
    """The promotion-gating metric: 2-player only (a 55% bar is only
    meaningful in a symmetric two-player match - a 3/4-player free-for-all's
    parity win rate is 1/num_players, not 50%), with `num_openings` distinct
    openings each played twice (candidate both seats) so first-player
    advantage cancels instead of adding noise. A game that hits max_moves is
    scored as a 0.5/0.5 draw rather than discarded (see _play_paired_2p_games).

    Openings are uniform-random legal moves, deduped under D4 symmetry, not
    MCTS-sampled from a reference network (see _generate_distinct_random_
    openings) - `opening_temperature` is accepted but no longer used; kept
    in the signature so existing callers (train.py) don't need updating
    just for this change.

    Distinctness invariant: generation-time dedup (see
    _generate_distinct_random_openings) only guarantees the OPENINGS were
    distinct - it says nothing about whether the PLAYED games are still
    distinct once both sides finish greedily, which is exactly what the
    old MCTS-sampled harness got wrong (openings looked fine; by move 20
    they'd collapsed to a handful of repeated scenarios anyway). So this
    checks the actual played games, at MID_GAME_DISTINCTNESS_CHECKPOINT
    (move 20) and at each game's own final position, and RAISES rather
    than returning a win rate if either falls below MIN_DISTINCTNESS_RATIO
    - a collapsed sample must produce no win rate at all, not a number
    that looks like it came from `attempted` independent trials when it
    didn't.

    Returns a dict, not a bare float:
      win_rate: total candidate_score / attempted - attempted is always
        2 * num_openings now, since nothing is ever discarded from the
        denominator.
      wins, draws, losses (by candidate_score: 1.0/0.5/0.0), attempted
      distinct_opening_count: how many of the num_openings requested were
        actually generated (always == num_openings; the generator raises
        rather than returning fewer - kept here so it's visible without
        cross-referencing the call).
      distinct_at_mid20 / games_reaching_mid20: distinctness of PLAYED
        games at move 20, among games that got that far.
      distinct_final / attempted: distinctness of every played game's own
        final position (decided or capped).
      openings: full per-opening breakdown (opening plies + both games'
        candidate_seat/decided/candidate_score/move_count/reason/
        mid20_key/final_key) for diagnosing exactly which positions/seats
        produced the result.
    """
    del opening_temperature  # unused - see docstring

    opponent = ColourWarsNet().to(device)
    opponent.load_state_dict(torch.load(checkpoint_path, map_location=device))
    opponent.eval()

    openings = _generate_distinct_random_openings(num_openings, opening_plies)

    total_score = 0.0
    wins = 0
    draws = 0
    losses = 0
    attempted = 0
    mid20_keys = []
    final_keys = []
    openings_breakdown = []
    for opening_idx, opening in enumerate(openings):
        opening_actions = opening["opening_actions"]
        games = _play_paired_2p_games(net, opponent, device, num_simulations, opening_actions, max_moves=max_moves)
        for g in games:
            attempted += 1
            total_score += g["candidate_score"]
            if g["candidate_score"] == 1.0:
                wins += 1
            elif g["candidate_score"] == 0.5:
                draws += 1
            else:
                losses += 1
            if g["mid20_key"] is not None:
                mid20_keys.append(g["mid20_key"])
            final_keys.append(g["final_key"])
        openings_breakdown.append({
            "opening_index": opening_idx,
            "opening_actions": opening_actions,
            "canonical_key": opening["canonical_key"],
            "games": games,
        })

    distinct_at_mid20, games_reaching_mid20, distinct_final = _check_distinctness(
        mid20_keys, final_keys, attempted
    )

    result = {
        "win_rate": (total_score / attempted) if attempted else 0.0,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "attempted": attempted,
        "distinct_opening_count": len(openings),
        "distinct_at_mid20": distinct_at_mid20,
        "games_reaching_mid20": games_reaching_mid20,
        "distinct_final": distinct_final,
        "openings": openings_breakdown,
    }

    # Conservative on purpose - a healthy run (validated 2026-08-31 with
    # real uniform-random openings) held 89/90 (99%) distinct at move 20;
    # the broken MCTS-sampled run this replaces held 12/100 (12%). 50%
    # leaves an enormous margin on both sides while still catching a real
    # collapse immediately, rather than needing to be tuned precisely.
    #
    # `result` is built BEFORE this check (not after, as originally written)
    # specifically so a failure can still carry it - see EvalDistinctnessError.
    # A caller that only has a bare RuntimeError with a message has no way to
    # see which openings collapsed; a caller with .result can still persist
    # and audit the full breakdown even though no win rate is trustworthy.
    failure_reason = _distinctness_failure_reason(
        distinct_at_mid20, games_reaching_mid20, distinct_final, attempted
    )
    if failure_reason is not None:
        raise EvalDistinctnessError(
            f"Eval sample has collapsed - {failure_reason}. Refusing to report a win rate "
            f"built on a collapsed sample.",
            result=result,
        )

    return result
