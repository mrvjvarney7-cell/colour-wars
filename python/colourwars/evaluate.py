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

from colourwars.env import ColourWarsEnv
from colourwars.mcts import run_mcts, visit_count_policy
from colourwars.network import ColourWarsNet


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


def _generate_opening(net: ColourWarsNet, device, num_simulations: int,
                       opening_plies: int, opening_temperature: float) -> list:
    """Plays `opening_plies` moves of a fresh 2-player game using `net`'s own
    MCTS at `opening_temperature` (no root noise - that would degrade search
    quality itself rather than just diversify positions; temperature samples
    among genuinely-good moves instead). Uses the fixed reference net (the
    caller passes best.pt) rather than the candidate, so the distribution of
    test positions stays constant across every candidate checked against a
    given best.pt, and a candidate can't shape openings in its own favour.
    Returns the list of actions taken, so the exact same opening can be
    replayed (deterministically, via env.step - no re-search) for a
    seat-swapped pairing."""
    env = ColourWarsEnv(2)
    actions = []
    for _ in range(opening_plies):
        if env.done:
            break
        root = run_mcts(env, net, device, num_simulations=num_simulations, add_root_noise=False)
        pi = visit_count_policy(root, env.rows * env.cols, temperature=opening_temperature)
        action = int(np.random.choice(len(pi), p=pi))
        env = env.step(action)
        actions.append(action)
    return actions


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
    length 2, one per seat assignment)."""
    games = []
    for candidate_seat in (0, 1):
        env = ColourWarsEnv(2)
        for a in opening_actions:
            env = env.step(a)
        move_count = len(opening_actions)
        while not env.done and move_count < max_moves:
            net = candidate_net if env.current_player == candidate_seat else opponent_net
            action = _mcts_move(env, net, device, num_simulations)
            env = env.step(action)
            move_count += 1
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

    Returns a dict, not a bare float:
      win_rate: total candidate_score / attempted - attempted is always
        2 * num_openings now, since nothing is ever discarded from the
        denominator.
      wins, draws, losses (by candidate_score: 1.0/0.5/0.0), attempted
      openings: full per-opening breakdown (opening plies + both games'
        candidate_seat/decided/candidate_score/move_count/reason) for
        diagnosing exactly which positions/seats produced the result.
    """
    opponent = ColourWarsNet().to(device)
    opponent.load_state_dict(torch.load(checkpoint_path, map_location=device))
    opponent.eval()

    total_score = 0.0
    wins = 0
    draws = 0
    losses = 0
    attempted = 0
    openings_breakdown = []
    for opening_idx in range(num_openings):
        opening_actions = _generate_opening(opponent, device, num_simulations, opening_plies, opening_temperature)
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
        openings_breakdown.append({
            "opening_index": opening_idx,
            "opening_actions": opening_actions,
            "games": games,
        })

    return {
        "win_rate": (total_score / attempted) if attempted else 0.0,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "attempted": attempted,
        "openings": openings_breakdown,
    }
