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
    opponent = ColourWarsNet().to(device)
    opponent.load_state_dict(torch.load(checkpoint_path, map_location=device))
    opponent.eval()

    wins = 0
    for _ in range(num_games):
        n_players = random.choice([2, 3, 4])
        if _play_game(net, opponent, device, n_players, num_simulations):
            wins += 1
    return wins / num_games
