"""Self-play driven by the Rust game engine + MCTS (colourwars_rs), with the
PyTorch network staying in Python as designed. This mirrors
selfplay.generate_selfplay_games_batched's contract (same TrainingExample
shape) but the entire tree search / game simulation hot path runs in Rust;
Python's only job per search round is one batched network forward pass.
"""

from __future__ import annotations

from typing import List

import colourwars_rs as rs
import numpy as np
import torch

from colourwars.env import MAX_PLAYERS, NUM_PLANES
from colourwars.game import ROWS, COLS
from colourwars.network import ColourWarsNet
from colourwars.selfplay import TrainingExample


def _make_numpy_forward_fn(net: ColourWarsNet, device: torch.device):
    net.eval()

    @torch.no_grad()
    def forward_fn(states: np.ndarray):
        state_tensor = torch.from_numpy(states).to(device)
        policy_logits, values = net(state_tensor)
        return (
            policy_logits.cpu().numpy().astype(np.float32),
            values.cpu().numpy().astype(np.float32),
        )

    return forward_fn


def generate_selfplay_games_rust(
    net: ColourWarsNet,
    device: torch.device,
    num_games: int,
    num_simulations: int = 100,
    batch_size: int = 32,
    player_counts=(2, 3, 4),
    temperature_moves: int = 10,
    max_moves: int = 300,
    seed: int = 0,
) -> List[List[TrainingExample]]:
    forward_fn = _make_numpy_forward_fn(net, device)

    records = rs.run_batched_selfplay_rust(
        forward_fn,
        num_games,
        num_simulations,
        batch_size,
        list(player_counts),
        temperature_moves=temperature_moves,
        max_moves=max_moves,
        seed=seed,
    )

    games: List[List[TrainingExample]] = []
    for rec in records:
        n = rec.n_plies
        states = np.asarray(rec.states, dtype=np.float32).reshape(n, NUM_PLANES, ROWS, COLS)
        policies = np.asarray(rec.policies, dtype=np.float32).reshape(n, ROWS * COLS)
        values = np.asarray(rec.values, dtype=np.float32).reshape(n, MAX_PLAYERS)
        games.append(
            [
                TrainingExample(state=states[i], policy=policies[i], value=values[i])
                for i in range(n)
            ]
        )
    return games
