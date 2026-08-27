"""Self-play game generation: play a full game with MCTS(net) guiding move
selection, recording (state, mcts_policy, mover_id, num_players) at each ply.
After the game ends, outcomes are converted into mover-relative value targets
matching the network's output convention (see network.py's module docstring).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List

import numpy as np
import torch

from colourwars.batched_mcts import ForwardFn, make_local_forward_fn, run_batched_mcts
from colourwars.env import MAX_PLAYERS, ColourWarsEnv
from colourwars.mcts import Node, run_mcts, visit_count_policy
from colourwars.network import ColourWarsNet


@dataclass
class TrainingExample:
    state: np.ndarray  # (NUM_PLANES, R, C)
    policy: np.ndarray  # (R*C,)
    value: np.ndarray  # (MAX_PLAYERS,) relative-to-mover target


def _draw_fallback_outcome(env: ColourWarsEnv) -> np.ndarray:
    """Used when a game hits max_moves without a natural conclusion: draw (0)
    for still-active players, loss (-1) for eliminated ones."""
    abs_outcome = np.zeros(MAX_PLAYERS, dtype=np.float64)
    active_ids = set(env.active_player_ids())
    for p in env.state.players:
        abs_outcome[p.id] = 0.0 if p.id in active_ids else -1.0
    return abs_outcome


def _records_to_examples(records, abs_outcome: np.ndarray) -> List[TrainingExample]:
    examples = []
    for (state, pi, me, n) in records:
        rel_value = np.zeros(MAX_PLAYERS, dtype=np.float32)
        for k in range(n):
            rel_value[k] = abs_outcome[(k + me) % n]
        examples.append(TrainingExample(state=state.astype(np.float32), policy=pi.astype(np.float32), value=rel_value))
    return examples


def play_one_game(
    net: ColourWarsNet,
    device: torch.device,
    num_players: int,
    num_simulations: int = 100,
    temperature_moves: int = 10,
    max_moves: int = 300,
) -> List[TrainingExample]:
    env = ColourWarsEnv(num_players)
    records = []  # (state_tensor, pi, mover_id, n_players)

    move_count = 0
    while not env.done and move_count < max_moves:
        temperature = 1.0 if move_count < temperature_moves else 0.0
        root = run_mcts(env, net, device, num_simulations=num_simulations, add_root_noise=True)
        pi = visit_count_policy(root, env.rows * env.cols, temperature=1.0)

        records.append((env.encode_state(), pi, env.current_player, env.num_players))

        if temperature == 0:
            action = int(np.argmax(pi))
        else:
            action = int(np.random.choice(len(pi), p=pi))

        env = env.step(action)
        move_count += 1

    abs_outcome = env.outcome_values().astype(np.float64) if env.done else _draw_fallback_outcome(env)
    return _records_to_examples(records, abs_outcome)


def generate_selfplay_games(
    net: ColourWarsNet,
    device: torch.device,
    num_games: int,
    num_simulations: int = 100,
    player_counts=(2, 3, 4),
) -> List[TrainingExample]:
    all_examples: List[TrainingExample] = []
    for i in range(num_games):
        n_players = random.choice(player_counts)
        examples = play_one_game(net, device, n_players, num_simulations=num_simulations)
        all_examples.extend(examples)
    return all_examples


class _Slot:
    """One concurrently-running self-play game within a batched generation
    run. Reused (reset to a fresh game) as soon as its game finishes, so the
    batch stays full and GPU utilization doesn't taper off while a few slow
    games in the batch finish last."""

    __slots__ = ("env", "records", "move_count")

    def __init__(self, num_players: int):
        self.env = ColourWarsEnv(num_players)
        self.records = []  # (state, pi, mover_id, n_players)
        self.move_count = 0


def run_batched_selfplay_loop(
    forward_fn: ForwardFn,
    num_games: int,
    num_simulations: int = 100,
    batch_size: int = 32,
    player_counts=(2, 3, 4),
    temperature_moves: int = 10,
    max_moves: int = 300,
) -> List[List[TrainingExample]]:
    """The throughput-oriented self-play driver: runs up to `batch_size` games
    at once, batching every game's pending MCTS leaf into one forward_fn call
    per search round (see batched_mcts.run_batched_mcts). Finished slots are
    immediately replaced with a fresh game so the batch stays full until
    `num_games` total games have been collected.

    Deliberately backend-agnostic (forward_fn, not net/device): this exact
    function drives BOTH the single-process local-GPU path
    (generate_selfplay_games_batched below) and the multiprocess path where
    forward_fn is a round-trip to a shared inference-server process (see
    parallel_selfplay.py) - so the two can never silently diverge in game or
    search logic, only in how a leaf's evaluation is obtained.

    Returns a list of per-game example lists (rather than one flat list) so
    callers can inspect/report on individual games if useful.
    """
    batch_size = min(batch_size, num_games)
    slots: List[_Slot] = [_Slot(random.choice(player_counts)) for _ in range(batch_size)]
    active = [True] * batch_size

    completed_games: List[List[TrainingExample]] = []

    while len(completed_games) < num_games:
        live_indices = [i for i in range(batch_size) if active[i]]
        if not live_indices:
            break  # shouldn't happen: we always refill while target not yet met

        roots = [Node(slots[i].env, prior=1.0) for i in live_indices]
        run_batched_mcts(roots, forward_fn, num_simulations=num_simulations, add_root_noise=True)

        for root, i in zip(roots, live_indices):
            slot = slots[i]
            env = slot.env
            pi = visit_count_policy(root, env.rows * env.cols, temperature=1.0)
            slot.records.append((env.encode_state(), pi, env.current_player, env.num_players))

            use_greedy = slot.move_count >= temperature_moves
            if use_greedy or pi.sum() <= 0:
                legal = env.legal_moves()
                action = int(np.argmax(pi)) if pi.sum() > 0 else int(random.choice(legal))
            else:
                action = int(np.random.choice(len(pi), p=pi))

            slot.env = env.step(action)
            slot.move_count += 1

            finished = slot.env.done or slot.move_count >= max_moves
            if finished:
                abs_outcome = (
                    slot.env.outcome_values().astype(np.float64)
                    if slot.env.done
                    else _draw_fallback_outcome(slot.env)
                )
                completed_games.append(_records_to_examples(slot.records, abs_outcome))

                if len(completed_games) < num_games:
                    slots[i] = _Slot(random.choice(player_counts))
                else:
                    active[i] = False

    return completed_games[:num_games]


def generate_selfplay_games_batched(
    net: ColourWarsNet,
    device: torch.device,
    num_games: int,
    num_simulations: int = 100,
    batch_size: int = 32,
    player_counts=(2, 3, 4),
    temperature_moves: int = 10,
    max_moves: int = 300,
) -> List[List[TrainingExample]]:
    """Single-process batched self-play: local in-process GPU forward_fn.
    See run_batched_selfplay_loop for the actual driver."""
    forward_fn = make_local_forward_fn(net, device)
    return run_batched_selfplay_loop(
        forward_fn,
        num_games,
        num_simulations=num_simulations,
        batch_size=batch_size,
        player_counts=player_counts,
        temperature_moves=temperature_moves,
        max_moves=max_moves,
    )
