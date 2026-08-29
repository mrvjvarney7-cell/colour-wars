"""Mines self-play positions for the daily puzzle (T10): finds a move where
the mover's own win probability (per a real MCTS search, not just the raw
network prior) swings from under 30% to over 90% in that single move - a
"spot the only good move" puzzle.

Fully separate from the training/eval pipeline by design: reads a
checkpoint and runs self-play/search using the existing, UNMODIFIED engine
(game.py/env.py/mcts.py/network.py), and never writes to checkpoints/ or
training_log.jsonl. Games are played fully greedy (temperature 0, no root
noise) rather than with self-play's usual exploration, since a puzzle needs
a definite, reproducible "the" solution move, not one move sampled from a
distribution.

CWN encoding here is a hand-written Python mirror of js/gameLogic.js's
encodeCwn (see that file for the full format writeup) - kept local to this
script rather than added to game.py, so mining stays a strictly additive,
separate tool rather than a change to the shared engine module.

Usage:
    python -m colourwars.mine_puzzles --checkpoint checkpoints/best.pt \
        --games 20 --simulations 60 --min-before 0.30 --min-after 0.90 \
        --out ../js/puzzles.json
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from colourwars.env import ColourWarsEnv
from colourwars.game import GameState
from colourwars.mcts import run_mcts, visit_count_policy
from colourwars.network import ColourWarsNet

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
DEFAULT_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best.pt")
DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "..", "..", "js", "puzzles.json")

_CWN_LETTERS = "abcdefghijkl"


def _cwn_char_for_cell(cell) -> str | None:
    if cell.owner is None or cell.count == 0:
        return None
    return _CWN_LETTERS[cell.owner * 3 + (cell.count - 1)]


def _encode_cwn_row(row) -> str:
    out = []
    empty_run = 0
    for cell in row:
        ch = _cwn_char_for_cell(cell)
        if ch is None:
            empty_run += 1
        else:
            if empty_run > 0:
                out.append(str(empty_run))
                empty_run = 0
            out.append(ch)
    if empty_run > 0:
        out.append(str(empty_run))
    return "".join(out)


def encode_cwn(state: GameState) -> str:
    board_str = "/".join(_encode_cwn_row(row) for row in state.board)
    not_opened = "".join(str(i) for i, p in enumerate(state.players) if not p.has_moved)
    if not not_opened:
        not_opened = "-"
    return f"{board_str} {state.current_player_index} {not_opened} {state.total_moves}"


def to_algebraic(r: int, c: int, rows: int) -> str:
    return chr(97 + c) + str(rows - r)


def mine_one_game(net: ColourWarsNet, device, num_players: int, num_simulations: int,
                   max_moves: int, min_before: float, min_after: float) -> list:
    """Plays one fully-greedy game, returning every {cwn, solution, before,
    after, swing} puzzle candidate found along the way. Reuses each ply's
    own search for the NEXT ply's "after" reading (see the loop below) -
    one search per position, not two per move."""
    env = ColourWarsEnv(num_players)
    puzzles = []
    move_count = 0

    root = run_mcts(env, net, device, num_simulations=num_simulations, add_root_noise=False)
    prev_value = root.value_sum / max(root.visit_count, 1)

    while not env.done and move_count < max_moves:
        mover = env.current_player
        before = (prev_value[mover] + 1) / 2

        pi = visit_count_policy(root, env.rows * env.cols, temperature=0.0)
        action = int(np.argmax(pi))
        r, c = action // env.cols, action % env.cols
        cwn_before = encode_cwn(env.state)

        env = env.step(action)
        move_count += 1

        if env.done:
            after = 1.0 if env.winner == mover else 0.0
            next_value = None
        else:
            root = run_mcts(env, net, device, num_simulations=num_simulations, add_root_noise=False)
            next_value = root.value_sum / max(root.visit_count, 1)
            after = (next_value[mover] + 1) / 2

        if before < min_before and after > min_after:
            puzzles.append({
                "cwn": cwn_before,
                "solution": to_algebraic(r, c, env.rows),
                "before": round(before, 4),
                "after": round(after, 4),
            })

        if next_value is not None:
            prev_value = next_value

    return puzzles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--simulations", type=int, default=60)
    parser.add_argument("--max-moves", type=int, default=300)
    parser.add_argument("--min-before", type=float, default=0.30,
                         help="only keep swings where the mover's win probability was BELOW this before the move")
    parser.add_argument("--min-after", type=float, default=0.90,
                         help="only keep swings where the mover's win probability was ABOVE this after the move")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = ColourWarsNet().to(device)
    net.load_state_dict(torch.load(args.checkpoint, map_location=device))
    net.eval()

    existing = []
    if os.path.exists(args.out):
        with open(args.out) as f:
            existing = json.load(f)
    seen_cwns = {p["cwn"] for p in existing}

    found = []
    for g in range(args.games):
        n_players = random.choice([2, 3, 4])
        candidates = mine_one_game(net, device, n_players, args.simulations, args.max_moves,
                                    args.min_before, args.min_after)
        new_ones = [p for p in candidates if p["cwn"] not in seen_cwns]
        for p in new_ones:
            seen_cwns.add(p["cwn"])
        found.extend(new_ones)
        print(f"game {g + 1}/{args.games} ({n_players}p): {len(candidates)} candidate(s), "
              f"{len(new_ones)} new. Total so far: {len(existing) + len(found)}")

    all_puzzles = existing + found
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_puzzles, f, indent=2)
    print(f"\nWrote {args.out}: {len(all_puzzles)} total puzzles ({len(found)} new this run).")


if __name__ == "__main__":
    main()
