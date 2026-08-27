"""Interactive CLI: play Colour Wars against the trained AI.

Usage:
    python -m colourwars.play
    python -m colourwars.play --players 3 --human-seat 1 --sims 300
"""

from __future__ import annotations

import argparse
import os

import torch

from colourwars.env import ColourWarsEnv
from colourwars.mcts import run_mcts, visit_count_policy
from colourwars.network import ColourWarsNet

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
PLAYER_SYMBOLS = ["R", "G", "B", "Y"]


def render_board(env: ColourWarsEnv) -> str:
    lines = ["    " + "  ".join(f"{c:2d}" for c in range(env.cols))]
    for r in range(env.rows):
        cells = []
        for c in range(env.cols):
            cell = env.state.board[r][c]
            cells.append(" . " if cell.owner is None else f"{PLAYER_SYMBOLS[cell.owner]}{cell.count}")
        lines.append(f"{r:2d}  " + " ".join(f"{s:>2}" for s in cells))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=os.path.join(CHECKPOINT_DIR, "best.pt"))
    parser.add_argument("--players", type=int, default=2, choices=[2, 3, 4])
    parser.add_argument("--human-seat", type=int, default=0)
    parser.add_argument("--sims", type=int, default=200, help="MCTS simulations per AI move")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = ColourWarsNet().to(device)
    net.load_state_dict(torch.load(args.checkpoint, map_location=device))
    net.eval()
    print(f"Loaded {args.checkpoint} on {device}.")
    print(f"You are player {args.human_seat} ({PLAYER_SYMBOLS[args.human_seat]}). "
          f"{args.players - 1} AI opponent(s), {args.sims} MCTS sims/move.\n")

    env = ColourWarsEnv(args.players)

    while not env.done:
        print(render_board(env))
        current = env.current_player
        active = env.active_player_ids()
        print(f"\nActive players: {[PLAYER_SYMBOLS[p] for p in active]}  |  "
              f"Player {current}'s turn ({PLAYER_SYMBOLS[current]})")

        if current == args.human_seat:
            legal_rc = {(a // env.cols, a % env.cols) for a in env.legal_moves()}
            while True:
                raw = input("Your move ('row col', e.g. '3 4'): ").strip()
                parts = raw.split()
                if len(parts) != 2 or not all(p.lstrip("-").isdigit() for p in parts):
                    print("Enter two numbers separated by a space.")
                    continue
                r, c = int(parts[0]), int(parts[1])
                if (r, c) not in legal_rc:
                    print("Illegal move - cell must be empty or one you already own, and in bounds.")
                    continue
                action = r * env.cols + c
                break
        else:
            print("AI is thinking...")
            root = run_mcts(env, net, device, num_simulations=args.sims, add_root_noise=False)
            pi = visit_count_policy(root, env.rows * env.cols, temperature=0.0)
            action = int(pi.argmax())
            r, c = action // env.cols, action % env.cols
            print(f"AI (player {current}) plays ({r}, {c})")

        env = env.step(action)
        print()

    print(render_board(env))
    if env.winner is not None:
        who = "You" if env.winner == args.human_seat else f"Player {env.winner} ({PLAYER_SYMBOLS[env.winner]})"
        print(f"\nGame over! {who} win{'s' if who != 'You' else ''}!")
    else:
        print("\nGame over - no winner.")


if __name__ == "__main__":
    main()
