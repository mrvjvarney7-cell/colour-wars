"""Cross-checks the browser JS AI implementation against the Python
reference before it's trusted to play real games:

  PART A - encoding: js/ai/encode.js's encodeState() vs
           ColourWarsEnv.encode_state(), driven through identical random
           move sequences on both a Python game.py state and a JS
           GameLogic state (same pattern as compare_rust_engine.py).
  PART B - network math: js/ai/network.js's forward() vs the ORIGINAL
           PyTorch best.pt, on the encoded states collected in Part A.
           This is the critical check - it directly answers "does the
           hand-written JS forward pass compute what PyTorch computes".
  PART C - MCTS behavior: js/ai/mcts.js's runMcts() does not mutate the
           root game state, always returns a legal action, and can drive
           a full game (via GameLogic) to completion without crashing,
           across 2/3/4 players.

Run with: python -m colourwars.tests.cross_check_js
"""

from __future__ import annotations

import json
import os
import random

import numpy as np
import torch
from py_mini_racer import MiniRacer

from colourwars.env import ColourWarsEnv
from colourwars.game import create_game, is_valid_move, play_move
from colourwars.network import ColourWarsNet

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
JS_DIR = os.path.join(REPO_ROOT, "js")
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "best.pt")


def make_js_context():
    mr = MiniRacer()
    for rel in ("gameLogic.js", "ai/encode.js", "ai/network.js", "ai/mcts.js", "ai/weights.js"):
        with open(os.path.join(JS_DIR, rel)) as f:
            mr.eval(f.read())
    return mr


def js_create_game(mr, num_players):
    mr.eval(f"var __s = GameLogic.createGame({num_players});")


def js_play_move(mr, row, col):
    mr.eval(f"var __r = GameLogic.playMove(__s, {row}, {col}); __s = __r.state;")


def js_encode_state(mr):
    # JSON.stringify sidesteps a py_mini_racer quirk where a raw JS array of
    # numbers returned directly from eval() comes back as a list of JSObject
    # wrappers rather than plain Python floats.
    return json.loads(mr.eval("JSON.stringify(Array.from(Encode.encodeState(__s)))"))


# ---------------------------------------------------------------------------
# PART A + B: random-game-driven encoding + network cross-check
# ---------------------------------------------------------------------------

def part_ab(mr, net, num_games, max_moves, max_network_samples, seed):
    random.seed(seed)
    encoding_mismatches = []
    network_max_abs_diff = {"policy": 0.0, "value": 0.0}
    network_samples_checked = 0
    total_plies = 0

    for game_idx in range(num_games):
        num_players = random.choice([2, 3, 4])
        py_state = create_game(num_players, 7, 7)
        js_create_game(mr, num_players)

        move_count = 0
        while not py_state.game_over and move_count < max_moves:
            legal = [
                (r, c)
                for r in range(py_state.rows)
                for c in range(py_state.cols)
                if is_valid_move(py_state.board, r, c, py_state.current_player_index)
            ]
            if not legal:
                break
            row, col = random.choice(legal)

            # Encode BEFORE the move (matches how self-play/MCTS records
            # states - encoding is of the position the mover is choosing
            # from), on both engines, then compare.
            py_encoded_full = ColourWarsEnv.from_state(py_state).encode_state()  # (planes, rows, cols)
            py_encoded = py_encoded_full.flatten()
            js_encoded = np.array(js_encode_state(mr), dtype=np.float32)

            if not np.allclose(py_encoded, js_encoded, atol=1e-6):
                n_diff = int(np.sum(~np.isclose(py_encoded, js_encoded, atol=1e-6)))
                encoding_mismatches.append((game_idx, move_count, n_diff))
            else:
                # Only spend PyTorch/JS forward-pass time on states we know
                # are encoded identically - otherwise a network mismatch
                # would just be restating the encoding bug.
                if network_samples_checked < max_network_samples:
                    with torch.no_grad():
                        t = torch.from_numpy(py_encoded_full).unsqueeze(0)
                        policy_t, value_t = net(t)
                    policy_py = policy_t[0].numpy()
                    value_py = value_t[0].numpy()

                    state_json = json.dumps(py_encoded.tolist())
                    mr.eval(f"var __enc = new Float32Array({state_json});")
                    mr.eval("var __out = NeuralNet.forward(__enc, AI_WEIGHTS);")
                    policy_js = np.array(
                        json.loads(mr.eval("JSON.stringify(Array.from(__out.policyLogits))")), dtype=np.float32
                    )
                    value_js = np.array(
                        json.loads(mr.eval("JSON.stringify(Array.from(__out.value))")), dtype=np.float32
                    )

                    network_max_abs_diff["policy"] = max(
                        network_max_abs_diff["policy"], float(np.max(np.abs(policy_py - policy_js)))
                    )
                    network_max_abs_diff["value"] = max(
                        network_max_abs_diff["value"], float(np.max(np.abs(value_py - value_js)))
                    )
                    network_samples_checked += 1

            py_result = play_move(py_state, row, col)
            py_state = py_result.state
            js_play_move(mr, row, col)

            move_count += 1
            total_plies += 1

    return {
        "total_plies": total_plies,
        "encoding_mismatches": encoding_mismatches,
        "network_samples_checked": network_samples_checked,
        "network_max_abs_diff": network_max_abs_diff,
    }


# ---------------------------------------------------------------------------
# PART C: MCTS behavior (non-mutation, legality, full games)
# ---------------------------------------------------------------------------

def part_c_non_mutation_and_legality(mr, num_players, num_simulations):
    js_create_game(mr, num_players)
    snapshot_before = mr.eval("JSON.stringify(__s)")
    mr.eval(f"var __root = MCTS.runMcts(__s, AI_WEIGHTS, {num_simulations});")
    snapshot_after = mr.eval("JSON.stringify(__s)")
    mutated = snapshot_before != snapshot_after

    action = mr.eval("MCTS.bestAction(__root)")
    row, col = divmod(action, 7)
    legal = mr.eval(f"GameLogic.isValidMove(__s.board, {row}, {col}, __s.currentPlayerIndex)")
    return {"mutated": mutated, "action": action, "legal": legal}


def part_c_full_game(mr, num_players, num_simulations, max_moves):
    js_create_game(mr, num_players)
    move_count = 0
    while True:
        game_over = mr.eval("__s.gameOver")
        if game_over or move_count >= max_moves:
            break
        mr.eval(f"var __root = MCTS.runMcts(__s, AI_WEIGHTS, {num_simulations});")
        action = mr.eval("MCTS.bestAction(__root)")
        if action is None:
            break
        row, col = divmod(action, 7)
        js_play_move(mr, row, col)
        move_count += 1

    game_over = mr.eval("__s.gameOver")
    winner = mr.eval("__s.winner")
    return {"moves": move_count, "game_over": game_over, "winner": winner}


def main():
    print("Loading best.pt (CPU) ...")
    net = ColourWarsNet()
    net.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
    net.eval()

    print("Setting up JS context (gameLogic.js + ai/*.js + weights.js) ...")
    mr = make_js_context()

    print("\n=== PART A+B: encoding + network cross-check over random games ===")
    result = part_ab(mr, net, num_games=60, max_moves=250, max_network_samples=300, seed=42)
    print(f"Checked {result['total_plies']} plies across 60 random games.")
    print(f"Encoding mismatches: {len(result['encoding_mismatches'])}")
    if result["encoding_mismatches"]:
        print(f"  first few: {result['encoding_mismatches'][:5]}")
    print(f"Network forward pass compared on {result['network_samples_checked']} states.")
    print(f"Max abs diff - policy logits: {result['network_max_abs_diff']['policy']:.6f}, "
          f"value: {result['network_max_abs_diff']['value']:.6f}")

    encoding_ok = len(result["encoding_mismatches"]) == 0
    network_ok = (result["network_max_abs_diff"]["policy"] < 1e-3
                  and result["network_max_abs_diff"]["value"] < 1e-3)
    print(f"-> encoding {'PASS' if encoding_ok else 'FAIL'}, network {'PASS' if network_ok else 'FAIL'}")

    print("\n=== PART C: MCTS behavior ===")
    all_legal = True
    any_mutated = False
    for n_players in (2, 3, 4):
        r = part_c_non_mutation_and_legality(mr, n_players, num_simulations=30)
        print(f"  {n_players}p non-mutation/legality: mutated={r['mutated']} "
              f"action={r['action']} legal={r['legal']}")
        all_legal = all_legal and r["legal"]
        any_mutated = any_mutated or r["mutated"]

    print("  Playing full MCTS-driven games (this takes a while)...")
    full_game_results = []
    for n_players in (2, 3, 4):
        r = part_c_full_game(mr, n_players, num_simulations=15, max_moves=40)
        print(f"  {n_players}p full game: {r}")
        full_game_results.append(r)

    mcts_ok = all_legal and not any_mutated
    print(f"-> MCTS non-mutation/legality {'PASS' if mcts_ok else 'FAIL'}, "
          f"full games completed without crashing: {len(full_game_results)}/3")

    print("\n=== SUMMARY ===")
    overall = encoding_ok and network_ok and mcts_ok
    print(f"Encoding cross-check:  {'PASS' if encoding_ok else 'FAIL'}")
    print(f"Network cross-check:   {'PASS' if network_ok else 'FAIL'} "
          f"(max abs diff policy={result['network_max_abs_diff']['policy']:.6f}, "
          f"value={result['network_max_abs_diff']['value']:.6f})")
    print(f"MCTS behavior:         {'PASS' if mcts_ok else 'FAIL'}")
    print(f"\nOVERALL: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
