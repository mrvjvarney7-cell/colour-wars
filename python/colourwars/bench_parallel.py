"""Correctness smoke test + throughput benchmark for the multiprocess
self-play path (parallel_selfplay.py) vs the single-process batched path
(selfplay.generate_selfplay_games_batched).

Usage:
    python -m colourwars.bench_parallel --mode smoke
    python -m colourwars.bench_parallel --mode bench --workers 4 6 8 --games 48 --sims 100
"""

from __future__ import annotations

import argparse
import time

import torch

from colourwars.network import ColourWarsNet
from colourwars.parallel_selfplay import run_parallel_selfplay
from colourwars.selfplay import generate_selfplay_games_batched


def smoke_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = ColourWarsNet().to(device)

    print("Running 6 games (2/3/4-player mix) through the multiprocess path, 2 workers...")
    games = run_parallel_selfplay(
        net, num_games=6, num_workers=2, games_per_worker_batch_size=3,
        num_simulations=10, device_str=device.type, player_counts=(2, 3, 4), max_moves=250,
    )
    print(f"Got {len(games)} completed games.")
    for i, g in enumerate(games):
        n_active_slots = sum(1 for v in g[-1].value if v != 0)
        print(f"  game {i}: {len(g)} moves, state shape {g[0].state.shape}, "
              f"policy sum {g[0].policy.sum():.3f}, final value {g[-1].value}")
    assert len(games) == 6
    for g in games:
        assert len(g) > 0
        assert g[0].state.shape == (10, 7, 7)
        assert abs(g[0].policy.sum() - 1.0) < 1e-4
    print("Smoke test OK: correct shapes, valid policy distributions, plausible outcome vectors.")


def bench(workers_list, num_games, num_sims, games_per_worker_batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = ColourWarsNet().to(device)

    print(f"\n--- Single-process baseline (batch=48, {num_sims} sims/move) ---")
    t0 = time.time()
    games = generate_selfplay_games_batched(net, device, num_games=num_games, num_simulations=num_sims, batch_size=48)
    dt = time.time() - t0
    n_moves = sum(len(g) for g in games)
    print(f"{num_games} games, {n_moves} moves, {dt:.1f}s -> "
          f"{n_moves * (num_sims + 1) / dt:.1f} sims/sec, {num_games / dt * 3600:.0f} games/hour")

    for w in workers_list:
        print(f"\n--- Multiprocess: {w} workers, batch/worker={games_per_worker_batch_size}, {num_sims} sims/move ---")
        t0 = time.time()
        games = run_parallel_selfplay(
            net, num_games=num_games, num_workers=w,
            games_per_worker_batch_size=games_per_worker_batch_size,
            num_simulations=num_sims, device_str=device.type,
            debug=True,
        )
        dt = time.time() - t0
        n_moves = sum(len(g) for g in games)
        print(f"{len(games)} games, {n_moves} moves, {dt:.1f}s -> "
              f"{n_moves * (num_sims + 1) / dt:.1f} sims/sec, {num_games / dt * 3600:.0f} games/hour")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "bench"], default="smoke")
    parser.add_argument("--workers", type=int, nargs="+", default=[4, 6, 8])
    parser.add_argument("--games", type=int, default=48)
    parser.add_argument("--sims", type=int, default=100)
    parser.add_argument("--batch-per-worker", type=int, default=8)
    args = parser.parse_args()

    if args.mode == "smoke":
        smoke_test()
    else:
        bench(args.workers, args.games, args.sims, args.batch_per_worker)
