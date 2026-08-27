"""Multiprocess self-play: N CPU worker processes each run the same
batched-MCTS self-play loop as selfplay.generate_selfplay_games_batched, but
instead of calling the network in-process, they send their per-round batch
of leaf states to ONE shared inference-server process that owns the GPU.
The server merges batches arriving from multiple workers within a short
flush window into a single larger forward pass, then routes each worker's
slice of the results back to it.

This is deliberately built on top of exactly the same driver
(selfplay.run_batched_selfplay_loop / batched_mcts.run_batched_mcts) as the
single-process path - a worker's forward_fn is just an IPC round-trip
instead of a local GPU call, so game/search logic cannot silently diverge
between the two paths; only how a leaf's network evaluation is obtained
differs.

Windows note: multiprocessing here uses the 'spawn' start method (the only
one available on Windows), so every object handed to Process(args=...) must
be picklable, and the worker/server entry points must be importable
top-level functions (they are - see below), not closures or things defined
in __main__. Any script that calls run_parallel_selfplay must guard its
top-level code with `if __name__ == "__main__":`.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import random
import time
from typing import Dict, List, Optional

import numpy as np
import torch

from colourwars.network import ColourWarsNet
from colourwars.selfplay import TrainingExample, run_batched_selfplay_loop


def _inference_server_main(
    checkpoint_path: Optional[str],
    request_queue: "mp.Queue",
    response_queues: Dict[int, "mp.Queue"],
    device_str: str,
    num_workers: int,
    max_batch_size: int,
    flush_timeout: float,
    ready_event,
    debug: bool = False,
) -> None:
    """Owns the network on the GPU. Loop: block for the first request in a
    round, then drain additional requests (from any worker) for up to
    `flush_timeout` seconds or until `max_batch_size` total states have been
    collected, run ONE forward pass on the concatenated batch, then split the
    output back out per-worker by the shard sizes it arrived in.

    A worker signals it has no more requests coming by sending (worker_id,
    None); once `num_workers` such sentinels have been seen AND the request
    queue is empty, the server exits.
    """
    device = torch.device(device_str)
    net = ColourWarsNet().to(device)
    if checkpoint_path is not None:
        net.load_state_dict(torch.load(checkpoint_path, map_location=device))
    net.eval()
    ready_event.set()

    active_workers = num_workers
    n_rounds = 0
    n_states = 0
    t_wait = t_forward = t_respond = 0.0
    wall_start = time.time()

    while active_workers > 0:
        t0 = time.time()
        wid, payload = request_queue.get()  # blocking - safe, someone will eventually send DONE
        t_wait += time.time() - t0
        if payload is None:
            active_workers -= 1
            continue

        shards = [(wid, payload)]
        total = payload.shape[0]
        deadline = time.time() + flush_timeout
        while total < max_batch_size:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            t0 = time.time()
            try:
                wid2, payload2 = request_queue.get(timeout=remaining)
            except queue.Empty:
                t_wait += time.time() - t0
                break
            t_wait += time.time() - t0
            if payload2 is None:
                active_workers -= 1
                continue
            shards.append((wid2, payload2))
            total += payload2.shape[0]

        t0 = time.time()
        batch = np.concatenate([p for _, p in shards], axis=0)
        state_tensor = torch.from_numpy(batch).to(device)
        with torch.no_grad():
            policy_logits, values = net(state_tensor)
        policy_logits = policy_logits.cpu().numpy()
        values = values.cpu().numpy()
        t_forward += time.time() - t0

        t0 = time.time()
        offset = 0
        for shard_wid, payload_arr in shards:
            k = payload_arr.shape[0]
            response_queues[shard_wid].put((policy_logits[offset:offset + k], values[offset:offset + k]))
            offset += k
        t_respond += time.time() - t0

        n_rounds += 1
        n_states += total

    if debug:
        wall = time.time() - wall_start
        print(
            f"[inference-server] {n_rounds} batches, {n_states} states, wall={wall:.1f}s | "
            f"queue-wait={t_wait:.1f}s ({t_wait/wall:.1%}) forward={t_forward:.1f}s ({t_forward/wall:.1%}) "
            f"respond-put={t_respond:.1f}s ({t_respond/wall:.1%}) | avg batch={n_states/max(n_rounds,1):.1f}",
            flush=True,
        )


def _selfplay_worker_main(
    worker_id: int,
    num_games: int,
    num_simulations: int,
    batch_size: int,
    player_counts,
    temperature_moves: int,
    max_moves: int,
    seed: int,
    request_queue: "mp.Queue",
    response_queue: "mp.Queue",
    result_queue: "mp.Queue",
) -> None:
    random.seed(seed)
    np.random.seed(seed)

    def forward_fn(envs):
        states = np.stack([e.encode_state() for e in envs], axis=0).astype(np.float32)
        request_queue.put((worker_id, states))
        return response_queue.get()

    games = run_batched_selfplay_loop(
        forward_fn,
        num_games,
        num_simulations=num_simulations,
        batch_size=batch_size,
        player_counts=player_counts,
        temperature_moves=temperature_moves,
        max_moves=max_moves,
    )

    request_queue.put((worker_id, None))  # tell the server this worker is done
    result_queue.put((worker_id, games))


def run_parallel_selfplay(
    net: ColourWarsNet,
    num_games: int,
    num_workers: int = 6,
    games_per_worker_batch_size: int = 8,
    num_simulations: int = 100,
    device_str: str = "cuda",
    player_counts=(2, 3, 4),
    temperature_moves: int = 10,
    max_moves: int = 300,
    max_batch_size: int = 512,
    flush_timeout: float = 0.005,
    checkpoint_path: Optional[str] = None,
    debug: bool = False,
) -> List[List[TrainingExample]]:
    """Runs `num_games` self-play games across `num_workers` CPU worker
    processes, all sharing one GPU inference-server process for network
    evaluation. Must be called from inside `if __name__ == "__main__":` on
    Windows (spawn start method requirement).

    `checkpoint_path`: if given, the server loads weights from this file
    instead of `net`'s current in-memory weights (avoids having to pickle a
    CUDA module across the process boundary - if omitted, `net`'s weights
    are saved to a temp file automatically).
    """
    import os
    import tempfile

    ctx = mp.get_context("spawn")

    tmp_checkpoint = None
    if checkpoint_path is None:
        fd, tmp_checkpoint = tempfile.mkstemp(suffix=".pt", prefix="cw_parallel_selfplay_")
        os.close(fd)
        torch.save(net.state_dict(), tmp_checkpoint)
        checkpoint_path = tmp_checkpoint

    request_queue = ctx.Queue()
    response_queues = {i: ctx.Queue() for i in range(num_workers)}
    result_queue = ctx.Queue()
    ready_event = ctx.Event()

    server = ctx.Process(
        target=_inference_server_main,
        args=(checkpoint_path, request_queue, response_queues, device_str,
              num_workers, max_batch_size, flush_timeout, ready_event, debug),
        daemon=True,
    )
    server.start()
    ready_event.wait(timeout=60)

    games_per_worker = [num_games // num_workers] * num_workers
    for i in range(num_games % num_workers):
        games_per_worker[i] += 1

    workers = []
    for i in range(num_workers):
        if games_per_worker[i] == 0:
            continue
        p = ctx.Process(
            target=_selfplay_worker_main,
            args=(i, games_per_worker[i], num_simulations, games_per_worker_batch_size,
                  player_counts, temperature_moves, max_moves, 1000 + i,
                  request_queue, response_queues[i], result_queue),
        )
        p.start()
        workers.append(p)

    all_games: List[List[TrainingExample]] = []
    for _ in range(len(workers)):
        _, games = result_queue.get()
        all_games.extend(games)

    for p in workers:
        p.join(timeout=30)
    server.join(timeout=30)
    if server.is_alive():
        server.terminate()

    if tmp_checkpoint is not None:
        try:
            os.remove(tmp_checkpoint)
        except OSError:
            pass

    return all_games
