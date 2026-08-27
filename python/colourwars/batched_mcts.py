"""Batched MCTS driver: runs many independent search trees (one per
concurrent self-play game) side by side, and evaluates all of their
currently-pending leaves through ONE network forward pass per round, instead
of one state at a time.

Why no virtual loss is needed here: virtual loss exists to stop multiple
*concurrent workers exploring the same tree* from repeatedly picking the same
in-flight leaf before its result comes back. This driver does not parallelize
within a single tree - each game contributes exactly one leaf per round, and
that leaf is fully expanded (network-evaluated and backed up) before that
game's tree is touched again in the next round. So a game's own search is
bit-for-bit the same sequence of select -> expand -> backup steps as the
single-game run_mcts() in mcts.py; the only thing batching changes is that
N games' leaves for round K are evaluated in one call rather than N separate
calls. This keeps self-play statistically identical to the unbatched path
while fixing the batch-size-1 GPU underutilization.

run_batched_mcts() takes a `forward_fn: (list[env]) -> (policy_logits, values)`
rather than a network/device pair, so the exact same driver code serves two
very different evaluation backends:
  - a local in-process GPU forward pass (see make_local_forward_fn / the
    single-process path in selfplay.py), or
  - a round-trip to a separate inference-server process shared by many
    self-play worker processes (see parallel_selfplay.py) - each worker's
    per-round batch becomes one shard the server concatenates with other
    workers' shards into one larger GPU forward pass.
Sharing this driver guarantees the two paths can never silently diverge in
tree-search or game logic - only in how a leaf's evaluation is obtained.
"""

from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np
import torch

from colourwars.env import ColourWarsEnv
from colourwars.mcts import (
    Node,
    add_root_dirichlet_noise,
    backup,
    expand_leaf_with_output,
    select_leaf,
    terminal_value,
)
from colourwars.network import ColourWarsNet

ForwardFn = Callable[[List[ColourWarsEnv]], Tuple[np.ndarray, np.ndarray]]


@torch.no_grad()
def _batched_forward(net: ColourWarsNet, device: torch.device, envs: List[ColourWarsEnv]):
    """One local in-process forward pass for a list of envs. Returns
    (policy_logits, values) as numpy arrays of shape (len(envs), 49) and
    (len(envs), MAX_PLAYERS)."""
    net.eval()
    states = np.stack([e.encode_state() for e in envs], axis=0)
    state_tensor = torch.from_numpy(states).to(device)
    policy_logits, values = net(state_tensor)
    return policy_logits.cpu().numpy(), values.cpu().numpy()


def make_local_forward_fn(net: ColourWarsNet, device: torch.device) -> ForwardFn:
    """Wraps an in-process network+device as a forward_fn for run_batched_mcts."""

    def fn(envs: List[ColourWarsEnv]):
        return _batched_forward(net, device, envs)

    return fn


def run_batched_mcts(
    roots: List[Node],
    forward_fn: ForwardFn,
    num_simulations: int = 100,
    c_puct: float = 1.5,
    dirichlet_alpha: float = 0.3,
    dirichlet_epsilon: float = 0.25,
    add_root_noise: bool = True,
) -> None:
    """Runs MCTS on every root in `roots` concurrently, mutating them in place
    (same end state as calling run_mcts on each independently), batching all
    leaf evaluations for a given round into one forward_fn call.

    Every root must be a freshly-constructed, unexpanded, non-terminal Node
    (this mirrors run_mcts's contract - callers create one Node per game
    right before searching for that game's next move)."""
    active = [i for i, r in enumerate(roots) if not r.env.done]
    if not active:
        return

    # Round 0: expand every root itself (equivalent to run_mcts's initial
    # _evaluate_leaf(root) + visit_count=1 bootstrap).
    envs = [roots[i].env for i in active]
    policy_logits, values = forward_fn(envs)
    for slot, i in enumerate(active):
        root = roots[i]
        abs_value = expand_leaf_with_output(root, policy_logits[slot], values[slot])
        root.visit_count = 1
        root.value_sum += abs_value
        if add_root_noise:
            add_root_dirichlet_noise(root, dirichlet_alpha, dirichlet_epsilon)

    for _ in range(num_simulations):
        pending_leaf = {}  # active-index -> (leaf, path)
        pending_terminal = {}  # active-index -> path (backup immediately, no network needed)

        for i in active:
            leaf, path = select_leaf(roots[i], c_puct)
            if leaf.env.done:
                pending_terminal[i] = (leaf, path)
            else:
                pending_leaf[i] = (leaf, path)

        for i, (leaf, path) in pending_terminal.items():
            backup(path, terminal_value(leaf.env))

        if pending_leaf:
            idxs = list(pending_leaf.keys())
            leaves = [pending_leaf[i][0] for i in idxs]
            envs = [leaf.env for leaf in leaves]
            policy_logits, values = forward_fn(envs)
            for slot, i in enumerate(idxs):
                leaf, path = pending_leaf[i]
                abs_value = expand_leaf_with_output(leaf, policy_logits[slot], values[slot])
                backup(path, abs_value)
