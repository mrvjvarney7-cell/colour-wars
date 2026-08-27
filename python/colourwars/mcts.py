"""AlphaZero-style MCTS (PUCT) adapted for a multi-player, general-sum game.

Key departure from classic 2-player AlphaZero MCTS: value backup does NOT
flip sign at each ply, because this is not zero-sum. Instead every node
carries a length-MAX_PLAYERS value vector indexed by ABSOLUTE player id, and
a leaf's evaluation (terminal outcome, or network prediction converted from
the mover-relative frame back to absolute ids) is added unchanged to every
ancestor edge on the path to the root. Action selection at a given node still
only looks at the value component belonging to THAT node's mover, since a
rational player picks the move that's best for themselves.

Two ways to drive a search are provided:
  - run_mcts(): the original single-game path. One state at a time through
    the network (batch size 1). Kept as-is for debugging/fallback/small
    scale use - see batched_mcts.py's BatchedMCTS for the throughput path
    used by real self-play at scale.
  - select_leaf() / backup() / expand_leaf_with_output(): the stepwise
    primitives run_mcts is built from, exposed so a batched driver can
    interleave many independent trees (one per concurrent self-play game)
    and evaluate all of their pending leaves in a single network forward
    pass. See batched_mcts.py.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from colourwars.env import MAX_PLAYERS, ColourWarsEnv
from colourwars.network import ColourWarsNet


class Node:
    """A search-tree node. `env` is materialized LAZILY for children: expanding
    a node with N legal actions creates N child Nodes but does NOT step the
    game for any of them up front - each child only pays for env.step() (a
    board clone + cascade resolution) the first time it is actually selected
    during a later descent (see the `env` property below). Since a move
    typically runs far fewer simulations than there are legal actions at a
    freshly-expanded node, most children are never selected at all, so this
    avoids the large majority of the child-materialization cost that eager
    expansion pays for nothing. Root nodes are constructed with a real env up
    front (there's nothing to defer - the caller already has it)."""

    __slots__ = ("_env", "_parent_env", "_action", "mover", "prior", "children", "visit_count", "value_sum", "is_expanded")

    def __init__(
        self,
        env: Optional[ColourWarsEnv],
        prior: float,
        parent_env: Optional[ColourWarsEnv] = None,
        action: Optional[int] = None,
    ):
        self._env = env
        self._parent_env = parent_env
        self._action = action
        self.mover = env.current_player if env is not None else None
        self.prior = prior
        self.children: Dict[int, "Node"] = {}
        self.visit_count = 0
        self.value_sum = np.zeros(MAX_PLAYERS, dtype=np.float64)
        self.is_expanded = False

    @classmethod
    def lazy(cls, prior: float, parent_env: ColourWarsEnv, action: int) -> "Node":
        return cls(env=None, prior=prior, parent_env=parent_env, action=action)

    @property
    def env(self) -> ColourWarsEnv:
        if self._env is None:
            self._env = self._parent_env.step(self._action)
            self.mover = self._env.current_player
            self._parent_env = None  # release; no longer needed once materialized
        return self._env

    def value(self) -> np.ndarray:
        if self.visit_count == 0:
            return np.zeros(MAX_PLAYERS, dtype=np.float64)
        return self.value_sum / self.visit_count


def _select_child_action(node: Node, c_puct: float) -> int:
    best_action, best_score = None, -math.inf
    sqrt_total = math.sqrt(max(node.visit_count, 1))
    mover = node.mover
    for action, child in node.children.items():
        q = child.value()[mover] if child.visit_count > 0 else 0.0
        u = c_puct * child.prior * sqrt_total / (1 + child.visit_count)
        score = q + u
        if score > best_score:
            best_score = score
            best_action = action
    return best_action


def select_leaf(root: Node, c_puct: float = 1.5) -> Tuple[Node, List[Node]]:
    """Descends from `root` via PUCT until hitting either a terminal state or
    a node that has not yet been expanded (i.e. needs a network evaluation).
    Returns (leaf, path) where path includes root..leaf inclusive, for
    backup(). Pure tree-walk, no network calls - safe to call independently
    per game each round of a batched search."""
    node = root
    path = [root]
    while node.is_expanded and not node.env.done and node.children:
        action = _select_child_action(node, c_puct)
        node = node.children[action]
        path.append(node)
    return node, path


def backup(path: List[Node], value: np.ndarray) -> None:
    for n in path:
        n.visit_count += 1
        n.value_sum += value


def terminal_value(env: ColourWarsEnv) -> np.ndarray:
    return env.outcome_values().astype(np.float64)


def expand_leaf_with_output(node: Node, policy_logits: np.ndarray, value: np.ndarray) -> np.ndarray:
    """Expands `node` (not yet expanded, not terminal) given this leaf's raw
    network outputs (policy_logits over all rows*cols actions, value in the
    mover-relative frame). Returns the absolute-player-id value vector to
    back up the tree with. Shared by both the single-state and batched
    network-call paths so expansion logic can't drift between them."""
    env = node.env
    mask = env.legal_moves_mask()

    logits = policy_logits.copy()
    logits[~mask] = -1e9
    logits -= logits.max()
    probs = np.exp(logits)
    probs *= mask
    total = probs.sum()
    if total > 0:
        probs /= total
    else:
        probs = mask.astype(np.float64) / max(mask.sum(), 1)

    for action in np.nonzero(mask)[0]:
        node.children[int(action)] = Node.lazy(prior=float(probs[action]), parent_env=env, action=int(action))
    node.is_expanded = True

    perm = env.action_to_relative_owner_perm()  # perm[player_id] = rel slot
    abs_value = np.zeros(MAX_PLAYERS, dtype=np.float64)
    for pid in range(env.num_players):
        abs_value[pid] = value[perm[pid]]
    return abs_value


def add_root_dirichlet_noise(root: Node, alpha: float = 0.3, epsilon: float = 0.25) -> None:
    if not root.children:
        return
    actions = list(root.children.keys())
    noise = np.random.dirichlet([alpha] * len(actions))
    for a, n in zip(actions, noise):
        child = root.children[a]
        child.prior = child.prior * (1 - epsilon) + n * epsilon


def _evaluate_leaf(node: Node, net: ColourWarsNet, device: torch.device) -> np.ndarray:
    state_tensor = torch.from_numpy(node.env.encode_state())
    policy_logits, rel_value = net.predict(state_tensor, device)
    return expand_leaf_with_output(node, policy_logits, rel_value)


def run_mcts(
    root_env: ColourWarsEnv,
    net: ColourWarsNet,
    device: torch.device,
    num_simulations: int = 200,
    c_puct: float = 1.5,
    dirichlet_alpha: float = 0.3,
    dirichlet_epsilon: float = 0.25,
    add_root_noise: bool = True,
) -> Node:
    """Single-game MCTS: evaluates one board state at a time (batch size 1).
    Simple and correct, but leaves the GPU mostly idle - use BatchedMCTS
    (batched_mcts.py) for real self-play throughput."""
    root = Node(root_env, prior=1.0)

    if root_env.done:
        return root

    root_value = _evaluate_leaf(root, net, device)
    root.visit_count = 1
    root.value_sum += root_value

    if add_root_noise:
        add_root_dirichlet_noise(root, dirichlet_alpha, dirichlet_epsilon)

    for _ in range(num_simulations):
        leaf, path = select_leaf(root, c_puct)
        if leaf.env.done:
            leaf_value = terminal_value(leaf.env)
        else:
            leaf_value = _evaluate_leaf(leaf, net, device)
        backup(path, leaf_value)

    return root


def visit_count_policy(root: Node, num_actions: int, temperature: float = 1.0) -> np.ndarray:
    """Returns a (num_actions,) probability distribution derived from child
    visit counts, for use as the MCTS-improved policy training target and
    for actual move sampling."""
    pi = np.zeros(num_actions, dtype=np.float64)
    if not root.children:
        return pi
    actions = list(root.children.keys())
    counts = np.array([root.children[a].visit_count for a in actions], dtype=np.float64)

    if temperature == 0:
        best = actions[int(np.argmax(counts))]
        pi[best] = 1.0
        return pi

    counts = counts ** (1.0 / temperature)
    counts_sum = counts.sum()
    if counts_sum <= 0:
        probs = np.ones_like(counts) / len(counts)
    else:
        probs = counts / counts_sum
    for a, p in zip(actions, probs):
        pi[a] = p
    return pi
