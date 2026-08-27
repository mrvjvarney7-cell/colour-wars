"""RL-facing wrapper around game.py: tensor encoding, legal-move masks, and a
step() API suited to MCTS/self-play. Game rules themselves live in game.py
and are untouched here - this module only adds the numpy/RL plumbing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from colourwars.game import (
    ROWS,
    COLS,
    GameState,
    create_game,
    is_valid_move,
    play_move,
)

MAX_PLAYERS = 4
# Board-plane count for the NN encoding, see encode_state():
#   MAX_PLAYERS owner-indicator planes + 1 count plane + 1 current-player plane
#   + MAX_PLAYERS active-player-count planes (one-hot broadcast).
NUM_PLANES = MAX_PLAYERS + 1 + 1 + MAX_PLAYERS


class ColourWarsEnv:
    """Wraps a GameState and exposes the interface self-play/MCTS need."""

    def __init__(self, num_players: int, rows: int = ROWS, cols: int = COLS):
        self.rows = rows
        self.cols = cols
        self.num_players = num_players
        self.state: GameState = create_game(num_players, rows, cols)

    @staticmethod
    def from_state(state: GameState) -> "ColourWarsEnv":
        env = ColourWarsEnv.__new__(ColourWarsEnv)
        env.rows = state.rows
        env.cols = state.cols
        env.num_players = len(state.players)
        env.state = state
        return env

    def clone(self) -> "ColourWarsEnv":
        return ColourWarsEnv.from_state(self.state)

    @property
    def current_player(self) -> int:
        return self.state.current_player_index

    @property
    def done(self) -> bool:
        return self.state.game_over

    @property
    def winner(self) -> Optional[int]:
        return self.state.winner

    def legal_moves_mask(self) -> np.ndarray:
        """Returns a flat (rows*cols,) bool mask of legal actions for the
        current player."""
        mask = np.zeros(self.rows * self.cols, dtype=bool)
        player = self.state.current_player_index
        has_moved = self.state.players[player].has_moved
        for r in range(self.rows):
            for c in range(self.cols):
                if is_valid_move(self.state.board, r, c, player, has_moved):
                    mask[r * self.cols + c] = True
        return mask

    def legal_moves(self) -> List[int]:
        mask = self.legal_moves_mask()
        return list(np.nonzero(mask)[0])

    def step(self, action: int) -> "ColourWarsEnv":
        """Applies action (flat index) and returns a NEW env (does not mutate self)."""
        r, c = divmod(action, self.cols)
        result = play_move(self.state, r, c)
        return ColourWarsEnv.from_state(result.state)

    def active_player_ids(self) -> List[int]:
        return [p.id for p in self.state.players if p.active]

    def outcome_values(self) -> np.ndarray:
        """Returns a (MAX_PLAYERS,) vector of terminal values in {-1, 0, 1}
        for a finished game: +1 for the winner, -1 for every eliminated/losing
        player, 0 for padding slots beyond num_players. Only valid when done."""
        assert self.done
        values = np.zeros(MAX_PLAYERS, dtype=np.float32)
        for p in self.state.players:
            values[p.id] = 1.0 if p.id == self.state.winner else -1.0
        return values

    def encode_state(self) -> np.ndarray:
        """Encodes the current state as a (NUM_PLANES, rows, cols) float32
        tensor, from the perspective of the CURRENT player (plane 0 is
        always "my" cells, so the network is player-invariant):
          planes [0 .. num_players-1]: owner==relative_player_id indicator
            (rotated so the current player is always plane 0), scaled by
            cell dot count / CRITICAL_MASS.
          plane [MAX_PLAYERS]: raw dot count / CRITICAL_MASS for every
            occupied cell regardless of owner (redundant but helps early
            training signal).
          plane [MAX_PLAYERS + 1]: constant 1 if it's this player's first
            move (i.e. their next placement is the opening 3-dot bonus),
            else 0. Broadcast over the whole plane.
          planes [MAX_PLAYERS + 2 .. end]: one-hot broadcast of num_players
            (which of 2/3/4 players this game has), so one network can
            handle variable player counts.
        """
        from colourwars.game import CRITICAL_MASS, placement_dots

        rows, cols = self.rows, self.cols
        planes = np.zeros((NUM_PLANES, rows, cols), dtype=np.float32)
        me = self.state.current_player_index
        n = self.num_players

        for r in range(rows):
            for c in range(cols):
                cell = self.state.board[r][c]
                if cell.owner is not None:
                    rel = (cell.owner - me) % n
                    planes[rel, r, c] = cell.count / CRITICAL_MASS
                    planes[MAX_PLAYERS, r, c] = cell.count / CRITICAL_MASS

        opening = 1.0 if placement_dots(self.state, me) > 1 else 0.0
        planes[MAX_PLAYERS + 1, :, :] = opening

        n_plane_idx = MAX_PLAYERS + 2 + (n - 2)
        planes[n_plane_idx, :, :] = 1.0

        return planes

    def action_to_relative_owner_perm(self) -> np.ndarray:
        """Returns the permutation mapping absolute player id -> relative
        slot (rel = (id - me) % n) used by encode_state, so callers can map
        network value-head outputs (relative) back to absolute player ids."""
        me = self.state.current_player_index
        n = self.num_players
        perm = np.zeros(MAX_PLAYERS, dtype=np.int64)
        for pid in range(n):
            perm[pid] = (pid - me) % n
        return perm
