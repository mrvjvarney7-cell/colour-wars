"""Pure Python port of js/gameLogic.js (Colour Wars custom Chain Reaction variant).

This is a faithful, line-by-line port. Any behavioral discrepancy from the JS
source is a bug in this file, not an intentional deviation. See
python/colourwars/tests/test_game.py, which translates
js/gameLogic.test.js scenario-for-scenario, for the cross-check.

Rules (fixed critical mass of 4 everywhere, not the classic position-based
2/3/4 system):
- 7x7 board, 2-4 players.
- Placing on an empty cell or a cell you own adds 1 dot, except each
  player's first move of the game, which places OPENING_DOTS (3) dots.
- At 4 dots a cell explodes: loses 4, each existing orthogonal neighbour
  gains 1 dot and is captured by the exploding player. A corner/edge with
  fewer than 4 neighbours simply discards the excess dots.
- Explosions cascade (wave by wave, all unstable cells in a wave explode
  simultaneously) until the board is stable.
- A player is eliminated once they own zero cells, but this is only
  checked once every player has had at least one move.
- Last player standing wins.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

ROWS = 7
COLS = 7

# Each player's very first move of the game places this many dots instead of
# one, to get games moving faster. Every later move places a single dot.
OPENING_DOTS = 3

# Every cell on the board explodes at the same threshold, regardless of
# position: corners, edges and interior cells all detonate at 4 dots and lose
# exactly 4 when they do.
CRITICAL_MASS = 4

COLOR_PALETTE = [
    {"name": "blue", "hex": "#2563eb"},
    {"name": "green", "hex": "#16a34a"},
    {"name": "orange", "hex": "#f97316"},
    {"name": "red", "hex": "#dc2626"},
]


def get_neighbors(row: int, col: int, rows: int, cols: int) -> List[Tuple[int, int]]:
    out = []
    if row > 0:
        out.append((row - 1, col))
    if row < rows - 1:
        out.append((row + 1, col))
    if col > 0:
        out.append((row, col - 1))
    if col < cols - 1:
        out.append((row, col + 1))
    return out


def get_critical_mass(*_args) -> int:
    """Fixed for every cell on the board. Accepts (and ignores) positional
    arguments so callers can pass a position, mirroring the JS signature."""
    return CRITICAL_MASS


@dataclass
class Cell:
    owner: Optional[int] = None
    count: int = 0


Board = List[List[Cell]]


def create_empty_board(rows: int = ROWS, cols: int = COLS) -> Board:
    return [[Cell(owner=None, count=0) for _ in range(cols)] for _ in range(rows)]


def clone_board(board: Board) -> Board:
    return [[Cell(owner=cell.owner, count=cell.count) for cell in row] for row in board]


def board_dims(board: Board) -> Tuple[int, int]:
    return len(board), len(board[0])


@dataclass
class ExplodedCell:
    row: int
    col: int


@dataclass
class Gain:
    row: int
    col: int
    from_row: int
    from_col: int


@dataclass
class Step:
    board: Board
    exploded: List[ExplodedCell]
    gains: List[Gain]


@dataclass
class ApplyMoveResult:
    board: Board
    steps: List[Step]


def apply_move(
    board: Board,
    row: int,
    col: int,
    player: int,
    rows: Optional[int] = None,
    cols: Optional[int] = None,
    dots: Optional[int] = None,
) -> ApplyMoveResult:
    """Applies a single move (placing `dots` dots, default 1, for `player` at
    row/col) and fully resolves any resulting chain reaction."""
    dims_rows, dims_cols = board_dims(board)
    rows = rows or dims_rows
    cols = cols or dims_cols
    dots = dots or 1

    working = clone_board(board)
    cell = working[row][col]
    cell.owner = player
    cell.count += dots

    steps: List[Step] = []
    guard = 0
    guard_limit = rows * cols * 50  # safety valve against pathological infinite loops

    while True:
        guard += 1
        if guard > guard_limit:
            break

        unstable: List[Tuple[int, int]] = []
        for r in range(rows):
            for c in range(cols):
                if working[r][c].count >= CRITICAL_MASS:
                    unstable.append((r, c))

        if len(unstable) == 0:
            break

        gains: List[Gain] = []
        # Resolve this wave: every unstable cell explodes simultaneously.
        for (er, ec) in unstable:
            exploding_cell = working[er][ec]
            # Always loses the full critical mass, whatever its neighbour count.
            exploding_cell.count -= CRITICAL_MASS
            if exploding_cell.count <= 0:
                exploding_cell.count = 0
                exploding_cell.owner = None
            # Only orthogonal neighbours that exist receive a dot; for a
            # corner or edge cell the leftover dots are simply discarded.
            neighbors = get_neighbors(er, ec, rows, cols)
            for (nr, nc) in neighbors:
                gains.append(Gain(row=nr, col=nc, from_row=er, from_col=ec))

        for g in gains:
            target = working[g.row][g.col]
            target.count += 1
            target.owner = player

        steps.append(
            Step(
                board=clone_board(working),
                exploded=[ExplodedCell(row=r, col=c) for (r, c) in unstable],
                gains=gains,
            )
        )

    return ApplyMoveResult(board=working, steps=steps)


def is_valid_move(board: Board, row: int, col: int, player: int, has_moved: bool) -> bool:
    """A player's very first move (their opening) may target any empty cell.
    Every move after that may only target a cell that player already owns -
    the only way to gain new territory is by exploding into it via a chain
    reaction, never by placing directly on an empty or opponent-owned cell."""
    if row < 0 or col < 0 or row >= len(board) or col >= len(board[0]):
        return False
    cell = board[row][col]
    if has_moved:
        return cell.owner == player
    return cell.owner is None or cell.owner == player


def count_cells_for_player(board: Board, player: int) -> int:
    n = 0
    for row in board:
        for cell in row:
            if cell.owner == player:
                n += 1
    return n


# ---- Game-level state ----


@dataclass
class Player:
    id: int
    name: str
    color: str
    color_name: str
    active: bool = True
    # Drives the opening-move bonus: False until this player has moved once.
    has_moved: bool = False


@dataclass
class GameState:
    rows: int
    cols: int
    board: Board
    players: List[Player]
    current_player_index: int = 0
    total_moves: int = 0
    game_over: bool = False
    winner: Optional[int] = None


def create_game(num_players: int, rows: int = ROWS, cols: int = COLS) -> GameState:
    num_players = max(2, min(4, num_players))

    players = []
    for i in range(num_players):
        players.append(
            Player(
                id=i,
                name=f"Player {i + 1}",
                color=COLOR_PALETTE[i]["hex"],
                color_name=COLOR_PALETTE[i]["name"],
                active=True,
                has_moved=False,
            )
        )

    return GameState(
        rows=rows,
        cols=cols,
        board=create_empty_board(rows, cols),
        players=players,
        current_player_index=0,
        total_moves=0,
        game_over=False,
        winner=None,
    )


def placement_dots(state: GameState, player_id: int) -> int:
    """How many dots `player_id`'s next placement drops: OPENING_DOTS for
    their very first move of the game, 1 for every move after that."""
    p = state.players[player_id] if 0 <= player_id < len(state.players) else None
    return OPENING_DOTS if (p is not None and not p.has_moved) else 1


def next_active_player_index(state: GameState, from_index: int) -> int:
    n = len(state.players)
    for step in range(1, n + 1):
        idx = (from_index + step) % n
        if state.players[idx].active:
            return idx
    return from_index


@dataclass
class PlayMoveResult:
    state: GameState
    steps: List[Step]


def _clone_player(p: Player) -> Player:
    return Player(
        id=p.id,
        name=p.name,
        color=p.color,
        color_name=p.color_name,
        active=p.active,
        has_moved=p.has_moved,
    )


def play_move(state: GameState, row: int, col: int) -> PlayMoveResult:
    """Applies a move to a game state (does NOT mutate the input state)."""
    if state.game_over:
        return PlayMoveResult(state=state, steps=[])

    player = state.current_player_index
    if not is_valid_move(state.board, row, col, player, state.players[player].has_moved):
        return PlayMoveResult(state=state, steps=[])

    dots = placement_dots(state, player)
    result = apply_move(state.board, row, col, player, state.rows, state.cols, dots)

    new_state = GameState(
        rows=state.rows,
        cols=state.cols,
        board=result.board,
        players=[_clone_player(p) for p in state.players],
        current_player_index=state.current_player_index,
        total_moves=state.total_moves + 1,
        game_over=False,
        winner=None,
    )
    # This player has now opened; their later moves place a single dot.
    new_state.players[player].has_moved = True

    # Only start checking eliminations once every player has had at least one turn.
    first_round_complete = new_state.total_moves >= len(new_state.players)
    if first_round_complete:
        for p in new_state.players:
            if p.active and count_cells_for_player(new_state.board, p.id) == 0:
                p.active = False

    active_players = [p for p in new_state.players if p.active]
    if first_round_complete and len(active_players) == 1:
        new_state.game_over = True
        new_state.winner = active_players[0].id
        new_state.current_player_index = player
    else:
        new_state.current_player_index = next_active_player_index(new_state, player)

    return PlayMoveResult(state=new_state, steps=result.steps)
