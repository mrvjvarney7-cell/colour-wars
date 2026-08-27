//! Faithful Rust port of js/gameLogic.js, cross-checked against the existing
//! Python port at python/colourwars/game.py (itself already verified against
//! all 33 of js/gameLogic.test.js's scenarios). This file is the THIRD
//! independent implementation of the same rules; python/colourwars/game.py
//! remains the immediate source of truth for exact behavior, since it is
//! itself already a line-by-line port of the original JS.
//!
//! Rules (fixed critical mass of 4 everywhere, not the classic position-based
//! 2/3/4 system):
//! - 7x7 board, 2-4 players.
//! - Placing on an empty cell or a cell you own adds 1 dot, except each
//!   player's first move of the game, which places OPENING_DOTS (3) dots.
//! - At 4 dots a cell explodes: loses 4, each existing orthogonal neighbour
//!   gains 1 dot and is captured by the exploding player. A corner/edge with
//!   fewer than 4 neighbours simply discards the excess dots.
//! - Explosions cascade (wave by wave, all unstable cells in a wave explode
//!   simultaneously) until the board is stable.
//! - A player is eliminated once they own zero cells, but this is only
//!   checked once every player has had at least one move.
//! - Last player standing wins.
//!
//! Deliberate simplification vs. the JS/Python versions: cosmetic-only fields
//! (player display name, color hex, color name) are dropped from `Player`
//! here - no test in python/colourwars/tests/test_game.py (the 33 cases
//! ported below) ever inspects them, and they play no role in the rules or
//! in anything the RL pipeline consumes, so carrying them through Rust would
//! only add string-allocation overhead for no behavioral benefit.

pub const ROWS: usize = 7;
pub const COLS: usize = 7;

/// Each player's very first move of the game places this many dots instead
/// of one, to get games moving faster. Every later move places a single dot.
pub const OPENING_DOTS: i32 = 3;

/// Every cell on the board explodes at the same threshold, regardless of
/// position: corners, edges and interior cells all detonate at 4 dots and
/// lose exactly 4 when they do.
pub const CRITICAL_MASS: i32 = 4;

pub fn get_neighbors(row: usize, col: usize, rows: usize, cols: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::with_capacity(4);
    if row > 0 {
        out.push((row - 1, col));
    }
    if row + 1 < rows {
        out.push((row + 1, col));
    }
    if col > 0 {
        out.push((row, col - 1));
    }
    if col + 1 < cols {
        out.push((row, col + 1));
    }
    out
}

/// Fixed for every cell on the board. Takes a position (like the JS/Python
/// versions do) purely so call sites and tests can mirror those APIs; the
/// threshold itself never depends on it.
pub fn get_critical_mass(_row: usize, _col: usize, _rows: usize, _cols: usize) -> i32 {
    CRITICAL_MASS
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Cell {
    pub owner: Option<u8>,
    pub count: i32,
}

impl Cell {
    pub fn new(owner: Option<u8>, count: i32) -> Self {
        Cell { owner, count }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Board {
    pub rows: usize,
    pub cols: usize,
    cells: Vec<Cell>,
}

impl Board {
    pub fn new(rows: usize, cols: usize) -> Self {
        Board {
            rows,
            cols,
            cells: vec![Cell::new(None, 0); rows * cols],
        }
    }

    #[inline]
    fn idx(&self, r: usize, c: usize) -> usize {
        r * self.cols + c
    }

    #[inline]
    pub fn get(&self, r: usize, c: usize) -> &Cell {
        &self.cells[self.idx(r, c)]
    }

    #[inline]
    pub fn get_mut(&mut self, r: usize, c: usize) -> &mut Cell {
        let i = self.idx(r, c);
        &mut self.cells[i]
    }

    pub fn set(&mut self, r: usize, c: usize, cell: Cell) {
        let i = self.idx(r, c);
        self.cells[i] = cell;
    }

    pub fn iter_cells(&self) -> impl Iterator<Item = &Cell> {
        self.cells.iter()
    }
}

#[derive(Clone, Debug)]
pub struct ExplodedCell {
    pub row: usize,
    pub col: usize,
}

#[derive(Clone, Debug)]
pub struct Gain {
    pub row: usize,
    pub col: usize,
    pub from_row: usize,
    pub from_col: usize,
}

#[derive(Clone, Debug)]
pub struct Step {
    pub board: Board,
    pub exploded: Vec<ExplodedCell>,
    pub gains: Vec<Gain>,
}

pub struct ApplyMoveResult {
    pub board: Board,
    pub steps: Vec<Step>,
}

/// Applies a single move (placing `dots` dots for `player` at row/col) and
/// fully resolves any resulting chain reaction.
pub fn apply_move(board: &Board, row: usize, col: usize, player: u8, dots: i32) -> ApplyMoveResult {
    let mut working = board.clone();
    {
        let cell = working.get_mut(row, col);
        cell.owner = Some(player);
        cell.count += dots;
    }

    let mut steps = Vec::new();
    let guard_limit = working.rows * working.cols * 50; // safety valve, mirrors the JS/Python guard
    let mut guard = 0u64;

    loop {
        guard += 1;
        if guard > guard_limit as u64 {
            break;
        }

        let mut unstable = Vec::new();
        for r in 0..working.rows {
            for c in 0..working.cols {
                if working.get(r, c).count >= CRITICAL_MASS {
                    unstable.push((r, c));
                }
            }
        }
        if unstable.is_empty() {
            break;
        }

        let mut gains = Vec::new();
        for &(er, ec) in &unstable {
            let cell = working.get_mut(er, ec);
            cell.count -= CRITICAL_MASS;
            if cell.count <= 0 {
                cell.count = 0;
                cell.owner = None;
            }
            for (nr, nc) in get_neighbors(er, ec, working.rows, working.cols) {
                gains.push(Gain { row: nr, col: nc, from_row: er, from_col: ec });
            }
        }

        for g in &gains {
            let target = working.get_mut(g.row, g.col);
            target.count += 1;
            target.owner = Some(player);
        }

        steps.push(Step {
            board: working.clone(),
            exploded: unstable.iter().map(|&(r, c)| ExplodedCell { row: r, col: c }).collect(),
            gains,
        });
    }

    ApplyMoveResult { board: working, steps }
}

/// row/col are signed so out-of-range/negative queries (as in the JS/Python
/// tests) can be represented without panicking on an unsigned underflow.
pub fn is_valid_move(board: &Board, row: i32, col: i32, player: u8) -> bool {
    if row < 0 || col < 0 || row as usize >= board.rows || col as usize >= board.cols {
        return false;
    }
    let cell = board.get(row as usize, col as usize);
    cell.owner.is_none() || cell.owner == Some(player)
}

pub fn count_cells_for_player(board: &Board, player: u8) -> usize {
    board.iter_cells().filter(|c| c.owner == Some(player)).count()
}

// ---- Game-level state ----

#[derive(Clone, Debug)]
pub struct Player {
    pub id: u8,
    pub active: bool,
    /// Drives the opening-move bonus: false until this player has moved once.
    pub has_moved: bool,
}

#[derive(Clone, Debug)]
pub struct GameState {
    pub rows: usize,
    pub cols: usize,
    pub board: Board,
    pub players: Vec<Player>,
    pub current_player_index: usize,
    pub total_moves: u32,
    pub game_over: bool,
    pub winner: Option<u8>,
}

pub fn create_game(num_players: usize, rows: usize, cols: usize) -> GameState {
    let num_players = num_players.clamp(2, 4);
    let players = (0..num_players)
        .map(|i| Player { id: i as u8, active: true, has_moved: false })
        .collect();

    GameState {
        rows,
        cols,
        board: Board::new(rows, cols),
        players,
        current_player_index: 0,
        total_moves: 0,
        game_over: false,
        winner: None,
    }
}

/// How many dots `player_id`'s next placement drops: OPENING_DOTS for their
/// very first move of the game, 1 for every move after that.
pub fn placement_dots(state: &GameState, player_id: usize) -> i32 {
    match state.players.get(player_id) {
        Some(p) if !p.has_moved => OPENING_DOTS,
        _ => 1,
    }
}

pub fn next_active_player_index(state: &GameState, from_index: usize) -> usize {
    let n = state.players.len();
    for step in 1..=n {
        let idx = (from_index + step) % n;
        if state.players[idx].active {
            return idx;
        }
    }
    from_index
}

pub struct PlayMoveResult {
    pub state: GameState,
    pub steps: Vec<Step>,
}

/// Applies a move to a game state (does NOT mutate the input state).
pub fn play_move(state: &GameState, row: usize, col: usize) -> PlayMoveResult {
    if state.game_over {
        return PlayMoveResult { state: state.clone(), steps: vec![] };
    }

    let player = state.current_player_index;
    if !is_valid_move(&state.board, row as i32, col as i32, player as u8) {
        return PlayMoveResult { state: state.clone(), steps: vec![] };
    }

    let dots = placement_dots(state, player);
    let result = apply_move(&state.board, row, col, player as u8, dots);

    let mut new_state = GameState {
        rows: state.rows,
        cols: state.cols,
        board: result.board,
        players: state.players.clone(),
        current_player_index: state.current_player_index,
        total_moves: state.total_moves + 1,
        game_over: false,
        winner: None,
    };
    new_state.players[player].has_moved = true;

    // Only start checking eliminations once every player has had at least one turn.
    let first_round_complete = new_state.total_moves as usize >= new_state.players.len();
    if first_round_complete {
        for p in new_state.players.iter_mut() {
            if p.active && count_cells_for_player(&new_state.board, p.id) == 0 {
                p.active = false;
            }
        }
    }

    let active_count = new_state.players.iter().filter(|p| p.active).count();
    if first_round_complete && active_count == 1 {
        new_state.game_over = true;
        new_state.winner = new_state.players.iter().find(|p| p.active).map(|p| p.id);
        new_state.current_player_index = player;
    } else {
        new_state.current_player_index = next_active_player_index(&new_state, player);
    }

    PlayMoveResult { state: new_state, steps: result.steps }
}

// =============================================================================
// Tests: a scenario-for-scenario port of python/colourwars/tests/test_game.py
// (itself a port of js/gameLogic.test.js) - same 33 cases, same assertions.
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const CM: i32 = 4;

    /// Tests that hand-build a mid-game board are simulating players who have
    /// already opened, so clear the per-player opening-move bonus for them.
    fn mid_game(mut state: GameState) -> GameState {
        for p in state.players.iter_mut() {
            p.has_moved = true;
        }
        state
    }

    /// Repeatedly place single dots on one cell for `player`, returning the
    /// result of the final placement.
    fn place_dots(board: &Board, r: usize, c: usize, player: u8, times: usize) -> ApplyMoveResult {
        let mut b = board.clone();
        let mut res = None;
        for _ in 0..times {
            let r_ = apply_move(&b, r, c, player, 1);
            b = r_.board.clone();
            res = Some(r_);
        }
        res.unwrap()
    }

    // ---------- Critical mass is a fixed 4 everywhere ----------

    #[test]
    fn critical_mass_fixed_4_corners_edges_interior() {
        assert_eq!(CRITICAL_MASS, 4);
        assert_eq!(get_critical_mass(0, 0, 7, 7), 4);
        assert_eq!(get_critical_mass(0, 6, 7, 7), 4);
        assert_eq!(get_critical_mass(6, 0, 7, 7), 4);
        assert_eq!(get_critical_mass(6, 6, 7, 7), 4);
        assert_eq!(get_critical_mass(0, 3, 7, 7), 4);
        assert_eq!(get_critical_mass(3, 0, 7, 7), 4);
        assert_eq!(get_critical_mass(6, 3, 7, 7), 4);
        assert_eq!(get_critical_mass(3, 6, 7, 7), 4);
        assert_eq!(get_critical_mass(3, 3, 7, 7), 4);
        assert_eq!(get_critical_mass(1, 1, 7, 7), 4);
        assert_eq!(get_critical_mass(5, 2, 7, 7), 4);
    }

    #[test]
    fn every_one_of_49_cells_reports_critical_mass_4() {
        for r in 0..7 {
            for c in 0..7 {
                assert_eq!(get_critical_mass(r, c, 7, 7), 4);
            }
        }
    }

    #[test]
    fn neighbor_lookup_only_existing_orthogonal_neighbors() {
        assert_eq!(get_neighbors(0, 0, 7, 7).len(), 2);
        assert_eq!(get_neighbors(0, 6, 7, 7).len(), 2);
        assert_eq!(get_neighbors(6, 0, 7, 7).len(), 2);
        assert_eq!(get_neighbors(6, 6, 7, 7).len(), 2);
        assert_eq!(get_neighbors(0, 3, 7, 7).len(), 3);
        assert_eq!(get_neighbors(3, 0, 7, 7).len(), 3);
        assert_eq!(get_neighbors(3, 3, 7, 7).len(), 4);
        for (nr, nc) in get_neighbors(0, 0, 7, 7) {
            assert!(nr < 7 && nc < 7);
            let dr = nr as i64 - 0i64;
            let dc = nc as i64 - 0i64;
            assert_eq!(dr.abs() + dc.abs(), 1);
        }
    }

    // ---------- Nothing explodes below 4, everything explodes at exactly 4 ----------

    #[test]
    fn no_cell_explodes_at_1_2_3_dots() {
        for &(r, c) in &[(0, 0), (0, 6), (6, 0), (6, 6), (0, 3), (3, 0), (6, 3), (3, 6), (3, 3), (1, 1), (5, 2)] {
            let res = place_dots(&Board::new(7, 7), r, c, 0, 3);
            assert_eq!(res.board.get(r, c).count, 3);
            assert_eq!(res.board.get(r, c).owner, Some(0));
            assert_eq!(res.steps.len(), 0);
        }
    }

    #[test]
    fn corner_explodes_at_exactly_4_feeding_both_neighbors() {
        let res = place_dots(&Board::new(7, 7), 0, 6, 0, 4);
        assert_eq!(res.steps.len(), 1);
        assert_eq!(res.board.get(0, 6).count, 0);
        assert_eq!(res.board.get(0, 6).owner, None);
        assert_eq!(res.board.get(1, 6).count, 1);
        assert_eq!(res.board.get(1, 6).owner, Some(0));
        assert_eq!(res.board.get(0, 5).count, 1);
        assert_eq!(res.board.get(0, 5).owner, Some(0));
        let total: i32 = res.board.iter_cells().map(|c| c.count).sum();
        assert_eq!(total, 2);
    }

    #[test]
    fn edge_cell_explodes_at_exactly_4_feeding_all_three() {
        let res = place_dots(&Board::new(7, 7), 0, 3, 0, 4);
        assert_eq!(res.steps.len(), 1);
        assert_eq!(res.board.get(0, 3).count, 0);
        assert_eq!(res.board.get(0, 3).owner, None);
        for &(r, c) in &[(0, 2), (0, 4), (1, 3)] {
            assert_eq!(res.board.get(r, c).count, 1);
            assert_eq!(res.board.get(r, c).owner, Some(0));
        }
        let total: i32 = res.board.iter_cells().map(|c| c.count).sum();
        assert_eq!(total, 3);
    }

    #[test]
    fn interior_cell_explodes_at_exactly_4_feeding_all_four() {
        let res = place_dots(&Board::new(7, 7), 3, 3, 0, 4);
        assert_eq!(res.steps.len(), 1);
        assert_eq!(res.board.get(3, 3).count, 0);
        assert_eq!(res.board.get(3, 3).owner, None);
        for &(r, c) in &[(2, 3), (4, 3), (3, 2), (3, 4)] {
            assert_eq!(res.board.get(r, c).count, 1);
            assert_eq!(res.board.get(r, c).owner, Some(0));
        }
        let total: i32 = res.board.iter_cells().map(|c| c.count).sum();
        assert_eq!(total, 4);
    }

    #[test]
    fn every_corner_and_edge_explodes_at_4_without_crashing() {
        for &(r, c) in &[(0, 0), (0, 6), (6, 0), (6, 6), (0, 3), (3, 0), (6, 3), (3, 6)] {
            let res = place_dots(&Board::new(7, 7), r, c, 0, 4);
            assert_eq!(res.steps.len(), 1);
            for rr in 0..7 {
                for cc in 0..7 {
                    assert!(res.board.get(rr, cc).count >= 0);
                }
            }
            let nbrs = get_neighbors(r, c, 7, 7);
            let delivered: i32 = nbrs.iter().map(|&(nr, nc)| res.board.get(nr, nc).count).sum();
            assert_eq!(delivered as usize, nbrs.len());
        }
    }

    #[test]
    fn basic_placement_empty_cell_and_own_cell() {
        let board = Board::new(7, 7);
        let r1 = apply_move(&board, 3, 3, 0, 1);
        assert_eq!(r1.board.get(3, 3).owner, Some(0));
        assert_eq!(r1.board.get(3, 3).count, 1);
        assert_eq!(r1.steps.len(), 0);
        let r2 = apply_move(&r1.board, 3, 3, 0, 1);
        assert_eq!(r2.board.get(3, 3).count, 2);
    }

    // ---------- Capture and chain reactions ----------

    #[test]
    fn exploding_into_opponent_cell_captures_regardless_of_owner() {
        let mut board = Board::new(7, 7);
        board.set(3, 3, Cell::new(Some(0), 3));
        board.set(2, 3, Cell::new(Some(1), 1));
        let result = apply_move(&board, 3, 3, 0, 1);
        assert_eq!(result.board.get(2, 3).owner, Some(0));
        assert_eq!(result.board.get(2, 3).count, 2);
    }

    #[test]
    fn explosion_pushing_neighbor_to_4_chains_further() {
        let mut board = Board::new(7, 7);
        board.set(3, 3, Cell::new(Some(0), 3));
        board.set(3, 4, Cell::new(Some(0), 3));
        let result = apply_move(&board, 3, 3, 0, 1);
        assert!(result.steps.len() >= 2);
        assert_eq!(result.board.get(3, 4).count, 0);
        assert_eq!(result.board.get(3, 4).owner, None);
        for &(r, c) in &[(2, 4), (4, 4), (3, 5)] {
            assert_eq!(result.board.get(r, c).owner, Some(0));
            assert_eq!(result.board.get(r, c).count, 1);
        }
    }

    #[test]
    fn corner_reaching_4_purely_from_cascade_explodes_same_move() {
        let mut board = Board::new(7, 7);
        board.set(0, 6, Cell::new(Some(1), 3));
        board.set(1, 6, Cell::new(Some(0), 3));
        let result = apply_move(&board, 1, 6, 0, 1);
        assert!(result.steps.len() >= 2);
        assert_eq!(result.board.get(0, 6).owner, None);
        assert_eq!(result.board.get(0, 6).count, 0);
        assert_eq!(result.board.get(0, 5).owner, Some(0));
    }

    #[test]
    fn chain_reaction_captures_cells_from_multiple_owners_across_waves() {
        let mut board = Board::new(7, 7);
        board.set(3, 3, Cell::new(Some(0), 3));
        board.set(3, 4, Cell::new(Some(1), 1));
        board.set(2, 3, Cell::new(Some(2), 3));
        let result = apply_move(&board, 3, 3, 0, 1);
        assert!(result.steps.len() >= 2);
        assert_eq!(result.board.get(3, 4).owner, Some(0));
        assert_eq!(result.board.get(2, 3).owner, None);
        assert_eq!(result.board.get(1, 3).owner, Some(0));
    }

    #[test]
    fn no_cell_ever_rests_at_or_above_critical_mass() {
        let mut board = Board::new(7, 7);
        board.set(3, 3, Cell::new(Some(0), 3));
        board.set(3, 4, Cell::new(Some(0), 3));
        board.set(2, 3, Cell::new(Some(0), 3));
        board.set(4, 3, Cell::new(Some(0), 3));
        board.set(3, 2, Cell::new(Some(0), 3));
        let res = apply_move(&board, 3, 3, 0, 1);
        for r in 0..7 {
            for c in 0..7 {
                assert!(res.board.get(r, c).count < CM);
                assert!(res.board.get(r, c).count >= 0);
            }
        }
    }

    #[test]
    fn does_not_explode_when_below_critical_mass() {
        let mut board = Board::new(7, 7);
        board.set(3, 3, Cell::new(Some(0), 2));
        let result = apply_move(&board, 3, 3, 0, 1);
        assert_eq!(result.board.get(3, 3).count, 3);
        assert_eq!(result.steps.len(), 0);
    }

    // ---------- Move validation ----------

    #[test]
    fn empty_cell_is_valid_for_any_player() {
        let board = Board::new(7, 7);
        assert_eq!(is_valid_move(&board, 2, 2, 0), true);
        assert_eq!(is_valid_move(&board, 2, 2, 1), true);
    }

    #[test]
    fn own_cell_valid_opponent_cell_not() {
        let mut board = Board::new(7, 7);
        board.set(2, 2, Cell::new(Some(0), 1));
        assert_eq!(is_valid_move(&board, 2, 2, 0), true);
        assert_eq!(is_valid_move(&board, 2, 2, 1), false);
    }

    #[test]
    fn out_of_bounds_is_invalid() {
        let board = Board::new(7, 7);
        assert_eq!(is_valid_move(&board, -1, 0, 0), false);
        assert_eq!(is_valid_move(&board, 0, 7, 0), false);
    }

    #[test]
    fn cannot_play_on_opponent_owned_cell_via_play_move() {
        let mut state = mid_game(create_game(2, 7, 7));
        state.board.set(3, 3, Cell::new(Some(1), 1));
        let r = play_move(&state, 3, 3);
        assert_eq!(r.state.current_player_index, 0);
        assert_eq!(r.state.board.get(3, 3).count, 1);
    }

    // ---------- Turn order, elimination and winning ----------

    #[test]
    fn turn_advances_to_next_player_after_a_move() {
        let state = create_game(3, 7, 7);
        let r = play_move(&state, 3, 3);
        assert_eq!(r.state.current_player_index, 1);
    }

    #[test]
    fn board_dimensions_stay_7x7_by_default() {
        let state = create_game(2, 7, 7);
        assert_eq!(state.board.rows, 7);
        assert_eq!(state.board.cols, 7);
    }

    #[test]
    fn no_eliminations_before_every_player_has_had_one_turn() {
        let state = create_game(2, 7, 7);
        let r = play_move(&state, 0, 0);
        assert_eq!(r.state.game_over, false);
        assert_eq!(r.state.players[0].active, true);
        assert_eq!(r.state.players[1].active, true);
    }

    #[test]
    fn player_with_zero_cells_eliminated_once_all_players_moved() {
        let mut state = mid_game(create_game(2, 7, 7));
        state.board.set(5, 5, Cell::new(Some(1), 3));
        state.current_player_index = 1;
        state.total_moves = 1;
        let r = play_move(&state, 5, 5);
        assert_eq!(r.state.players[0].active, false);
        assert_eq!(r.state.game_over, true);
        assert_eq!(r.state.winner, Some(1));
    }

    #[test]
    fn full_game_last_player_standing_wins_only_when_truly_last() {
        let mut state = mid_game(create_game(2, 7, 7));
        state.board.set(0, 0, Cell::new(Some(0), 1));
        state.board.set(3, 3, Cell::new(Some(1), 3));
        state.board.set(3, 2, Cell::new(Some(0), 1));
        state.board.set(3, 4, Cell::new(Some(0), 1));
        state.board.set(2, 3, Cell::new(Some(0), 1));
        state.board.set(4, 3, Cell::new(Some(0), 1));
        state.current_player_index = 1;
        state.total_moves = 1;
        let r = play_move(&state, 3, 3);
        assert_eq!(r.state.players[0].active, true);
        assert_eq!(r.state.game_over, false);
    }

    #[test]
    fn p4_no_elimination_before_every_player_moved_once() {
        let mut state = create_game(4, 7, 7);
        state.total_moves = 2;
        state.current_player_index = 0;
        let r = play_move(&state, 0, 0);
        assert_eq!(r.state.players[1].active, true);
        assert_eq!(r.state.game_over, false);
    }

    #[test]
    fn p4_turn_order_skips_eliminated_player() {
        let mut state = mid_game(create_game(4, 7, 7));
        state.board.set(3, 3, Cell::new(Some(0), 3));
        state.board.set(2, 3, Cell::new(Some(1), 1));
        state.board.set(6, 6, Cell::new(Some(2), 1));
        state.board.set(0, 6, Cell::new(Some(3), 1));
        state.board.set(0, 0, Cell::new(Some(0), 1));
        state.current_player_index = 0;
        state.total_moves = 4;

        let r1 = play_move(&state, 3, 3);
        assert_eq!(r1.state.players[1].active, false);
        assert_eq!(r1.state.game_over, false);
        assert_eq!(r1.state.current_player_index, 2);

        let r2 = play_move(&r1.state, 6, 5);
        assert_eq!(r2.state.current_player_index, 3);
        let r3 = play_move(&r2.state, 0, 5);
        assert_eq!(r3.state.current_player_index, 0);
    }

    #[test]
    fn p4_single_multiwave_cascade_eliminates_three_opponents() {
        let mut state = mid_game(create_game(4, 7, 7));
        state.board.set(3, 3, Cell::new(Some(0), 3));
        state.board.set(0, 0, Cell::new(Some(0), 1));
        state.board.set(2, 3, Cell::new(Some(1), 1));
        state.board.set(3, 4, Cell::new(Some(1), 1));
        state.board.set(4, 3, Cell::new(Some(2), 3));
        state.board.set(3, 2, Cell::new(Some(3), 1));
        state.board.set(5, 3, Cell::new(Some(3), 1));
        state.current_player_index = 0;
        state.total_moves = 3;

        let r = play_move(&state, 3, 3);
        assert!(r.steps.len() >= 2);
        assert_eq!(count_cells_for_player(&r.state.board, 1), 0);
        assert_eq!(count_cells_for_player(&r.state.board, 2), 0);
        assert_eq!(count_cells_for_player(&r.state.board, 3), 0);
        assert!(count_cells_for_player(&r.state.board, 0) > 0);
        for &(rr, cc) in &[(2, 3), (3, 4), (3, 2), (5, 3)] {
            assert_eq!(r.state.board.get(rr, cc).owner, Some(0));
        }
        assert_eq!(r.state.board.get(4, 3).owner, None);
        assert_eq!(r.state.game_over, true);
        assert_eq!(r.state.winner, Some(0));
    }

    #[test]
    fn p4_game_does_not_falsely_end_while_two_or_more_hold_cells() {
        let mut state = mid_game(create_game(4, 7, 7));
        state.board.set(3, 3, Cell::new(Some(0), 3));
        state.board.set(2, 3, Cell::new(Some(1), 1));
        state.board.set(6, 6, Cell::new(Some(2), 1));
        state.board.set(0, 6, Cell::new(Some(3), 1));
        state.current_player_index = 0;
        state.total_moves = 4;
        let r = play_move(&state, 3, 3);
        assert_eq!(r.state.players[1].active, false);
        assert_eq!(r.state.players[2].active, true);
        assert_eq!(r.state.players[3].active, true);
        assert_eq!(r.state.game_over, false);
        assert_eq!(r.state.winner, None);
    }

    // ---------- Placement rules ----------

    #[test]
    fn after_first_move_next_player_can_place_on_any_empty_cell() {
        let state = create_game(2, 7, 7);
        let r1 = play_move(&state, 0, 0);
        assert_eq!(r1.state.current_player_index, 1);
        assert_eq!(is_valid_move(&r1.state.board, 5, 5, 1), true);
        let r2 = play_move(&r1.state, 5, 5);
        assert_eq!(r2.state.board.get(5, 5).owner, Some(1));
        assert_eq!(r2.state.board.get(5, 5).count, 3);
        assert_eq!(r2.state.current_player_index, 0);
        let r3 = play_move(&r2.state, 2, 6);
        assert_eq!(r3.state.board.get(2, 6).owner, Some(0));
        assert_eq!(r3.state.board.get(2, 6).count, 1);
    }

    // ---------- Opening-move rule ----------

    #[test]
    fn opening_move_first_placement_puts_3_dots() {
        let state = create_game(2, 7, 7);
        assert_eq!(placement_dots(&state, 0), 3);
        let r = play_move(&state, 3, 3);
        assert_eq!(r.state.board.get(3, 3).count, 3);
        assert_eq!(r.state.board.get(3, 3).owner, Some(0));
        assert_eq!(r.steps.len(), 0);
    }

    #[test]
    fn opening_3_dots_never_detonates_even_on_corner() {
        let state = create_game(2, 7, 7);
        let r = play_move(&state, 0, 6);
        assert_eq!(r.steps.len(), 0);
        assert_eq!(r.state.board.get(0, 6).count, 3);
        assert_eq!(r.state.board.get(0, 6).owner, Some(0));
        assert_eq!(r.state.board.get(1, 6).owner, None);
        assert_eq!(r.state.board.get(0, 5).owner, None);
    }

    #[test]
    fn opening_bonus_is_per_player() {
        let mut s = create_game(4, 7, 7);
        let spots = [(1usize, 1usize), (1, 4), (4, 1), (4, 4)];
        for i in 0..4 {
            assert_eq!(placement_dots(&s, i), 3);
            let res = play_move(&s, spots[i].0, spots[i].1);
            assert_eq!(res.state.board.get(spots[i].0, spots[i].1).count, 3);
            assert_eq!(res.state.board.get(spots[i].0, spots[i].1).owner, Some(i as u8));
            s = res.state;
        }
        for j in 0..4 {
            assert_eq!(placement_dots(&s, j), 1);
        }
    }

    #[test]
    fn after_opening_later_moves_add_exactly_1_dot() {
        let state = create_game(2, 7, 7);
        let r1 = play_move(&state, 1, 1);
        assert_eq!(r1.state.board.get(1, 1).count, 3);
        let r2 = play_move(&r1.state, 4, 4);
        assert_eq!(r2.state.board.get(4, 4).count, 3);
        assert_eq!(placement_dots(&r2.state, 0), 1);
        let r3 = play_move(&r2.state, 5, 1);
        assert_eq!(r3.state.board.get(5, 1).count, 1);
        let r4 = play_move(&r3.state, 4, 4); // player 1 tops up their own opening cell
        assert_eq!(r4.state.board.get(4, 4).count, 0);
        assert!(r4.steps.len() >= 1);
    }
}
