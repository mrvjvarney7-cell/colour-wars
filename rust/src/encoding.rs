//! Faithful Rust port of python/colourwars/env.py's ColourWarsEnv.encode_state
//! / action_to_relative_owner_perm / outcome_values. Kept in Rust (not called
//! back into Python) because it runs once per MCTS leaf per search round -
//! exactly the hot path this whole port exists to speed up.

use crate::game::{placement_dots, CRITICAL_MASS, GameState};

pub const MAX_PLAYERS: usize = 4;
/// MAX_PLAYERS owner-relative planes + 1 raw-count plane + 1 opening-bonus
/// plane + MAX_PLAYERS one-hot player-count planes.
pub const NUM_PLANES: usize = MAX_PLAYERS + 1 + 1 + MAX_PLAYERS;

/// Row-major (plane, row, col) flattened, matching numpy's default (C, H, W)
/// layout when reshaped in Python as `.reshape(NUM_PLANES, rows, cols)`.
pub fn encode_state(state: &GameState) -> Vec<f32> {
    let rows = state.rows;
    let cols = state.cols;
    let plane_stride = rows * cols;
    let mut planes = vec![0f32; NUM_PLANES * plane_stride];

    let me = state.current_player_index;
    let n = state.players.len();
    let cm = CRITICAL_MASS as f32;

    for r in 0..rows {
        for c in 0..cols {
            let cell = state.board.get(r, c);
            if let Some(owner) = cell.owner {
                let rel = ((owner as i64 - me as i64).rem_euclid(n as i64)) as usize;
                let scaled = cell.count as f32 / cm;
                planes[rel * plane_stride + r * cols + c] = scaled;
                planes[MAX_PLAYERS * plane_stride + r * cols + c] = scaled;
            }
        }
    }

    let opening = if placement_dots(state, me) > 1 { 1.0f32 } else { 0.0f32 };
    let opening_start = (MAX_PLAYERS + 1) * plane_stride;
    for i in 0..plane_stride {
        planes[opening_start + i] = opening;
    }

    let n_plane_idx = MAX_PLAYERS + 2 + (n - 2);
    let n_plane_start = n_plane_idx * plane_stride;
    for i in 0..plane_stride {
        planes[n_plane_start + i] = 1.0;
    }

    planes
}

/// perm[player_id] = relative slot used by encode_state (rel = (id - me) % n).
pub fn action_to_relative_owner_perm(state: &GameState) -> [usize; MAX_PLAYERS] {
    let me = state.current_player_index;
    let n = state.players.len();
    let mut perm = [0usize; MAX_PLAYERS];
    for pid in 0..n {
        perm[pid] = ((pid as i64 - me as i64).rem_euclid(n as i64)) as usize;
    }
    perm
}

/// +1 for the winner, -1 for every other (eliminated/losing) player, 0 for
/// padding slots beyond this game's player count. Only valid when game_over.
pub fn outcome_values(state: &GameState) -> [f32; MAX_PLAYERS] {
    debug_assert!(state.game_over);
    let mut values = [0f32; MAX_PLAYERS];
    for p in &state.players {
        values[p.id as usize] = if Some(p.id) == state.winner { 1.0 } else { -1.0 };
    }
    values
}
