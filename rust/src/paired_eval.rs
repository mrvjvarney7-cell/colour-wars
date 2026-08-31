//! Paired-evaluation driver: plays a candidate network against an opponent
//! network over a fixed, pre-generated set of openings (each played twice,
//! candidate in both seats, to cancel first-move advantage), reusing
//! mcts.rs's Tree/run_batched_mcts primitives unchanged - nothing about
//! batched search assumes anything about who "owns" a tree.
//!
//! Eval's one real complication versus self-play: two DIFFERENT networks
//! alternate by seat within a single game. Handled by partitioning the live
//! (non-terminal) games each round by whose turn it is, and calling
//! run_batched_mcts once per network per round - mcts.rs itself needs no
//! changes. Move selection is always greedy (temperature 0, no root noise),
//! matching evaluate.py's _mcts_move exactly - eval must be deterministic,
//! unlike self-play's temperature-sampled exploration.
//!
//! Board canonicalisation (D4 symmetry, for the played-game distinctness
//! invariant) deliberately stays in Python (board_symmetry.py) - this
//! module only returns raw flat owner/count snapshots (game::flat_owners/
//! flat_counts) at the mid-game checkpoint and final position, so there is
//! exactly one implementation of canonicalisation, not two that could
//! silently drift apart.
//!
//! outcome_values/draw_fallback_outcome (encoding.rs/mcts.rs) are the wrong
//! tools here - both are MCTS value-backup conventions (+-1 per absolute
//! player id), not the candidate-centric 1.0/0.5/0.0 win/loss/draw scoring
//! evaluate.py's _play_paired_2p_games uses. Scored directly below instead.

use rand::SeedableRng;

use crate::game::{self, GameState, COLS, ROWS};
use crate::mcts::{run_batched_mcts, ForwardFn, Tree};

pub struct EvalGameResult {
    pub opening_index: usize,
    pub candidate_seat: usize,
    pub decided: bool,
    pub candidate_score: f64,
    pub move_count: usize,
    pub mid_owners: Option<Vec<i32>>,
    pub mid_counts: Option<Vec<i32>>,
    pub final_owners: Vec<i32>,
    pub final_counts: Vec<i32>,
}

struct EvalSlot {
    state: GameState,
    opening_index: usize,
    candidate_seat: usize,
    move_count: usize,
    mid_owners: Option<Vec<i32>>,
    mid_counts: Option<Vec<i32>>,
}

fn candidate_score(state: &GameState, candidate_seat: usize) -> (bool, f64) {
    if state.game_over {
        let won = state.winner == Some(candidate_seat as u8);
        (true, if won { 1.0 } else { 0.0 })
    } else {
        (false, 0.5)
    }
}

fn argmax(pi: &[f32]) -> usize {
    let mut best_i = 0;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in pi.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    best_i
}

/// Replays `actions` from a fresh 2-player game - no search, the opening is
/// fixed and pre-generated in Python (see evaluate.py's
/// _generate_distinct_random_openings). Captures the mid-checkpoint
/// snapshot immediately if the opening itself already reached it.
fn build_slot(opening_index: usize, candidate_seat: usize, actions: &[usize], mid_checkpoint_ply: usize) -> EvalSlot {
    let mut state = game::create_game(2, ROWS, COLS);
    for &a in actions {
        let row = a / COLS;
        let col = a % COLS;
        state = game::play_move(&state, row, col).state;
    }
    let move_count = actions.len();
    let (mid_owners, mid_counts) = if move_count >= mid_checkpoint_ply {
        (Some(game::flat_owners(&state.board)), Some(game::flat_counts(&state.board)))
    } else {
        (None, None)
    };
    EvalSlot { state, opening_index, candidate_seat, move_count, mid_owners, mid_counts }
}

fn finalize(slot: EvalSlot) -> EvalGameResult {
    let (decided, candidate_score) = candidate_score(&slot.state, slot.candidate_seat);
    EvalGameResult {
        opening_index: slot.opening_index,
        candidate_seat: slot.candidate_seat,
        decided,
        candidate_score,
        move_count: slot.move_count,
        mid_owners: slot.mid_owners,
        mid_counts: slot.mid_counts,
        final_owners: game::flat_owners(&slot.state.board),
        final_counts: game::flat_counts(&slot.state.board),
    }
}

/// Pulls the next (opening_index, candidate_seat) off the queue into
/// `slots[idx]`, replaying its opening. A slot that's already finished right
/// after opening replay (game over, or the opening alone already reaches
/// max_moves) is finalized immediately and the next queue item is tried,
/// rather than ever handing a terminal state to the round-partition step.
/// Leaves `slots[idx]` as None once the queue is exhausted.
#[allow(clippy::too_many_arguments)]
fn fill_slot(
    slots: &mut [Option<EvalSlot>],
    idx: usize,
    queue: &[(usize, usize)],
    next_queue_idx: &mut usize,
    openings: &[Vec<usize>],
    mid_checkpoint_ply: usize,
    max_moves: usize,
    completed: &mut Vec<EvalGameResult>,
) {
    while *next_queue_idx < queue.len() {
        let (opening_index, candidate_seat) = queue[*next_queue_idx];
        *next_queue_idx += 1;
        let slot = build_slot(opening_index, candidate_seat, &openings[opening_index], mid_checkpoint_ply);
        if slot.state.game_over || slot.move_count >= max_moves {
            completed.push(finalize(slot));
            continue;
        }
        slots[idx] = Some(slot);
        return;
    }
    slots[idx] = None;
}

fn apply_move_and_advance(slots: &mut [Option<EvalSlot>], i: usize, tree: &Tree, mid_checkpoint_ply: usize) {
    let pi = tree.visit_count_policy();
    let action = argmax(&pi);
    let row = action / COLS;
    let col = action % COLS;

    let slot = slots[i].as_mut().unwrap();
    let result = game::play_move(&slot.state, row, col);
    slot.state = result.state;
    slot.move_count += 1;
    if slot.mid_owners.is_none() && slot.move_count >= mid_checkpoint_ply {
        slot.mid_owners = Some(game::flat_owners(&slot.state.board));
        slot.mid_counts = Some(game::flat_counts(&slot.state.board));
    }
}

/// Runs every (opening, seat) pairing to completion (decided, or max_moves
/// reached), up to `batch_size` concurrently, batching each round's leaf
/// evaluations into one forward_fn call per network per round - the eval
/// equivalent of mcts::run_batched_selfplay. Output is exactly
/// `2 * openings.len()` records, opening-major then seat-major, each
/// carrying its own opening_index/candidate_seat so nothing depends on
/// ordering.
#[allow(clippy::too_many_arguments)]
pub fn run_batched_paired_eval(
    candidate_forward_fn: &mut ForwardFn,
    opponent_forward_fn: &mut ForwardFn,
    openings: &[Vec<usize>],
    num_simulations: usize,
    batch_size: usize,
    mid_checkpoint_ply: usize,
    max_moves: usize,
    c_puct: f64,
) -> Vec<EvalGameResult> {
    let mut queue: Vec<(usize, usize)> = Vec::with_capacity(openings.len() * 2);
    for opening_index in 0..openings.len() {
        queue.push((opening_index, 0));
        queue.push((opening_index, 1));
    }
    let total_games = queue.len();
    let mut next_queue_idx = 0usize;

    let mut completed: Vec<EvalGameResult> = Vec::with_capacity(total_games);

    // Never consumed - add_root_noise is always false below (eval is
    // deterministic given fixed networks+openings), but run_batched_mcts's
    // signature requires an Rng regardless.
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);

    let initial_batch = batch_size.min(total_games.max(1)).max(1);
    let mut slots: Vec<Option<EvalSlot>> = Vec::new();
    slots.resize_with(initial_batch, || None);
    for i in 0..initial_batch {
        fill_slot(&mut slots, i, &queue, &mut next_queue_idx, openings, mid_checkpoint_ply, max_moves, &mut completed);
    }

    while completed.len() < total_games {
        let live_indices: Vec<usize> = (0..slots.len()).filter(|&i| slots[i].is_some()).collect();
        if live_indices.is_empty() {
            break; // shouldn't happen: fill_slot only leaves a slot None once the queue is exhausted
        }

        let cand_indices: Vec<usize> = live_indices.iter().copied()
            .filter(|&i| {
                let s = slots[i].as_ref().unwrap();
                s.state.current_player_index == s.candidate_seat
            })
            .collect();
        let opp_indices: Vec<usize> = live_indices.iter().copied()
            .filter(|&i| {
                let s = slots[i].as_ref().unwrap();
                s.state.current_player_index != s.candidate_seat
            })
            .collect();

        let mut cand_trees: Vec<Tree> =
            cand_indices.iter().map(|&i| Tree::new_root(slots[i].as_ref().unwrap().state.clone())).collect();
        let mut opp_trees: Vec<Tree> =
            opp_indices.iter().map(|&i| Tree::new_root(slots[i].as_ref().unwrap().state.clone())).collect();

        if !cand_trees.is_empty() {
            run_batched_mcts(&mut cand_trees, candidate_forward_fn, num_simulations, c_puct, 0.3, 0.25, false, &mut rng);
        }
        if !opp_trees.is_empty() {
            run_batched_mcts(&mut opp_trees, opponent_forward_fn, num_simulations, c_puct, 0.3, 0.25, false, &mut rng);
        }

        for (pos, &i) in cand_indices.iter().enumerate() {
            apply_move_and_advance(&mut slots, i, &cand_trees[pos], mid_checkpoint_ply);
        }
        for (pos, &i) in opp_indices.iter().enumerate() {
            apply_move_and_advance(&mut slots, i, &opp_trees[pos], mid_checkpoint_ply);
        }

        for i in 0..slots.len() {
            let finished = match &slots[i] {
                Some(s) => s.state.game_over || s.move_count >= max_moves,
                None => false,
            };
            if finished {
                let slot = slots[i].take().unwrap();
                completed.push(finalize(slot));
                fill_slot(&mut slots, i, &queue, &mut next_queue_idx, openings, mid_checkpoint_ply, max_moves, &mut completed);
            }
        }
    }

    completed
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell as StdCell;

    fn dummy_forward_fn(_states: &[f32], k: usize) -> (Vec<f32>, Vec<f32>) {
        (vec![0f32; k * crate::mcts::ACTION_DIM], vec![0f32; k * crate::encoding::MAX_PLAYERS])
    }

    fn run_dummy(openings: &[Vec<usize>], num_simulations: usize, max_moves: usize) -> Vec<EvalGameResult> {
        let mut cand: Box<ForwardFn> = Box::new(dummy_forward_fn);
        let mut opp: Box<ForwardFn> = Box::new(dummy_forward_fn);
        run_batched_paired_eval(&mut *cand, &mut *opp, openings, num_simulations, 8, 20, max_moves, 1.5)
    }

    #[test]
    fn single_opening_produces_two_seat_records() {
        let openings = vec![vec![24usize, 17usize]]; // a couple of legal-looking opening plies
        let results = run_dummy(&openings, 4, 40);
        assert_eq!(results.len(), 2);
        let mut seats: Vec<usize> = results.iter().map(|r| r.candidate_seat).collect();
        seats.sort();
        assert_eq!(seats, vec![0, 1]);
        assert!(results.iter().all(|r| r.opening_index == 0));
    }

    #[test]
    fn candidate_score_matches_decided_and_move_count() {
        let openings = vec![vec![24, 17], vec![10, 30, 5]];
        let max_moves = 30;
        let results = run_dummy(&openings, 4, max_moves);
        for r in &results {
            assert!([0.0, 0.5, 1.0].contains(&r.candidate_score));
            if r.decided {
                assert!(r.candidate_score == 0.0 || r.candidate_score == 1.0);
            } else {
                assert_eq!(r.candidate_score, 0.5);
                assert_eq!(r.move_count, max_moves);
            }
        }
    }

    #[test]
    fn mid_checkpoint_captured_together_or_not_at_all() {
        let openings = vec![vec![24, 17], vec![10, 30, 5, 6, 7]];
        let results = run_dummy(&openings, 4, 40);
        for r in &results {
            assert_eq!(r.mid_owners.is_some(), r.mid_counts.is_some());
            if let Some(owners) = &r.mid_owners {
                assert_eq!(owners.len(), ROWS * COLS);
                assert_eq!(r.mid_counts.as_ref().unwrap().len(), ROWS * COLS);
                assert!(r.move_count >= 20);
            } else {
                assert!(r.move_count < 20);
            }
        }
    }

    #[test]
    fn final_board_is_always_legal() {
        let openings = vec![vec![24, 17], vec![], vec![10, 30, 5]];
        let results = run_dummy(&openings, 4, 40);
        for r in &results {
            assert_eq!(r.final_owners.len(), ROWS * COLS);
            assert_eq!(r.final_counts.len(), ROWS * COLS);
            for &c in &r.final_counts {
                assert!((0..game::CRITICAL_MASS).contains(&c), "illegal count {c}");
            }
        }
    }

    #[test]
    fn opening_replay_advances_move_count_correctly() {
        let opening = vec![24usize, 17usize, 32usize];
        let results = run_dummy(&[opening.clone()], 4, 40);
        for r in &results {
            assert!(r.move_count >= opening.len());
        }
    }

    #[test]
    fn both_networks_are_actually_exercised() {
        let cand_calls = StdCell::new(0usize);
        let opp_calls = StdCell::new(0usize);
        let mut cand_fn = |states: &[f32], k: usize| -> (Vec<f32>, Vec<f32>) {
            cand_calls.set(cand_calls.get() + 1);
            dummy_forward_fn(states, k)
        };
        let mut opp_fn = |states: &[f32], k: usize| -> (Vec<f32>, Vec<f32>) {
            opp_calls.set(opp_calls.get() + 1);
            dummy_forward_fn(states, k)
        };
        let openings = vec![vec![24usize, 17usize], vec![10usize, 30usize]];
        run_batched_paired_eval(&mut cand_fn, &mut opp_fn, &openings, 4, 8, 20, 40, 1.5);
        assert!(cand_calls.get() > 0, "candidate network was never called");
        assert!(opp_calls.get() > 0, "opponent network was never called");
    }

    #[test]
    fn deterministic_given_fixed_inputs() {
        let openings = vec![vec![24, 17], vec![10, 30, 5], vec![1, 2, 3, 4]];
        let a = run_dummy(&openings, 6, 30);
        let b = run_dummy(&openings, 6, 30);
        assert_eq!(a.len(), b.len());
        for (ra, rb) in a.iter().zip(b.iter()) {
            assert_eq!(ra.opening_index, rb.opening_index);
            assert_eq!(ra.candidate_seat, rb.candidate_seat);
            assert_eq!(ra.decided, rb.decided);
            assert_eq!(ra.candidate_score, rb.candidate_score);
            assert_eq!(ra.move_count, rb.move_count);
            assert_eq!(ra.final_owners, rb.final_owners);
            assert_eq!(ra.final_counts, rb.final_counts);
        }
    }

    #[test]
    fn opening_alone_reaching_max_moves_finalizes_without_any_search() {
        // max_moves smaller than the opening length forces the early-finalize
        // path in fill_slot (slot.move_count >= max_moves right after replay)
        // without ever calling run_batched_mcts on it - this exercises the
        // exact same branch a game_over-during-opening case would (the
        // condition is `state.game_over || move_count >= max_moves`), more
        // reliably than hand-scripting a real elimination sequence.
        let opening = vec![24usize, 17usize, 32usize, 25usize, 18usize];
        let results = run_dummy(&[opening.clone()], 4, 3);
        assert_eq!(results.len(), 2);
        for r in &results {
            assert_eq!(r.move_count, opening.len());
            assert!(!r.decided);
            assert_eq!(r.candidate_score, 0.5);
        }
    }
}
