//! Rust port of mcts.py + batched_mcts.py + selfplay.py's self-play driver:
//! PUCT search with vector-valued (per-absolute-player-id) backup for this
//! general-sum, multi-player game, preserving the lazy child-expansion
//! optimization (a child's game state is only materialized - via
//! `GameState::play_move` - the first time it is actually selected during a
//! descent, not when its parent is expanded).
//!
//! Deliberately generic over `ForwardFn` (a plain Rust closure, not a PyO3
//! type) so this whole module is testable with `cargo test` alone, no Python
//! required - only the thin adapter in pybindings.rs needs pyo3, to turn a
//! Python callable into a `ForwardFn`.

use rand::Rng;
use rand_distr::{Dirichlet, Distribution};

use crate::encoding::{action_to_relative_owner_perm, encode_state, outcome_values, MAX_PLAYERS};
use crate::game::{self, GameState, COLS, ROWS};

pub const ACTION_DIM: usize = ROWS * COLS;

/// Given a flattened batch of `k` encoded states (length k * NUM_PLANES *
/// ROWS * COLS), returns (policy_logits flattened k*ACTION_DIM, values
/// flattened k*MAX_PLAYERS). This is exactly the same contract as
/// batched_mcts.ForwardFn in the Python code, just spelled as a Rust closure
/// instead of a Python callable that wraps a PyTorch model.
pub type ForwardFn<'a> = dyn FnMut(&[f32], usize) -> (Vec<f32>, Vec<f32>) + 'a;

struct MctsNode {
    state: Option<GameState>,
    parent_and_action: Option<(usize, usize)>, // (parent arena idx, action = row*COLS+col)
    mover: Option<usize>,
    prior: f32,
    children: Vec<(usize, usize)>, // (action, child arena idx)
    visit_count: u32,
    value_sum: [f64; MAX_PLAYERS],
    is_expanded: bool,
}

/// One game's search tree, stored as an arena (Vec<MctsNode>) - index 0 is
/// always the root. Arena-of-indices avoids the aliasing issues an
/// owned/boxed recursive tree would hit under Rust's borrow checker when a
/// child needs to read its (already-materialized) parent's state to
/// lazily compute its own.
pub struct Tree {
    nodes: Vec<MctsNode>,
}

impl Tree {
    pub fn new_root(state: GameState) -> Self {
        let mover = state.current_player_index;
        Tree {
            nodes: vec![MctsNode {
                state: Some(state),
                parent_and_action: None,
                mover: Some(mover),
                prior: 1.0,
                children: vec![],
                visit_count: 0,
                value_sum: [0.0; MAX_PLAYERS],
                is_expanded: false,
            }],
        }
    }

    pub fn root_state(&self) -> &GameState {
        self.nodes[0].state.as_ref().unwrap()
    }

    fn materialize(&mut self, idx: usize) {
        if self.nodes[idx].state.is_some() {
            return;
        }
        let (parent_idx, action) = self.nodes[idx].parent_and_action.unwrap();
        let parent_state = self.nodes[parent_idx].state.clone().unwrap();
        let row = action / COLS;
        let col = action % COLS;
        let result = game::play_move(&parent_state, row, col);
        self.nodes[idx].mover = Some(result.state.current_player_index);
        self.nodes[idx].state = Some(result.state);
    }

    /// Descends via PUCT until an unexpanded or terminal node, materializing
    /// the leaf's state before returning (mirrors mcts.py's `Node.env`
    /// property being touched at exactly this point - lazily, on first need).
    pub fn select_leaf(&mut self, c_puct: f64) -> (usize, Vec<usize>) {
        let mut idx = 0usize;
        let mut path = vec![0usize];
        loop {
            if !self.nodes[idx].is_expanded {
                break;
            }
            if self.nodes[idx].state.as_ref().unwrap().game_over {
                break;
            }
            if self.nodes[idx].children.is_empty() {
                break;
            }

            let mover = self.nodes[idx].mover.unwrap();
            let sqrt_total = (self.nodes[idx].visit_count.max(1) as f64).sqrt();

            let mut best_pos = 0usize;
            let mut best_score = f64::NEG_INFINITY;
            for (pos, &(_action, child_idx)) in self.nodes[idx].children.iter().enumerate() {
                let child = &self.nodes[child_idx];
                let q = if child.visit_count > 0 {
                    child.value_sum[mover] / child.visit_count as f64
                } else {
                    0.0
                };
                let u = c_puct * child.prior as f64 * sqrt_total / (1.0 + child.visit_count as f64);
                let score = q + u;
                if score > best_score {
                    best_score = score;
                    best_pos = pos;
                }
            }
            let (_, next_idx) = self.nodes[idx].children[best_pos];
            idx = next_idx;
            path.push(idx);
        }
        self.materialize(idx);
        (idx, path)
    }

    pub fn is_terminal(&self, idx: usize) -> bool {
        self.nodes[idx].state.as_ref().unwrap().game_over
    }

    pub fn encode_leaf(&self, idx: usize) -> Vec<f32> {
        encode_state(self.nodes[idx].state.as_ref().unwrap())
    }

    pub fn terminal_value(&self, idx: usize) -> [f64; MAX_PLAYERS] {
        let ov = outcome_values(self.nodes[idx].state.as_ref().unwrap());
        let mut out = [0f64; MAX_PLAYERS];
        for i in 0..MAX_PLAYERS {
            out[i] = ov[i] as f64;
        }
        out
    }

    /// Expands `idx` (already materialized, not terminal, not yet expanded)
    /// given its raw network outputs, and returns the absolute-player-id
    /// value vector to back up the path with.
    pub fn expand(&mut self, idx: usize, policy_logits: &[f32], value_rel: &[f32]) -> [f64; MAX_PLAYERS] {
        let state = self.nodes[idx].state.clone().unwrap();
        let player = state.current_player_index as u8;

        let has_moved = state.players[player as usize].has_moved;
        let mut legal_actions = Vec::new();
        for r in 0..state.rows {
            for c in 0..state.cols {
                if game::is_valid_move(&state.board, r as i32, c as i32, player, has_moved) {
                    legal_actions.push(r * state.cols + c);
                }
            }
        }

        let mut max_logit = f32::NEG_INFINITY;
        for &a in &legal_actions {
            if policy_logits[a] > max_logit {
                max_logit = policy_logits[a];
            }
        }
        let mut probs = vec![0f32; legal_actions.len()];
        let mut sum = 0f32;
        for (i, &a) in legal_actions.iter().enumerate() {
            let p = (policy_logits[a] - max_logit).exp();
            probs[i] = p;
            sum += p;
        }
        if sum > 0.0 {
            for p in probs.iter_mut() {
                *p /= sum;
            }
        } else {
            let u = 1.0 / (legal_actions.len().max(1) as f32);
            for p in probs.iter_mut() {
                *p = u;
            }
        }

        for (i, &action) in legal_actions.iter().enumerate() {
            let child_idx = self.nodes.len();
            self.nodes.push(MctsNode {
                state: None,
                parent_and_action: Some((idx, action)),
                mover: None,
                prior: probs[i],
                children: vec![],
                visit_count: 0,
                value_sum: [0.0; MAX_PLAYERS],
                is_expanded: false,
            });
            self.nodes[idx].children.push((action, child_idx));
        }
        self.nodes[idx].is_expanded = true;

        let perm = action_to_relative_owner_perm(&state);
        let n = state.players.len();
        let mut abs_value = [0f64; MAX_PLAYERS];
        for pid in 0..n {
            abs_value[pid] = value_rel[perm[pid]] as f64;
        }
        abs_value
    }

    pub fn backup(&mut self, path: &[usize], value: [f64; MAX_PLAYERS]) {
        for &idx in path {
            let node = &mut self.nodes[idx];
            node.visit_count += 1;
            for i in 0..MAX_PLAYERS {
                node.value_sum[i] += value[i];
            }
        }
    }

    pub fn add_root_dirichlet_noise(&mut self, alpha: f64, epsilon: f64, rng: &mut impl Rng) {
        let children = self.nodes[0].children.clone();
        // rand_distr::Dirichlet requires >=2 categories; with 0 or 1 legal
        // actions there's nothing for noise to do anyway (a single action
        // gets selected regardless of its prior), so skip - matches the
        // Python version's behavior (np.random.dirichlet trivially returns
        // [1.0] for a single category, which blends to a no-op outcome).
        if children.len() < 2 {
            return;
        }
        let dirichlet = Dirichlet::new(&vec![alpha; children.len()]).unwrap();
        let noise: Vec<f64> = dirichlet.sample(rng);
        for (i, &(_, child_idx)) in children.iter().enumerate() {
            let child = &mut self.nodes[child_idx];
            child.prior = (child.prior as f64 * (1.0 - epsilon) + noise[i] * epsilon) as f32;
        }
    }

    /// Visit-count-derived policy target over all ACTION_DIM actions
    /// (temperature 1.0, matching the Python driver's fixed training-target
    /// temperature - move-selection temperature is handled by the caller).
    pub fn visit_count_policy(&self) -> Vec<f32> {
        let mut pi = vec![0f32; ACTION_DIM];
        let children = &self.nodes[0].children;
        if children.is_empty() {
            return pi;
        }
        let total: u32 = children.iter().map(|&(_, idx)| self.nodes[idx].visit_count).sum();
        if total == 0 {
            return pi;
        }
        for &(action, idx) in children {
            pi[action] = self.nodes[idx].visit_count as f32 / total as f32;
        }
        pi
    }
}

/// Runs MCTS on every tree in `trees` concurrently, mutating them in place,
/// batching all leaf evaluations for a round into one forward_fn call - the
/// direct Rust equivalent of batched_mcts.run_batched_mcts. Every tree must
/// be freshly constructed (Tree::new_root), unexpanded, non-terminal.
pub fn run_batched_mcts(
    trees: &mut [Tree],
    forward_fn: &mut ForwardFn,
    num_simulations: usize,
    c_puct: f64,
    dirichlet_alpha: f64,
    dirichlet_epsilon: f64,
    add_root_noise: bool,
    rng: &mut impl Rng,
) {
    let active: Vec<usize> = (0..trees.len()).filter(|&i| !trees[i].root_state().game_over).collect();
    if active.is_empty() {
        return;
    }

    // Round 0: expand every root itself.
    let mut states_flat = Vec::new();
    for &i in &active {
        states_flat.extend(encode_state(trees[i].root_state()));
    }
    let (policy_flat, values_flat) = forward_fn(&states_flat, active.len());
    for (pos, &i) in active.iter().enumerate() {
        let policy_slice = &policy_flat[pos * ACTION_DIM..(pos + 1) * ACTION_DIM];
        let value_slice = &values_flat[pos * MAX_PLAYERS..(pos + 1) * MAX_PLAYERS];
        let abs_value = trees[i].expand(0, policy_slice, value_slice);
        trees[i].backup(&[0], abs_value);
        if add_root_noise {
            trees[i].add_root_dirichlet_noise(dirichlet_alpha, dirichlet_epsilon, rng);
        }
    }

    for _ in 0..num_simulations {
        let mut pending_leaf: Vec<(usize, usize, Vec<usize>)> = Vec::new(); // (tree idx, leaf idx, path)
        let mut terminal_backups: Vec<(usize, Vec<usize>, [f64; MAX_PLAYERS])> = Vec::new();

        for &i in &active {
            let (leaf_idx, path) = trees[i].select_leaf(c_puct);
            if trees[i].is_terminal(leaf_idx) {
                let v = trees[i].terminal_value(leaf_idx);
                terminal_backups.push((i, path, v));
            } else {
                pending_leaf.push((i, leaf_idx, path));
            }
        }

        for (i, path, v) in terminal_backups {
            trees[i].backup(&path, v);
        }

        if !pending_leaf.is_empty() {
            let mut states_flat = Vec::new();
            for (i, leaf_idx, _) in &pending_leaf {
                states_flat.extend(trees[*i].encode_leaf(*leaf_idx));
            }
            let (policy_flat, values_flat) = forward_fn(&states_flat, pending_leaf.len());
            for (pos, (i, leaf_idx, path)) in pending_leaf.iter().enumerate() {
                let policy_slice = &policy_flat[pos * ACTION_DIM..(pos + 1) * ACTION_DIM];
                let value_slice = &values_flat[pos * MAX_PLAYERS..(pos + 1) * MAX_PLAYERS];
                let abs_value = trees[*i].expand(*leaf_idx, policy_slice, value_slice);
                trees[*i].backup(path, abs_value);
            }
        }
    }
}

// ---- Self-play driver: port of selfplay.py's run_batched_selfplay_loop ----

pub struct TrainingExampleRs {
    pub state: Vec<f32>,
    pub policy: Vec<f32>,
    pub value: [f32; MAX_PLAYERS],
}

struct PlyRecord {
    state: Vec<f32>,
    policy: Vec<f32>,
    mover: usize,
    n_players: usize,
}

struct SelfplaySlot {
    state: GameState,
    records: Vec<PlyRecord>,
    move_count: usize,
}

impl SelfplaySlot {
    fn new(n_players: usize) -> Self {
        SelfplaySlot { state: game::create_game(n_players, ROWS, COLS), records: vec![], move_count: 0 }
    }
}

fn draw_fallback_outcome(state: &GameState) -> [f64; MAX_PLAYERS] {
    let mut v = [0f64; MAX_PLAYERS];
    for p in &state.players {
        v[p.id as usize] = if p.active { 0.0 } else { -1.0 };
    }
    v
}

fn finalize_game(slot: SelfplaySlot, abs_outcome: [f64; MAX_PLAYERS]) -> Vec<TrainingExampleRs> {
    let mut examples = Vec::with_capacity(slot.records.len());
    for rec in slot.records {
        let mut value = [0f32; MAX_PLAYERS];
        for k in 0..rec.n_players {
            let src = (k + rec.mover) % rec.n_players;
            value[k] = abs_outcome[src] as f32;
        }
        examples.push(TrainingExampleRs { state: rec.state, policy: rec.policy, value });
    }
    examples
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

fn sample_categorical(pi: &[f32], rng: &mut impl Rng) -> usize {
    let r: f32 = rng.gen::<f32>();
    let mut cum = 0f32;
    for (i, &p) in pi.iter().enumerate() {
        cum += p;
        if r < cum {
            return i;
        }
    }
    pi.len() - 1
}

fn random_legal_action(state: &GameState, rng: &mut impl Rng) -> usize {
    let player = state.current_player_index as u8;
    let has_moved = state.players[player as usize].has_moved;
    let mut legal = Vec::new();
    for r in 0..state.rows {
        for c in 0..state.cols {
            if game::is_valid_move(&state.board, r as i32, c as i32, player, has_moved) {
                legal.push(r * state.cols + c);
            }
        }
    }
    legal[rng.gen_range(0..legal.len())]
}

/// Runs `num_games` self-play games (up to `batch_size` concurrently),
/// batching every game's pending MCTS leaf into one forward_fn call per
/// search round - the Rust equivalent of
/// selfplay.run_batched_selfplay_loop + batched_mcts.run_batched_mcts
/// combined. Finished slots are immediately replaced with a fresh game so
/// the batch stays full until `num_games` total games are collected.
#[allow(clippy::too_many_arguments)]
pub fn run_batched_selfplay(
    forward_fn: &mut ForwardFn,
    num_games: usize,
    num_simulations: usize,
    batch_size: usize,
    player_counts: &[usize],
    temperature_moves: usize,
    max_moves: usize,
    c_puct: f64,
    dirichlet_alpha: f64,
    dirichlet_epsilon: f64,
    rng: &mut impl Rng,
) -> Vec<Vec<TrainingExampleRs>> {
    let batch_size = batch_size.min(num_games).max(1);
    let mut slots: Vec<Option<SelfplaySlot>> = (0..batch_size)
        .map(|_| Some(SelfplaySlot::new(player_counts[rng.gen_range(0..player_counts.len())])))
        .collect();

    let mut completed: Vec<Vec<TrainingExampleRs>> = Vec::new();

    while completed.len() < num_games {
        let live_indices: Vec<usize> = (0..batch_size).filter(|&i| slots[i].is_some()).collect();
        if live_indices.is_empty() {
            break;
        }

        let mut trees: Vec<Tree> =
            live_indices.iter().map(|&i| Tree::new_root(slots[i].as_ref().unwrap().state.clone())).collect();

        run_batched_mcts(
            &mut trees, forward_fn, num_simulations, c_puct, dirichlet_alpha, dirichlet_epsilon, true, rng,
        );

        for (pos, &i) in live_indices.iter().enumerate() {
            let tree = &trees[pos];
            let pi = tree.visit_count_policy();
            let state_ref = tree.root_state();
            let encoded = encode_state(state_ref);
            let mover = state_ref.current_player_index;
            let n_players = state_ref.players.len();

            let slot = slots[i].as_mut().unwrap();
            slot.records.push(PlyRecord { state: encoded, policy: pi.clone(), mover, n_players });

            let use_greedy = slot.move_count >= temperature_moves;
            let pi_sum: f32 = pi.iter().sum();
            let action = if use_greedy || pi_sum <= 0.0 {
                if pi_sum > 0.0 { argmax(&pi) } else { random_legal_action(&slot.state, rng) }
            } else {
                sample_categorical(&pi, rng)
            };

            let row = action / COLS;
            let col = action % COLS;
            let result = game::play_move(&slot.state, row, col);
            slot.state = result.state;
            slot.move_count += 1;

            let finished = slot.state.game_over || slot.move_count >= max_moves;
            if finished {
                let abs_outcome = if slot.state.game_over {
                    let ov = outcome_values(&slot.state);
                    let mut v = [0f64; MAX_PLAYERS];
                    for k in 0..MAX_PLAYERS {
                        v[k] = ov[k] as f64;
                    }
                    v
                } else {
                    draw_fallback_outcome(&slot.state)
                };
                let finished_slot = slots[i].take().unwrap();
                completed.push(finalize_game(finished_slot, abs_outcome));

                if completed.len() < num_games {
                    slots[i] = Some(SelfplaySlot::new(player_counts[rng.gen_range(0..player_counts.len())]));
                }
            }
        }
    }

    completed.truncate(num_games);
    completed
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    /// A dummy forward_fn: uniform policy logits (0.0 everywhere - masked
    /// softmax over legal moves makes this uniform-over-legal-moves), zero
    /// value everywhere. No Python/PyTorch involved - this is pure
    /// structural verification that the tree search and self-play loop
    /// terminate correctly and produce well-formed output.
    fn dummy_forward_fn(states: &[f32], k: usize) -> (Vec<f32>, Vec<f32>) {
        let _ = states;
        (vec![0f32; k * ACTION_DIM], vec![0f32; k * MAX_PLAYERS])
    }

    #[test]
    fn single_tree_mcts_runs_and_expands() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let state = game::create_game(2, ROWS, COLS);
        let mut trees = vec![Tree::new_root(state)];
        let mut ff: Box<ForwardFn> = Box::new(dummy_forward_fn);
        run_batched_mcts(&mut trees, &mut *ff, 20, 1.5, 0.3, 0.25, true, &mut rng);
        let pi = trees[0].visit_count_policy();
        let sum: f32 = pi.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "policy should sum to 1, got {sum}");
    }

    #[test]
    fn root_dirichlet_noise_does_not_panic_with_a_single_legal_action() {
        // A board with only one empty/own cell for the mover: exactly 1
        // legal action at the root. rand_distr::Dirichlet panics on a
        // single-category distribution, so add_root_dirichlet_noise must
        // special-case this rather than blindly calling Dirichlet::new.
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let mut state = game::create_game(2, ROWS, COLS);
        // Fill every cell except (0,0) with an opponent-owned cell so the
        // mover (player 0) has exactly one legal move.
        for r in 0..ROWS {
            for c in 0..COLS {
                if (r, c) != (0, 0) {
                    let cell = state.board.get_mut(r, c);
                    cell.owner = Some(1);
                    cell.count = 1;
                }
            }
        }
        let mut trees = vec![Tree::new_root(state)];
        let mut ff: Box<ForwardFn> = Box::new(dummy_forward_fn);
        run_batched_mcts(&mut trees, &mut *ff, 5, 1.5, 0.3, 0.25, true, &mut rng);
        let pi = trees[0].visit_count_policy();
        assert!((pi.iter().sum::<f32>() - 1.0).abs() < 1e-4);
    }

    #[test]
    fn batched_selfplay_produces_valid_games() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        let mut ff: Box<ForwardFn> = Box::new(dummy_forward_fn);
        let games = run_batched_selfplay(
            &mut *ff, 6, 8, 4, &[2, 3, 4], 5, 250, 1.5, 0.3, 0.25, &mut rng,
        );
        assert_eq!(games.len(), 6);
        for game in &games {
            assert!(!game.is_empty());
            for ex in game {
                assert_eq!(ex.state.len(), crate::encoding::NUM_PLANES * ROWS * COLS);
                assert_eq!(ex.policy.len(), ACTION_DIM);
                let psum: f32 = ex.policy.iter().sum();
                assert!((psum - 1.0).abs() < 1e-3, "policy should sum to 1, got {psum}");
            }
        }
    }
}
