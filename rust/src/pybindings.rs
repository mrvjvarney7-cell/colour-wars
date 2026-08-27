//! Minimal PyO3 exposure of ONLY the (already cargo-tested) game engine, used
//! exclusively by python/colourwars/tests/compare_rust_engine.py to fuzz-check
//! this Rust port against the existing Python engine before any MCTS port or
//! full training-pipeline integration happens. This is intentionally narrow:
//! no tree search, no self-play - just enough surface to drive identical
//! move sequences through both engines and diff every field after every ply.

use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::encoding;
use crate::game::{self, GameState};
use crate::mcts;

#[pyclass(name = "RustGameState")]
#[derive(Clone)]
pub struct PyGameState {
    pub(crate) inner: GameState,
}

#[pymethods]
impl PyGameState {
    #[getter]
    fn rows(&self) -> usize {
        self.inner.rows
    }

    #[getter]
    fn cols(&self) -> usize {
        self.inner.cols
    }

    #[getter]
    fn current_player_index(&self) -> usize {
        self.inner.current_player_index
    }

    #[getter]
    fn total_moves(&self) -> u32 {
        self.inner.total_moves
    }

    #[getter]
    fn game_over(&self) -> bool {
        self.inner.game_over
    }

    #[getter]
    fn winner(&self) -> Option<u8> {
        self.inner.winner
    }

    #[getter]
    fn num_players(&self) -> usize {
        self.inner.players.len()
    }

    fn players_active(&self) -> Vec<bool> {
        self.inner.players.iter().map(|p| p.active).collect()
    }

    fn players_has_moved(&self) -> Vec<bool> {
        self.inner.players.iter().map(|p| p.has_moved).collect()
    }

    /// Flat row-major list of owners, -1 for empty.
    fn board_owners(&self) -> Vec<i32> {
        let mut out = Vec::with_capacity(self.inner.rows * self.inner.cols);
        for r in 0..self.inner.rows {
            for c in 0..self.inner.cols {
                out.push(match self.inner.board.get(r, c).owner {
                    Some(o) => o as i32,
                    None => -1,
                });
            }
        }
        out
    }

    /// Flat row-major list of dot counts.
    fn board_counts(&self) -> Vec<i32> {
        let mut out = Vec::with_capacity(self.inner.rows * self.inner.cols);
        for r in 0..self.inner.rows {
            for c in 0..self.inner.cols {
                out.push(self.inner.board.get(r, c).count);
            }
        }
        out
    }

    fn is_valid_move(&self, row: i32, col: i32, player: u8) -> bool {
        game::is_valid_move(&self.inner.board, row, col, player)
    }

    fn legal_moves(&self) -> Vec<(usize, usize)> {
        let player = self.inner.current_player_index as u8;
        let mut out = Vec::new();
        for r in 0..self.inner.rows {
            for c in 0..self.inner.cols {
                if game::is_valid_move(&self.inner.board, r as i32, c as i32, player) {
                    out.push((r, c));
                }
            }
        }
        out
    }

    fn placement_dots(&self, player_id: usize) -> i32 {
        game::placement_dots(&self.inner, player_id)
    }

    /// Flat (NUM_PLANES * rows * cols,) encoding, for cross-checking against
    /// ColourWarsEnv.encode_state() before the Rust MCTS is built on top of it.
    fn encode_state(&self) -> Vec<f32> {
        encoding::encode_state(&self.inner)
    }
}

#[pyfunction]
fn create_game(num_players: usize, rows: usize, cols: usize) -> PyGameState {
    PyGameState { inner: game::create_game(num_players, rows, cols) }
}

/// Returns (new_state, num_explosion_waves).
#[pyfunction]
fn play_move(state: &PyGameState, row: usize, col: usize) -> (PyGameState, usize) {
    let result = game::play_move(&state.inner, row, col);
    let n_steps = result.steps.len();
    (PyGameState { inner: result.state }, n_steps)
}

/// One completed self-play game: three flat numpy arrays (states, policies,
/// values) plus the ply count needed to reshape them on the Python side.
#[pyclass(name = "RustGameRecord")]
pub struct PyGameRecord {
    #[pyo3(get)]
    states: Py<PyArray1<f32>>, // (n_plies * NUM_PLANES * rows * cols,)
    #[pyo3(get)]
    policies: Py<PyArray1<f32>>, // (n_plies * ACTION_DIM,)
    #[pyo3(get)]
    values: Py<PyArray1<f32>>, // (n_plies * MAX_PLAYERS,)
    #[pyo3(get)]
    n_plies: usize,
}

/// Runs `num_games` self-play games via Rust MCTS (batched leaf evaluation,
/// lazy child expansion), calling back into `forward_fn` once per search
/// round for the actual PyTorch network forward pass (which stays in
/// Python/PyTorch, per the project's design - only game rules and tree
/// search move to Rust).
///
/// `forward_fn(states: np.ndarray[k, NUM_PLANES, rows, cols]) -> (policy_logits[k, ACTION_DIM], values[k, MAX_PLAYERS])`,
/// both float32 C-contiguous numpy arrays.
#[pyfunction]
#[pyo3(signature = (
    forward_fn, num_games, num_simulations, batch_size, player_counts,
    temperature_moves=10, max_moves=300, c_puct=1.5, dirichlet_alpha=0.3,
    dirichlet_epsilon=0.25, seed=0
))]
#[allow(clippy::too_many_arguments)]
fn run_batched_selfplay_rust(
    py: Python<'_>,
    forward_fn: Bound<'_, PyAny>,
    num_games: usize,
    num_simulations: usize,
    batch_size: usize,
    player_counts: Vec<usize>,
    temperature_moves: usize,
    max_moves: usize,
    c_puct: f64,
    dirichlet_alpha: f64,
    dirichlet_epsilon: f64,
    seed: u64,
) -> PyResult<Vec<PyGameRecord>> {
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let mut closure = |states: &[f32], k: usize| -> (Vec<f32>, Vec<f32>) {
        let flat = PyArray1::from_slice_bound(py, states);
        let reshaped = flat
            .reshape([k, encoding::NUM_PLANES, game::ROWS, game::COLS])
            .expect("reshape of state batch failed");
        let result = forward_fn.call1((reshaped,)).expect("forward_fn call failed");
        let tuple: &Bound<PyTuple> =
            result.downcast().expect("forward_fn must return a (policy_logits, values) tuple");
        let policy_arr: PyReadonlyArray2<f32> =
            tuple.get_item(0).unwrap().extract().expect("policy_logits must be a float32 numpy 2D array");
        let value_arr: PyReadonlyArray2<f32> =
            tuple.get_item(1).unwrap().extract().expect("values must be a float32 numpy 2D array");
        let policy_vec = policy_arr.as_slice().expect("policy_logits must be C-contiguous").to_vec();
        let value_vec = value_arr.as_slice().expect("values must be C-contiguous").to_vec();
        (policy_vec, value_vec)
    };

    let games = mcts::run_batched_selfplay(
        &mut closure,
        num_games,
        num_simulations,
        batch_size,
        &player_counts,
        temperature_moves,
        max_moves,
        c_puct,
        dirichlet_alpha,
        dirichlet_epsilon,
        &mut rng,
    );

    let mut out = Vec::with_capacity(games.len());
    for game in games {
        let n_plies = game.len();
        let mut states_flat = Vec::with_capacity(n_plies * encoding::NUM_PLANES * game::ROWS * game::COLS);
        let mut policies_flat = Vec::with_capacity(n_plies * mcts::ACTION_DIM);
        let mut values_flat = Vec::with_capacity(n_plies * encoding::MAX_PLAYERS);
        for ex in game {
            states_flat.extend(ex.state);
            policies_flat.extend(ex.policy);
            values_flat.extend(ex.value);
        }
        out.push(PyGameRecord {
            states: states_flat.into_pyarray_bound(py).unbind(),
            policies: policies_flat.into_pyarray_bound(py).unbind(),
            values: values_flat.into_pyarray_bound(py).unbind(),
            n_plies,
        });
    }
    Ok(out)
}

#[pymodule]
fn colourwars_rs(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGameState>()?;
    m.add_class::<PyGameRecord>()?;
    m.add_function(wrap_pyfunction!(create_game, m)?)?;
    m.add_function(wrap_pyfunction!(play_move, m)?)?;
    m.add_function(wrap_pyfunction!(run_batched_selfplay_rust, m)?)?;
    Ok(())
}
