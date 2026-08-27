pub mod encoding;
pub mod game;
pub mod mcts;

// A minimal PyO3 shim over ONLY the game engine is enabled here (behind the
// "python" feature) purely to support the mandatory fuzz cross-check against
// the existing Python engine - see pybindings.rs's module docs. MCTS and the
// full training-pipeline integration are added only after that cross-check
// passes with zero divergence.
#[cfg(feature = "python")]
mod pybindings;
