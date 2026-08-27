// State encoding for the neural network - a faithful JS port of
// python/colourwars/env.py's ColourWarsEnv.encode_state /
// action_to_relative_owner_perm. Kept separate from network.js/mcts.js so
// it can be cross-checked against the Python reference in isolation.
(function (root) {
  'use strict';

  var GL = root.GameLogic;
  var MAX_PLAYERS = 4;
  // MAX_PLAYERS owner-relative planes + 1 raw-count plane + 1 opening-bonus
  // plane + MAX_PLAYERS one-hot player-count planes.
  var NUM_PLANES = MAX_PLAYERS + 1 + 1 + MAX_PLAYERS; // 10

  // JS `%` is remainder (can be negative for a negative left operand);
  // Python's `%` is always non-negative for a positive divisor. This project
  // relies on the Python semantics (see env.py's `(cell.owner - me) % n`),
  // so this must be an explicit true-modulo, not raw `%`.
  function mod(a, n) {
    return ((a % n) + n) % n;
  }

  function relSlot(playerId, me, n) {
    return mod(playerId - me, n);
  }

  // Returns a flat Float32Array of length NUM_PLANES*rows*cols, row-major
  // within each plane (index = plane*rows*cols + r*cols + c) - matches
  // numpy's default (C, H, W) layout so the exported-weights convolution
  // math in network.js can assume the same indexing.
  function encodeState(state) {
    var rows = state.rows, cols = state.cols;
    var stride = rows * cols;
    var planes = new Float32Array(NUM_PLANES * stride);
    var me = state.currentPlayerIndex;
    var n = state.players.length;
    var cm = GL.CRITICAL_MASS;

    for (var r = 0; r < rows; r++) {
      for (var c = 0; c < cols; c++) {
        var cell = state.board[r][c];
        if (cell.owner !== null) {
          var rel = relSlot(cell.owner, me, n);
          var scaled = cell.count / cm;
          planes[rel * stride + r * cols + c] = scaled;
          planes[MAX_PLAYERS * stride + r * cols + c] = scaled;
        }
      }
    }

    var opening = GL.placementDots(state, me) > 1 ? 1.0 : 0.0;
    var openingStart = (MAX_PLAYERS + 1) * stride;
    for (var i = 0; i < stride; i++) planes[openingStart + i] = opening;

    var nPlaneIdx = MAX_PLAYERS + 2 + (n - 2);
    var nPlaneStart = nPlaneIdx * stride;
    for (var j = 0; j < stride; j++) planes[nPlaneStart + j] = 1.0;

    return planes;
  }

  // perm[playerId] = relative slot used by encodeState (rel = (id-me) mod n).
  // Used to map the network's mover-relative value output back to absolute
  // player ids during MCTS backup.
  function relativeOwnerPerm(state) {
    var me = state.currentPlayerIndex;
    var n = state.players.length;
    var perm = [0, 0, 0, 0];
    for (var pid = 0; pid < n; pid++) perm[pid] = relSlot(pid, me, n);
    return perm;
  }

  root.Encode = {
    MAX_PLAYERS: MAX_PLAYERS,
    NUM_PLANES: NUM_PLANES,
    mod: mod,
    encodeState: encodeState,
    relativeOwnerPerm: relativeOwnerPerm
  };
})(typeof window !== 'undefined' ? window : globalThis);
