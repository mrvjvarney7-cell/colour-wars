// PUCT MCTS for the AI's turn - a faithful JS port of
// python/colourwars/mcts.py's single-tree search (run_mcts), adapted for
// this general-sum multi-player game: value backup is a per-absolute-
// player-id vector, not a sign-flipped scalar. No Dirichlet root noise here
// (that's a training-time exploration device only - see mcts.py's
// add_root_noise) - this is inference/play, so it always picks the
// strongest move it can find (temperature 0, i.e. most-visited child).
//
// Lazily materializes child game states (via GameLogic.playMove) only the
// first time a child is actually selected during descent, exactly like
// mcts.py's Node - most of a freshly-expanded node's ~49 candidate children
// are never visited, so this avoids the large majority of board-cloning
// work eager expansion would otherwise pay for nothing.
//
// Uses ONLY GameLogic (js/gameLogic.js) for all rules/state - the same
// engine the human-vs-human UI already trusts, non-mutating by design, so
// running many search branches from one root state can never corrupt it.
(function (root) {
  'use strict';

  var GL = root.GameLogic;
  var ENC = root.Encode;
  var NN = root.NeuralNet;
  var MAX_PLAYERS = ENC.MAX_PLAYERS;

  function Node(state, prior, parentState, action) {
    this.state = state || null;
    this.parentState = parentState || null;
    this.action = (action === undefined) ? null : action;
    this.mover = state ? state.currentPlayerIndex : null;
    this.prior = prior;
    this.children = {};      // action(int) -> Node
    this.childActions = [];  // ordered action ints, for iteration
    this.visitCount = 0;
    this.valueSum = [0, 0, 0, 0];
    this.isExpanded = false;
  }

  // Lazily computes this node's game state from its parent + action the
  // first time it's needed, then caches it. GameLogic.playMove does not
  // mutate its input, so calling it here can never corrupt the parent.
  Node.prototype.getState = function () {
    if (this.state === null) {
      var cols = this.parentState.cols;
      var r = Math.floor(this.action / cols);
      var c = this.action % cols;
      var result = GL.playMove(this.parentState, r, c);
      this.state = result.state;
      this.mover = this.state.currentPlayerIndex;
      this.parentState = null; // no longer needed once materialized
    }
    return this.state;
  };

  function selectChildAction(node, cPuct) {
    var bestAction = null;
    var bestScore = -Infinity;
    var sqrtTotal = Math.sqrt(Math.max(node.visitCount, 1));
    var mover = node.mover;
    for (var i = 0; i < node.childActions.length; i++) {
      var a = node.childActions[i];
      var child = node.children[a];
      var q = child.visitCount > 0 ? child.valueSum[mover] / child.visitCount : 0;
      var u = cPuct * child.prior * sqrtTotal / (1 + child.visitCount);
      var score = q + u;
      if (score > bestScore) {
        bestScore = score;
        bestAction = a;
      }
    }
    return bestAction;
  }

  // Descends via PUCT until an unexpanded or terminal node - pure tree walk,
  // no network calls.
  function selectLeaf(rootNode, cPuct) {
    var node = rootNode;
    var path = [rootNode];
    while (node.isExpanded && !node.getState().gameOver && node.childActions.length > 0) {
      var action = selectChildAction(node, cPuct);
      node = node.children[action];
      path.push(node);
    }
    return { leaf: node, path: path };
  }

  function backup(path, value) {
    for (var i = 0; i < path.length; i++) {
      var node = path[i];
      node.visitCount++;
      for (var k = 0; k < MAX_PLAYERS; k++) node.valueSum[k] += value[k];
    }
  }

  function terminalValue(state) {
    var value = [0, 0, 0, 0];
    for (var i = 0; i < state.players.length; i++) {
      var p = state.players[i];
      value[p.id] = (p.id === state.winner) ? 1 : -1;
    }
    return value;
  }

  // Expands `node` (already materialized, not terminal, not yet expanded)
  // given its raw network outputs, and returns the absolute-player-id value
  // vector to back up the path with. Masked softmax over legal moves only,
  // exactly mirroring mcts.py's expand_leaf_with_output.
  function expandLeafWithOutput(node, policyLogits, valueRel) {
    var state = node.getState();
    var rows = state.rows, cols = state.cols;
    var player = state.currentPlayerIndex;

    var hasMoved = state.players[player].hasMoved;
    var legalActions = [];
    for (var r = 0; r < rows; r++) {
      for (var c = 0; c < cols; c++) {
        if (GL.isValidMove(state.board, r, c, player, hasMoved)) legalActions.push(r * cols + c);
      }
    }

    var maxLogit = -Infinity;
    for (var i = 0; i < legalActions.length; i++) {
      var lv = policyLogits[legalActions[i]];
      if (lv > maxLogit) maxLogit = lv;
    }
    var probs = new Array(legalActions.length);
    var sum = 0;
    for (i = 0; i < legalActions.length; i++) {
      var p = Math.exp(policyLogits[legalActions[i]] - maxLogit);
      probs[i] = p;
      sum += p;
    }
    if (sum > 0) {
      for (i = 0; i < probs.length; i++) probs[i] /= sum;
    } else {
      var uniform = 1 / Math.max(legalActions.length, 1);
      for (i = 0; i < probs.length; i++) probs[i] = uniform;
    }

    for (i = 0; i < legalActions.length; i++) {
      var action = legalActions[i];
      node.children[action] = new Node(null, probs[i], state, action);
      node.childActions.push(action);
    }
    node.isExpanded = true;

    var perm = ENC.relativeOwnerPerm(state);
    var n = state.players.length;
    var absValue = [0, 0, 0, 0];
    for (var pid = 0; pid < n; pid++) absValue[pid] = valueRel[perm[pid]];
    return absValue;
  }

  function evaluateLeaf(node, weights) {
    var state = node.getState();
    var encoded = ENC.encodeState(state);
    var out = NN.forward(encoded, weights);
    return expandLeafWithOutput(node, out.policyLogits, out.value);
  }

  // Runs PUCT search from rootState (a GameLogic game-state object) and
  // returns the root Node, with every descendant's visit counts/values
  // populated. rootState itself is never mutated (see the Node/getState
  // comments above).
  function runMcts(rootState, weights, numSimulations, cPuct) {
    cPuct = (cPuct === undefined) ? 1.5 : cPuct;
    var rootNode = new Node(rootState, 1.0);
    if (rootState.gameOver) return rootNode;

    var rootValue = evaluateLeaf(rootNode, weights);
    rootNode.visitCount = 1;
    for (var k = 0; k < MAX_PLAYERS; k++) rootNode.valueSum[k] = rootValue[k];

    for (var s = 0; s < numSimulations; s++) {
      var sel = selectLeaf(rootNode, cPuct);
      var leaf = sel.leaf, path = sel.path;
      var leafState = leaf.getState();
      var value = leafState.gameOver ? terminalValue(leafState) : evaluateLeaf(leaf, weights);
      backup(path, value);
    }
    return rootNode;
  }

  // Strongest-move selection for play (temperature 0): the child with the
  // most visits. Returns null if the root has no legal moves at all (a
  // pre-existing edge case in the ported rules, not something new here).
  function bestAction(rootNode) {
    var bestA = null;
    var bestVisits = -1;
    for (var i = 0; i < rootNode.childActions.length; i++) {
      var a = rootNode.childActions[i];
      var v = rootNode.children[a].visitCount;
      if (v > bestVisits) {
        bestVisits = v;
        bestA = a;
      }
    }
    return bestA;
  }

  // Summarizes a completed search for display: the mover's own estimated win
  // probability (their backed-up root Q-value, which reflects the whole
  // search, not just the raw network's single-glance output, remapped from
  // [-1,1] to [0,1]), and every legal move ranked by visit share - MCTS's
  // "how seriously did it consider this" signal, since PUCT spends more
  // simulations on moves it believes are stronger.
  function rootInsight(rootNode) {
    var mover = rootNode.mover;
    var totalChildVisits = 0;
    var moves = [];
    for (var i = 0; i < rootNode.childActions.length; i++) {
      var a = rootNode.childActions[i];
      var child = rootNode.children[a];
      totalChildVisits += child.visitCount;
      moves.push({ action: a, visitCount: child.visitCount });
    }
    moves.sort(function (x, y) { return y.visitCount - x.visitCount; });
    for (i = 0; i < moves.length; i++) {
      moves[i].share = totalChildVisits > 0 ? moves[i].visitCount / totalChildVisits : 0;
    }
    var winProbability = (mover !== null && rootNode.visitCount > 0)
      ? (rootNode.valueSum[mover] / rootNode.visitCount + 1) / 2
      : null;
    return { mover: mover, winProbability: winProbability, moves: moves };
  }

  root.MCTS = {
    Node: Node,
    selectLeaf: selectLeaf,
    backup: backup,
    terminalValue: terminalValue,
    expandLeafWithOutput: expandLeafWithOutput,
    runMcts: runMcts,
    bestAction: bestAction,
    rootInsight: rootInsight
  };
})(typeof window !== 'undefined' ? window : globalThis);
