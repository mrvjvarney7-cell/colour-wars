// Node test suite for gameLogic.js. Run with: node js/gameLogic.test.js
var assert = require('assert');
var GL = require('./gameLogic.js');

var passed = 0;
var failed = 0;

function test(name, fn) {
  try {
    fn();
    passed++;
    console.log('  ok - ' + name);
  } catch (err) {
    failed++;
    console.log('  FAIL - ' + name);
    console.log('    ' + err.message);
  }
}

// Tests that hand-build a mid-game board simulate players who have already
// opened, so clear the per-player opening-move bonus for them.
function midGame(state) {
  state.players.forEach(function (p) { p.hasMoved = true; });
  return state;
}

function boardFromSpec(spec) {
  // spec: array of rows, each row array of [owner, count] or null for empty
  return spec.map(function (row) {
    return row.map(function (c) {
      if (c === null) return { owner: null, count: 0 };
      return { owner: c[0], count: c[1] };
    });
  });
}

console.log('Critical mass calculations');
test('corner has critical mass 2', function () {
  assert.strictEqual(GL.getCriticalMass(0, 0, 7, 7), 2);
  assert.strictEqual(GL.getCriticalMass(0, 6, 7, 7), 2);
  assert.strictEqual(GL.getCriticalMass(6, 0, 7, 7), 2);
  assert.strictEqual(GL.getCriticalMass(6, 6, 7, 7), 2);
});
test('edge has critical mass 3', function () {
  assert.strictEqual(GL.getCriticalMass(0, 3, 7, 7), 3);
  assert.strictEqual(GL.getCriticalMass(3, 0, 7, 7), 3);
  assert.strictEqual(GL.getCriticalMass(6, 3, 7, 7), 3);
  assert.strictEqual(GL.getCriticalMass(3, 6, 7, 7), 3);
});
test('interior has critical mass 4', function () {
  assert.strictEqual(GL.getCriticalMass(3, 3, 7, 7), 4);
  assert.strictEqual(GL.getCriticalMass(1, 1, 7, 7), 4);
});

console.log('Basic move placement');
test('placing a dot on an empty cell sets owner and count', function () {
  var board = GL.createEmptyBoard(7, 7);
  var result = GL.applyMove(board, 3, 3, 0, 7, 7);
  assert.strictEqual(result.board[3][3].owner, 0);
  assert.strictEqual(result.board[3][3].count, 1);
  assert.strictEqual(result.steps.length, 0);
});
test('placing a dot on own cell increments count', function () {
  var board = GL.createEmptyBoard(7, 7);
  board[3][3] = { owner: 0, count: 1 };
  var result = GL.applyMove(board, 3, 3, 0, 7, 7);
  assert.strictEqual(result.board[3][3].count, 2);
});

console.log('Corner explosion (critical mass 2)');
test('corner cell explodes at 2 dots, splits to its 2 neighbours', function () {
  var board = GL.createEmptyBoard(7, 7);
  board[0][0] = { owner: 0, count: 1 };
  var result = GL.applyMove(board, 0, 0, 0, 7, 7);
  assert.strictEqual(result.board[0][0].count, 0);
  assert.strictEqual(result.board[0][0].owner, null);
  assert.strictEqual(result.board[0][1].count, 1);
  assert.strictEqual(result.board[0][1].owner, 0);
  assert.strictEqual(result.board[1][0].count, 1);
  assert.strictEqual(result.board[1][0].owner, 0);
  assert.strictEqual(result.steps.length, 1);
});

console.log('Edge explosion (critical mass 3)');
test('edge cell explodes at 3 dots, splits to its 3 neighbours', function () {
  var board = GL.createEmptyBoard(7, 7);
  board[0][3] = { owner: 1, count: 2 };
  var result = GL.applyMove(board, 0, 3, 1, 7, 7);
  assert.strictEqual(result.board[0][3].count, 0);
  assert.strictEqual(result.board[0][2].count, 1);
  assert.strictEqual(result.board[0][4].count, 1);
  assert.strictEqual(result.board[1][3].count, 1);
  [[0,2],[0,4],[1,3]].forEach(function (p) {
    assert.strictEqual(result.board[p[0]][p[1]].owner, 1);
  });
});

console.log('Interior explosion (critical mass 4)');
test('interior cell explodes at 4 dots, splits to its 4 neighbours', function () {
  var board = GL.createEmptyBoard(7, 7);
  board[3][3] = { owner: 0, count: 3 };
  var result = GL.applyMove(board, 3, 3, 0, 7, 7);
  assert.strictEqual(result.board[3][3].count, 0);
  var neighbours = [[2,3],[4,3],[3,2],[3,4]];
  neighbours.forEach(function (p) {
    assert.strictEqual(result.board[p[0]][p[1]].count, 1);
    assert.strictEqual(result.board[p[0]][p[1]].owner, 0);
  });
});

console.log('Capture on explosion');
test('exploding into an opponent cell captures it regardless of previous owner', function () {
  var board = GL.createEmptyBoard(7, 7);
  board[3][3] = { owner: 0, count: 3 };
  board[2][3] = { owner: 1, count: 1 }; // enemy cell, below its own critical mass of 4
  var result = GL.applyMove(board, 3, 3, 0, 7, 7);
  assert.strictEqual(result.board[2][3].owner, 0);
  assert.strictEqual(result.board[2][3].count, 2);
});

console.log('Chain reactions');
test('an explosion that pushes a neighbour past critical mass chains further', function () {
  // Center interior cell about to explode; one neighbour already at critical mass - 1.
  var board = GL.createEmptyBoard(7, 7);
  board[3][3] = { owner: 0, count: 3 };
  board[3][4] = { owner: 0, count: 3 }; // interior, critical mass 4; will receive 1 from (3,3) explosion -> explodes too
  var result = GL.applyMove(board, 3, 3, 0, 7, 7);
  assert.ok(result.steps.length >= 2, 'expected at least 2 waves, got ' + result.steps.length);
  // (3,3) exploded and gave 1 to (3,4), pushing it from 3->4, which then also explodes.
  assert.strictEqual(result.board[3][4].count, 0);
  assert.strictEqual(result.board[3][4].owner, null);
  // (3,4)'s neighbours should each have gained a dot and be owned by player 0.
  var farNeighbours = [[2,4],[4,4],[3,5]];
  farNeighbours.forEach(function (p) {
    assert.strictEqual(result.board[p[0]][p[1]].owner, 0);
    assert.strictEqual(result.board[p[0]][p[1]].count, 1);
  });
});

test('chain reaction can capture and flip multiple opponent cells across waves', function () {
  var board = GL.createEmptyBoard(7, 7);
  board[0][0] = { owner: 0, count: 1 }; // corner, cm=2, about to explode
  board[0][1] = { owner: 1, count: 2 }; // edge, cm=3, will receive 1 -> 3 -> explodes
  board[0][2] = { owner: 1, count: 1 }; // edge, will receive from (0,1)'s explosion
  var result = GL.applyMove(board, 0, 0, 0, 7, 7);
  // (0,0) explodes -> (0,1) gets +1 = 3 = cm, explodes -> gives to (0,0),(0,2),(1,1)
  assert.strictEqual(result.board[0][1].owner, null);
  assert.strictEqual(result.board[0][1].count, 0);
  assert.strictEqual(result.board[0][2].owner, 0);
  assert.strictEqual(result.board[0][2].count, 2);
  assert.strictEqual(result.board[1][1].owner, 0);
  assert.ok(result.steps.length >= 2);
});

test('does not explode when below critical mass', function () {
  var board = GL.createEmptyBoard(7, 7);
  board[3][3] = { owner: 0, count: 2 }; // interior cm=4
  var result = GL.applyMove(board, 3, 3, 0, 7, 7);
  assert.strictEqual(result.board[3][3].count, 3);
  assert.strictEqual(result.steps.length, 0);
});

console.log('isValidMove');
test('empty cell is valid for any player', function () {
  var board = GL.createEmptyBoard(7, 7);
  assert.strictEqual(GL.isValidMove(board, 2, 2, 0), true);
  assert.strictEqual(GL.isValidMove(board, 2, 2, 1), true);
});
test('own cell is valid, opponent cell is not', function () {
  var board = GL.createEmptyBoard(7, 7);
  board[2][2] = { owner: 0, count: 1 };
  assert.strictEqual(GL.isValidMove(board, 2, 2, 0), true);
  assert.strictEqual(GL.isValidMove(board, 2, 2, 1), false);
});
test('out of bounds is invalid', function () {
  var board = GL.createEmptyBoard(7, 7);
  assert.strictEqual(GL.isValidMove(board, -1, 0, 0), false);
  assert.strictEqual(GL.isValidMove(board, 0, 7, 0), false);
});

console.log('Game state: turn order and elimination');
test('turn advances to next player after a move', function () {
  var state = GL.createGame(3);
  var r = GL.playMove(state, 3, 3);
  assert.strictEqual(r.state.currentPlayerIndex, 1);
});

test('no eliminations occur before every player has had one turn', function () {
  var state = GL.createGame(2, 7, 7);
  // P0 plays a corner to a state that would wipe P1 out, but P1 hasn't moved yet
  // so nobody should be eliminated / game shouldn't end this early regardless.
  var r = GL.playMove(state, 0, 0);
  assert.strictEqual(r.state.gameOver, false);
  assert.strictEqual(r.state.players[0].active, true);
  assert.strictEqual(r.state.players[1].active, true);
});

test('a player with zero cells is eliminated once all players have moved', function () {
  var state = midGame(GL.createGame(2, 7, 7));
  // Manually construct a state where it's player 1's move, totalMoves already 1
  // (P0 has moved once), and player 0 currently owns zero cells (e.g. got wiped
  // by a hypothetical earlier explosion chain) - after P1 moves, elimination check runs.
  state.board[5][5] = { owner: 1, count: 3 }; // interior, cm 4
  state.currentPlayerIndex = 1;
  state.totalMoves = 1;
  var r = GL.playMove(state, 5, 5);
  assert.strictEqual(r.state.players[0].active, false);
  assert.strictEqual(r.state.gameOver, true);
  assert.strictEqual(r.state.winner, 1);
});

test('full game: last player standing wins', function () {
  var state = midGame(GL.createGame(2, 7, 7));
  // P0 owns one weak cell, P1 is about to blow up and capture everything via chain.
  state.board[0][0] = { owner: 0, count: 1 }; // P0's only cell, corner cm=2 (not yet critical)
  state.board[3][3] = { owner: 1, count: 3 }; // interior cm=4, about to explode
  state.board[3][2] = { owner: 0, count: 1 };
  state.board[3][4] = { owner: 0, count: 1 };
  state.board[2][3] = { owner: 0, count: 1 };
  state.board[4][3] = { owner: 0, count: 1 };
  state.currentPlayerIndex = 1;
  state.totalMoves = 1;
  var r = GL.playMove(state, 3, 3);
  // P1's explosion captures all 4 neighbours (previously P0), but P0 still has (0,0).
  assert.strictEqual(r.state.players[0].active, true);
  assert.strictEqual(r.state.gameOver, false);
});

test('board dimensions stay 7x7 by default', function () {
  var state = GL.createGame(2);
  assert.strictEqual(state.board.length, 7);
  assert.strictEqual(state.board[0].length, 7);
});

test('cannot play on an opponent-owned cell via playMove (no-op)', function () {
  var state = GL.createGame(2);
  state.board[3][3] = { owner: 1, count: 1 };
  var r = GL.playMove(state, 3, 3); // player 0's turn, cell owned by player 1
  assert.strictEqual(r.state.currentPlayerIndex, 0); // unchanged, move rejected
  assert.strictEqual(r.state.board[3][3].count, 1); // unchanged
});

console.log('\n4-player full-game scenarios');
test('4p: no elimination before every player has moved once, even with a zero-cell player', function () {
  var state = GL.createGame(4, 7, 7);
  state.players[1].active = true;
  state.totalMoves = 2;
  state.currentPlayerIndex = 0;
  var r = GL.playMove(state, 0, 0);
  assert.strictEqual(r.state.players[1].active, true);
  assert.strictEqual(r.state.gameOver, false);
});

test('4p: turn order skips an eliminated player and does not end the game early', function () {
  var state = midGame(GL.createGame(4, 7, 7));
  state.board[3][3] = { owner: 0, count: 3 };
  state.board[2][3] = { owner: 1, count: 1 };
  state.board[6][6] = { owner: 2, count: 1 };
  state.board[0][6] = { owner: 3, count: 1 };
  state.board[0][0] = { owner: 0, count: 1 };
  state.currentPlayerIndex = 0;
  state.totalMoves = 4;

  var r1 = GL.playMove(state, 3, 3);
  assert.strictEqual(r1.state.players[1].active, false);
  assert.strictEqual(r1.state.gameOver, false);
  assert.strictEqual(r1.state.currentPlayerIndex, 2);

  var r2 = GL.playMove(r1.state, 6, 5);
  assert.strictEqual(r2.state.currentPlayerIndex, 3);

  var r3 = GL.playMove(r2.state, 0, 5);
  assert.strictEqual(r3.state.currentPlayerIndex, 0);
});

test('4p: a single multi-wave chain reaction can eliminate three opponents at once and correctly attributes every captured cell to the mover', function () {
  var state = midGame(GL.createGame(4, 7, 7));
  state.board[3][3] = { owner: 0, count: 3 };
  state.board[0][0] = { owner: 0, count: 1 };
  state.board[2][3] = { owner: 1, count: 1 };
  state.board[3][4] = { owner: 1, count: 1 };
  state.board[4][3] = { owner: 2, count: 3 };
  state.board[3][2] = { owner: 3, count: 1 };
  state.board[5][3] = { owner: 3, count: 1 };
  state.currentPlayerIndex = 0;
  state.totalMoves = 3;

  var r = GL.playMove(state, 3, 3);

  assert.ok(r.steps.length >= 2);
  assert.strictEqual(GL.countCellsForPlayer(r.state.board, 1), 0);
  assert.strictEqual(GL.countCellsForPlayer(r.state.board, 2), 0);
  assert.strictEqual(GL.countCellsForPlayer(r.state.board, 3), 0);
  assert.ok(GL.countCellsForPlayer(r.state.board, 0) > 0);

  [[2,3],[3,4],[3,2],[5,3]].forEach(function (p) {
    assert.strictEqual(r.state.board[p[0]][p[1]].owner, 0);
  });
  assert.strictEqual(r.state.board[4][3].owner, null);

  assert.strictEqual(r.state.players[1].active, false);
  assert.strictEqual(r.state.players[2].active, false);
  assert.strictEqual(r.state.players[3].active, false);
  assert.strictEqual(r.state.players[0].active, true);
  assert.strictEqual(r.state.gameOver, true);
  assert.strictEqual(r.state.winner, 0);
});

test('4p: game does not falsely end while two or more players still hold cells', function () {
  var state = midGame(GL.createGame(4, 7, 7));
  state.board[3][3] = { owner: 0, count: 3 };
  state.board[2][3] = { owner: 1, count: 1 };
  state.board[6][6] = { owner: 2, count: 1 };
  state.board[0][6] = { owner: 3, count: 1 };
  state.currentPlayerIndex = 0;
  state.totalMoves = 4;

  var r = GL.playMove(state, 3, 3);
  assert.strictEqual(r.state.players[1].active, false);
  assert.strictEqual(r.state.players[2].active, true);
  assert.strictEqual(r.state.players[3].active, true);
  assert.strictEqual(r.state.gameOver, false);
  assert.strictEqual(r.state.winner, null);
});

console.log('\nRegression coverage for reported bugs: turn restriction, premature critical mass, chain-reaction re-evaluation');

test('(a) after the first move, the next player can place on ANY empty cell, not just one they already own', function () {
  var state = GL.createGame(2, 7, 7);
  var r1 = GL.playMove(state, 0, 0);
  assert.strictEqual(r1.state.currentPlayerIndex, 1);
  assert.strictEqual(GL.isValidMove(r1.state.board, 5, 5, 1), true);
  var r2 = GL.playMove(r1.state, 5, 5);
  assert.strictEqual(r2.state.board[5][5].owner, 1);
  // Player 1's OPENING move places the 3-dot opening stack.
  assert.strictEqual(r2.state.board[5][5].count, 3);
  assert.strictEqual(r2.state.currentPlayerIndex, 0);
  var r3 = GL.playMove(r2.state, 2, 6);
  assert.strictEqual(r3.state.board[2][6].count, 1);
});

test('(b) an edge cell does not explode at 2 dots but does at exactly 3 (its critical mass)', function () {
  var board = GL.createEmptyBoard(7, 7);
  assert.strictEqual(GL.getCriticalMass(0, 3, 7, 7), 3);

  var afterFirst = GL.applyMove(board, 0, 3, 0, 7, 7);
  assert.strictEqual(afterFirst.board[0][3].count, 1);
  assert.strictEqual(afterFirst.steps.length, 0);

  var afterSecond = GL.applyMove(afterFirst.board, 0, 3, 0, 7, 7);
  assert.strictEqual(afterSecond.board[0][3].count, 2);
  assert.strictEqual(afterSecond.board[0][3].owner, 0);
  assert.strictEqual(afterSecond.steps.length, 0);

  var afterThird = GL.applyMove(afterSecond.board, 0, 3, 0, 7, 7);
  assert.strictEqual(afterThird.steps.length, 1);
  assert.strictEqual(afterThird.board[0][3].count, 0);
  assert.strictEqual(afterThird.board[0][3].owner, null);
  assert.strictEqual(afterThird.board[0][2].count, 1);
  assert.strictEqual(afterThird.board[0][4].count, 1);
  assert.strictEqual(afterThird.board[1][3].count, 1);
});

test('(c) a cell reaching critical mass purely from a cascade-received dot mid-chain-reaction explodes in the same move', function () {
  var board = GL.createEmptyBoard(7, 7);
  board[1][3] = { owner: 0, count: 3 };
  board[0][3] = { owner: 1, count: 2 };
  assert.strictEqual(GL.getCriticalMass(0, 3, 7, 7), 3);

  var result = GL.applyMove(board, 1, 3, 0, 7, 7);

  assert.ok(result.steps.length >= 2);
  assert.strictEqual(result.board[0][3].owner, null);
  assert.strictEqual(result.board[0][3].count, 0);
  assert.strictEqual(result.board[1][3].owner, 0);
  assert.strictEqual(result.board[1][3].count, 1);
  assert.strictEqual(result.board[0][2].owner, 0);
  assert.strictEqual(result.board[0][2].count, 1);
  assert.strictEqual(result.board[0][4].owner, 0);
  assert.strictEqual(result.board[0][4].count, 1);
});

console.log('\n' + passed + ' passed, ' + failed + ' failed');
if (failed > 0) process.exit(1);
