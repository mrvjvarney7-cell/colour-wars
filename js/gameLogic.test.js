// Node test suite for gameLogic.js. Run with: node js/gameLogic.test.js
// Mirrors js/tests.js (the browser-runnable version of the same suite).
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

function ASSERT_EQ(actual, expected, msg) {
  assert.strictEqual(actual, expected, msg);
}
function ASSERT_OK(val, msg) {
  assert.ok(val, msg);
}

// Tests that hand-build a mid-game board are simulating players who have
// already opened, so clear the per-player opening-move bonus for them.
function midGame(state) {
  state.players.forEach(function (p) { p.hasMoved = true; });
  return state;
}

var CM = 4; // fixed critical mass for every cell

// Repeatedly place single dots on one cell for `player`, returning the result
// of the final placement.
function placeDots(board, r, c, player, times) {
  var res = { board: board, steps: [] };
  for (var i = 0; i < times; i++) {
    res = GL.applyMove(res.board, r, c, player, 7, 7);
  }
  return res;
}

// ---------- Critical mass is a fixed 4 everywhere ----------

test('critical mass is 4 for corners, edges and interior cells alike', function () {
  ASSERT_EQ(GL.CRITICAL_MASS, 4);
  // corners
  ASSERT_EQ(GL.getCriticalMass(0, 0, 7, 7), 4, 'top-left corner');
  ASSERT_EQ(GL.getCriticalMass(0, 6, 7, 7), 4, 'top-right corner');
  ASSERT_EQ(GL.getCriticalMass(6, 0, 7, 7), 4, 'bottom-left corner');
  ASSERT_EQ(GL.getCriticalMass(6, 6, 7, 7), 4, 'bottom-right corner');
  // edges
  ASSERT_EQ(GL.getCriticalMass(0, 3, 7, 7), 4, 'top edge');
  ASSERT_EQ(GL.getCriticalMass(3, 0, 7, 7), 4, 'left edge');
  ASSERT_EQ(GL.getCriticalMass(6, 3, 7, 7), 4, 'bottom edge');
  ASSERT_EQ(GL.getCriticalMass(3, 6, 7, 7), 4, 'right edge');
  // interior
  ASSERT_EQ(GL.getCriticalMass(3, 3, 7, 7), 4, 'centre');
  ASSERT_EQ(GL.getCriticalMass(1, 1, 7, 7), 4, 'inner corner-ish');
  ASSERT_EQ(GL.getCriticalMass(5, 2, 7, 7), 4, 'arbitrary interior');
});

test('every one of the 49 cells reports critical mass 4', function () {
  for (var r = 0; r < 7; r++) {
    for (var c = 0; c < 7; c++) {
      ASSERT_EQ(GL.getCriticalMass(r, c, 7, 7), 4, 'cell (' + r + ',' + c + ')');
    }
  }
});

// Neighbour lookup is unchanged - only the explosion threshold changed.
test('neighbour lookup still returns only existing orthogonal neighbours', function () {
  ASSERT_EQ(GL.getNeighbors(0, 0, 7, 7).length, 2, 'corner has 2 neighbours');
  ASSERT_EQ(GL.getNeighbors(0, 6, 7, 7).length, 2, 'corner has 2 neighbours');
  ASSERT_EQ(GL.getNeighbors(6, 0, 7, 7).length, 2, 'corner has 2 neighbours');
  ASSERT_EQ(GL.getNeighbors(6, 6, 7, 7).length, 2, 'corner has 2 neighbours');
  ASSERT_EQ(GL.getNeighbors(0, 3, 7, 7).length, 3, 'edge has 3 neighbours');
  ASSERT_EQ(GL.getNeighbors(3, 0, 7, 7).length, 3, 'edge has 3 neighbours');
  ASSERT_EQ(GL.getNeighbors(3, 3, 7, 7).length, 4, 'interior has 4 neighbours');
  // and none of them are diagonals or off-board
  GL.getNeighbors(0, 0, 7, 7).forEach(function (n) {
    ASSERT_OK(n[0] >= 0 && n[0] < 7 && n[1] >= 0 && n[1] < 7, 'neighbour on board');
    ASSERT_OK(Math.abs(n[0] - 0) + Math.abs(n[1] - 0) === 1, 'neighbour is orthogonal');
  });
});

// ---------- Nothing explodes below 4, everything explodes at exactly 4 ----------

test('no cell explodes at 1, 2 or 3 dots - corner, edge or interior', function () {
  [[0, 0], [0, 6], [6, 0], [6, 6], [0, 3], [3, 0], [6, 3], [3, 6], [3, 3], [1, 1], [5, 2]]
    .forEach(function (p) {
      var res = placeDots(GL.createEmptyBoard(7, 7), p[0], p[1], 0, 3);
      ASSERT_EQ(res.board[p[0]][p[1]].count, 3, 'cell ' + p + ' should hold 3 dots');
      ASSERT_EQ(res.board[p[0]][p[1]].owner, 0, 'cell ' + p + ' still owned');
      ASSERT_EQ(res.steps.length, 0, 'cell ' + p + ' must not explode below 4');
    });
});

test('a CORNER (2 neighbours) explodes at exactly 4, losing 4, feeding both neighbours', function () {
  var res = placeDots(GL.createEmptyBoard(7, 7), 0, 6, 0, 4);
  ASSERT_EQ(res.steps.length, 1, 'corner must explode on the 4th dot');
  ASSERT_EQ(res.board[0][6].count, 0, 'corner loses all 4 dots');
  ASSERT_EQ(res.board[0][6].owner, null);
  // Its two orthogonal neighbours each gain exactly one dot.
  ASSERT_EQ(res.board[1][6].count, 1);
  ASSERT_EQ(res.board[1][6].owner, 0);
  ASSERT_EQ(res.board[0][5].count, 1);
  ASSERT_EQ(res.board[0][5].owner, 0);
  // The other 2 dots are discarded - they must not appear anywhere else.
  var total = 0;
  for (var r = 0; r < 7; r++) for (var c = 0; c < 7; c++) total += res.board[r][c].count;
  ASSERT_EQ(total, 2, 'corner explosion leaves only the 2 neighbour dots on the board');
});

test('an EDGE cell (3 neighbours) explodes at exactly 4, losing 4, feeding all three', function () {
  var res = placeDots(GL.createEmptyBoard(7, 7), 0, 3, 0, 4);
  ASSERT_EQ(res.steps.length, 1, 'edge must explode on the 4th dot');
  ASSERT_EQ(res.board[0][3].count, 0, 'edge loses all 4 dots');
  ASSERT_EQ(res.board[0][3].owner, null);
  [[0, 2], [0, 4], [1, 3]].forEach(function (p) {
    ASSERT_EQ(res.board[p[0]][p[1]].count, 1, 'neighbour ' + p + ' gains one dot');
    ASSERT_EQ(res.board[p[0]][p[1]].owner, 0);
  });
  var total = 0;
  for (var r = 0; r < 7; r++) for (var c = 0; c < 7; c++) total += res.board[r][c].count;
  ASSERT_EQ(total, 3, 'edge explosion leaves only the 3 neighbour dots (1 discarded)');
});

test('an INTERIOR cell (4 neighbours) explodes at exactly 4, losing 4, feeding all four', function () {
  var res = placeDots(GL.createEmptyBoard(7, 7), 3, 3, 0, 4);
  ASSERT_EQ(res.steps.length, 1);
  ASSERT_EQ(res.board[3][3].count, 0);
  ASSERT_EQ(res.board[3][3].owner, null);
  [[2, 3], [4, 3], [3, 2], [3, 4]].forEach(function (p) {
    ASSERT_EQ(res.board[p[0]][p[1]].count, 1);
    ASSERT_EQ(res.board[p[0]][p[1]].owner, 0);
  });
  var total = 0;
  for (var r = 0; r < 7; r++) for (var c = 0; c < 7; c++) total += res.board[r][c].count;
  ASSERT_EQ(total, 4, 'interior explosion conserves all 4 dots');
});

test('every corner and edge explodes at 4 without crashing or going negative', function () {
  [[0, 0], [0, 6], [6, 0], [6, 6], [0, 3], [3, 0], [6, 3], [3, 6]].forEach(function (p) {
    var res = placeDots(GL.createEmptyBoard(7, 7), p[0], p[1], 0, 4);
    ASSERT_EQ(res.steps.length, 1, 'cell ' + p + ' explodes at 4');
    for (var r = 0; r < 7; r++) {
      for (var c = 0; c < 7; c++) {
        ASSERT_OK(res.board[r][c].count >= 0,
          'cell (' + r + ',' + c + ') went negative after ' + p + ' exploded');
      }
    }
    var nbrs = GL.getNeighbors(p[0], p[1], 7, 7);
    var delivered = 0;
    nbrs.forEach(function (n) { delivered += res.board[n[0]][n[1]].count; });
    ASSERT_EQ(delivered, nbrs.length, 'each existing neighbour of ' + p + ' got exactly one dot');
  });
});

test('basic placement: empty cell takes owner, own cell increments', function () {
  var board = GL.createEmptyBoard(7, 7);
  var r1 = GL.applyMove(board, 3, 3, 0, 7, 7);
  ASSERT_EQ(r1.board[3][3].owner, 0);
  ASSERT_EQ(r1.board[3][3].count, 1);
  ASSERT_EQ(r1.steps.length, 0);
  var r2 = GL.applyMove(r1.board, 3, 3, 0, 7, 7);
  ASSERT_EQ(r2.board[3][3].count, 2);
});

// ---------- Capture and chain reactions ----------

test('exploding into an opponent cell captures it regardless of previous owner', function () {
  var board = GL.createEmptyBoard(7, 7);
  board[3][3] = { owner: 0, count: 3 };
  board[2][3] = { owner: 1, count: 1 };
  var result = GL.applyMove(board, 3, 3, 0, 7, 7);
  ASSERT_EQ(result.board[2][3].owner, 0);
  ASSERT_EQ(result.board[2][3].count, 2);
});

test('an explosion that pushes a neighbour to 4 chains further', function () {
  var board = GL.createEmptyBoard(7, 7);
  board[3][3] = { owner: 0, count: 3 };
  board[3][4] = { owner: 0, count: 3 }; // reaches 4 from the blast, so it explodes too
  var result = GL.applyMove(board, 3, 3, 0, 7, 7);
  ASSERT_OK(result.steps.length >= 2, 'expected at least 2 waves, got ' + result.steps.length);
  ASSERT_EQ(result.board[3][4].count, 0);
  ASSERT_EQ(result.board[3][4].owner, null);
  [[2, 4], [4, 4], [3, 5]].forEach(function (p) {
    ASSERT_EQ(result.board[p[0]][p[1]].owner, 0);
    ASSERT_EQ(result.board[p[0]][p[1]].count, 1);
  });
});

test('a corner reaching 4 purely from cascade-received dots explodes in the same move', function () {
  var board = GL.createEmptyBoard(7, 7);
  board[0][6] = { owner: 1, count: 3 }; // corner, one below the fixed threshold
  board[1][6] = { owner: 0, count: 3 }; // its neighbour; reaches 4 when tapped
  var result = GL.applyMove(board, 1, 6, 0, 7, 7);
  ASSERT_OK(result.steps.length >= 2, 'corner should detonate on a later wave');
  ASSERT_EQ(result.board[0][6].owner, null, 'corner exploded and emptied');
  ASSERT_EQ(result.board[0][6].count, 0);
  ASSERT_EQ(result.board[0][5].owner, 0, 'corner fed its neighbour and captured it');
});

test('chain reaction captures cells from multiple owners across waves', function () {
  var board = GL.createEmptyBoard(7, 7);
  board[3][3] = { owner: 0, count: 3 };
  board[3][4] = { owner: 1, count: 1 };
  board[2][3] = { owner: 2, count: 3 }; // reaches 4 via the blast -> chains
  var result = GL.applyMove(board, 3, 3, 0, 7, 7);
  ASSERT_OK(result.steps.length >= 2);
  ASSERT_EQ(result.board[3][4].owner, 0, 'opponent cell captured');
  ASSERT_EQ(result.board[2][3].owner, null, 'chained cell exploded and emptied');
  ASSERT_EQ(result.board[1][3].owner, 0, 'second-wave neighbour captured by the mover');
});

test('no cell ever RESTS at or above critical mass once a move resolves', function () {
  var board = GL.createEmptyBoard(7, 7);
  board[3][3] = { owner: 0, count: 3 };
  board[3][4] = { owner: 0, count: 3 };
  board[2][3] = { owner: 0, count: 3 };
  board[4][3] = { owner: 0, count: 3 };
  board[3][2] = { owner: 0, count: 3 };
  var res = GL.applyMove(board, 3, 3, 0, 7, 7);
  for (var r = 0; r < 7; r++) {
    for (var c = 0; c < 7; c++) {
      ASSERT_OK(res.board[r][c].count < CM,
        'cell (' + r + ',' + c + ') rests at ' + res.board[r][c].count);
      ASSERT_OK(res.board[r][c].count >= 0, 'cell (' + r + ',' + c + ') is negative');
    }
  }
});

test('does not explode when below critical mass', function () {
  var board = GL.createEmptyBoard(7, 7);
  board[3][3] = { owner: 0, count: 2 };
  var result = GL.applyMove(board, 3, 3, 0, 7, 7);
  ASSERT_EQ(result.board[3][3].count, 3);
  ASSERT_EQ(result.steps.length, 0);
});

// ---------- Move validation ----------

test('empty cell is valid before a player has opened, invalid after', function () {
  var board = GL.createEmptyBoard(7, 7);
  ASSERT_EQ(GL.isValidMove(board, 2, 2, 0, false), true);
  ASSERT_EQ(GL.isValidMove(board, 2, 2, 1, false), true);
  ASSERT_EQ(GL.isValidMove(board, 2, 2, 0, true), false,
    'a player who has already opened cannot claim an empty cell directly');
});
test('own cell is valid, opponent cell is not', function () {
  var board = GL.createEmptyBoard(7, 7);
  board[2][2] = { owner: 0, count: 1 };
  ASSERT_EQ(GL.isValidMove(board, 2, 2, 0, true), true);
  ASSERT_EQ(GL.isValidMove(board, 2, 2, 1, true), false);
});
test('out of bounds is invalid', function () {
  var board = GL.createEmptyBoard(7, 7);
  ASSERT_EQ(GL.isValidMove(board, -1, 0, 0, false), false);
  ASSERT_EQ(GL.isValidMove(board, 0, 7, 0, false), false);
});
test('cannot play on an opponent-owned cell via playMove (no-op)', function () {
  var state = midGame(GL.createGame(2));
  state.board[3][3] = { owner: 1, count: 1 };
  var r = GL.playMove(state, 3, 3);
  ASSERT_EQ(r.state.currentPlayerIndex, 0);
  ASSERT_EQ(r.state.board[3][3].count, 1);
});

// ---------- Turn order, elimination and winning ----------

test('turn advances to next player after a move', function () {
  var state = GL.createGame(3);
  var r = GL.playMove(state, 3, 3);
  ASSERT_EQ(r.state.currentPlayerIndex, 1);
});

test('board dimensions stay 7x7 by default', function () {
  var state = GL.createGame(2);
  ASSERT_EQ(state.board.length, 7);
  ASSERT_EQ(state.board[0].length, 7);
});

test('no eliminations occur before every player has had one turn', function () {
  var state = GL.createGame(2, 7, 7);
  var r = GL.playMove(state, 0, 0);
  ASSERT_EQ(r.state.gameOver, false);
  ASSERT_EQ(r.state.players[0].active, true);
  ASSERT_EQ(r.state.players[1].active, true);
});

test('a player with zero cells is eliminated once all players have moved', function () {
  var state = midGame(GL.createGame(2, 7, 7));
  state.board[5][5] = { owner: 1, count: 3 }; // reaches 4 on this move and explodes
  state.currentPlayerIndex = 1;
  state.totalMoves = 1;
  var r = GL.playMove(state, 5, 5);
  ASSERT_EQ(r.state.players[0].active, false);
  ASSERT_EQ(r.state.gameOver, true);
  ASSERT_EQ(r.state.winner, 1);
});

test('full game: last player standing wins only when truly last', function () {
  var state = midGame(GL.createGame(2, 7, 7));
  state.board[0][0] = { owner: 0, count: 1 }; // P0's surviving cell, far away
  state.board[3][3] = { owner: 1, count: 3 };
  state.board[3][2] = { owner: 0, count: 1 };
  state.board[3][4] = { owner: 0, count: 1 };
  state.board[2][3] = { owner: 0, count: 1 };
  state.board[4][3] = { owner: 0, count: 1 };
  state.currentPlayerIndex = 1;
  state.totalMoves = 1;
  var r = GL.playMove(state, 3, 3);
  ASSERT_EQ(r.state.players[0].active, true);
  ASSERT_EQ(r.state.gameOver, false);
});

test('4p: no elimination before every player has moved once, even with a zero-cell player', function () {
  var state = GL.createGame(4, 7, 7);
  state.totalMoves = 2;
  state.currentPlayerIndex = 0;
  var r = GL.playMove(state, 0, 0);
  ASSERT_EQ(r.state.players[1].active, true);
  ASSERT_EQ(r.state.gameOver, false);
});

test('4p: turn order skips an eliminated player and does not end the game early', function () {
  var state = midGame(GL.createGame(4, 7, 7));
  state.board[3][3] = { owner: 0, count: 3 }; // detonates on this move
  state.board[2][3] = { owner: 1, count: 1 }; // player 1's only cell -> captured
  state.board[6][6] = { owner: 2, count: 1 };
  state.board[0][6] = { owner: 3, count: 1 };
  state.board[0][0] = { owner: 0, count: 1 };
  state.currentPlayerIndex = 0;
  state.totalMoves = 4;

  var r1 = GL.playMove(state, 3, 3);
  ASSERT_EQ(r1.state.players[1].active, false, 'player 1 eliminated');
  ASSERT_EQ(r1.state.gameOver, false, 'players 0, 2 and 3 still hold cells');
  ASSERT_EQ(r1.state.currentPlayerIndex, 2, 'turn skips eliminated player 1');

  var r2 = GL.playMove(r1.state, 6, 6); // player 2's own cell
  ASSERT_EQ(r2.state.currentPlayerIndex, 3);
  var r3 = GL.playMove(r2.state, 0, 6); // player 3's own cell
  ASSERT_EQ(r3.state.currentPlayerIndex, 0, 'wraps back past eliminated player 1');
});

test('4p: a single multi-wave cascade can eliminate three opponents and attributes every captured cell to the mover', function () {
  var state = midGame(GL.createGame(4, 7, 7));
  state.board[3][3] = { owner: 0, count: 3 }; // detonates on this move
  state.board[0][0] = { owner: 0, count: 1 }; // mover's safe cell
  state.board[2][3] = { owner: 1, count: 1 }; // captured wave 1
  state.board[3][4] = { owner: 1, count: 1 }; // captured wave 1
  state.board[4][3] = { owner: 2, count: 3 }; // captured wave 1, hits 4 -> chains
  state.board[3][2] = { owner: 3, count: 1 }; // captured wave 1
  state.board[5][3] = { owner: 3, count: 1 }; // captured wave 2 by (4,3)
  state.currentPlayerIndex = 0;
  state.totalMoves = 3;

  var r = GL.playMove(state, 3, 3);
  ASSERT_OK(r.steps.length >= 2, 'expected a chained cascade');
  ASSERT_EQ(GL.countCellsForPlayer(r.state.board, 1), 0);
  ASSERT_EQ(GL.countCellsForPlayer(r.state.board, 2), 0);
  ASSERT_EQ(GL.countCellsForPlayer(r.state.board, 3), 0);
  ASSERT_OK(GL.countCellsForPlayer(r.state.board, 0) > 0);
  [[2, 3], [3, 4], [3, 2], [5, 3]].forEach(function (p) {
    ASSERT_EQ(r.state.board[p[0]][p[1]].owner, 0, 'cell ' + p + ' belongs to the mover');
  });
  ASSERT_EQ(r.state.board[4][3].owner, null, '(4,3) exploded in wave 2 so it ends empty');
  ASSERT_EQ(r.state.gameOver, true);
  ASSERT_EQ(r.state.winner, 0);
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
  ASSERT_EQ(r.state.players[1].active, false);
  ASSERT_EQ(r.state.players[2].active, true);
  ASSERT_EQ(r.state.players[3].active, true);
  ASSERT_EQ(r.state.gameOver, false);
  ASSERT_EQ(r.state.winner, null);
});

// ---------- Placement rules ----------

test('after opening, further placements are restricted to your own cells - not any empty cell', function () {
  var state = GL.createGame(2, 7, 7);
  var r1 = GL.playMove(state, 0, 0);
  ASSERT_EQ(r1.state.currentPlayerIndex, 1);
  ASSERT_EQ(GL.isValidMove(r1.state.board, 5, 5, 1, false), true, "player 1's opening: empty cell is fine");
  var r2 = GL.playMove(r1.state, 5, 5);
  ASSERT_EQ(r2.state.board[5][5].owner, 1);
  ASSERT_EQ(r2.state.board[5][5].count, 3, "player 1's opening places 3 dots");
  ASSERT_EQ(r2.state.currentPlayerIndex, 0);

  // Player 0 has already opened - an unowned empty cell is no longer valid.
  ASSERT_EQ(GL.isValidMove(r2.state.board, 2, 6, 0, true), false);
  var r3 = GL.playMove(r2.state, 2, 6);
  ASSERT_EQ(r3.state.board[2][6].owner, null, 'placing on an unowned empty cell after opening is a no-op');
  ASSERT_EQ(r3.state.currentPlayerIndex, 0, 'an invalid move does not consume the turn');

  // Their own opening cell is still a valid target - the 4th dot reaches
  // critical mass and detonates immediately.
  ASSERT_EQ(GL.isValidMove(r3.state.board, 0, 0, 0, true), true);
  var r4 = GL.playMove(r3.state, 0, 0);
  ASSERT_EQ(r4.state.board[0][0].count, 0, 'reached 4 and exploded');
  ASSERT_EQ(r4.state.board[0][0].owner, null);
  ASSERT_OK(r4.steps.length >= 1);
});

// ---------- Opening-move rule ----------

test("opening move: a player's first placement puts 3 dots on the cell", function () {
  var state = GL.createGame(2, 7, 7);
  ASSERT_EQ(GL.placementDots(state, 0), 3);
  var r = GL.playMove(state, 3, 3);
  ASSERT_EQ(r.state.board[3][3].count, 3);
  ASSERT_EQ(r.state.board[3][3].owner, 0);
  ASSERT_EQ(r.steps.length, 0, '3 dots is below the critical mass of 4');
});

test('opening 3 dots never detonates, even on a corner, under the fixed threshold', function () {
  var state = GL.createGame(2, 7, 7);
  var r = GL.playMove(state, 0, 6); // corner opening
  ASSERT_EQ(r.steps.length, 0, 'corner holds 3 dots now that critical mass is 4');
  ASSERT_EQ(r.state.board[0][6].count, 3);
  ASSERT_EQ(r.state.board[0][6].owner, 0);
  ASSERT_EQ(r.state.board[1][6].owner, null, 'no neighbour was fed');
  ASSERT_EQ(r.state.board[0][5].owner, null);
});

test('opening bonus is per-player: each player gets 3 dots on their own first move', function () {
  var s = GL.createGame(4, 7, 7);
  var spots = [[1, 1], [1, 4], [4, 1], [4, 4]];
  for (var i = 0; i < 4; i++) {
    ASSERT_EQ(GL.placementDots(s, i), 3, 'player ' + i + ' opening');
    var res = GL.playMove(s, spots[i][0], spots[i][1]);
    ASSERT_EQ(res.state.board[spots[i][0]][spots[i][1]].count, 3);
    ASSERT_EQ(res.state.board[spots[i][0]][spots[i][1]].owner, i);
    s = res.state;
  }
  for (var j = 0; j < 4; j++) ASSERT_EQ(GL.placementDots(s, j), 1);
});

test("after opening, a player's later move on their own cell adds exactly 1 dot before resolving", function () {
  var state = GL.createGame(2, 7, 7);
  var r1 = GL.playMove(state, 1, 1);
  ASSERT_EQ(r1.state.board[1][1].count, 3);
  var r2 = GL.playMove(r1.state, 4, 4);
  ASSERT_EQ(r2.state.board[4][4].count, 3);
  ASSERT_EQ(GL.placementDots(r2.state, 0), 1);
  // Player 0 has only one owned cell (their opening cell) - that is now their
  // only legal move, and the 4th dot reaches the threshold and detonates.
  var r3 = GL.playMove(r2.state, 1, 1);
  ASSERT_EQ(r3.state.board[1][1].count, 0, 'reached 4 and exploded');
  ASSERT_OK(r3.steps.length >= 1);
});


// ---------- CWN (Colour Wars Notation) ----------

test('CWN: fresh empty board matches the designed example exactly, for 2/3/4 players', function () {
  ASSERT_EQ(GL.encodeCwn(GL.createGame(2, 7, 7)), '7/7/7/7/7/7/7 0 01 0');
  ASSERT_EQ(GL.encodeCwn(GL.createGame(3, 7, 7)), '7/7/7/7/7/7/7 0 012 0');
  ASSERT_EQ(GL.encodeCwn(GL.createGame(4, 7, 7)), '7/7/7/7/7/7/7 0 0123 0');
});

test('CWN: decodeCwn(encodeCwn(x)) round-trips a fresh game exactly', function () {
  var state = GL.createGame(3, 7, 7);
  var decoded = GL.decodeCwn(GL.encodeCwn(state));
  ASSERT_EQ(GL.encodeCwn(decoded), GL.encodeCwn(state));
  ASSERT_EQ(decoded.players.length, 3);
  ASSERT_EQ(decoded.currentPlayerIndex, 0);
  ASSERT_EQ(decoded.totalMoves, 0);
  ASSERT_OK(decoded.players.every(function (p) { return !p.hasMoved && p.active; }));
});

test('CWN: round-trips a mid-game position with mixed empty runs, dot counts and owners', function () {
  var state = GL.createGame(4, 7, 7);
  state = GL.playMove(state, 0, 0).state;   // p0 opens corner, 3 dots
  state = GL.playMove(state, 6, 6).state;   // p1 opens opposite corner, 3 dots
  state = GL.playMove(state, 3, 3).state;   // p2 opens center, 3 dots
  state = GL.playMove(state, 1, 1).state;   // p3 opens, 3 dots - first round now complete
  state = GL.playMove(state, 0, 0).state;   // p0's only cell: 3+1=4, explodes into (0,1)/(1,0)
  state = GL.playMove(state, 6, 6).state;   // p1's only cell: 3+1=4, explodes into (6,5)/(5,6)
  var cwn = GL.encodeCwn(state);
  var decoded = GL.decodeCwn(cwn);
  ASSERT_EQ(GL.encodeCwn(decoded), cwn, 'CWN should be stable under a decode/re-encode round trip');
  // Board contents must match cell-for-cell, not just the notation string.
  for (var r = 0; r < 7; r++) {
    for (var c = 0; c < 7; c++) {
      ASSERT_EQ(decoded.board[r][c].owner, state.board[r][c].owner, 'owner mismatch at ' + r + ',' + c);
      ASSERT_EQ(decoded.board[r][c].count, state.board[r][c].count, 'count mismatch at ' + r + ',' + c);
    }
  }
  ASSERT_EQ(decoded.currentPlayerIndex, state.currentPlayerIndex);
  ASSERT_EQ(decoded.totalMoves, state.totalMoves);
});

test('CWN: notYetOpened is "-" once every player has opened', function () {
  var state = GL.createGame(2, 7, 7);
  state = GL.playMove(state, 0, 0).state;
  state = GL.playMove(state, 6, 6).state;
  var cwn = GL.encodeCwn(state);
  ASSERT_OK(cwn.indexOf(' 0 - ') !== -1, 'expected "-" once both players have opened, got: ' + cwn);
  var decoded = GL.decodeCwn(cwn);
  ASSERT_OK(decoded.players.every(function (p) { return p.hasMoved; }));
});

test('CWN: a hand-built position with an eliminated player round-trips its board and active flags', function () {
  // A completed-looking snapshot: player 0 has zero cells, ply is past the
  // first-round-complete threshold, so decodeCwn should mark them inactive
  // exactly like a real game reaching the same shape would.
  var board = GL.createEmptyBoard(7, 7);
  board[3][3] = { owner: 1, count: 2 };
  board[3][4] = { owner: 1, count: 1 };
  var state = {
    rows: 7, cols: 7, board: board,
    players: [
      { id: 0, name: 'Player 1', color: GL.COLOR_PALETTE[0].hex, colorName: GL.COLOR_PALETTE[0].name, active: false, hasMoved: true },
      { id: 1, name: 'Player 2', color: GL.COLOR_PALETTE[1].hex, colorName: GL.COLOR_PALETTE[1].name, active: true, hasMoved: true }
    ],
    currentPlayerIndex: 1, totalMoves: 5, gameOver: true, winner: 1
  };
  var cwn = GL.encodeCwn(state);
  var decoded = GL.decodeCwn(cwn);
  ASSERT_EQ(decoded.players[0].active, false, 'player 0 (zero cells, ply past first round) should decode inactive');
  ASSERT_EQ(decoded.players[1].active, true);
  ASSERT_EQ(decoded.gameOver, true);
  ASSERT_EQ(decoded.winner, 1);
});

test('CWN: rejects malformed input rather than silently producing a wrong board', function () {
  var threw = false;
  try { GL.decodeCwn('7/7/7/7/7/7 0 01 0'); } catch (e) { threw = true; } // only 6 rows
  ASSERT_OK(threw, 'expected an error for a board with the wrong row count');

  threw = false;
  try { GL.decodeCwn('7/7/7/7/7/7/7 0 01'); } catch (e) { threw = true; } // only 3 fields
  ASSERT_OK(threw, 'expected an error for missing the ply field');

  threw = false;
  try { GL.decodeCwn('zzzzzzz/7/7/7/7/7/7 0 01 0'); } catch (e) { threw = true; } // invalid letter
  ASSERT_OK(threw, 'expected an error for an invalid board character');
});

console.log('\n' + passed + ' passed, ' + failed + ' failed');
if (failed > 0) process.exit(1);
