// Browser-runnable test suite for gameLogic.js (mirrors js/gameLogic.test.js,
// which is the Node-runnable version of the same suite).
(function () {
  'use strict';
  var GL = window.GameLogic;
  var passed = 0;
  var failed = 0;
  var results = [];

  function assertEqual(actual, expected, msg) {
    if (actual !== expected) {
      throw new Error((msg || 'assertion failed') + ' — expected ' + JSON.stringify(expected) + ' got ' + JSON.stringify(actual));
    }
  }
  function assertTrue(val, msg) {
    if (!val) throw new Error(msg || 'expected truthy value');
  }

  function test(name, fn) {
    try {
      fn();
      passed++;
      results.push({ name: name, ok: true });
    } catch (err) {
      failed++;
      results.push({ name: name, ok: false, error: err.message });
    }
  }

  test('corner has critical mass 2', function () {
    assertEqual(GL.getCriticalMass(0, 0, 7, 7), 2);
    assertEqual(GL.getCriticalMass(0, 6, 7, 7), 2);
    assertEqual(GL.getCriticalMass(6, 0, 7, 7), 2);
    assertEqual(GL.getCriticalMass(6, 6, 7, 7), 2);
  });
  test('edge has critical mass 3', function () {
    assertEqual(GL.getCriticalMass(0, 3, 7, 7), 3);
    assertEqual(GL.getCriticalMass(3, 0, 7, 7), 3);
    assertEqual(GL.getCriticalMass(6, 3, 7, 7), 3);
    assertEqual(GL.getCriticalMass(3, 6, 7, 7), 3);
  });
  test('interior has critical mass 4', function () {
    assertEqual(GL.getCriticalMass(3, 3, 7, 7), 4);
    assertEqual(GL.getCriticalMass(1, 1, 7, 7), 4);
  });

  test('placing a dot on an empty cell sets owner and count', function () {
    var board = GL.createEmptyBoard(7, 7);
    var result = GL.applyMove(board, 3, 3, 0, 7, 7);
    assertEqual(result.board[3][3].owner, 0);
    assertEqual(result.board[3][3].count, 1);
    assertEqual(result.steps.length, 0);
  });
  test('placing a dot on own cell increments count', function () {
    var board = GL.createEmptyBoard(7, 7);
    board[3][3] = { owner: 0, count: 1 };
    var result = GL.applyMove(board, 3, 3, 0, 7, 7);
    assertEqual(result.board[3][3].count, 2);
  });

  test('corner cell explodes at 2 dots, splits to its 2 neighbours', function () {
    var board = GL.createEmptyBoard(7, 7);
    board[0][0] = { owner: 0, count: 1 };
    var result = GL.applyMove(board, 0, 0, 0, 7, 7);
    assertEqual(result.board[0][0].count, 0);
    assertEqual(result.board[0][0].owner, null);
    assertEqual(result.board[0][1].count, 1);
    assertEqual(result.board[0][1].owner, 0);
    assertEqual(result.board[1][0].count, 1);
    assertEqual(result.board[1][0].owner, 0);
    assertEqual(result.steps.length, 1);
  });

  test('edge cell explodes at 3 dots, splits to its 3 neighbours', function () {
    var board = GL.createEmptyBoard(7, 7);
    board[0][3] = { owner: 1, count: 2 };
    var result = GL.applyMove(board, 0, 3, 1, 7, 7);
    assertEqual(result.board[0][3].count, 0);
    assertEqual(result.board[0][2].count, 1);
    assertEqual(result.board[0][4].count, 1);
    assertEqual(result.board[1][3].count, 1);
    [[0,2],[0,4],[1,3]].forEach(function (p) {
      assertEqual(result.board[p[0]][p[1]].owner, 1);
    });
  });

  test('interior cell explodes at 4 dots, splits to its 4 neighbours', function () {
    var board = GL.createEmptyBoard(7, 7);
    board[3][3] = { owner: 0, count: 3 };
    var result = GL.applyMove(board, 3, 3, 0, 7, 7);
    assertEqual(result.board[3][3].count, 0);
    var neighbours = [[2,3],[4,3],[3,2],[3,4]];
    neighbours.forEach(function (p) {
      assertEqual(result.board[p[0]][p[1]].count, 1);
      assertEqual(result.board[p[0]][p[1]].owner, 0);
    });
  });

  test('exploding into an opponent cell captures it regardless of previous owner', function () {
    var board = GL.createEmptyBoard(7, 7);
    board[3][3] = { owner: 0, count: 3 };
    board[2][3] = { owner: 1, count: 1 };
    var result = GL.applyMove(board, 3, 3, 0, 7, 7);
    assertEqual(result.board[2][3].owner, 0);
    assertEqual(result.board[2][3].count, 2);
  });

  test('an explosion that pushes a neighbour past critical mass chains further', function () {
    var board = GL.createEmptyBoard(7, 7);
    board[3][3] = { owner: 0, count: 3 };
    board[3][4] = { owner: 0, count: 3 };
    var result = GL.applyMove(board, 3, 3, 0, 7, 7);
    assertTrue(result.steps.length >= 2, 'expected at least 2 waves, got ' + result.steps.length);
    assertEqual(result.board[3][4].count, 0);
    assertEqual(result.board[3][4].owner, null);
    var farNeighbours = [[2,4],[4,4],[3,5]];
    farNeighbours.forEach(function (p) {
      assertEqual(result.board[p[0]][p[1]].owner, 0);
      assertEqual(result.board[p[0]][p[1]].count, 1);
    });
  });

  test('chain reaction can capture and flip multiple opponent cells across waves', function () {
    var board = GL.createEmptyBoard(7, 7);
    board[0][0] = { owner: 0, count: 1 };
    board[0][1] = { owner: 1, count: 2 };
    board[0][2] = { owner: 1, count: 1 };
    var result = GL.applyMove(board, 0, 0, 0, 7, 7);
    assertEqual(result.board[0][1].owner, null);
    assertEqual(result.board[0][1].count, 0);
    assertEqual(result.board[0][2].owner, 0);
    assertEqual(result.board[0][2].count, 2);
    assertEqual(result.board[1][1].owner, 0);
    assertTrue(result.steps.length >= 2);
  });

  test('does not explode when below critical mass', function () {
    var board = GL.createEmptyBoard(7, 7);
    board[3][3] = { owner: 0, count: 2 };
    var result = GL.applyMove(board, 3, 3, 0, 7, 7);
    assertEqual(result.board[3][3].count, 3);
    assertEqual(result.steps.length, 0);
  });

  test('empty cell is valid for any player', function () {
    var board = GL.createEmptyBoard(7, 7);
    assertEqual(GL.isValidMove(board, 2, 2, 0), true);
    assertEqual(GL.isValidMove(board, 2, 2, 1), true);
  });
  test('own cell is valid, opponent cell is not', function () {
    var board = GL.createEmptyBoard(7, 7);
    board[2][2] = { owner: 0, count: 1 };
    assertEqual(GL.isValidMove(board, 2, 2, 0), true);
    assertEqual(GL.isValidMove(board, 2, 2, 1), false);
  });
  test('out of bounds is invalid', function () {
    var board = GL.createEmptyBoard(7, 7);
    assertEqual(GL.isValidMove(board, -1, 0, 0), false);
    assertEqual(GL.isValidMove(board, 0, 7, 0), false);
  });

  test('turn advances to next player after a move', function () {
    var state = GL.createGame(3);
    var r = GL.playMove(state, 3, 3);
    assertEqual(r.state.currentPlayerIndex, 1);
  });

  test('no eliminations occur before every player has had one turn', function () {
    var state = GL.createGame(2, 7, 7);
    var r = GL.playMove(state, 0, 0);
    assertEqual(r.state.gameOver, false);
    assertEqual(r.state.players[0].active, true);
    assertEqual(r.state.players[1].active, true);
  });

  test('a player with zero cells is eliminated once all players have moved', function () {
    var state = GL.createGame(2, 7, 7);
    state.board[5][5] = { owner: 1, count: 3 };
    state.currentPlayerIndex = 1;
    state.totalMoves = 1;
    var r = GL.playMove(state, 5, 5);
    assertEqual(r.state.players[0].active, false);
    assertEqual(r.state.gameOver, true);
    assertEqual(r.state.winner, 1);
  });

  test('full game: last player standing wins only when truly last', function () {
    var state = GL.createGame(2, 7, 7);
    state.board[0][0] = { owner: 0, count: 1 };
    state.board[3][3] = { owner: 1, count: 3 };
    state.board[3][2] = { owner: 0, count: 1 };
    state.board[3][4] = { owner: 0, count: 1 };
    state.board[2][3] = { owner: 0, count: 1 };
    state.board[4][3] = { owner: 0, count: 1 };
    state.currentPlayerIndex = 1;
    state.totalMoves = 1;
    var r = GL.playMove(state, 3, 3);
    assertEqual(r.state.players[0].active, true);
    assertEqual(r.state.gameOver, false);
  });

  test('board dimensions stay 7x7 by default', function () {
    var state = GL.createGame(2);
    assertEqual(state.board.length, 7);
    assertEqual(state.board[0].length, 7);
  });

  test('cannot play on an opponent-owned cell via playMove (no-op)', function () {
    var state = GL.createGame(2);
    state.board[3][3] = { owner: 1, count: 1 };
    var r = GL.playMove(state, 3, 3);
    assertEqual(r.state.currentPlayerIndex, 0);
    assertEqual(r.state.board[3][3].count, 1);
  });

  // ---- 4-player full-game scenarios ----

  test('4p: no elimination before every player has moved once, even with a zero-cell player', function () {
    var state = GL.createGame(4, 7, 7);
    state.players[1].active = true;
    // Player 1 owns nothing yet (hasn't had a turn), but only 2 of 4 players have moved so far.
    state.totalMoves = 2;
    state.currentPlayerIndex = 0;
    var r = GL.playMove(state, 0, 0);
    assertEqual(r.state.players[1].active, true, 'player with 0 cells must not be eliminated before their first turn');
    assertEqual(r.state.gameOver, false);
  });

  test('4p: turn order skips an eliminated player and does not end the game early', function () {
    var state = GL.createGame(4, 7, 7);
    // Player 1's only cell sits where player 0 is about to explode into it.
    state.board[3][3] = { owner: 0, count: 3 }; // interior, cm 4
    state.board[2][3] = { owner: 1, count: 1 }; // player 1's only cell
    state.board[6][6] = { owner: 2, count: 1 };
    state.board[0][6] = { owner: 3, count: 1 };
    state.board[0][0] = { owner: 0, count: 1 };
    state.currentPlayerIndex = 0;
    state.totalMoves = 4; // everyone has already moved once

    var r1 = GL.playMove(state, 3, 3); // player 0 explodes, captures (2,3), wiping player 1
    assertEqual(r1.state.players[1].active, false, 'player 1 should be eliminated (0 cells)');
    assertEqual(r1.state.gameOver, false, 'three players (0,2,3) still hold cells');
    assertEqual(r1.state.currentPlayerIndex, 2, 'turn should skip eliminated player 1 and go to player 2');

    var r2 = GL.playMove(r1.state, 6, 5); // player 2's turn, a harmless empty cell
    assertEqual(r2.state.currentPlayerIndex, 3, 'turn should go to player 3');

    var r3 = GL.playMove(r2.state, 0, 5); // player 3's turn, a harmless empty cell
    assertEqual(r3.state.currentPlayerIndex, 0, 'turn should skip eliminated player 1 and wrap back to player 0');
  });

  test('4p: a single multi-wave chain reaction can eliminate three opponents at once and correctly attributes every captured cell to the mover', function () {
    var state = GL.createGame(4, 7, 7);
    // Center stack about to explode (interior, critical mass 4).
    state.board[3][3] = { owner: 0, count: 3 };
    state.board[0][0] = { owner: 0, count: 1 }; // player 0's extra cell, well away from the blast

    // Direct neighbours of (3,3), each belonging to a different opponent.
    state.board[2][3] = { owner: 1, count: 1 }; // player 1's ONLY cell -> captured in wave 1
    state.board[3][4] = { owner: 1, count: 1 }; // player 1's second cell -> also captured in wave 1
    state.board[4][3] = { owner: 2, count: 3 }; // player 2's ONLY cell -> captured wave 1, hits its own
                                                 // critical mass (4) on capture, chains into wave 2
    state.board[3][2] = { owner: 3, count: 1 }; // player 3's first cell -> captured in wave 1

    // (5,3) is a second-degree neighbour, only reachable via (4,3)'s wave-2 explosion.
    state.board[5][3] = { owner: 3, count: 1 }; // player 3's second cell -> captured in wave 2

    state.currentPlayerIndex = 0;
    state.totalMoves = 3; // about to become 4 -> first round complete, eliminations apply

    var r = GL.playMove(state, 3, 3);

    assertTrue(r.steps.length >= 2, 'expected at least 2 explosion waves (direct + chained)');
    assertEqual(GL.countCellsForPlayer(r.state.board, 1), 0, 'player 1 should have zero cells left');
    assertEqual(GL.countCellsForPlayer(r.state.board, 2), 0, 'player 2 should have zero cells left');
    assertEqual(GL.countCellsForPlayer(r.state.board, 3), 0, 'player 3 should have zero cells left');
    assertTrue(GL.countCellsForPlayer(r.state.board, 0) > 0, 'player 0 should still hold cells');

    // Every captured cell, whether taken in the first wave or chained into the second,
    // must end up owned by the ORIGINAL mover (player 0) - never by an intermediate owner.
    // (4,3) is excluded: it was captured in wave 1 but itself exploded in wave 2, so it
    // ends up empty again, same as any exploding cell - that's checked separately below.
    [[2,3],[3,4],[3,2],[5,3]].forEach(function (p) {
      assertEqual(r.state.board[p[0]][p[1]].owner, 0, 'cell (' + p[0] + ',' + p[1] + ') should end up owned by the mover');
    });
    assertEqual(r.state.board[4][3].owner, null, '(4,3) exploded in wave 2 so it should end up empty, not owned');

    assertEqual(r.state.players[1].active, false);
    assertEqual(r.state.players[2].active, false);
    assertEqual(r.state.players[3].active, false);
    assertEqual(r.state.players[0].active, true);
    assertEqual(r.state.gameOver, true, 'game should end immediately once only one player remains');
    assertEqual(r.state.winner, 0);
  });

  test('4p: game does not falsely end while two or more players still hold cells', function () {
    var state = GL.createGame(4, 7, 7);
    state.board[3][3] = { owner: 0, count: 3 };
    state.board[2][3] = { owner: 1, count: 1 }; // player 1's only cell -> will be wiped
    state.board[6][6] = { owner: 2, count: 1 }; // player 2 untouched
    state.board[0][6] = { owner: 3, count: 1 }; // player 3 untouched
    state.currentPlayerIndex = 0;
    state.totalMoves = 4;

    var r = GL.playMove(state, 3, 3);
    assertEqual(r.state.players[1].active, false);
    assertEqual(r.state.players[2].active, true);
    assertEqual(r.state.players[3].active, true);
    assertEqual(r.state.gameOver, false, 'must not end while players 0, 2 and 3 all still hold cells');
    assertEqual(r.state.winner, null);
  });

  // Render results to the page and console.
  var out = document.getElementById('results');
  var lines = [];
  results.forEach(function (r) {
    lines.push((r.ok ? 'OK   ' : 'FAIL ') + r.name + (r.ok ? '' : ' :: ' + r.error));
  });
  var summary = passed + ' passed, ' + failed + ' failed';
  lines.push('');
  lines.push(summary);
  out.textContent = lines.join('\n');
  console.log(lines.join('\n'));
  window.__TEST_SUMMARY__ = { passed: passed, failed: failed };
})();
