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
    assertEqual(GL.CRITICAL_MASS, 4);
    // corners
    assertEqual(GL.getCriticalMass(0, 0, 7, 7), 4, 'top-left corner');
    assertEqual(GL.getCriticalMass(0, 6, 7, 7), 4, 'top-right corner');
    assertEqual(GL.getCriticalMass(6, 0, 7, 7), 4, 'bottom-left corner');
    assertEqual(GL.getCriticalMass(6, 6, 7, 7), 4, 'bottom-right corner');
    // edges
    assertEqual(GL.getCriticalMass(0, 3, 7, 7), 4, 'top edge');
    assertEqual(GL.getCriticalMass(3, 0, 7, 7), 4, 'left edge');
    assertEqual(GL.getCriticalMass(6, 3, 7, 7), 4, 'bottom edge');
    assertEqual(GL.getCriticalMass(3, 6, 7, 7), 4, 'right edge');
    // interior
    assertEqual(GL.getCriticalMass(3, 3, 7, 7), 4, 'centre');
    assertEqual(GL.getCriticalMass(1, 1, 7, 7), 4, 'inner corner-ish');
    assertEqual(GL.getCriticalMass(5, 2, 7, 7), 4, 'arbitrary interior');
  });

  test('every one of the 49 cells reports critical mass 4', function () {
    for (var r = 0; r < 7; r++) {
      for (var c = 0; c < 7; c++) {
        assertEqual(GL.getCriticalMass(r, c, 7, 7), 4, 'cell (' + r + ',' + c + ')');
      }
    }
  });

  // Neighbour lookup is unchanged - only the explosion threshold changed.
  test('neighbour lookup still returns only existing orthogonal neighbours', function () {
    assertEqual(GL.getNeighbors(0, 0, 7, 7).length, 2, 'corner has 2 neighbours');
    assertEqual(GL.getNeighbors(0, 6, 7, 7).length, 2, 'corner has 2 neighbours');
    assertEqual(GL.getNeighbors(6, 0, 7, 7).length, 2, 'corner has 2 neighbours');
    assertEqual(GL.getNeighbors(6, 6, 7, 7).length, 2, 'corner has 2 neighbours');
    assertEqual(GL.getNeighbors(0, 3, 7, 7).length, 3, 'edge has 3 neighbours');
    assertEqual(GL.getNeighbors(3, 0, 7, 7).length, 3, 'edge has 3 neighbours');
    assertEqual(GL.getNeighbors(3, 3, 7, 7).length, 4, 'interior has 4 neighbours');
    // and none of them are diagonals or off-board
    GL.getNeighbors(0, 0, 7, 7).forEach(function (n) {
      assertTrue(n[0] >= 0 && n[0] < 7 && n[1] >= 0 && n[1] < 7, 'neighbour on board');
      assertTrue(Math.abs(n[0] - 0) + Math.abs(n[1] - 0) === 1, 'neighbour is orthogonal');
    });
  });

  // ---------- Nothing explodes below 4, everything explodes at exactly 4 ----------

  test('no cell explodes at 1, 2 or 3 dots - corner, edge or interior', function () {
    [[0, 0], [0, 6], [6, 0], [6, 6], [0, 3], [3, 0], [6, 3], [3, 6], [3, 3], [1, 1], [5, 2]]
      .forEach(function (p) {
        var res = placeDots(GL.createEmptyBoard(7, 7), p[0], p[1], 0, 3);
        assertEqual(res.board[p[0]][p[1]].count, 3, 'cell ' + p + ' should hold 3 dots');
        assertEqual(res.board[p[0]][p[1]].owner, 0, 'cell ' + p + ' still owned');
        assertEqual(res.steps.length, 0, 'cell ' + p + ' must not explode below 4');
      });
  });

  test('a CORNER (2 neighbours) explodes at exactly 4, losing 4, feeding both neighbours', function () {
    var res = placeDots(GL.createEmptyBoard(7, 7), 0, 6, 0, 4);
    assertEqual(res.steps.length, 1, 'corner must explode on the 4th dot');
    assertEqual(res.board[0][6].count, 0, 'corner loses all 4 dots');
    assertEqual(res.board[0][6].owner, null);
    // Its two orthogonal neighbours each gain exactly one dot.
    assertEqual(res.board[1][6].count, 1);
    assertEqual(res.board[1][6].owner, 0);
    assertEqual(res.board[0][5].count, 1);
    assertEqual(res.board[0][5].owner, 0);
    // The other 2 dots are discarded - they must not appear anywhere else.
    var total = 0;
    for (var r = 0; r < 7; r++) for (var c = 0; c < 7; c++) total += res.board[r][c].count;
    assertEqual(total, 2, 'corner explosion leaves only the 2 neighbour dots on the board');
  });

  test('an EDGE cell (3 neighbours) explodes at exactly 4, losing 4, feeding all three', function () {
    var res = placeDots(GL.createEmptyBoard(7, 7), 0, 3, 0, 4);
    assertEqual(res.steps.length, 1, 'edge must explode on the 4th dot');
    assertEqual(res.board[0][3].count, 0, 'edge loses all 4 dots');
    assertEqual(res.board[0][3].owner, null);
    [[0, 2], [0, 4], [1, 3]].forEach(function (p) {
      assertEqual(res.board[p[0]][p[1]].count, 1, 'neighbour ' + p + ' gains one dot');
      assertEqual(res.board[p[0]][p[1]].owner, 0);
    });
    var total = 0;
    for (var r = 0; r < 7; r++) for (var c = 0; c < 7; c++) total += res.board[r][c].count;
    assertEqual(total, 3, 'edge explosion leaves only the 3 neighbour dots (1 discarded)');
  });

  test('an INTERIOR cell (4 neighbours) explodes at exactly 4, losing 4, feeding all four', function () {
    var res = placeDots(GL.createEmptyBoard(7, 7), 3, 3, 0, 4);
    assertEqual(res.steps.length, 1);
    assertEqual(res.board[3][3].count, 0);
    assertEqual(res.board[3][3].owner, null);
    [[2, 3], [4, 3], [3, 2], [3, 4]].forEach(function (p) {
      assertEqual(res.board[p[0]][p[1]].count, 1);
      assertEqual(res.board[p[0]][p[1]].owner, 0);
    });
    var total = 0;
    for (var r = 0; r < 7; r++) for (var c = 0; c < 7; c++) total += res.board[r][c].count;
    assertEqual(total, 4, 'interior explosion conserves all 4 dots');
  });

  test('every corner and edge explodes at 4 without crashing or going negative', function () {
    [[0, 0], [0, 6], [6, 0], [6, 6], [0, 3], [3, 0], [6, 3], [3, 6]].forEach(function (p) {
      var res = placeDots(GL.createEmptyBoard(7, 7), p[0], p[1], 0, 4);
      assertEqual(res.steps.length, 1, 'cell ' + p + ' explodes at 4');
      for (var r = 0; r < 7; r++) {
        for (var c = 0; c < 7; c++) {
          assertTrue(res.board[r][c].count >= 0,
            'cell (' + r + ',' + c + ') went negative after ' + p + ' exploded');
        }
      }
      var nbrs = GL.getNeighbors(p[0], p[1], 7, 7);
      var delivered = 0;
      nbrs.forEach(function (n) { delivered += res.board[n[0]][n[1]].count; });
      assertEqual(delivered, nbrs.length, 'each existing neighbour of ' + p + ' got exactly one dot');
    });
  });

  test('basic placement: empty cell takes owner, own cell increments', function () {
    var board = GL.createEmptyBoard(7, 7);
    var r1 = GL.applyMove(board, 3, 3, 0, 7, 7);
    assertEqual(r1.board[3][3].owner, 0);
    assertEqual(r1.board[3][3].count, 1);
    assertEqual(r1.steps.length, 0);
    var r2 = GL.applyMove(r1.board, 3, 3, 0, 7, 7);
    assertEqual(r2.board[3][3].count, 2);
  });

  // ---------- Capture and chain reactions ----------

  test('exploding into an opponent cell captures it regardless of previous owner', function () {
    var board = GL.createEmptyBoard(7, 7);
    board[3][3] = { owner: 0, count: 3 };
    board[2][3] = { owner: 1, count: 1 };
    var result = GL.applyMove(board, 3, 3, 0, 7, 7);
    assertEqual(result.board[2][3].owner, 0);
    assertEqual(result.board[2][3].count, 2);
  });

  test('an explosion that pushes a neighbour to 4 chains further', function () {
    var board = GL.createEmptyBoard(7, 7);
    board[3][3] = { owner: 0, count: 3 };
    board[3][4] = { owner: 0, count: 3 }; // reaches 4 from the blast, so it explodes too
    var result = GL.applyMove(board, 3, 3, 0, 7, 7);
    assertTrue(result.steps.length >= 2, 'expected at least 2 waves, got ' + result.steps.length);
    assertEqual(result.board[3][4].count, 0);
    assertEqual(result.board[3][4].owner, null);
    [[2, 4], [4, 4], [3, 5]].forEach(function (p) {
      assertEqual(result.board[p[0]][p[1]].owner, 0);
      assertEqual(result.board[p[0]][p[1]].count, 1);
    });
  });

  test('a corner reaching 4 purely from cascade-received dots explodes in the same move', function () {
    var board = GL.createEmptyBoard(7, 7);
    board[0][6] = { owner: 1, count: 3 }; // corner, one below the fixed threshold
    board[1][6] = { owner: 0, count: 3 }; // its neighbour; reaches 4 when tapped
    var result = GL.applyMove(board, 1, 6, 0, 7, 7);
    assertTrue(result.steps.length >= 2, 'corner should detonate on a later wave');
    assertEqual(result.board[0][6].owner, null, 'corner exploded and emptied');
    assertEqual(result.board[0][6].count, 0);
    assertEqual(result.board[0][5].owner, 0, 'corner fed its neighbour and captured it');
  });

  test('chain reaction captures cells from multiple owners across waves', function () {
    var board = GL.createEmptyBoard(7, 7);
    board[3][3] = { owner: 0, count: 3 };
    board[3][4] = { owner: 1, count: 1 };
    board[2][3] = { owner: 2, count: 3 }; // reaches 4 via the blast -> chains
    var result = GL.applyMove(board, 3, 3, 0, 7, 7);
    assertTrue(result.steps.length >= 2);
    assertEqual(result.board[3][4].owner, 0, 'opponent cell captured');
    assertEqual(result.board[2][3].owner, null, 'chained cell exploded and emptied');
    assertEqual(result.board[1][3].owner, 0, 'second-wave neighbour captured by the mover');
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
        assertTrue(res.board[r][c].count < CM,
          'cell (' + r + ',' + c + ') rests at ' + res.board[r][c].count);
        assertTrue(res.board[r][c].count >= 0, 'cell (' + r + ',' + c + ') is negative');
      }
    }
  });

  test('does not explode when below critical mass', function () {
    var board = GL.createEmptyBoard(7, 7);
    board[3][3] = { owner: 0, count: 2 };
    var result = GL.applyMove(board, 3, 3, 0, 7, 7);
    assertEqual(result.board[3][3].count, 3);
    assertEqual(result.steps.length, 0);
  });

  // ---------- Move validation ----------

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
  test('cannot play on an opponent-owned cell via playMove (no-op)', function () {
    var state = midGame(GL.createGame(2));
    state.board[3][3] = { owner: 1, count: 1 };
    var r = GL.playMove(state, 3, 3);
    assertEqual(r.state.currentPlayerIndex, 0);
    assertEqual(r.state.board[3][3].count, 1);
  });

  // ---------- Turn order, elimination and winning ----------

  test('turn advances to next player after a move', function () {
    var state = GL.createGame(3);
    var r = GL.playMove(state, 3, 3);
    assertEqual(r.state.currentPlayerIndex, 1);
  });

  test('board dimensions stay 7x7 by default', function () {
    var state = GL.createGame(2);
    assertEqual(state.board.length, 7);
    assertEqual(state.board[0].length, 7);
  });

  test('no eliminations occur before every player has had one turn', function () {
    var state = GL.createGame(2, 7, 7);
    var r = GL.playMove(state, 0, 0);
    assertEqual(r.state.gameOver, false);
    assertEqual(r.state.players[0].active, true);
    assertEqual(r.state.players[1].active, true);
  });

  test('a player with zero cells is eliminated once all players have moved', function () {
    var state = midGame(GL.createGame(2, 7, 7));
    state.board[5][5] = { owner: 1, count: 3 }; // reaches 4 on this move and explodes
    state.currentPlayerIndex = 1;
    state.totalMoves = 1;
    var r = GL.playMove(state, 5, 5);
    assertEqual(r.state.players[0].active, false);
    assertEqual(r.state.gameOver, true);
    assertEqual(r.state.winner, 1);
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
    assertEqual(r.state.players[0].active, true);
    assertEqual(r.state.gameOver, false);
  });

  test('4p: no elimination before every player has moved once, even with a zero-cell player', function () {
    var state = GL.createGame(4, 7, 7);
    state.totalMoves = 2;
    state.currentPlayerIndex = 0;
    var r = GL.playMove(state, 0, 0);
    assertEqual(r.state.players[1].active, true);
    assertEqual(r.state.gameOver, false);
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
    assertEqual(r1.state.players[1].active, false, 'player 1 eliminated');
    assertEqual(r1.state.gameOver, false, 'players 0, 2 and 3 still hold cells');
    assertEqual(r1.state.currentPlayerIndex, 2, 'turn skips eliminated player 1');

    var r2 = GL.playMove(r1.state, 6, 5);
    assertEqual(r2.state.currentPlayerIndex, 3);
    var r3 = GL.playMove(r2.state, 0, 5);
    assertEqual(r3.state.currentPlayerIndex, 0, 'wraps back past eliminated player 1');
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
    assertTrue(r.steps.length >= 2, 'expected a chained cascade');
    assertEqual(GL.countCellsForPlayer(r.state.board, 1), 0);
    assertEqual(GL.countCellsForPlayer(r.state.board, 2), 0);
    assertEqual(GL.countCellsForPlayer(r.state.board, 3), 0);
    assertTrue(GL.countCellsForPlayer(r.state.board, 0) > 0);
    [[2, 3], [3, 4], [3, 2], [5, 3]].forEach(function (p) {
      assertEqual(r.state.board[p[0]][p[1]].owner, 0, 'cell ' + p + ' belongs to the mover');
    });
    assertEqual(r.state.board[4][3].owner, null, '(4,3) exploded in wave 2 so it ends empty');
    assertEqual(r.state.gameOver, true);
    assertEqual(r.state.winner, 0);
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
    assertEqual(r.state.players[1].active, false);
    assertEqual(r.state.players[2].active, true);
    assertEqual(r.state.players[3].active, true);
    assertEqual(r.state.gameOver, false);
    assertEqual(r.state.winner, null);
  });

  // ---------- Placement rules ----------

  test('after the first move, the next player can place on ANY empty cell', function () {
    var state = GL.createGame(2, 7, 7);
    var r1 = GL.playMove(state, 0, 0);
    assertEqual(r1.state.currentPlayerIndex, 1);
    assertEqual(GL.isValidMove(r1.state.board, 5, 5, 1), true);
    var r2 = GL.playMove(r1.state, 5, 5);
    assertEqual(r2.state.board[5][5].owner, 1);
    assertEqual(r2.state.board[5][5].count, 3, "player 1's opening places 3 dots");
    assertEqual(r2.state.currentPlayerIndex, 0);
    var r3 = GL.playMove(r2.state, 2, 6);
    assertEqual(r3.state.board[2][6].owner, 0);
    assertEqual(r3.state.board[2][6].count, 1, 'non-opening placement adds a single dot');
  });

  // ---------- Opening-move rule ----------

  test("opening move: a player's first placement puts 3 dots on the cell", function () {
    var state = GL.createGame(2, 7, 7);
    assertEqual(GL.placementDots(state, 0), 3);
    var r = GL.playMove(state, 3, 3);
    assertEqual(r.state.board[3][3].count, 3);
    assertEqual(r.state.board[3][3].owner, 0);
    assertEqual(r.steps.length, 0, '3 dots is below the critical mass of 4');
  });

  test('opening 3 dots never detonates, even on a corner, under the fixed threshold', function () {
    var state = GL.createGame(2, 7, 7);
    var r = GL.playMove(state, 0, 6); // corner opening
    assertEqual(r.steps.length, 0, 'corner holds 3 dots now that critical mass is 4');
    assertEqual(r.state.board[0][6].count, 3);
    assertEqual(r.state.board[0][6].owner, 0);
    assertEqual(r.state.board[1][6].owner, null, 'no neighbour was fed');
    assertEqual(r.state.board[0][5].owner, null);
  });

  test('opening bonus is per-player: each player gets 3 dots on their own first move', function () {
    var s = GL.createGame(4, 7, 7);
    var spots = [[1, 1], [1, 4], [4, 1], [4, 4]];
    for (var i = 0; i < 4; i++) {
      assertEqual(GL.placementDots(s, i), 3, 'player ' + i + ' opening');
      var res = GL.playMove(s, spots[i][0], spots[i][1]);
      assertEqual(res.state.board[spots[i][0]][spots[i][1]].count, 3);
      assertEqual(res.state.board[spots[i][0]][spots[i][1]].owner, i);
      s = res.state;
    }
    for (var j = 0; j < 4; j++) assertEqual(GL.placementDots(s, j), 1);
  });

  test("after opening, a player's later moves add exactly 1 dot", function () {
    var state = GL.createGame(2, 7, 7);
    var r1 = GL.playMove(state, 1, 1);
    assertEqual(r1.state.board[1][1].count, 3);
    var r2 = GL.playMove(r1.state, 4, 4);
    assertEqual(r2.state.board[4][4].count, 3);
    assertEqual(GL.placementDots(r2.state, 0), 1);
    var r3 = GL.playMove(r2.state, 5, 1);
    assertEqual(r3.state.board[5][1].count, 1);
    // A 4th dot on player 0's opening cell reaches the threshold and detonates.
    var r4 = GL.playMove(r3.state, 4, 4); // player 1 tops up their own opening cell
    assertEqual(r4.state.board[4][4].count, 0, 'reached 4 and exploded');
    assertTrue(r4.steps.length >= 1);
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
