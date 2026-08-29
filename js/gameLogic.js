// Pure, DOM-free game logic for Chain Reaction / Color Wars.
// Usable from the browser (attaches to window.GameLogic) and from Node (module.exports).

(function (root) {
  'use strict';

  var ROWS = 7;
  var COLS = 7;

  // Each player's very first move of the game places this many dots instead of
  // one, to get games moving faster. Every later move places a single dot.
  var OPENING_DOTS = 3;

  // Every cell on the board explodes at the same threshold, regardless of
  // position: corners, edges and interior cells all detonate at 4 dots and lose
  // exactly 4 when they do. A cell with fewer than 4 orthogonal neighbours still
  // sends only one dot to each neighbour it actually has - the remaining dots
  // are discarded.
  var CRITICAL_MASS = 4;

  function getNeighbors(row, col, rows, cols) {
    var out = [];
    if (row > 0) out.push([row - 1, col]);
    if (row < rows - 1) out.push([row + 1, col]);
    if (col > 0) out.push([row, col - 1]);
    if (col < cols - 1) out.push([row, col + 1]);
    return out;
  }

  // Fixed for every cell on the board. Arguments are accepted so callers can pass
  // a position, but the threshold no longer depends on it.
  function getCriticalMass() {
    return CRITICAL_MASS;
  }

  function createEmptyBoard(rows, cols) {
    rows = rows || ROWS;
    cols = cols || COLS;
    var board = [];
    for (var r = 0; r < rows; r++) {
      var rowArr = [];
      for (var c = 0; c < cols; c++) {
        rowArr.push({ owner: null, count: 0 });
      }
      board.push(rowArr);
    }
    return board;
  }

  function cloneBoard(board) {
    return board.map(function (row) {
      return row.map(function (cell) {
        return { owner: cell.owner, count: cell.count };
      });
    });
  }

  function boardDims(board) {
    return { rows: board.length, cols: board[0].length };
  }

  // Applies a single move (placing `dots` dots, default 1, for `player` at
  // row/col) and fully resolves any resulting chain reaction. Returns:
  //   { board: <final board>, steps: [ { board: <snapshot after this wave>,
  //       exploded: [{row,col}], gains: [{row,col,fromRow,fromCol}] } ... ] }
  // `steps` lets the UI animate wave-by-wave; if no explosion occurs, steps is [].
  function applyMove(board, row, col, player, rows, cols, dots) {
    var dims = boardDims(board);
    rows = rows || dims.rows;
    cols = cols || dims.cols;
    dots = dots || 1;

    var working = cloneBoard(board);
    var cell = working[row][col];
    cell.owner = player;
    cell.count += dots;

    var steps = [];
    var guard = 0;
    var guardLimit = rows * cols * 50; // safety valve against pathological infinite loops

    while (true) {
      guard++;
      if (guard > guardLimit) break;

      var unstable = [];
      for (var r = 0; r < rows; r++) {
        for (var c = 0; c < cols; c++) {
          if (working[r][c].count >= CRITICAL_MASS) {
            unstable.push([r, c]);
          }
        }
      }

      if (unstable.length === 0) break;

      var gains = [];
      // Resolve this wave: every unstable cell explodes simultaneously.
      unstable.forEach(function (pos) {
        var er = pos[0], ec = pos[1];
        var explodingCell = working[er][ec];
        // Always loses the full critical mass, whatever its neighbour count.
        explodingCell.count -= CRITICAL_MASS;
        if (explodingCell.count <= 0) {
          explodingCell.count = 0;
          explodingCell.owner = null;
        }
        // Only orthogonal neighbours that exist receive a dot; for a corner or
        // edge cell the leftover dots are simply discarded.
        var neighbors = getNeighbors(er, ec, rows, cols);
        neighbors.forEach(function (n) {
          gains.push({ row: n[0], col: n[1], fromRow: er, fromCol: ec });
        });
      });

      gains.forEach(function (g) {
        var target = working[g.row][g.col];
        target.count += 1;
        target.owner = player;
      });

      steps.push({
        board: cloneBoard(working),
        exploded: unstable.map(function (p) { return { row: p[0], col: p[1] }; }),
        gains: gains
      });
    }

    return { board: working, steps: steps };
  }

  // A player's very first move (their opening) may target any empty cell.
  // Every move after that may only target a cell that player already owns -
  // the only way to gain new territory is by exploding into it via a chain
  // reaction, never by placing directly on an empty or opponent-owned cell.
  function isValidMove(board, row, col, player, hasMoved) {
    if (row < 0 || col < 0 || row >= board.length || col >= board[0].length) return false;
    var cell = board[row][col];
    if (hasMoved) return cell.owner === player;
    return cell.owner === null || cell.owner === player;
  }

  function countCellsForPlayer(board, player) {
    var n = 0;
    board.forEach(function (row) {
      row.forEach(function (cell) {
        if (cell.owner === player) n++;
      });
    });
    return n;
  }

  // ---- Game-level state ----

  var COLOR_PALETTE = [
    { name: 'blue', hex: '#2563eb' },
    { name: 'green', hex: '#16a34a' },
    { name: 'orange', hex: '#f97316' },
    { name: 'red', hex: '#dc2626' }
  ];

  function createGame(numPlayers, rows, cols) {
    rows = rows || ROWS;
    cols = cols || COLS;
    numPlayers = Math.max(2, Math.min(4, numPlayers));

    var players = [];
    for (var i = 0; i < numPlayers; i++) {
      players.push({
        id: i,
        name: 'Player ' + (i + 1),
        color: COLOR_PALETTE[i].hex,
        colorName: COLOR_PALETTE[i].name,
        active: true,
        // Drives the opening-move bonus: false until this player has moved once.
        hasMoved: false
      });
    }

    return {
      rows: rows,
      cols: cols,
      board: createEmptyBoard(rows, cols),
      players: players,
      currentPlayerIndex: 0,
      totalMoves: 0,
      gameOver: false,
      winner: null
    };
  }

  // How many dots `playerId`'s next placement drops: OPENING_DOTS for their very
  // first move of the game, 1 for every move after that.
  function placementDots(state, playerId) {
    var p = state.players[playerId];
    return (p && !p.hasMoved) ? OPENING_DOTS : 1;
  }

  function nextActivePlayerIndex(state, fromIndex) {
    var n = state.players.length;
    for (var step = 1; step <= n; step++) {
      var idx = (fromIndex + step) % n;
      if (state.players[idx].active) return idx;
    }
    return fromIndex;
  }

  // Applies a move to a game state (does NOT mutate the input state).
  // Returns { state: <new game state>, steps: <animation steps from applyMove> }
  function playMove(state, row, col) {
    if (state.gameOver) {
      return { state: state, steps: [] };
    }
    var player = state.currentPlayerIndex;
    if (!isValidMove(state.board, row, col, player, state.players[player].hasMoved)) {
      return { state: state, steps: [] };
    }

    var dots = placementDots(state, player);
    var result = applyMove(state.board, row, col, player, state.rows, state.cols, dots);
    var newState = {
      rows: state.rows,
      cols: state.cols,
      board: result.board,
      players: state.players.map(function (p) { return Object.assign({}, p); }),
      currentPlayerIndex: state.currentPlayerIndex,
      totalMoves: state.totalMoves + 1,
      gameOver: false,
      winner: null
    };
    // This player has now opened; their later moves place a single dot.
    newState.players[player].hasMoved = true;

    // Only start checking eliminations once every player has had at least one turn.
    var firstRoundComplete = newState.totalMoves >= newState.players.length;
    if (firstRoundComplete) {
      newState.players.forEach(function (p) {
        if (p.active && countCellsForPlayer(newState.board, p.id) === 0) {
          p.active = false;
        }
      });
    }

    var activePlayers = newState.players.filter(function (p) { return p.active; });
    if (firstRoundComplete && activePlayers.length === 1) {
      newState.gameOver = true;
      newState.winner = activePlayers[0].id;
      newState.currentPlayerIndex = player;
    } else {
      newState.currentPlayerIndex = nextActivePlayerIndex(newState, player);
    }

    return { state: newState, steps: result.steps };
  }

  // ---- CWN (Colour Wars Notation) ----
  // Position notation: "<board> <mover> <notYetOpened> <ply>"
  //   board: 7 rows separated by '/', each row a sequence of digit-runs
  //     (count of consecutive empty cells) and letters (occupied cells) - a
  //     cell can never rest AT its critical mass (4, see applyMove), so only
  //     counts 1-3 ever need encoding. Player 0 = a/b/c (1/2/3 dots), player
  //     1 = d/e/f, player 2 = g/h/i, player 3 = j/k/l.
  //   mover: current player index, one digit.
  //   notYetOpened: every player index that hasn't made their opening move
  //     yet, concatenated (e.g. "01"), or '-' once everyone has.
  //   ply: total moves played so far.
  // Example, an empty 2-player board: "7/7/7/7/7/7/7 0 01 0".
  //
  // numPlayers has no field of its own - decodeCwn recovers it as one more
  // than the highest player index referenced anywhere (board owners, mover,
  // notYetOpened). A player who has both opened AND been reduced to zero
  // cells (eliminated) without being the current mover leaves no trace in
  // any of those three places, so a position reached that way would
  // silently decode with too few players - a known limitation of the
  // format as specified (no player-count field), not a bug to route around
  // by inventing one.
  var CWN_LETTERS = 'abcdefghijkl';

  function cwnCharForCell(cell) {
    if (cell.owner === null || cell.count === 0) return null;
    return CWN_LETTERS.charAt(cell.owner * 3 + (cell.count - 1));
  }

  function encodeCwnRow(row) {
    var out = '';
    var emptyRun = 0;
    for (var c = 0; c < row.length; c++) {
      var ch = cwnCharForCell(row[c]);
      if (ch === null) {
        emptyRun++;
      } else {
        if (emptyRun > 0) { out += emptyRun; emptyRun = 0; }
        out += ch;
      }
    }
    if (emptyRun > 0) out += emptyRun;
    return out;
  }

  function encodeCwn(state) {
    var boardStr = state.board.map(encodeCwnRow).join('/');
    var notOpened = state.players
      .map(function (p, i) { return p.hasMoved ? null : i; })
      .filter(function (i) { return i !== null; })
      .join('');
    if (notOpened === '') notOpened = '-';
    return boardStr + ' ' + state.currentPlayerIndex + ' ' + notOpened + ' ' + state.totalMoves;
  }

  function decodeCwnRow(rowStr, cols) {
    var cells = [];
    for (var i = 0; i < rowStr.length; i++) {
      var ch = rowStr.charAt(i);
      if (ch >= '0' && ch <= '9') {
        var n = Number(ch);
        for (var k = 0; k < n; k++) cells.push({ owner: null, count: 0 });
      } else {
        var idx = CWN_LETTERS.indexOf(ch);
        if (idx === -1) throw new Error('invalid CWN board character: "' + ch + '"');
        cells.push({ owner: Math.floor(idx / 3), count: (idx % 3) + 1 });
      }
    }
    if (cells.length !== cols) {
      throw new Error('CWN row "' + rowStr + '" decoded to ' + cells.length + ' cells, expected ' + cols);
    }
    return cells;
  }

  function decodeCwn(cwn) {
    var parts = cwn.trim().split(/\s+/);
    if (parts.length !== 4) throw new Error('CWN must have 4 space-separated fields, got ' + parts.length);
    var boardStr = parts[0], moverStr = parts[1], notOpenedStr = parts[2], plyStr = parts[3];

    var rowStrs = boardStr.split('/');
    if (rowStrs.length !== ROWS) throw new Error('CWN board must have ' + ROWS + ' rows, got ' + rowStrs.length);
    var board = rowStrs.map(function (rowStr) { return decodeCwnRow(rowStr, COLS); });

    var mover = Number(moverStr);
    var ply = Number(plyStr);
    if (!Number.isFinite(mover) || !Number.isFinite(ply)) {
      throw new Error('CWN mover/ply fields must be numbers, got "' + moverStr + '"/"' + plyStr + '"');
    }
    var notOpened = (notOpenedStr === '-') ? [] : notOpenedStr.split('').map(Number);

    var maxPlayerIndex = mover;
    board.forEach(function (row) {
      row.forEach(function (cell) {
        if (cell.owner !== null && cell.owner > maxPlayerIndex) maxPlayerIndex = cell.owner;
      });
    });
    notOpened.forEach(function (i) { if (i > maxPlayerIndex) maxPlayerIndex = i; });
    var numPlayers = Math.max(2, Math.min(4, maxPlayerIndex + 1));

    var players = [];
    for (var i = 0; i < numPlayers; i++) {
      players.push({
        id: i,
        name: 'Player ' + (i + 1),
        color: COLOR_PALETTE[i].hex,
        colorName: COLOR_PALETTE[i].name,
        active: true,
        hasMoved: notOpened.indexOf(i) === -1
      });
    }

    // Mirrors playMove's own elimination/game-over derivation (same
    // first-round-complete gate), applied once up front for this snapshot
    // instead of incrementally across a real move sequence.
    var firstRoundComplete = ply >= numPlayers;
    if (firstRoundComplete) {
      players.forEach(function (p) {
        if (countCellsForPlayer(board, p.id) === 0) p.active = false;
      });
    }
    var activePlayers = players.filter(function (p) { return p.active; });
    var gameOver = firstRoundComplete && activePlayers.length === 1;

    return {
      rows: ROWS,
      cols: COLS,
      board: board,
      players: players,
      currentPlayerIndex: mover,
      totalMoves: ply,
      gameOver: gameOver,
      winner: gameOver ? activePlayers[0].id : null
    };
  }

  var GameLogic = {
    ROWS: ROWS,
    COLS: COLS,
    OPENING_DOTS: OPENING_DOTS,
    CRITICAL_MASS: CRITICAL_MASS,
    COLOR_PALETTE: COLOR_PALETTE,
    placementDots: placementDots,
    getNeighbors: getNeighbors,
    getCriticalMass: getCriticalMass,
    createEmptyBoard: createEmptyBoard,
    cloneBoard: cloneBoard,
    applyMove: applyMove,
    isValidMove: isValidMove,
    countCellsForPlayer: countCellsForPlayer,
    createGame: createGame,
    playMove: playMove,
    nextActivePlayerIndex: nextActivePlayerIndex,
    encodeCwn: encodeCwn,
    decodeCwn: decodeCwn
  };

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = GameLogic;
  } else {
    root.GameLogic = GameLogic;
  }
})(typeof window !== 'undefined' ? window : globalThis);
