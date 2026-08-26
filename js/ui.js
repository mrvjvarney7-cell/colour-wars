(function () {
  'use strict';

  var GL = window.GameLogic;
  var PALETTE = GL.COLOR_PALETTE;

  var FLIGHT_MS = 220;
  var WAVE_PAUSE_MS = 90;
  var POP_MS = 260;

  // ---------- DOM refs ----------
  var setupScreen = document.getElementById('setup-screen');
  var gameScreen = document.getElementById('game-screen');
  var winScreen = document.getElementById('win-screen');

  var playerCountButtonsEl = document.getElementById('player-count-buttons');
  var playerListEl = document.getElementById('player-list');
  var startGameBtn = document.getElementById('start-game-btn');

  var turnLabelEl = document.getElementById('turn-label');
  var turnDotEl = document.getElementById('turn-dot');
  var playersStripEl = document.getElementById('players-strip');
  var boardEl = document.getElementById('board');
  var fxLayerEl = document.getElementById('fx-layer');
  var boardWrapEl = document.querySelector('.board-wrap');
  var newGameBtn = document.getElementById('new-game-btn');

  var winDotEl = document.getElementById('win-dot');
  var winTitleEl = document.getElementById('win-title');
  var playAgainBtn = document.getElementById('play-again-btn');

  // ---------- Setup state ----------
  var setup = {
    numPlayers: 2,
    players: [
      { name: 'Player 1', color: PALETTE[0].hex },
      { name: 'Player 2', color: PALETTE[1].hex },
      { name: 'Player 3', color: PALETTE[2].hex },
      { name: 'Player 4', color: PALETTE[3].hex }
    ]
  };

  function renderPlayerCountButtons() {
    playerCountButtonsEl.innerHTML = '';
    [2, 3, 4].forEach(function (n) {
      var btn = document.createElement('button');
      btn.type = 'button';
      btn.textContent = n;
      if (n === setup.numPlayers) btn.classList.add('active');
      btn.addEventListener('click', function () {
        setup.numPlayers = n;
        renderPlayerCountButtons();
        renderPlayerList();
      });
      playerCountButtonsEl.appendChild(btn);
    });
  }

  function swapColors(playerIndex, newColor) {
    var active = setup.players.slice(0, setup.numPlayers);
    var otherIndex = -1;
    active.forEach(function (p, i) {
      if (i !== playerIndex && p.color === newColor) otherIndex = i;
    });
    if (otherIndex !== -1) {
      var tmp = setup.players[otherIndex].color;
      setup.players[otherIndex].color = setup.players[playerIndex].color;
      setup.players[playerIndex].color = tmp;
    } else {
      setup.players[playerIndex].color = newColor;
    }
  }

  function renderPlayerList() {
    playerListEl.innerHTML = '';
    for (var i = 0; i < setup.numPlayers; i++) {
      (function (i) {
        var p = setup.players[i];
        var row = document.createElement('div');
        row.className = 'player-row';

        var input = document.createElement('input');
        input.type = 'text';
        input.maxLength = 16;
        input.value = p.name;
        input.setAttribute('aria-label', 'Player ' + (i + 1) + ' name');
        input.addEventListener('input', function () {
          p.name = input.value.trim() || ('Player ' + (i + 1));
        });
        row.appendChild(input);

        var swatches = document.createElement('div');
        swatches.className = 'swatches';
        PALETTE.forEach(function (c) {
          var sw = document.createElement('button');
          sw.type = 'button';
          sw.className = 'swatch';
          sw.style.setProperty('--swatch-color', c.hex);
          sw.setAttribute('aria-label', c.name);
          if (p.color === c.hex) sw.classList.add('selected');
          sw.addEventListener('click', function () {
            swapColors(i, c.hex);
            renderPlayerList();
          });
          swatches.appendChild(sw);
        });
        row.appendChild(swatches);

        playerListEl.appendChild(row);
      })(i);
    }
  }

  // ---------- Game state ----------
  var state = null;
  var animating = false;
  var cellEls = [];

  function buildBoardDom(rows, cols) {
    boardEl.innerHTML = '';
    cellEls = [];
    for (var r = 0; r < rows; r++) {
      var rowEls = [];
      for (var c = 0; c < cols; c++) {
        var cell = document.createElement('div');
        cell.className = 'cell';
        cell.dataset.row = r;
        cell.dataset.col = c;
        var cluster = document.createElement('div');
        cluster.className = 'dot-cluster';
        cell.appendChild(cluster);
        cell.addEventListener('click', onCellClick);
        boardEl.appendChild(cell);
        rowEls.push(cell);
      }
      cellEls.push(rowEls);
    }
  }

  function playerColor(playerId) {
    return state.players[playerId].color;
  }

  function renderCell(r, c, board) {
    var cellEl = cellEls[r][c];
    var data = board[r][c];
    var cluster = cellEl.querySelector('.dot-cluster');
    if (data.owner === null || data.count === 0) {
      cellEl.classList.remove('owned');
      cellEl.style.removeProperty('--cell-color');
      cluster.innerHTML = '';
      cluster.removeAttribute('data-count');
      return;
    }
    cellEl.classList.add('owned');
    cellEl.style.setProperty('--cell-color', playerColor(data.owner));
    cluster.setAttribute('data-count', String(data.count));
    cluster.innerHTML = '';
    for (var i = 0; i < data.count; i++) {
      var dot = document.createElement('div');
      dot.className = 'dot';
      cluster.appendChild(dot);
    }
  }

  function renderBoard(board) {
    for (var r = 0; r < board.length; r++) {
      for (var c = 0; c < board[0].length; c++) {
        renderCell(r, c, board);
      }
    }
  }

  function renderTurnIndicator() {
    var p = state.players[state.currentPlayerIndex];
    turnLabelEl.textContent = p.name + "'s turn";
    turnDotEl.style.background = p.color;
    turnDotEl.style.color = p.color;
  }

  function renderPlayersStrip() {
    playersStripEl.innerHTML = '';
    state.players.forEach(function (p, i) {
      var chip = document.createElement('div');
      chip.className = 'player-chip';
      if (!p.active) chip.classList.add('eliminated');
      if (i === state.currentPlayerIndex && p.active) chip.classList.add('current');
      chip.style.setProperty('--chip-color', p.color);
      var dot = document.createElement('span');
      dot.className = 'dot';
      dot.style.background = p.color;
      chip.appendChild(dot);
      var label = document.createElement('span');
      label.textContent = p.name;
      chip.appendChild(label);
      playersStripEl.appendChild(chip);
    });
  }

  function cellCenter(r, c) {
    var cellRect = cellEls[r][c].getBoundingClientRect();
    var wrapRect = boardWrapEl.getBoundingClientRect();
    return {
      x: cellRect.left + cellRect.width / 2 - wrapRect.left,
      y: cellRect.top + cellRect.height / 2 - wrapRect.top
    };
  }

  function delay(ms) {
    return new Promise(function (resolve) { setTimeout(resolve, ms); });
  }

  function animateWave(step, color) {
    step.exploded.forEach(function (pos) {
      var el = cellEls[pos.row][pos.col];
      el.classList.remove('pop');
      // eslint-disable-next-line no-unused-expressions
      void el.offsetWidth; // restart animation
      el.classList.add('pop');
      setTimeout(function () { el.classList.remove('pop'); }, POP_MS);
    });

    var flyers = step.gains.map(function (g) {
      var from = cellCenter(g.fromRow, g.fromCol);
      var to = cellCenter(g.row, g.col);
      var dot = document.createElement('div');
      dot.className = 'flying-dot';
      dot.style.background = color;
      dot.style.color = color;
      dot.style.left = from.x + 'px';
      dot.style.top = from.y + 'px';
      fxLayerEl.appendChild(dot);
      return { el: dot, to: to };
    });

    // Force layout so the browser registers the start position before we
    // change left/top, otherwise the transition won't animate.
    void fxLayerEl.offsetWidth;

    flyers.forEach(function (f) {
      f.el.style.left = f.to.x + 'px';
      f.el.style.top = f.to.y + 'px';
    });

    return delay(FLIGHT_MS).then(function () {
      flyers.forEach(function (f) { f.el.remove(); });
      renderBoard(step.board);
      return delay(WAVE_PAUSE_MS);
    });
  }

  function animateSteps(steps, color) {
    var chain = Promise.resolve();
    steps.forEach(function (step) {
      chain = chain.then(function () { return animateWave(step, color); });
    });
    return chain;
  }

  function showWinScreen() {
    var winner = state.players[state.winner];
    winDotEl.style.background = winner.color;
    winDotEl.style.color = winner.color;
    winTitleEl.textContent = winner.name + ' wins!';
    winScreen.classList.remove('hidden');
  }

  function onCellClick(e) {
    if (animating || !state || state.gameOver) return;
    var r = Number(e.currentTarget.dataset.row);
    var c = Number(e.currentTarget.dataset.col);
    if (!GL.isValidMove(state.board, r, c, state.currentPlayerIndex)) return;

    var movingPlayerId = state.currentPlayerIndex;
    var movingColor = playerColor(movingPlayerId);
    var result = GL.playMove(state, r, c);

    animating = true;
    // Show the dot landing on the tapped cell immediately, before any explosion waves animate.
    var preBoard = GL.cloneBoard(state.board);
    preBoard[r][c].owner = movingPlayerId;
    preBoard[r][c].count += 1;
    renderCell(r, c, preBoard);

    animateSteps(result.steps, movingColor).then(function () {
      state = result.state;
      renderBoard(state.board);
      renderTurnIndicator();
      renderPlayersStrip();
      animating = false;
      if (state.gameOver) showWinScreen();
    });
  }

  function startGame() {
    var game = GL.createGame(setup.numPlayers);
    for (var i = 0; i < setup.numPlayers; i++) {
      game.players[i].name = setup.players[i].name;
      game.players[i].color = setup.players[i].color;
    }
    state = game;
    buildBoardDom(state.rows, state.cols);
    renderBoard(state.board);
    renderTurnIndicator();
    renderPlayersStrip();
    winScreen.classList.add('hidden');
    setupScreen.classList.add('hidden');
    gameScreen.classList.remove('hidden');
  }

  function backToSetup() {
    winScreen.classList.add('hidden');
    gameScreen.classList.add('hidden');
    setupScreen.classList.remove('hidden');
  }

  startGameBtn.addEventListener('click', startGame);
  newGameBtn.addEventListener('click', backToSetup);
  playAgainBtn.addEventListener('click', backToSetup);

  renderPlayerCountButtons();
  renderPlayerList();
})();
