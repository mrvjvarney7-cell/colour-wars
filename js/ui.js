(function () {
  'use strict';

  var GL = window.GameLogic;
  var PALETTE = GL.COLOR_PALETTE;

  var FLIGHT_MS = 220;
  var WAVE_PAUSE_MS = 90;
  var POP_MS = 260;

  // ---------- AI opponent ----------
  // Each seat is independently Human or AI (setup.players[i].isAI, toggled
  // per-player in the setup screen) - any mix works, from all-human to a
  // single human among AI opponents to a fully AI-vs-AI game. MCTS
  // simulations/move is a plain tradeoff between move strength and "AI is
  // thinking" wait time - tune here if needed.
  var AI_SIMULATIONS = 60;
  var THINKING_YIELD_MS = 50; // lets the "AI is thinking" indicator paint before the blocking search runs
  var aiThinkingEl = document.getElementById('ai-thinking');

  function isAiTurn() {
    return state && state.players[state.currentPlayerIndex].isAI;
  }

  // ---------- DOM refs ----------
  var setupScreen = document.getElementById('setup-screen');
  var gameScreen = document.getElementById('game-screen');
  var winScreen = document.getElementById('win-screen');

  var aiVersionTagEl = document.getElementById('ai-version-tag');
  var aiVersionTagIngameEl = document.getElementById('ai-version-tag-ingame');
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
      { name: 'Player 1', color: PALETTE[0].hex, isAI: false },
      { name: 'Player 2', color: PALETTE[1].hex, isAI: false },
      { name: 'Player 3', color: PALETTE[2].hex, isAI: false },
      { name: 'Player 4', color: PALETTE[3].hex, isAI: false }
    ]
  };

  // States which trained checkpoint the browser AI is actually running, so
  // it's never a mystery which version you're playing against. AI_VERSION
  // is written by python -m colourwars.export_weights alongside AI_WEIGHTS.
  function formatAiVersionText() {
    var v = window.AI_VERSION;
    if (!v) return 'AI version unknown';
    var label = (v.iteration != null) ? ('AI: iteration ' + v.iteration) : ('AI: ' + v.checkpointFile);
    if (typeof v.winRateVsRandom === 'number') {
      label += ' · ' + Math.round(v.winRateVsRandom * 100) + '% vs random';
    }
    if (v.promoted) label += ' · promoted';
    return label;
  }

  function anyAiSeatsInPlay() {
    return setup.players.slice(0, setup.numPlayers).some(function (p) { return p.isAI; });
  }

  function updateAiVersionTags() {
    // Guards against a stale cached copy of index.html from before these
    // elements existed being served alongside a newer ui.js (GitHub Pages'
    // cache-control means that skew is a real possibility, not hypothetical -
    // it's exactly what broke Start Game entirely before this guard existed).
    // A visitor missing the version tag is a much smaller problem than a
    // visitor who can't place a single dot.
    if (!aiVersionTagEl || !aiVersionTagIngameEl) return;
    var show = anyAiSeatsInPlay();
    var text = formatAiVersionText();
    aiVersionTagEl.textContent = text;
    aiVersionTagEl.classList.toggle('hidden', !show);
    aiVersionTagIngameEl.textContent = text;
    aiVersionTagIngameEl.classList.toggle('hidden', !show);
  }

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

        var aiToggle = document.createElement('button');
        aiToggle.type = 'button';
        aiToggle.className = 'ai-toggle-btn';
        aiToggle.textContent = p.isAI ? 'AI' : 'Human';
        aiToggle.setAttribute('aria-label', 'Player ' + (i + 1) + ' is ' + (p.isAI ? 'AI' : 'Human') + ' - tap to toggle');
        if (p.isAI) aiToggle.classList.add('active');
        aiToggle.addEventListener('click', function () {
          p.isAI = !p.isAI;
          renderPlayerList();
        });
        row.appendChild(aiToggle);

        playerListEl.appendChild(row);
      })(i);
    }
    updateAiVersionTags();
  }

  // ---------- Game state ----------
  var state = null;
  var animating = false;
  var cellEls = [];
  // Incremented whenever the current game is torn down (New Game / Start Game).
  // An animation captures the epoch it started in and aborts if it no longer
  // matches, so a cascade still in flight can never paint onto - or overwrite
  // the state of - a game that has since been replaced.
  var gameEpoch = 0;

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
      cellEl.classList.remove('critical');
      cellEl.style.removeProperty('--cell-color');
      cluster.innerHTML = '';
      cluster.removeAttribute('data-count');
      return;
    }
    cellEl.classList.add('owned');
    cellEl.style.setProperty('--cell-color', playerColor(data.owner));
    // A cell can transiently hold more dots than its critical mass mid-cascade
    // (e.g. a corner whose two neighbours both explode on the same wave gains 2
    // at once). It always detonates on the very next wave, so mark it as unstable
    // rather than letting it read as a cell resting above its critical mass.
    if (data.count >= GL.getCriticalMass(r, c, state.rows, state.cols)) {
      cellEl.classList.add('critical');
    } else {
      cellEl.classList.remove('critical');
    }
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

  // True if any cell in this snapshot is at/over its critical mass, i.e. the
  // board is still mid-cascade and will explode again on the next wave.
  function boardHasCriticalCells(board) {
    for (var r = 0; r < board.length; r++) {
      for (var c = 0; c < board[0].length; c++) {
        if (board[r][c].count >= GL.getCriticalMass(r, c, board.length, board[0].length)) {
          return true;
        }
      }
    }
    return false;
  }

  function animateWave(step, color, epoch) {
    if (epoch !== gameEpoch) return Promise.resolve();
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
      // The game may have been reset mid-flight; never paint a stale board.
      if (epoch !== gameEpoch) return;
      renderBoard(step.board);
      // Don't linger on a frame that still contains cells past their critical
      // mass - those are mid-cascade states, not a position anyone can read.
      return delay(boardHasCriticalCells(step.board) ? 0 : WAVE_PAUSE_MS);
    });
  }

  function animateSteps(steps, color, epoch) {
    var chain = Promise.resolve();
    steps.forEach(function (step) {
      chain = chain.then(function () { return animateWave(step, color, epoch); });
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

  // Applies a move and animates it - shared by human clicks and AI turns so
  // both go through byte-for-byte the same rules application and rendering
  // path. Assumes the caller has already confirmed the move is legal.
  function commitMove(r, c) {
    var movingPlayerId = state.currentPlayerIndex;
    var movingColor = playerColor(movingPlayerId);
    var result = GL.playMove(state, r, c);
    var epoch = gameEpoch;

    animating = true;
    // Show the dot(s) landing on the tapped cell immediately, before any explosion
    // waves animate. Skipped when the placement detonates straight away, so the
    // cell is never painted resting above its critical mass.
    if (result.steps.length === 0) {
      var preBoard = GL.cloneBoard(state.board);
      preBoard[r][c].owner = movingPlayerId;
      preBoard[r][c].count += GL.placementDots(state, movingPlayerId);
      renderCell(r, c, preBoard);
    }

    return animateSteps(result.steps, movingColor, epoch).then(function () {
      // If the game was reset while this cascade was animating, this result
      // belongs to a game that no longer exists - discard it rather than
      // overwriting the new game's state and board.
      if (epoch !== gameEpoch) return;
      state = result.state;
      renderBoard(state.board);
      renderTurnIndicator();
      renderPlayersStrip();
      animating = false;
      if (state.gameOver) {
        showWinScreen();
      } else {
        maybePlayAiTurn(epoch);
      }
    });
  }

  function onCellClick(e) {
    if (animating || !state || state.gameOver || isAiTurn()) return;
    var r = Number(e.currentTarget.dataset.row);
    var c = Number(e.currentTarget.dataset.col);
    if (!GL.isValidMove(state.board, r, c, state.currentPlayerIndex, state.players[state.currentPlayerIndex].hasMoved)) return;
    commitMove(r, c);
  }

  // If it's currently an AI seat's turn, shows the "thinking" indicator,
  // yields to the browser so it actually paints before the blocking search
  // runs, then computes and plays the AI's move through the same
  // commitMove() path a human click uses. Recurses via commitMove's own
  // post-move check, so a run of consecutive AI seats (3p/4p games) plays
  // itself out automatically until it's the human's turn again.
  function maybePlayAiTurn(epoch) {
    if (!isAiTurn() || state.gameOver) return;
    aiThinkingEl.classList.remove('hidden');
    setTimeout(function () {
      if (epoch !== gameEpoch) return; // game was reset while we were waiting to start
      var root = MCTS.runMcts(state, AI_WEIGHTS, AI_SIMULATIONS);
      var action = MCTS.bestAction(root);
      aiThinkingEl.classList.add('hidden');
      if (epoch !== gameEpoch || action === null) return;
      var r = Math.floor(action / state.cols);
      var c = action % state.cols;
      commitMove(r, c);
    }, THINKING_YIELD_MS);
  }

  // Abandons any cascade still animating from a previous game, so it cannot
  // paint onto or overwrite the game that replaces it.
  function resetAnimationState() {
    gameEpoch++;
    animating = false;
    fxLayerEl.innerHTML = '';
    aiThinkingEl.classList.add('hidden');
  }

  function startGame() {
    resetAnimationState();
    var game = GL.createGame(setup.numPlayers);
    for (var i = 0; i < setup.numPlayers; i++) {
      game.players[i].name = setup.players[i].name;
      game.players[i].color = setup.players[i].color;
      game.players[i].isAI = setup.players[i].isAI;
    }
    state = game;
    buildBoardDom(state.rows, state.cols);
    renderBoard(state.board);
    renderTurnIndicator();
    renderPlayersStrip();
    // Screen transition happens before the more decorative updateAiVersionTags()
    // call - if that (or any future addition here) throws, the player still
    // reaches a playable board instead of getting stuck on the setup screen.
    winScreen.classList.add('hidden');
    setupScreen.classList.add('hidden');
    gameScreen.classList.remove('hidden');
    updateAiVersionTags();
  }

  function backToSetup() {
    resetAnimationState();
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
