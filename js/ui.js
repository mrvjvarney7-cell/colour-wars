(function () {
  'use strict';

  var GL = window.GameLogic;
  var PALETTE = GL.COLOR_PALETTE;

  var FLIGHT_MS = 220;
  var WAVE_PAUSE_MS = 90;
  var POP_MS = 260;

  // ---------- Theme ----------
  // Dark is the default (matches style.css's bare :root palette); light
  // applies automatically when the OS prefers it, unless overridden here -
  // an explicit choice is persisted so it sticks on the next visit. Applied
  // as early as possible (top of this file, before any rendering) to avoid
  // a flash of the wrong theme.
  var THEME_STORAGE_KEY = 'colourwars-theme';
  var themeToggleBtn = document.getElementById('theme-toggle-btn');

  function getStoredTheme() {
    try { return localStorage.getItem(THEME_STORAGE_KEY); } catch (e) { return null; }
  }
  function setStoredTheme(theme) {
    try { localStorage.setItem(THEME_STORAGE_KEY, theme); } catch (e) { /* private mode etc. - just won't persist */ }
  }
  function systemPrefersLight() {
    return !!(window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches);
  }
  function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    if (themeToggleBtn) {
      var switchTo = (theme === 'light') ? 'dark' : 'light';
      themeToggleBtn.textContent = (switchTo === 'light') ? 'Light mode' : 'Dark mode';
      themeToggleBtn.setAttribute('aria-label', 'Switch to ' + switchTo + ' theme');
    }
  }

  applyTheme(getStoredTheme() || (systemPrefersLight() ? 'light' : 'dark'));

  if (themeToggleBtn) {
    themeToggleBtn.addEventListener('click', function () {
      var next = (document.documentElement.getAttribute('data-theme') === 'light') ? 'dark' : 'light';
      applyTheme(next);
      setStoredTheme(next);
    });
  }

  // ---------- AI opponent ----------
  // Each seat is independently Human or AI (setup.players[i].isAI, toggled
  // per-player in the setup screen) - any mix works, from all-human to a
  // single human among AI opponents to a fully AI-vs-AI game. MCTS
  // simulations/move is a plain tradeoff between move strength and "AI is
  // thinking" wait time - tune here if needed.
  var AI_SIMULATIONS = 60;
  var THINKING_YIELD_MS = 50; // lets the "AI is thinking" indicator paint before the blocking search runs
  var AI_INSIGHT_DISPLAY_MS = 1400; // how long the win% + considered moves stay visible before the move plays
  var AI_INSIGHT_TOP_N = 3; // how many candidate moves get a badge on the board
  var aiThinkingEl = document.getElementById('ai-thinking');
  var aiInsightEl = document.getElementById('ai-insight');

  function isAiTurn() {
    return state && state.players[state.currentPlayerIndex].isAI;
  }

  // ---------- DOM refs ----------
  var setupScreen = document.getElementById('setup-screen');
  var gameScreen = document.getElementById('game-screen');
  var winScreen = document.getElementById('win-screen');
  var historyScreen = document.getElementById('history-screen');
  var rulesScreen = document.getElementById('rules-screen');

  // The single place every screen transition goes through - every call site
  // used to hand-toggle .hidden on whichever two screens it cared about,
  // duplicated per transition. winScreen is an overlay drawn ON TOP of the
  // game screen, not one of the swappable screens here, so it isn't in this
  // map - showScreen() always hides it (every real transition should start
  // from a clean, non-overlaid view); showWinScreen() is the only thing
  // that ever reveals it, when a game actually just ended.
  var screensByName = {
    setup: setupScreen,
    game: gameScreen,
    history: historyScreen,
    rules: rulesScreen
  };

  // DOM-only: swaps which screen is visible, touches nothing else. Kept
  // separate from showScreen() below so the hashchange listener can apply a
  // route without writing location.hash right back (which would just cause
  // a redundant, harmless-but-wasteful second hashchange).
  function applyScreen(name) {
    Object.keys(screensByName).forEach(function (key) {
      // Guarded: older browser-test fixtures are static copies of markup
      // from before history/rules screens existed, so they don't have every
      // element this map lists - same reason every other optional element
      // in this file (aiInsightToggleEl, reviewGameBtn, etc.) is guarded.
      if (screensByName[key]) screensByName[key].classList.toggle('hidden', key !== name);
    });
    if (winScreen) winScreen.classList.add('hidden');
    renderNav(); // keeps the active nav-item highlight and Share/Export's enabled state current on every transition
  }

  // Every screen has a public route name distinct from its internal key -
  // "play"/"games" rather than "setup"/"history" - because those are the
  // names that will actually show up in the address bar and get shared.
  // "game" (the board) has no nav item of its own - you only ever reach it
  // by way of Play's start button, a replayed/shared position, or (later)
  // Puzzle/Analysis - so it gets a neutral route rather than one implying a
  // specific way of getting there. Deliberately NOT here: puzzle/analysis
  // mode. Those are launcher actions that land on the "game" screen with
  // different chrome, not screens of their own - giving them their own
  // route would mean tracking the same "what are we showing" state in two
  // places (the hash AND the inPuzzleMode-style flag), which is exactly the
  // drift this whole rework is meant to rule out.
  var ROUTE_FOR_SCREEN = { setup: 'play', game: 'game', history: 'games', rules: 'rules' };
  var SCREEN_FOR_ROUTE = { play: 'setup', game: 'game', games: 'history', rules: 'rules' };

  // The single place every screen transition goes through - every call site
  // used to hand-toggle .hidden on whichever two screens it cared about,
  // duplicated per transition; now they all call this, so the address bar
  // can never drift from what's actually on screen. winScreen is an overlay
  // drawn ON TOP of the game screen, not one of the swappable screens here,
  // so it isn't in screensByName/ROUTE_FOR_SCREEN - applyScreen() always
  // hides it (every real transition should start from a clean, non-overlaid
  // view); showWinScreen() is the only thing that ever reveals it, when a
  // game actually just ended.
  function showScreen(name) {
    applyScreen(name);
    var route = ROUTE_FOR_SCREEN[name];
    if (route && location.hash !== '#/' + route) location.hash = '#/' + route;
  }

  window.addEventListener('hashchange', function () {
    var route = location.hash.replace(/^#\/?/, '');
    var name = SCREEN_FOR_ROUTE[route];
    // An unrecognized/empty route (including one that names a screen that
    // doesn't exist YET, e.g. a future #/bots typed in by hand) leaves
    // whatever's currently showing alone rather than forcing a change -
    // there's nothing sensible to fall back to that isn't just guessing.
    if (!name) return;
    // Only ever reached mid-session (this listener can't fire before the
    // page has loaded once), so 'game' is always safe here - unlike the
    // one-time initial-load bootstrap below, `state` already holds
    // whatever game was in progress before the user navigated away from it.
    if (name === 'history') renderHistoryScreen(); // don't show stale data on a back/forward return to Games
    applyScreen(name);
  });

  // ---------- Navigation (drawer / sidebar) ----------
  // ONE list, rendered into the ONE #nav-list element that exists in the
  // DOM - it's presented as a slide-out drawer on mobile or a persistent
  // sidebar on desktop purely via CSS (see style.css's 860px breakpoint),
  // never as two separately-maintained element trees, so there is no way
  // for the two presentations to disagree about what's in the list.
  //
  // `action` is a real function reference (not a screen name routed through
  // a generic showScreen() call) so each item can reuse whatever its
  // existing open*() function already does beyond a bare screen swap -
  // openHistory() re-renders the list first, backToSetup() resets
  // animation/puzzle state, etc. `built: false` items have no action at
  // all and render disabled - Analysis/Bots/Engine/Settings are still
  // separate, not-yet-built steps of this same nav rework.
  var NAV_ITEMS = [
    { id: 'play', label: 'Play', built: true, action: function () { backToSetup(); } },
    { id: 'puzzle', label: 'Puzzle', built: true, action: function () { openPuzzle(); } },
    { id: 'analysis', label: 'Analysis', built: false },
    { id: 'bots', label: 'Bots', built: false },
    { id: 'games', label: 'Games', built: true, action: function () { openHistory(); } },
    { id: 'rules', label: 'Rules', built: true, action: function () { openRules(); } },
    { id: 'engine', label: 'Engine', built: false },
    { id: 'settings', label: 'Settings', built: false }
  ];

  function currentScreenName() {
    for (var key in screensByName) {
      if (screensByName[key] && !screensByName[key].classList.contains('hidden')) return key;
    }
    return null;
  }

  // Which item (if any) reads as "you are here". Per-id rather than a
  // generic screen-match for every item, because "puzzle" doesn't have a
  // screen of its own to match against - it's a mode flag on top of
  // 'game' - and forcing one abstraction to cover both cases would just
  // recreate the two-sources-of-truth problem this rework exists to avoid.
  function isNavItemActive(item) {
    var current = currentScreenName();
    if (item.id === 'play') return current === 'setup';
    if (item.id === 'games') return current === 'history';
    if (item.id === 'rules') return current === 'rules';
    if (item.id === 'puzzle') return inPuzzleMode === true;
    return false;
  }

  // True while there's a real, unfinished position on the board (a live
  // game OR an in-progress puzzle) AND the board is what's actually
  // showing - navigating to Rules/Games/etc from anywhere else can't lose
  // anything, since the board isn't what's currently in view.
  function isGameInProgress() {
    return !!state && !state.gameOver && !!gameScreen && !gameScreen.classList.contains('hidden');
  }

  function activateNavItem(item) {
    if (!item.built) return;
    if (isGameInProgress() && !window.confirm('You have a game in progress - leave it?')) return;
    item.action();
    closeDrawer(); // no-op when not open (desktop, or already closed)
  }

  var navListEl = document.getElementById('nav-list');

  function renderNav() {
    if (navListEl) {
      navListEl.innerHTML = '';
      NAV_ITEMS.forEach(function (item) {
        if (item.id === 'puzzle' && puzzles.length === 0) return; // hidden entirely, not just disabled
        var btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'nav-item';
        if (!item.built) {
          btn.className += ' nav-item-disabled';
          btn.disabled = true;
          btn.textContent = item.label;
          var soon = document.createElement('span');
          soon.className = 'nav-item-soon';
          soon.textContent = 'Soon';
          btn.appendChild(soon);
        } else {
          btn.textContent = item.label;
          if (isNavItemActive(item)) {
            btn.className += ' active';
            btn.setAttribute('aria-current', 'page');
          }
          btn.addEventListener('click', function () { activateNavItem(item); });
        }
        navListEl.appendChild(btn);
      });
    }
    // In-game actions: real, pre-existing buttons (not re-created here) -
    // just kept enabled/disabled to match whether there's anything to
    // share/export right now. Both already no-op safely if clicked with no
    // state (see shareCurrentPosition/exportCurrentGame), so this is a
    // usability nicety, not a correctness requirement.
    if (shareBtn) shareBtn.disabled = !state;
    if (exportGameBtn) exportGameBtn.disabled = !state || !gameStartCwn;
  }

  // ---------- Drawer (mobile) / sidebar (desktop) chrome ----------
  var navPanelEl = document.getElementById('nav-panel');
  var navScrimEl = document.getElementById('nav-scrim');
  var hamburgerBtn = document.getElementById('hamburger-btn');
  var sidebarCollapseBtn = document.getElementById('sidebar-collapse-btn');
  var topbarLogoLink = document.getElementById('topbar-logo-link');
  var navLogoLink = document.getElementById('nav-logo-link');
  var isDrawerOpen = false;
  var drawerSwipeStartX = null;
  var drawerSwipeStartY = null;

  function isMobileNavMode() {
    return !!(window.matchMedia && window.matchMedia('(max-width: 859px)').matches);
  }

  function trapDrawerFocus(e) {
    if (!navPanelEl) return;
    var focusable = navPanelEl.querySelectorAll('button:not([disabled]), a[href]');
    if (focusable.length === 0) return;
    var first = focusable[0];
    var last = focusable[focusable.length - 1];
    if (e.shiftKey && document.activeElement === first) {
      e.preventDefault();
      last.focus();
    } else if (!e.shiftKey && document.activeElement === last) {
      e.preventDefault();
      first.focus();
    }
  }

  function onDrawerKeydown(e) {
    if (e.key === 'Escape') { closeDrawer(); return; }
    if (e.key === 'Tab') trapDrawerFocus(e);
  }

  function openDrawer() {
    if (!navPanelEl || isDrawerOpen || !isMobileNavMode()) return;
    isDrawerOpen = true;
    navPanelEl.classList.add('open');
    navPanelEl.setAttribute('role', 'dialog');
    navPanelEl.setAttribute('aria-modal', 'true');
    if (navScrimEl) navScrimEl.classList.remove('hidden');
    document.body.classList.add('nav-open');
    if (hamburgerBtn) hamburgerBtn.setAttribute('aria-expanded', 'true');
    navPanelEl.focus();
    document.addEventListener('keydown', onDrawerKeydown);
  }

  function closeDrawer() {
    if (!isDrawerOpen) return;
    isDrawerOpen = false;
    if (navPanelEl) {
      navPanelEl.classList.remove('open');
      navPanelEl.removeAttribute('role');
      navPanelEl.removeAttribute('aria-modal');
    }
    if (navScrimEl) navScrimEl.classList.add('hidden');
    document.body.classList.remove('nav-open');
    document.removeEventListener('keydown', onDrawerKeydown);
    if (hamburgerBtn) {
      hamburgerBtn.setAttribute('aria-expanded', 'false');
      hamburgerBtn.focus();
    }
  }

  if (hamburgerBtn) {
    hamburgerBtn.addEventListener('click', function () {
      if (isDrawerOpen) closeDrawer(); else openDrawer();
    });
  }
  if (navScrimEl) navScrimEl.addEventListener('click', closeDrawer);

  // Swipe-left-to-close - a plain, single-touch horizontal gesture check,
  // not a full gesture library: mostly-horizontal, mostly-leftward, past a
  // small threshold so an incidental brush doesn't close it.
  if (navPanelEl) {
    navPanelEl.addEventListener('touchstart', function (e) {
      if (e.touches.length !== 1) return;
      drawerSwipeStartX = e.touches[0].clientX;
      drawerSwipeStartY = e.touches[0].clientY;
    }, { passive: true });
    navPanelEl.addEventListener('touchend', function (e) {
      if (drawerSwipeStartX === null) return;
      var touch = e.changedTouches[0];
      var dx = touch.clientX - drawerSwipeStartX;
      var dy = touch.clientY - drawerSwipeStartY;
      drawerSwipeStartX = null;
      drawerSwipeStartY = null;
      if (dx < -50 && Math.abs(dx) > Math.abs(dy) * 1.5) closeDrawer();
    }, { passive: true });
  }

  // The logo doubles as "Home" - there's no dedicated Home screen yet (see
  // the nav rework's Home-vs-Play step, not built yet), so for now it
  // lands on the same place "Play" does. Whichever logo is actually
  // visible (topbar on mobile, sidebar on desktop) navigates the same way;
  // href="#/play" is there as a plain-HTML fallback (e.g. no-JS, or a
  // right-click "open in new tab") and the click handler still runs the
  // real confirm-gated navigation on top of it.
  function goHome(e) {
    e.preventDefault();
    activateNavItem({ id: 'play', built: true, action: function () { backToSetup(); } });
  }
  if (topbarLogoLink) topbarLogoLink.addEventListener('click', goHome);
  if (navLogoLink) navLogoLink.addEventListener('click', goHome);

  // Desktop sidebar collapse - persisted alongside the theme preference
  // (see THEME_STORAGE_KEY above) so it survives a reload the same way.
  var SIDEBAR_COLLAPSED_KEY = 'colourwars-sidebar-collapsed';
  function getStoredSidebarCollapsed() {
    try { return localStorage.getItem(SIDEBAR_COLLAPSED_KEY) === 'true'; } catch (e) { return false; }
  }
  function setStoredSidebarCollapsed(collapsed) {
    try { localStorage.setItem(SIDEBAR_COLLAPSED_KEY, collapsed ? 'true' : 'false'); } catch (e) { /* private mode etc. */ }
  }
  function applySidebarCollapsed(collapsed) {
    if (navPanelEl) navPanelEl.classList.toggle('collapsed', collapsed);
    if (sidebarCollapseBtn) sidebarCollapseBtn.setAttribute('aria-expanded', collapsed ? 'false' : 'true');
  }
  applySidebarCollapsed(getStoredSidebarCollapsed());
  if (sidebarCollapseBtn) {
    sidebarCollapseBtn.addEventListener('click', function () {
      var collapsed = !navPanelEl.classList.contains('collapsed');
      applySidebarCollapsed(collapsed);
      setStoredSidebarCollapsed(collapsed);
    });
  }

  var aiInsightToggleEl = document.getElementById('ai-insight-toggle');
  var aiInsightToggleInputEl = document.getElementById('ai-insight-toggle-input');

  // Checked by default (matches the feature's original always-on behaviour);
  // unchecking skips the win%/considered-moves display, the pause that
  // exists purely so there's time to read it (so turning this off also
  // makes AI turns noticeably faster), AND the "AI is thinking..." spinner
  // itself - there was no separate control for that, and it's the same
  // "insight into what the AI is doing" surface, so one checkbox covers both.
  function showAiInsight() {
    return !!(aiInsightToggleInputEl && aiInsightToggleInputEl.checked);
  }

  var policyHeatmapToggleInputEl = document.getElementById('policy-heatmap-toggle-input');
  function showPolicyHeatmap() {
    return !!(policyHeatmapToggleInputEl && policyHeatmapToggleInputEl.checked);
  }
  if (policyHeatmapToggleInputEl) {
    policyHeatmapToggleInputEl.addEventListener('change', renderAnalysis);
  }

  var evalBarEl = document.getElementById('eval-bar');
  var evalBarFillEl = document.getElementById('eval-bar-fill');
  var evalBarLabelEl = document.getElementById('eval-bar-label');

  var playerCountButtonsEl = document.getElementById('player-count-buttons');
  var playerListEl = document.getElementById('player-list');
  var startGameBtn = document.getElementById('start-game-btn');

  var turnLabelEl = document.getElementById('turn-label');
  var turnDotEl = document.getElementById('turn-dot');
  var playersStripEl = document.getElementById('players-strip');
  var statsPanelEl = document.getElementById('stats-panel');
  var boardEl = document.getElementById('board');
  var fxLayerEl = document.getElementById('fx-layer');
  var boardWrapEl = document.querySelector('.board-wrap');
  var shareBtn = document.getElementById('share-btn');
  var exportGameBtn = document.getElementById('export-game-btn');
  var importGameBtn = document.getElementById('import-game-btn');

  var winDotEl = document.getElementById('win-dot');
  var winTitleEl = document.getElementById('win-title');
  var playAgainBtn = document.getElementById('play-again-btn');

  var backFromHistoryBtn = document.getElementById('back-from-history-btn');
  var clearHistoryBtn = document.getElementById('clear-history-btn');
  var historyListEl = document.getElementById('history-list');
  var historyStatsPanelEl = document.getElementById('history-stats-panel');

  var backFromRulesBtn = document.getElementById('back-from-rules-btn');

  var puzzleFeedbackEl = document.getElementById('puzzle-feedback');

  var moveHistoryListEl = document.getElementById('move-history-list');

  // ---------- Setup state ----------
  var setup = {
    numPlayers: 2,
    players: [
      // aiVersionIteration: null means "whatever the default/latest build
      // ships" - the zero-fetch, zero-latency path. Set only when a player's
      // row has its AI version dropdown explicitly changed (see
      // renderPlayerList) - each AI seat can be a different checkpoint.
      { name: 'Player 1', color: PALETTE[0].hex, isAI: false, aiVersionIteration: null },
      { name: 'Player 2', color: PALETTE[1].hex, isAI: false, aiVersionIteration: null },
      { name: 'Player 3', color: PALETTE[2].hex, isAI: false, aiVersionIteration: null },
      { name: 'Player 4', color: PALETTE[3].hex, isAI: false, aiVersionIteration: null }
    ]
  };

  // ---------- AI version picker (per seat) ----------
  // The browser AI defaults to whatever python -m colourwars.export_weights
  // last shipped as js/ai/weights.js (window.AI_WEIGHTS / window.AI_VERSION,
  // loaded eagerly via <script> so the default is ready with zero extra
  // latency). js/ai/versions/index.json separately lists every PROMOTED
  // iteration (python -m colourwars.export_all_versions) - picking one
  // fetches its weights on demand instead of bundling every past version
  // into the page's default load. Each AI seat picks independently (see
  // renderPlayerList's per-row <select>), so two AI opponents in the same
  // game can be different checkpoints - the cache/info maps below are keyed
  // by iteration precisely so the same version picked for two different
  // seats (or the default one, already loaded) is never fetched twice.
  var defaultIteration = window.AI_VERSION ? window.AI_VERSION.iteration : null;
  var versionWeightsCache = {}; // iteration(number) -> weights object
  var versionInfoByIteration = {}; // iteration(number) -> {iteration, elo, winRateVsRandom, measuredOnFixedHarness, promoted}
  var availableVersions = []; // [{iteration, file, elo, winRateVsRandom, measuredOnFixedHarness}, ...] from index.json, newest first

  if (defaultIteration != null) {
    versionWeightsCache[defaultIteration] = window.AI_WEIGHTS;
    versionInfoByIteration[defaultIteration] = window.AI_VERSION;
  }

  fetch('js/ai/versions/index.json')
    .then(function (res) { return res.ok ? res.json() : []; })
    .then(function (versions) {
      availableVersions = versions.slice().sort(function (a, b) { return b.iteration - a.iteration; });
      availableVersions.forEach(function (v) {
        if (!(v.iteration in versionInfoByIteration)) {
          versionInfoByIteration[v.iteration] = {
            iteration: v.iteration, elo: v.elo, winRateVsRandom: v.winRateVsRandom,
            measuredOnFixedHarness: v.measuredOnFixedHarness, preReset: v.preReset, promoted: true
          };
        }
      });
      refreshAllPlayerVersionSelects();
    })
    .catch(function () {
      // No network / fetch blocked (e.g. some browsers restrict fetch() for
      // file:// pages) - the default AI_WEIGHTS still works fine, there's
      // just nothing else to pick from.
    });

  // Resolves to weights.js's shape ({policyLogits-net weight tree}) for
  // `iteration`, fetching once and caching forever after - repeat calls for
  // an iteration already loaded (including the eagerly-bundled default)
  // resolve immediately with no network round-trip.
  function ensureVersionWeightsLoaded(iteration) {
    if (versionWeightsCache[iteration]) return Promise.resolve(versionWeightsCache[iteration]);
    var entry = availableVersions.filter(function (v) { return v.iteration === iteration; })[0];
    if (!entry) return Promise.reject(new Error('unknown AI version: iteration ' + iteration));
    return fetch('js/ai/versions/' + entry.file)
      .then(function (res) {
        if (!res.ok) throw new Error('fetch failed: ' + res.status);
        return res.json();
      })
      .then(function (weights) {
        versionWeightsCache[iteration] = weights;
        return weights;
      });
  }

  // v.measuredOnFixedHarness (set once by python -m colourwars.export_weights,
  // see derive_version_info there) is true only if this checkpoint's Elo/win
  // rate came from the 2p-paired, draw-scoring eval harness. Earlier
  // promotions were measured on a harness later found to be structurally
  // biased (deterministic games, effectively ~9 distinct outcomes, unfinished
  // games silently discarded rather than scored as draws) - showing their Elo
  // as a plain fact would ship a number that isn't one. false/undefined (an
  // older exported version.json predating this field) are treated the same
  // as each other - both mean "don't trust this Elo without a mark".
  function formatEloForDisplay(v) {
    if (typeof v.elo !== 'number') return '';
    // preReset: this iteration was promoted before the 2026-08-29 Elo reset
    // (see the "Elo was reset" note below the version picker) - its number
    // here is 0 only because the old chain was discarded, not because it
    // was actually average-strength, so show that plainly instead of a
    // number that looks like real signal.
    if (v.preReset === true) return ' · Elo: pre-reset (not comparable)';
    return v.measuredOnFixedHarness === true
      ? (' · Elo ' + Math.round(v.elo))
      : (' · Elo ~' + Math.round(v.elo) + ' (provisional)');
  }

  function anyAiSeatsInPlay() {
    return setup.players.slice(0, setup.numPlayers).some(function (p) { return p.isAI; });
  }

  var eloResetNoteEl = document.getElementById('elo-reset-note');

  function updateAiInsightToggleVisibility() {
    var show = anyAiSeatsInPlay();
    if (aiInsightToggleEl) aiInsightToggleEl.classList.toggle('hidden', !show);
    if (eloResetNoteEl) {
      var resetIteration = window.AI_VERSION && window.AI_VERSION.eloChainResetIteration;
      if (show && resetIteration != null) {
        eloResetNoteEl.textContent = 'Elo was reset at iteration ' + resetIteration +
          ' after fixing a broken evaluation system - older iterations show "not comparable" ' +
          'instead of an Elo on the same scale as newer ones.';
        eloResetNoteEl.classList.remove('hidden');
      } else {
        eloResetNoteEl.classList.add('hidden');
      }
    }
  }

  // Options for one player's version <select>: "Latest" (the eagerly-loaded
  // default, value="") first, then every other promoted iteration newest
  // first. Skips a duplicate entry if the default happens to also appear in
  // availableVersions (it always will, once index.json loads).
  // ---------- Bot ladder (T8) ----------
  // Same underlying data as before (one promoted checkpoint per entry) -
  // presented as opponents with a name/avatar/rating instead of raw build
  // artefacts ("iteration 29"), with progressive unlock: each tier opens up
  // once a human has beaten the tier below it at least once (per T7's local
  // history - no separate unlock-tracking storage needed). Ranked oldest
  // (weakest) to newest (strongest) among whatever's actually been
  // promoted, excluding the current default build (see buildVersionOptions -
  // "Latest" is a separate, always-available dev-default entry, not part of
  // the ladder to climb). Falls back to "Tier N" for any rank beyond the
  // named list, so this never breaks as more iterations get promoted.
  var BOT_LADDER = [
    { name: 'Rookie', avatar: '🌱' },
    { name: 'Challenger', avatar: '⚔️' },
    { name: 'Veteran', avatar: '🛡️' },
    { name: 'Champion', avatar: '👑' },
    { name: 'Grandmaster', avatar: '🏆' },
    { name: 'Legend', avatar: '⭐' }
  ];

  function botTierForRank(rank) {
    return BOT_LADDER[rank] || { name: 'Tier ' + (rank + 1), avatar: '🤖' };
  }

  // Ladder-eligible versions (excludes the default), oldest first = rank 0.
  function ladderVersionsAscending() {
    return availableVersions
      .filter(function (v) { return v.iteration !== defaultIteration; })
      .slice()
      .sort(function (a, b) { return a.iteration - b.iteration; });
  }

  // A bot is unlocked if it's rank 0 (always playable) or a human has beaten
  // the PREVIOUS tier at least once - "beaten" per computeHistoryStats'
  // per-bot win tracking (games with exactly one AI seat, that seat lost).
  function isBotUnlocked(rank, laddder) {
    if (rank === 0) return true;
    var previous = laddder[rank - 1];
    if (!previous) return true;
    var stats = computeHistoryStats(loadGameHistory());
    var record = stats.byBot[String(previous.iteration)];
    return !!(record && record.wins > 0);
  }

  function buildVersionOptions() {
    var opts = [];
    if (defaultIteration != null) {
      var defInfo = versionInfoByIteration[defaultIteration] || {};
      opts.push({ value: '', label: 'Latest (iteration ' + defaultIteration + ')' + formatEloForDisplay(defInfo), disabled: false });
    }
    var ladder = ladderVersionsAscending();
    ladder.forEach(function (v, rank) {
      var tier = botTierForRank(rank);
      var unlocked = isBotUnlocked(rank, ladder);
      var label;
      if (unlocked) {
        label = tier.avatar + ' ' + tier.name + formatEloForDisplay(v) +
          ' · ' + Math.round(v.winRateVsRandom * 100) + '% vs random';
      } else {
        var previousTier = botTierForRank(rank - 1);
        label = '🔒 Locked - beat ' + previousTier.name + ' to unlock';
      }
      opts.push({ value: String(v.iteration), label: label, disabled: !unlocked });
    });
    return opts;
  }

  // Re-populates every already-rendered row's version <select> in place
  // (not a full renderPlayerList() re-render, which would rebuild the name
  // <input> elements too and drop whatever a player is mid-typing) - used
  // when index.json finishes loading after the setup screen is already on
  // screen, so a row rendered before that still gets the full version list.
  function refreshAllPlayerVersionSelects() {
    var rows = playerListEl.querySelectorAll('.player-row');
    for (var i = 0; i < rows.length && i < setup.numPlayers; i++) {
      var sel = rows[i].querySelector('.player-ai-version-select');
      if (!sel) continue;
      var p = setup.players[i];
      sel.innerHTML = '';
      buildVersionOptions().forEach(function (o) {
        var opt = document.createElement('option');
        opt.value = o.value;
        opt.textContent = o.label;
        opt.disabled = o.disabled;
        sel.appendChild(opt);
      });
      // A previously-picked bot that's since become re-locked (shouldn't
      // normally happen, but a cleared history could do it) falls back to
      // the default rather than leaving the select on a disabled option.
      var wanted = (p.aiVersionIteration == null) ? '' : String(p.aiVersionIteration);
      var wantedOption = Array.prototype.filter.call(sel.options, function (o) { return o.value === wanted; })[0];
      sel.value = (wantedOption && !wantedOption.disabled) ? wanted : '';
    }
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

        // Each AI seat picks its own opponent strength independently - this
        // is NOT a single "AI difficulty for the whole game" control, see
        // buildVersionOptions(). Only shown once this row is toggled to AI.
        if (p.isAI) {
          var verSelect = document.createElement('select');
          verSelect.className = 'player-ai-version-select';
          verSelect.setAttribute('aria-label', 'Player ' + (i + 1) + ' AI version');
          buildVersionOptions().forEach(function (o) {
            var opt = document.createElement('option');
            opt.value = o.value;
            opt.textContent = o.label;
            opt.disabled = o.disabled;
            if (!o.disabled && ((p.aiVersionIteration == null && o.value === '') ||
                (p.aiVersionIteration != null && o.value === String(p.aiVersionIteration)))) {
              opt.selected = true;
            }
            verSelect.appendChild(opt);
          });
          verSelect.addEventListener('change', function () {
            var val = verSelect.value;
            p.aiVersionIteration = (val === '') ? null : Number(val);
            // Pre-fetch as soon as it's picked rather than waiting for Start
            // Game, so the common case (picked well before starting) has the
            // weights ready with no extra wait once the game actually begins.
            if (p.aiVersionIteration != null) {
              ensureVersionWeightsLoaded(p.aiVersionIteration).catch(function (err) {
                console.error('Could not pre-fetch AI version ' + p.aiVersionIteration + ':', err);
              });
            }
          });
          row.appendChild(verSelect);
        }

        playerListEl.appendChild(row);
      })(i);
    }
    updateAiInsightToggleVisibility();
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

  // ---------- Move history / position browsing ----------
  // boardHistory[0] is the empty starting board; boardHistory[i] is the
  // board immediately after moveList[i-1]. viewIndex is which snapshot is
  // currently displayed - equal to boardHistory.length-1 when "live" (the
  // normal case), or earlier while the player is browsing past positions
  // with the arrow keys or the move list, during which the board is
  // read-only and the live game keeps advancing untouched underneath.
  var boardHistory = [];
  var moveList = [];
  var viewIndex = 0;
  // The CWN of whatever position this game actually began from - almost
  // always a fresh empty board, but a share-link load (see the ?cwn= check
  // near the bottom of this file) starts from wherever that link pointed
  // instead. Recorded once at game start so a full-game export (T6) always
  // has the real starting point, not an assumption that every game starts empty.
  var gameStartCwn = null;
  // Longest single-move cascade this game (most explosion waves triggered
  // by one placed dot) - tracked incrementally in commitMove, surfaced in
  // the finished-game record T7's history stores (see recordFinishedGame).
  var longestChainThisGame = 0;
  // Per-move {drop, label} from the most recent runGameReview() (T4), keyed
  // by index into moveList - null until a review has actually run, or after
  // any new game start/replay (a review is specific to one game's moves).
  var moveReviewData = null;

  function isViewingLive() {
    return viewIndex === boardHistory.length - 1;
  }

  // Standard board-game algebraic notation: files a-g left to right, ranks
  // 1-7 bottom to top - row 0 is rendered at the top of the DOM grid, so
  // rank counts down from rows as r increases.
  function toAlgebraic(r, c) {
    return String.fromCharCode(97 + c) + (state.rows - r);
  }

  // Inverse of toAlgebraic - used by game import to turn a stored move list
  // (algebraic notation, same as the move-history panel already shows) back
  // into board coordinates for replaying via GL.playMove.
  function fromAlgebraic(notation, rows) {
    var c = notation.charCodeAt(0) - 97;
    var rank = Number(notation.slice(1));
    return { row: rows - rank, col: c };
  }

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

        // Coordinate labels: column letters along the bottom row, row
        // numbers along the left column, both in the cell's top-left corner
        // (the bottom-left cell carries both, e.g. "a1").
        var coordText = '';
        if (r === rows - 1) coordText += String.fromCharCode(97 + c);
        if (c === 0) coordText += String(rows - r);
        if (coordText) {
          var coord = document.createElement('span');
          coord.className = 'cell-coord';
          coord.textContent = coordText;
          cell.appendChild(coord);
        }

        var cluster = document.createElement('div');
        cluster.className = 'dot-cluster';
        cell.appendChild(cluster);
        cell.addEventListener('click', onCellClick);
        // r/c are `var`-declared loop counters shared by every iteration's
        // closures (function-scoped, not block-scoped) - by the time any of
        // these fire, they'd hold the loop's final post-exit value (rows,
        // cols) for every single cell, not the cell they're attached to.
        // Reading row/col from e.currentTarget.dataset at event time instead
        // (same fix onCellClick already uses) sidesteps that entirely.
        cell.addEventListener('mouseenter', function (e) {
          showChainPreview(Number(e.currentTarget.dataset.row), Number(e.currentTarget.dataset.col));
        });
        cell.addEventListener('mouseleave', clearChainPreview);
        cell.addEventListener('touchstart', function (e) {
          var row = Number(e.currentTarget.dataset.row);
          var col = Number(e.currentTarget.dataset.col);
          longPressFiredForTouch = false;
          clearTimeout(longPressTimer);
          longPressTimer = setTimeout(function () {
            longPressFiredForTouch = true;
            showChainPreview(row, col);
          }, LONG_PRESS_MS);
        }, { passive: true });
        cell.addEventListener('touchmove', cancelLongPress);
        cell.addEventListener('touchcancel', cancelLongPress);
        cell.addEventListener('touchend', function (e) {
          clearTimeout(longPressTimer);
          if (longPressFiredForTouch) {
            // A long-press already showed the preview - lifting the finger
            // should just dismiss it, not also commit the move via the
            // synthetic click browsers fire after touchend.
            e.preventDefault();
            clearChainPreview();
          }
        });
        boardEl.appendChild(cell);
        rowEls.push(cell);
      }
      cellEls.push(rowEls);
    }
  }

  function playerColor(playerId) {
    return state.players[playerId].color;
  }

  // ---------- Chain preview (hover on desktop, long-press on mobile) ----------
  // Highlights every cell a candidate move would detonate, cascading through
  // the full chain - a pure rules simulation via GL.applyMove (same function
  // commitMove uses for the real thing), never touching actual game state,
  // so hovering around costs nothing and can never desync the real board.
  var LONG_PRESS_MS = 380;
  var chainPreviewCells = [];
  var longPressTimer = null;
  var longPressFiredForTouch = false;

  function cancelLongPress() {
    clearTimeout(longPressTimer);
    if (longPressFiredForTouch) clearChainPreview();
    longPressFiredForTouch = false;
  }

  function clearChainPreview() {
    chainPreviewCells.forEach(function (el) {
      el.classList.remove('chain-preview-first', 'chain-preview-cascade');
      el.style.removeProperty('--chain-preview-color');
    });
    chainPreviewCells = [];
  }

  function showChainPreview(r, c) {
    if (animating || !state || state.gameOver || isAiTurn() || !isViewingLive()) return;
    var player = state.currentPlayerIndex;
    var hasMoved = state.players[player].hasMoved;
    if (!GL.isValidMove(state.board, r, c, player, hasMoved)) return;

    clearChainPreview();
    var dots = GL.placementDots(state, player);
    var result = GL.applyMove(state.board, r, c, player, state.rows, state.cols, dots);
    var color = playerColor(player);
    result.steps.forEach(function (step, waveIndex) {
      var cls = waveIndex === 0 ? 'chain-preview-first' : 'chain-preview-cascade';
      step.exploded.forEach(function (pos) {
        var el = cellEls[pos.row][pos.col];
        el.style.setProperty('--chain-preview-color', color);
        el.classList.add(cls);
        chainPreviewCells.push(el);
      });
    });
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

  // ---------- Quick position analysis (no search) ----------
  // One NN.forward() call on the LIVE position - shared groundwork for T3's
  // heatmap here and T1's eval bar (next). Deliberately always uses the
  // DEFAULT bundled network (window.AI_WEIGHTS), not whichever checkpoint an
  // AI opponent happens to be playing with (see the per-seat AI version
  // picker) - this is a neutral analysis engine available in every game,
  // including human-vs-human, the same way a chess site's eval bar isn't
  // tied to whatever bot you chose to play against. Only covers the LIVE
  // position, not a browsed-history frame - boardHistory only stores board
  // snapshots, not the mover/hasMoved bookkeeping encodeState also needs.
  function computeQuickAnalysis() {
    if (!state || state.gameOver) return null;
    var encoded = Encode.encodeState(state);
    var out = NeuralNet.forward(encoded, window.AI_WEIGHTS);
    var mover = state.currentPlayerIndex;
    return {
      winProbability: (out.value[mover] + 1) / 2,
      policy: MCTS.maskedPolicy(state, out.policyLogits)
    };
  }

  function clearPolicyHeatmap() {
    for (var r = 0; r < cellEls.length; r++) {
      for (var c = 0; c < cellEls[r].length; c++) {
        cellEls[r][c].classList.remove('heatmap-cell');
        cellEls[r][c].style.removeProperty('--heatmap-alpha');
      }
    }
  }

  // analysis is whatever computeQuickAnalysis() returned (or null) -
  // rendered here rather than recomputed, since renderAnalysis() below
  // already paid for the one forward pass this and renderEvalBar both need.
  function renderPolicyHeatmapFrom(analysis) {
    clearPolicyHeatmap();
    if (!showPolicyHeatmap() || !analysis) return;
    var cols = state.cols;
    // Normalized to THIS position's own strongest candidate, not a fixed
    // scale - a policy can be sharply peaked or fairly flat depending on the
    // position, and a fixed scale would make a flat-but-still-informative
    // position look like the network has no opinion at all.
    var maxProb = 0;
    for (var a = 0; a < analysis.policy.length; a++) {
      if (analysis.policy[a] > maxProb) maxProb = analysis.policy[a];
    }
    if (maxProb <= 0) return;
    for (a = 0; a < analysis.policy.length; a++) {
      if (analysis.policy[a] <= 0) continue;
      var r = Math.floor(a / cols), c = a % cols;
      var el = cellEls[r][c];
      el.classList.add('heatmap-cell');
      el.style.setProperty('--heatmap-alpha', String(analysis.policy[a] / maxProb));
    }
  }

  function renderEvalBarFrom(analysis) {
    if (!evalBarEl) return;
    if (!analysis) {
      evalBarEl.classList.add('hidden');
      return;
    }
    evalBarEl.classList.remove('hidden');
    var pct = Math.round(analysis.winProbability * 100);
    evalBarFillEl.style.height = pct + '%';
    evalBarFillEl.style.setProperty('--eval-bar-color', playerColor(state.currentPlayerIndex));
    evalBarLabelEl.textContent = pct + '%';
  }

  // Single entry point: computes the (possibly expensive-ish, though this
  // network is tiny) forward pass ONCE and feeds both T1 (eval bar) and T3
  // (policy heatmap) from it, rather than each recomputing independently.
  // null whenever there's nothing sensible to analyse - no game, game over,
  // or browsing move history (a browsed frame has no mover/hasMoved
  // bookkeeping to encode, only boardHistory's board snapshot).
  function renderAnalysis() {
    var analysis = (state && !state.gameOver && isViewingLive()) ? computeQuickAnalysis() : null;
    renderEvalBarFrom(analysis);
    renderPolicyHeatmapFrom(analysis);
  }

  // ---------- Game review (T4) ----------
  // Fewer simulations than live play (AI_SIMULATIONS=60) - "a short search
  // at each position" per the brief, and a full game can be dozens of
  // positions, each evaluated once here (see runGameReview).
  var REVIEW_SIMULATIONS = 30;

  // Thresholds are in win-probability PERCENTAGE POINTS lost, from the
  // mover's own equity just before their move to their own equity just
  // after it (both from the same fixed default-network search, so every
  // move in a review is judged by the same referee regardless of which
  // checkpoint actually played it). Chosen as round, chess.com/lichess-
  // flavoured numbers, not calibrated against this specific game's actual
  // swing distribution - a reasonable starting point, easy to retune once
  // real reviewed games show whether they're too strict/lax in practice.
  var REVIEW_THRESHOLDS = [
    { max: 0.02, label: 'best' },
    { max: 0.08, label: 'inaccuracy' },
    { max: 0.20, label: 'mistake' },
    { max: Infinity, label: 'blunder' }
  ];

  function classifyDrop(drop) {
    for (var i = 0; i < REVIEW_THRESHOLDS.length; i++) {
      if (drop <= REVIEW_THRESHOLDS[i].max) return REVIEW_THRESHOLDS[i].label;
    }
    return 'blunder';
  }

  // Full-game, absolute-player-id-indexed win probability from one search -
  // rootInsight() only gives the CURRENT mover's value; a move's "after"
  // evaluation needs the MOVER's value in a position where it's no longer
  // their turn, so this reads straight from the root node instead.
  function evaluateAllPlayers(evalState) {
    var root = MCTS.runMcts(evalState, window.AI_WEIGHTS, REVIEW_SIMULATIONS);
    var out = [];
    for (var k = 0; k < evalState.players.length; k++) {
      out.push((root.valueSum[k] / root.visitCount + 1) / 2);
    }
    return out;
  }

  var reviewGameBtn = document.getElementById('review-game-btn');
  var reviewProgressEl = null; // created on demand, see runGameReview

  // Replays the just-finished (or just-loaded) game from gameStartCwn move
  // by move - same replay mechanism as T6/T7 - evaluating each position
  // ONCE (N+1 evaluations for an N-move game, not 2N) and deriving each
  // move's win-probability drop from two consecutive evaluations. Runs
  // asynchronously (one setTimeout-yielded position at a time) so a search
  // taking real wall-clock time per position doesn't freeze the page, and
  // shows progress since a full game can take a while at REVIEW_SIMULATIONS.
  function runGameReview() {
    if (!gameStartCwn || moveList.length === 0) return;
    var reviewMoveList = moveList.slice();
    var replayState = GL.decodeCwn(gameStartCwn);

    if (!reviewProgressEl) {
      reviewProgressEl = document.createElement('div');
      reviewProgressEl.className = 'review-progress';
      moveHistoryListEl.parentNode.insertBefore(reviewProgressEl, moveHistoryListEl);
    }
    reviewProgressEl.classList.remove('hidden');
    if (reviewGameBtn) reviewGameBtn.disabled = true;

    var results = [];
    var prevEval = evaluateAllPlayers(replayState);
    var i = 0;

    function step() {
      if (i >= reviewMoveList.length) {
        moveReviewData = results;
        reviewProgressEl.classList.add('hidden');
        if (reviewGameBtn) reviewGameBtn.disabled = false;
        renderMoveHistoryList();
        return;
      }
      reviewProgressEl.textContent = 'Analysing move ' + (i + 1) + ' of ' + reviewMoveList.length + '…';

      var mover = replayState.currentPlayerIndex;
      var coord = fromAlgebraic(reviewMoveList[i].notation, replayState.rows);
      var moveResult = GL.playMove(replayState, coord.row, coord.col);
      replayState = moveResult.state;

      var nextEval = replayState.gameOver
        ? (function () { var v = []; for (var k = 0; k < replayState.players.length; k++) v.push(k === replayState.winner ? 1 : 0); return v; })()
        : evaluateAllPlayers(replayState);

      var drop = Math.max(0, prevEval[mover] - nextEval[mover]);
      results.push({ drop: drop, label: classifyDrop(drop) });
      prevEval = nextEval;
      i++;
      setTimeout(step, 0); // yields to the browser between positions
    }

    setTimeout(step, 0);
  }

  function clearLegalMoveHighlights() {
    for (var r = 0; r < cellEls.length; r++) {
      for (var c = 0; c < cellEls[r].length; c++) {
        cellEls[r][c].classList.remove('legal-move');
      }
    }
  }

  // Highlights every cell the current player may legally click. Without
  // this, a player's second-and-later moves being restricted to their own
  // cells (see isValidMove) is invisible until they click somewhere that
  // silently does nothing - easy to mistake for the site being broken.
  function updateLegalMoveHighlights() {
    clearLegalMoveHighlights();
    if (!state || state.gameOver || isAiTurn() || !isViewingLive()) return;
    var player = state.currentPlayerIndex;
    var hasMoved = state.players[player].hasMoved;
    for (var r2 = 0; r2 < state.rows; r2++) {
      for (var c2 = 0; c2 < state.cols; c2++) {
        if (GL.isValidMove(state.board, r2, c2, player, hasMoved)) {
          cellEls[r2][c2].classList.add('legal-move');
        }
      }
    }
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
      // Per-seat, since each AI opponent can now be a different checkpoint
      // (see the per-row version <select> in renderPlayerList) - there's no
      // single "the AI version" to show once for the whole game any more.
      if (p.isAI && p.aiVersionInfo && p.aiVersionInfo.iteration != null) {
        var verLabel = document.createElement('span');
        verLabel.className = 'player-chip-ai-version';
        verLabel.textContent = 'iter ' + p.aiVersionInfo.iteration;
        chip.appendChild(verLabel);
      }
      playersStripEl.appendChild(chip);
    });
  }

  // Squares and dots controlled by each player, plus each figure as a
  // percentage of the WHOLE board's fixed capacity - not a share among
  // players - so early-game numbers correctly read as small (e.g. 2%) and
  // the percentages do NOT need to sum to 100% across players (most of the
  // board is typically still unclaimed). A cell can hold at most
  // criticalMass - 1 dots before it detonates, so the board's total dot
  // capacity is rows*cols*(criticalMass-1).
  function computeBoardStats(board) {
    board = board || state.board;
    var perPlayer = state.players.map(function () { return { cells: 0, dots: 0 }; });
    for (var r = 0; r < board.length; r++) {
      for (var c = 0; c < board[0].length; c++) {
        var cell = board[r][c];
        if (cell.owner !== null) {
          perPlayer[cell.owner].cells += 1;
          perPlayer[cell.owner].dots += cell.count;
        }
      }
    }
    var totalCells = state.rows * state.cols;
    var maxDotsPerCell = GL.getCriticalMass(0, 0, state.rows, state.cols) - 1;
    var totalDotCapacity = totalCells * maxDotsPerCell;
    return { perPlayer: perPlayer, totalCells: totalCells, totalDotCapacity: totalDotCapacity };
  }

  // board defaults to the live position; pass a boardHistory snapshot to
  // show the stats for whatever position is currently being browsed.
  function renderStatsPanel(board) {
    statsPanelEl.innerHTML = '';
    var boardStats = computeBoardStats(board);
    state.players.forEach(function (p, i) {
      var s = boardStats.perPlayer[i];
      var cellPct = Math.round((s.cells / boardStats.totalCells) * 100);
      var dotPct = Math.round((s.dots / boardStats.totalDotCapacity) * 100);

      var row = document.createElement('div');
      row.className = 'stat-row';
      if (!p.active) row.classList.add('eliminated');

      var dot = document.createElement('span');
      dot.className = 'stat-dot';
      dot.style.background = p.color;
      row.appendChild(dot);

      var name = document.createElement('span');
      name.className = 'stat-name';
      name.textContent = p.name;
      row.appendChild(name);

      var squares = document.createElement('span');
      squares.className = 'stat-value';
      squares.innerHTML = '<b>' + s.cells + '</b> sq (' + cellPct + '%)';
      row.appendChild(squares);

      var dots = document.createElement('span');
      dots.className = 'stat-value';
      dots.innerHTML = '<b>' + s.dots + '</b> dots (' + dotPct + '%)';
      row.appendChild(dots);

      statsPanelEl.appendChild(row);
    });
  }

  // Chess-style move table: one row per round (every player's moved once),
  // one column per seat, e.g. for a 2p game "1.  e4  c5" / "2.  Nf3  Nc6".
  // Emits flat children (ply-label, then one cell per seat) with no
  // per-row wrapper - the CSS grid's own row-wrapping lays them out, since
  // grid-template-columns below is exactly (numPlayers + 1) wide.
  function renderMoveHistoryList() {
    var numPlayers = state.players.length;
    moveHistoryListEl.style.gridTemplateColumns = 'auto repeat(' + numPlayers + ', 1fr)';
    moveHistoryListEl.innerHTML = '';

    for (var round = 0; round * numPlayers < moveList.length; round++) {
      var ply = document.createElement('span');
      ply.className = 'move-ply';
      ply.textContent = (round + 1) + '.';
      moveHistoryListEl.appendChild(ply);

      for (var seat = 0; seat < numPlayers; seat++) {
        var moveIdx = round * numPlayers + seat;
        var cell = document.createElement('span');
        cell.className = 'move-cell';
        if (moveIdx < moveList.length) {
          var m = moveList[moveIdx];
          var boardIdx = moveIdx + 1; // this move produced boardHistory[boardIdx]
          cell.appendChild(document.createTextNode(m.notation));
          cell.style.color = m.color;
          // Game review (T4) - a quality dot once runGameReview() has
          // analysed this move; absent until/unless a review has run.
          if (moveReviewData && moveReviewData[moveIdx]) {
            var badge = document.createElement('span');
            badge.className = 'move-quality move-quality-' + moveReviewData[moveIdx].label;
            badge.title = moveReviewData[moveIdx].label + ' (' + Math.round(moveReviewData[moveIdx].drop * 100) + 'pp win% drop)';
            cell.appendChild(badge);
          }
          if (boardIdx === viewIndex) cell.classList.add('viewing');
          cell.addEventListener('click', (function (idx) {
            return function () { setViewIndex(idx); };
          })(boardIdx));
        } else {
          cell.classList.add('empty');
        }
        moveHistoryListEl.appendChild(cell);
      }
    }

    var viewingCell = moveHistoryListEl.querySelector('.move-cell.viewing');
    if (viewingCell) viewingCell.scrollIntoView({ block: 'nearest' });
  }

  // Displays boardHistory[viewIndex] read-only. When browsing (not live),
  // the turn indicator is repurposed to say so and legal-move highlighting
  // is switched off, since clicks are disabled while browsing (see
  // onCellClick) - the players strip is left showing live elimination
  // status regardless, since that isn't tracked per-snapshot.
  function renderHistoryFrame() {
    var board = boardHistory[viewIndex];
    renderBoard(board);
    renderStatsPanel(board);
    if (isViewingLive()) {
      renderTurnIndicator();
      updateLegalMoveHighlights();
    } else {
      turnLabelEl.textContent = 'Viewing move ' + viewIndex + ' of ' + (boardHistory.length - 1);
      turnDotEl.style.background = 'transparent';
      clearLegalMoveHighlights();
    }
    renderMoveHistoryList();
    renderAnalysis(); // no-ops/clears (heatmap also needs the toggle on) unless live
  }

  function setViewIndex(idx) {
    idx = Math.max(0, Math.min(idx, boardHistory.length - 1));
    if (idx === viewIndex) return;
    viewIndex = idx;
    renderHistoryFrame();
  }

  document.addEventListener('keydown', function (e) {
    if (gameScreen.classList.contains('hidden')) return;
    if (e.key === 'ArrowLeft') {
      e.preventDefault();
      setViewIndex(viewIndex - 1);
    } else if (e.key === 'ArrowRight') {
      e.preventDefault();
      setViewIndex(viewIndex + 1);
    }
  });

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
    clearChainPreview(); // the board is about to change under whatever was being previewed
    var movingPlayerId = state.currentPlayerIndex;
    var movingColor = playerColor(movingPlayerId);
    var result = GL.playMove(state, r, c);
    if (result.steps.length > longestChainThisGame) longestChainThisGame = result.steps.length;
    var epoch = gameEpoch;

    animating = true;
    // Suppresses the nav scrim's backdrop-filter blur for the duration of
    // this cascade (see style.css) - blurring a cell mid-explosion is
    // visibly janky, so the drawer/sidebar's scrim falls back to its flat
    // background instead if it's open (or opened) while this is true.
    document.body.classList.add('is-animating');
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
      var wasLive = isViewingLive();
      state = result.state;
      moveList.push({ color: movingColor, notation: toAlgebraic(r, c) });
      boardHistory.push(state.board);
      // If a browsed-back view was already showing an older position, leave
      // it there rather than yanking the player forward to this new move -
      // renderHistoryFrame() re-renders whatever viewIndex currently is.
      if (wasLive) viewIndex = boardHistory.length - 1;
      renderPlayersStrip();
      renderHistoryFrame();
      animating = false;
      document.body.classList.remove('is-animating');
      if (state.gameOver) {
        recordFinishedGame();
        showWinScreen();
      } else {
        maybePlayAiTurn(epoch);
      }
    });
  }

  function onCellClick(e) {
    // Clicks only ever apply to the live position - browsing past moves is read-only.
    if (animating || !state || state.gameOver || isAiTurn() || !isViewingLive()) return;
    var r = Number(e.currentTarget.dataset.row);
    var c = Number(e.currentTarget.dataset.col);
    if (!GL.isValidMove(state.board, r, c, state.currentPlayerIndex, state.players[state.currentPlayerIndex].hasMoved)) {
      // Give SOME feedback rather than doing nothing at all - a click on an
      // illegal cell (most commonly: any cell you don't already own, once
      // you've made your opening move) used to be entirely silent, which is
      // easy to mistake for the board not responding at all.
      var cellEl = e.currentTarget;
      cellEl.classList.remove('shake');
      void cellEl.offsetWidth; // restart animation
      cellEl.classList.add('shake');
      setTimeout(function () { cellEl.classList.remove('shake'); }, 300);
      return;
    }
    if (inPuzzleMode) {
      attemptPuzzleMove(r, c, e.currentTarget);
      return;
    }
    commitMove(r, c);
  }

  // Shows the AI's own estimate of its win chances (the MCTS root's backed-up
  // Q-value for the mover, remapped from [-1,1] to a 0-100% "how much it
  // fancies its odds") and badges the top few candidate moves by visit share
  // - the moves the search spent the most simulations exploring, i.e. what
  // it seriously considered before settling on its actual choice.
  function renderAiInsight(insight, chosenAction) {
    var pct = insight.winProbability !== null ? Math.round(insight.winProbability * 100) : null;
    aiInsightEl.textContent = (pct !== null) ? ('AI: ' + pct + '% chance') : 'AI: evaluating…';
    aiInsightEl.classList.remove('hidden');

    var cols = state.cols;
    insight.moves.slice(0, AI_INSIGHT_TOP_N).forEach(function (m) {
      var r = Math.floor(m.action / cols);
      var c = m.action % cols;
      var badge = document.createElement('div');
      badge.className = 'ai-candidate-badge' + (m.action === chosenAction ? ' chosen' : '');
      badge.textContent = Math.round(m.share * 100) + '%';
      cellEls[r][c].appendChild(badge);
    });
  }

  function clearAiInsight() {
    aiInsightEl.classList.add('hidden');
    aiInsightEl.textContent = '';
    var badges = document.querySelectorAll('.ai-candidate-badge');
    for (var i = 0; i < badges.length; i++) badges[i].remove();
  }

  // If it's currently an AI seat's turn, shows the "thinking" indicator,
  // yields to the browser so it actually paints before the blocking search
  // runs, then computes the AI's move. Rather than playing it immediately,
  // shows its win-chance estimate and considered moves for a beat so
  // there's actually time to read them, then plays the move through the
  // same commitMove() path a human click uses. Recurses via commitMove's own
  // post-move check, so a run of consecutive AI seats (3p/4p games) plays
  // itself out automatically until it's the human's turn again.
  function maybePlayAiTurn(epoch) {
    if (!isAiTurn() || state.gameOver) return;
    if (showAiInsight()) aiThinkingEl.classList.remove('hidden');
    setTimeout(function () {
      if (epoch !== gameEpoch) return; // game was reset while we were waiting to start
      var root = MCTS.runMcts(state, state.players[state.currentPlayerIndex].aiWeights, AI_SIMULATIONS);
      var action = MCTS.bestAction(root);
      aiThinkingEl.classList.add('hidden');
      if (epoch !== gameEpoch || action === null) return;

      var r = Math.floor(action / state.cols);
      var c = action % state.cols;

      if (!showAiInsight()) {
        commitMove(r, c);
        return;
      }

      renderAiInsight(MCTS.rootInsight(root), action);

      setTimeout(function () {
        clearAiInsight();
        if (epoch !== gameEpoch) return; // game was reset while the insight was on screen
        commitMove(r, c);
      }, AI_INSIGHT_DISPLAY_MS);
    }, THINKING_YIELD_MS);
  }

  // Abandons any cascade still animating from a previous game, so it cannot
  // paint onto or overwrite the game that replaces it.
  function resetAnimationState() {
    gameEpoch++;
    animating = false;
    document.body.classList.remove('is-animating');
    fxLayerEl.innerHTML = '';
    aiThinkingEl.classList.add('hidden');
    clearAiInsight();
    clearTimeout(longPressTimer);
    longPressFiredForTouch = false;
    chainPreviewCells = []; // old cells are about to be discarded by buildBoardDom anyway
    // Puzzle mode (T10) is specific to whatever screen openPuzzle() set up -
    // any other way of (re)entering the game screen (new game, replay,
    // import) must not leave it active. openPuzzle() re-enables it itself,
    // immediately after calling this.
    inPuzzleMode = false;
    currentPuzzle = null;
    if (puzzleFeedbackEl) puzzleFeedbackEl.classList.add('hidden');
  }

  // Whichever weights/version-info an AI seat actually plays with: its own
  // explicit pick if that finished loading, otherwise the eagerly-bundled
  // default - covers both "never picked one" (aiVersionIteration is null)
  // and "picked one but its fetch hadn't resolved yet" (falls back rather
  // than leaving a seat with no weights at all).
  function resolveAiWeightsForPlayer(p) {
    if (p.aiVersionIteration != null && versionWeightsCache[p.aiVersionIteration]) {
      return { weights: versionWeightsCache[p.aiVersionIteration], info: versionInfoByIteration[p.aiVersionIteration] };
    }
    return { weights: window.AI_WEIGHTS, info: window.AI_VERSION };
  }

  // Building the game itself (startGameNow) is synchronous, same as before -
  // but an AI seat's chosen version might still be fetching (picked just
  // before hitting Start Game, or pre-fetch failed/is slow). This waits for
  // every seat's own pick to be ready first, so a seat never silently starts
  // on the wrong (default) network because of a race.
  function startGame() {
    var neededIterations = [];
    for (var i = 0; i < setup.numPlayers; i++) {
      var p = setup.players[i];
      if (p.isAI && p.aiVersionIteration != null && !versionWeightsCache[p.aiVersionIteration]) {
        neededIterations.push(p.aiVersionIteration);
      }
    }

    if (neededIterations.length === 0) {
      startGameNow();
      return;
    }

    startGameBtn.disabled = true;
    var originalLabel = startGameBtn.textContent;
    startGameBtn.textContent = 'Loading AI…';
    Promise.all(neededIterations.map(function (it) {
      // Swallow per-version failures here (not just at the end) so one bad
      // fetch can't stop Promise.all from ever resolving for the rest -
      // resolveAiWeightsForPlayer() falls back to the default for whichever
      // seat's version still isn't in the cache once we get here.
      return ensureVersionWeightsLoaded(it).catch(function (err) {
        console.error('Could not load AI version ' + it + ' before starting - that seat will use the default instead.', err);
      });
    })).then(function () {
      startGameBtn.disabled = false;
      startGameBtn.textContent = originalLabel;
      startGameNow();
    });
  }

  function startGameNow() {
    resetAnimationState();
    var game = GL.createGame(setup.numPlayers);
    for (var i = 0; i < setup.numPlayers; i++) {
      game.players[i].name = setup.players[i].name;
      game.players[i].color = setup.players[i].color;
      game.players[i].isAI = setup.players[i].isAI;
      if (setup.players[i].isAI) {
        var resolved = resolveAiWeightsForPlayer(setup.players[i]);
        game.players[i].aiWeights = resolved.weights;
        game.players[i].aiVersionInfo = resolved.info;
      }
    }
    state = game;
    boardHistory = [state.board];
    moveList = [];
    viewIndex = 0;
    gameStartCwn = GL.encodeCwn(state);
    longestChainThisGame = 0;
    moveReviewData = null;
    buildBoardDom(state.rows, state.cols);
    renderBoard(state.board);
    renderTurnIndicator();
    renderPlayersStrip();
    renderStatsPanel();
    renderMoveHistoryList();
    updateLegalMoveHighlights();
    renderAnalysis();
    // Screen transition happens before anything else below that could throw -
    // if it does, the player still reaches a playable board instead of
    // getting stuck on the setup screen.
    showScreen('game');
    // Every other AI turn is kicked off reactively, from inside commitMove()
    // after a preceding move - but the very first turn of a new game has no
    // preceding move to react to. Without this, an AI seated at seat 0 would
    // never play at all, and since it's not the human's turn either, the
    // game would just sit there looking stuck.
    maybePlayAiTurn(gameEpoch);
  }

  function backToSetup() {
    resetAnimationState();
    showScreen('setup');
  }

  // ---------- Local game history (T7) ----------
  // Everything here stays on-device (localStorage) - nothing is sent
  // anywhere, matching the "static site, no backend" constraint.
  var HISTORY_STORAGE_KEY = 'colourwars-history';
  var HISTORY_MAX_GAMES = 200; // a soft cap so localStorage can't grow forever

  function loadGameHistory() {
    try {
      var raw = localStorage.getItem(HISTORY_STORAGE_KEY);
      var parsed = raw ? JSON.parse(raw) : [];
      return Array.isArray(parsed) ? parsed : [];
    } catch (e) {
      return []; // private mode, corrupted data, etc. - just start fresh
    }
  }

  function saveGameHistory(list) {
    try {
      localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(list.slice(-HISTORY_MAX_GAMES)));
    } catch (e) { /* private mode / storage full - the game itself still works fine */ }
  }

  // Called once from commitMove() exactly when state.gameOver newly becomes
  // true - records everything a replay needs (T6's startCwn + moves) plus
  // the summary fields T7's stats are computed from.
  function recordFinishedGame() {
    var players = state.players.map(function (p) {
      return {
        name: p.name,
        isAI: !!p.isAI,
        aiIteration: (p.isAI && p.aiVersionInfo) ? p.aiVersionInfo.iteration : null
      };
    });
    var entry = {
      date: new Date().toISOString(),
      numPlayers: players.length,
      players: players,
      winnerIndex: state.winner,
      totalMoves: state.totalMoves,
      longestChain: longestChainThisGame,
      startCwn: gameStartCwn,
      moves: moveList.map(function (m) { return m.notation; })
    };
    var history = loadGameHistory();
    history.push(entry);
    saveGameHistory(history);
  }

  function formatOpponents(entry) {
    return entry.players.map(function (p, i) {
      var label = p.name + (p.isAI ? (' (AI' + (p.aiIteration != null ? ' iter ' + p.aiIteration : '') + ')') : '');
      return (i === entry.winnerIndex) ? label + ' ★' : label; // star marks the winner
    }).join(' vs ');
  }

  // Win rate per bot (T7): only counts games with exactly one AI seat, since
  // "did the human(s) beat this bot" isn't well-defined once there are
  // several AI opponents at once or none at all - grouped by iteration
  // because that's the actual variable being compared (see the per-seat
  // version picker), not by name (every AI seat is just "Player N").
  function computeHistoryStats(history) {
    var byBot = {}; // iteration -> {wins, total}
    var totalMoves = 0;
    var longestChain = 0;
    var longestChainDate = null;
    history.forEach(function (entry) {
      totalMoves += entry.totalMoves;
      if (entry.longestChain > longestChain) {
        longestChain = entry.longestChain;
        longestChainDate = entry.date;
      }
      var aiSeats = entry.players
        .map(function (p, i) { return { p: p, i: i }; })
        .filter(function (x) { return x.p.isAI; });
      if (aiSeats.length === 1) {
        var bot = aiSeats[0];
        var key = (bot.p.aiIteration != null) ? String(bot.p.aiIteration) : 'unknown';
        if (!byBot[key]) byBot[key] = { wins: 0, total: 0 };
        byBot[key].total++;
        if (entry.winnerIndex !== bot.i) byBot[key].wins++; // a non-AI seat won
      }
    });
    return {
      gamesPlayed: history.length,
      avgMoves: history.length ? (totalMoves / history.length) : 0,
      longestChain: longestChain,
      longestChainDate: longestChainDate,
      byBot: byBot
    };
  }

  function renderHistoryScreen() {
    var history = loadGameHistory();
    var stats = computeHistoryStats(history);

    historyStatsPanelEl.innerHTML = '';
    var lines = [
      'Games played: ' + stats.gamesPlayed,
      'Average game length: ' + (stats.gamesPlayed ? Math.round(stats.avgMoves) + ' moves' : '–'),
      'Longest chain: ' + (stats.gamesPlayed ? stats.longestChain + ' waves' : '–')
    ];
    var botKeys = Object.keys(stats.byBot);
    if (botKeys.length > 0) {
      botKeys.sort(function (a, b) { return Number(a) - Number(b); }).forEach(function (key) {
        var b = stats.byBot[key];
        var pct = Math.round((b.wins / b.total) * 100);
        lines.push('Win rate vs iter ' + key + ': ' + pct + '% (' + b.wins + '/' + b.total + ')');
      });
    }
    lines.forEach(function (line) {
      var row = document.createElement('div');
      row.className = 'stat-row';
      row.textContent = line;
      historyStatsPanelEl.appendChild(row);
    });

    historyListEl.innerHTML = '';
    if (history.length === 0) {
      var empty = document.createElement('div');
      empty.className = 'history-empty';
      empty.textContent = 'No games played yet.';
      historyListEl.appendChild(empty);
      return;
    }
    history.slice().reverse().forEach(function (entry) {
      var row = document.createElement('button');
      row.type = 'button';
      row.className = 'history-row';

      var top = document.createElement('div');
      top.className = 'history-row-top';
      var opp = document.createElement('span');
      opp.textContent = formatOpponents(entry);
      top.appendChild(opp);
      row.appendChild(top);

      var meta = document.createElement('div');
      meta.className = 'history-row-meta';
      var when = new Date(entry.date);
      meta.textContent = when.toLocaleDateString() + ' · ' + entry.totalMoves + ' moves · longest chain ' + entry.longestChain;
      row.appendChild(meta);

      row.addEventListener('click', function () {
        try {
          var replay = replayGameFromExport(entry.startCwn, entry.moves);
          loadGameFromReplay(replay);
        } catch (e) {
          window.alert('Could not replay this game: ' + e.message);
        }
      });

      historyListEl.appendChild(row);
    });
  }

  function openHistory() {
    renderHistoryScreen();
    showScreen('history');
  }

  function closeHistory() {
    showScreen('setup');
  }

  // ---------- Rules (T9) ----------
  function openRules() {
    showScreen('rules');
  }

  function closeRules() {
    showScreen('setup');
  }

  // ---------- Daily puzzle (T10) ----------
  // js/puzzles.json is mined offline by python -m colourwars.mine_puzzles
  // (a standalone script, separate from the training/eval pipeline) - each
  // entry is {cwn, solution, before, after} where "before"/"after" are the
  // mover's own win probability (per a real MCTS search) immediately before
  // and after the solution move. Fetched the same way as
  // js/ai/versions/index.json - if it 404s or the fetch is blocked (plain
  // file:// pages), the button just never appears rather than erroring.
  var puzzles = [];
  var inPuzzleMode = false;
  var currentPuzzle = null;
  var puzzleSolved = false;

  fetch('js/puzzles.json')
    .then(function (res) { return res.ok ? res.json() : []; })
    .then(function (list) {
      puzzles = Array.isArray(list) ? list : [];
      renderNav(); // Puzzle nav item is entirely absent while puzzles.length === 0
    })
    .catch(function () { /* no puzzles yet / fetch blocked - nav item stays absent */ });

  // Deterministic by UTC calendar date - same puzzle for everyone on the
  // same day, cycling through the bank once it's been exhausted (the bank
  // grows over time simply by re-running mine_puzzles.py for longer).
  function todaysPuzzle() {
    if (puzzles.length === 0) return null;
    var dayIndex = Math.floor(Date.now() / 86400000);
    return puzzles[dayIndex % puzzles.length];
  }

  function openPuzzle() {
    var puzzle = todaysPuzzle();
    if (!puzzle) return;
    resetAnimationState();
    var decoded = GL.decodeCwn(puzzle.cwn);
    state = decoded;
    boardHistory = [state.board];
    moveList = [];
    viewIndex = 0;
    gameStartCwn = puzzle.cwn;
    longestChainThisGame = 0;
    moveReviewData = null;
    inPuzzleMode = true;
    currentPuzzle = puzzle;
    puzzleSolved = false;

    buildBoardDom(state.rows, state.cols);
    renderPlayersStrip();
    renderHistoryFrame();
    turnLabelEl.textContent = 'Find ' + state.players[state.currentPlayerIndex].name + "'s winning move!";
    turnDotEl.style.background = playerColor(state.currentPlayerIndex);
    if (puzzleFeedbackEl) {
      puzzleFeedbackEl.textContent = '';
      puzzleFeedbackEl.classList.remove('hidden');
    }
    showScreen('game');
  }

  function attemptPuzzleMove(r, c, cellEl) {
    if (puzzleSolved || !currentPuzzle) return;
    var notation = toAlgebraic(r, c);
    if (notation === currentPuzzle.solution) {
      puzzleSolved = true;
      if (puzzleFeedbackEl) puzzleFeedbackEl.textContent = 'Solved! ' + notation + ' was the move.';
      var movingColor = playerColor(state.currentPlayerIndex);
      var result = GL.playMove(state, r, c);
      var epoch = gameEpoch;
      animateSteps(result.steps, movingColor, epoch).then(function () {
        if (epoch !== gameEpoch) return;
        state = result.state;
        renderBoard(state.board);
        renderPlayersStrip();
      });
    } else {
      if (puzzleFeedbackEl) puzzleFeedbackEl.textContent = 'Not quite - try again.';
      if (cellEl) {
        cellEl.classList.remove('shake');
        void cellEl.offsetWidth;
        cellEl.classList.add('shake');
        setTimeout(function () { cellEl.classList.remove('shake'); }, 300);
      }
    }
  }

  // ---------- Share / export / import (T6: CWN) ----------

  // Clipboard writes need a secure context and, in some browsers, a fresh
  // user gesture - both true for a direct button click, but this is
  // best-effort regardless: any failure (unsupported browser, blocked
  // permission, plain file:// testing) falls back to a prompt() dialog with
  // the text pre-filled and selected, so it's always at least copyable by hand.
  function copyToClipboardOrPrompt(text, promptMessage) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).catch(function () {
        window.prompt(promptMessage, text);
      });
    } else {
      window.prompt(promptMessage, text);
    }
  }

  function shareCurrentPosition() {
    if (!state) return;
    var cwn = GL.encodeCwn(state);
    var url = location.origin + location.pathname + '?cwn=' + encodeURIComponent(cwn);
    copyToClipboardOrPrompt(url, 'Link to this position (copied to your clipboard if supported):');
  }

  function exportCurrentGame() {
    if (!state || !gameStartCwn) return;
    var payload = {
      format: 'colourwars-game-v1',
      startPosition: gameStartCwn,
      moves: moveList.map(function (m) { return m.notation; })
    };
    copyToClipboardOrPrompt(JSON.stringify(payload),
      'Game export (copied to your clipboard if supported) - paste this to import elsewhere:');
  }

  // Rebuilds a full game from a CWN starting position plus a list of
  // algebraic-notation moves by replaying each one for real through
  // GL.playMove - the same rules application the live game already uses,
  // so an imported game can never diverge from what actually happened.
  function replayGameFromExport(startPosition, moves) {
    var startState = GL.decodeCwn(startPosition);
    var replayState = startState;
    var replayBoardHistory = [startState.board];
    var replayMoveList = [];
    var longestChain = 0;
    for (var i = 0; i < moves.length; i++) {
      var coord = fromAlgebraic(moves[i], startState.rows);
      var mover = replayState.currentPlayerIndex;
      var color = replayState.players[mover].color;
      var result = GL.playMove(replayState, coord.row, coord.col);
      if (result.state === replayState) {
        throw new Error('move ' + (i + 1) + ' ("' + moves[i] + '") is illegal from the position reached so far');
      }
      if (result.steps.length > longestChain) longestChain = result.steps.length;
      replayState = result.state;
      replayBoardHistory.push(replayState.board);
      replayMoveList.push({ color: color, notation: moves[i] });
    }
    return {
      state: replayState, boardHistory: replayBoardHistory, moveList: replayMoveList,
      startCwn: startPosition, longestChain: longestChain
    };
  }

  // Shared by importGame() and the ?cwn= share-link check at the bottom of
  // this file: puts an already-built {state, boardHistory, moveList,
  // startCwn} on screen the same way startGameNow() does. There's no
  // setup.players to read names/colours/AI seats from here - the imported
  // or shared state already carries its own (decodeCwn's defaults: every
  // seat human, standard names/colours - CWN encodes a position, not which
  // seats are AI or which checkpoint they use).
  function loadGameFromReplay(replay) {
    resetAnimationState();
    state = replay.state;
    boardHistory = replay.boardHistory;
    moveList = replay.moveList;
    viewIndex = boardHistory.length - 1;
    gameStartCwn = replay.startCwn;
    // The share-link path (see below) builds a replay object with no
    // "longestChain" field (there's no move history to derive it from, just
    // a bare position) - 0 is the correct starting value there too.
    longestChainThisGame = replay.longestChain || 0;
    moveReviewData = null;
    buildBoardDom(state.rows, state.cols);
    renderPlayersStrip();
    renderHistoryFrame();
    showScreen('game');
    maybePlayAiTurn(gameEpoch);
  }

  function importGame() {
    var text = window.prompt('Paste a Colour Wars game export:');
    if (!text) return;
    var payload;
    try {
      payload = JSON.parse(text);
    } catch (e) {
      window.alert('That is not valid game-export JSON.');
      return;
    }
    if (!payload || payload.format !== 'colourwars-game-v1' ||
        typeof payload.startPosition !== 'string' || !Array.isArray(payload.moves)) {
      window.alert('Unrecognized game export format.');
      return;
    }
    var replay;
    try {
      replay = replayGameFromExport(payload.startPosition, payload.moves);
    } catch (e) {
      window.alert('Could not replay this game: ' + e.message);
      return;
    }
    loadGameFromReplay(replay);
  }

  startGameBtn.addEventListener('click', startGame);
  playAgainBtn.addEventListener('click', backToSetup);
  // Guarded the same way as other optional elements throughout this file -
  // a stale cached copy of index.html from before these buttons existed
  // must not crash the rest of setup.
  if (shareBtn) shareBtn.addEventListener('click', shareCurrentPosition);
  if (exportGameBtn) exportGameBtn.addEventListener('click', exportCurrentGame);
  if (importGameBtn) importGameBtn.addEventListener('click', importGame);
  if (backFromHistoryBtn) backFromHistoryBtn.addEventListener('click', closeHistory);
  if (backFromRulesBtn) backFromRulesBtn.addEventListener('click', closeRules);
  if (reviewGameBtn) {
    reviewGameBtn.addEventListener('click', function () {
      // showWinScreen() never hides game-screen underneath (win-screen is a
      // fixed-position overlay) - hiding it just reveals the board that's
      // already there, exactly as the game ended.
      winScreen.classList.add('hidden');
      runGameReview();
    });
  }
  if (clearHistoryBtn) {
    clearHistoryBtn.addEventListener('click', function () {
      if (window.confirm('Clear all local game history? This cannot be undone.')) {
        saveGameHistory([]);
        renderHistoryScreen();
      }
    });
  }

  // A ?cwn=<encoded position> URL (see shareCurrentPosition) loads straight
  // into that position instead of the normal setup screen - and always wins
  // over whatever route (if any) is also in the URL's hash: a shared board
  // link must land on the board, full stop. loadGameFromReplay() calls
  // showScreen('game'), which overwrites the hash to #/game itself, so a
  // combined link like index.html?cwn=...#/games still ends up showing the
  // board with the hash normalized to reflect that, not stuck on #/games.
  // Falls back to normal setup (and hash-based routing, below) on any
  // decode failure - a malformed/tampered link should never leave the page
  // stuck instead of just landing somewhere real.
  var cwnParam = new URLSearchParams(location.search).get('cwn');
  var loadedFromCwn = false;
  if (cwnParam) {
    try {
      var sharedState = GL.decodeCwn(cwnParam);
      // loadGameFromReplay() itself hides setup-screen/shows game-screen -
      // renderPlayerCountButtons()/renderPlayerList() below still run
      // regardless, so setup is ready and waiting for whenever "New Game" is
      // clicked later, same as any other game.
      loadGameFromReplay({ state: sharedState, boardHistory: [sharedState.board], moveList: [], startCwn: cwnParam });
      loadedFromCwn = true;
    } catch (e) {
      console.error('Invalid ?cwn= link, falling back to normal setup.', e);
    }
  }

  // Only reached when there's no (valid) ?cwn= - a bookmarked/shared
  // #/games or #/rules link should land there directly rather than always
  // starting on Play. 'setup' is already the default view, so nothing to do
  // there. 'game' has no standalone way to initialize from a hash alone -
  // it needs ?cwn=, an actually-started game, or puzzle mode, none of which
  // have happened yet at this exact point in a fresh page load - so a bare
  // #/game with nothing else in the URL is left on the default setup
  // screen rather than showing an empty, stateless board. Never writes a
  // hash back here either way - a plain visit's URL stays clean until the
  // player actually navigates.
  if (!loadedFromCwn) {
    var initialRoute = location.hash.replace(/^#\/?/, '');
    var initialScreen = SCREEN_FOR_ROUTE[initialRoute];
    if (initialScreen === 'history') { renderHistoryScreen(); applyScreen('history'); }
    else if (initialScreen === 'rules') applyScreen('rules');
  }

  // Every branch above except 'history'/'rules' leaves applyScreen() (and
  // so renderNav()) never having run at all on a plain load - without this,
  // #nav-list would sit empty until the next real transition or the
  // js/puzzles.json fetch happens to resolve.
  renderNav();

  renderPlayerCountButtons();
  renderPlayerList();
})();
