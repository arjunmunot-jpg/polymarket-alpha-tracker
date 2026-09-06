"""Polybot configuration — all tunable parameters in one place."""

import os

# --- Mode ---
DRY_RUN = True  # True = simulate trades, log what WOULD happen. False = live trading.
DRY_RUN_UNLIMITED = True   # When True + DRY_RUN=True: skip MAX_OPEN_POSITIONS cap to collect learning data
LIVE_COPY_ARMED = False    # SECOND independent arm for the live copy executor
                           # (execution/copy_executor.py). A live order requires BOTH
                           # DRY_RUN=False AND this True — and either flip is Arjun's
                           # explicit chat sign-off ONLY (INV-01). Never script this.

# --- Bullpen CLI ---
BULLPEN_BIN = os.getenv("BULLPEN_BIN", os.path.expanduser("~/.bullpen/bin/bullpen"))

# --- Capital allocation ---
BALANCE_PCT_PER_TRADE = 5          # legacy (Kelly sizing now handles this in capital_allocator)
MAX_PER_MARKET_USD = 5.0           # hard cap per market
TOTAL_BUDGET_USD = 10000.0         # LEARNING MODE: raised from 25 — fake money, don't let budget block learning. Restore to 25 before going live.
MIN_TRADE_SIZE_USD = 0.50          # ignore trades smaller than this
SLIPPAGE_PCT = 2                   # slippage tolerance %

# --- Trader selection ---
TOP_N_TRADERS = 30                 # monitor up to 30 traders (fetches 90 from leaderboard)
LEADERBOARD_PERIOD = "month"       # monthly window: more data, better scores
MIN_TRADES_COUNT = 10              # skip traders with fewer trades
MIN_PNL_USD = 500.0                # $500 PnL floor (was $1000)
SCORE_WEIGHTS = {
    "pnl": 0.40,
    "volume": 0.20,
    "trade_count": 0.20,
    "recency": 0.20,
}

# --- Trader quality floor ---
MIN_TRADER_SCORE = 0.45            # was 0.60 — too aggressive, dropped 4/5 traders

# --- Safety rules (MoonDev spec) ---
MAX_TRADER_INACTIVITY_HOURS = 168  # 7 days — monthly leaderboard traders may go days without a trade
MIN_MARKET_VOLUME_USD = 5000       # LEARNING MODE: lowered from 50000 to collect more data. Restore to 50000 before going live.
MAX_RESOLVE_DAYS = 90              # LEARNING MODE: was 30 (blocked all crypto — they resolve in 2-8 months). Category filter now blocks sports, so safe to widen. Restore to 7 before going live.
MARKET_VOLUME_DEFAULT_USD = 5000.0 # realistic volume fallback when data unavailable (factor=0.50)
MAX_SPREAD_PCT = 5.0               # LEARNING MODE: widened from 2.0 to let more markets through. Restore to 2.0 before going live.
MIN_ENTRY_PRICE = 0.03             # LEARNING MODE: lowered from 0.10 — let longshots through to see how they perform. Restore to 0.10 before going live.
MAX_ENTRY_PRICE = 0.95             # skip markets where price >= 95% (near-certain YES, no EV)
MIN_COPY_TRADE_SIZE_USD = 10       # LEARNING MODE: deliberately low. Several high-performing
                                   # wallets place small, frequent 'sniper' bets that a higher floor
                                   # would filter out entirely. Raise substantially before live use.
COPY_FINANCE_ONLY = False          # COUNCIL-PROFIT-RESEARCH-2026-04-25: copy should mirror whales not enforce finance bias. Old behavior (True) cut 90% of April whale volume which is sports. Set True only if Phase 5 strategy splits whale categories.
MAX_OPEN_POSITIONS = 5             # max concurrent open positions (prevents over-leverage)

# --- Risk guard (MoonDev Risk Agent) ---
MAX_DAILY_LOSS_USD = 1000.0        # LEARNING MODE: raised from 10 — don't halt on fake-money losses. Restore to 10 before going live.
MIN_WALLET_BALANCE_USD = 5.0       # never trade if balance drops below this

# --- Per-trade confidence scoring ---
# signal = trader_score*0.55 + insider*0.15 + volume*0.10 + api_signal*0.20
# min_required = CONF_BASE + market_price * CONF_PRICE_SENSITIVITY
# Fix: old values (0.50 base, 0.40 sens) meant min=0.70 at p=0.50, but max copy signal=0.67 → blocked
# New: min=0.50 at p=0.50; a high-scoring wallet with no volume component still clears the floor
TRADE_CONFIDENCE_WEIGHTS = {
    "trader_score":   0.55,   # primary signal
    "insider_signal": 0.15,   # insider detection boost
    "volume_factor":  0.10,   # market liquidity
    "api_signal":     0.20,   # Polymarket CLOB/Gamma opportunity score
    # WAVE 8 FIX 2026-04-26 (news pipeline integration): wired the news
    # worker output into trade scoring. Was fetched but never consumed.
    # Treated as signed deviation from neutral 0.5 — adds [-w, +w] to
    # confidence based on alignment with the trade's outcome (YES vs NO).
    "news_signal":    0.10,
}
CONF_BASE = 0.25              # LEARNING MODE: lowered from 0.35 to let borderline trades through. Restore to 0.35 before going live.
CONF_PRICE_SENSITIVITY = 0.20 # was 0.40 — gentler slope, high-price markets need only slight boost

# --- Insider detection ---
INSIDER_CONFIDENCE_THRESHOLD = 0.40   # was 0.55 — lowered while collecting data post-enrichment fix
INSIDER_MIN_RESOLVED_TRADES = 5       # was 8 — let early signals through
INSIDER_LEADERBOARD_PERIOD = "month"  # monthly data for better sample size
INSIDER_DETECTION_INTERVAL = 7200     # re-scan for insiders every 2 hours
INSIDER_POLYMARKETSCAN_ENABLED = True # pull additional wallets from polymarketscan.io
INSIDER_MIN_WIN_RATE = 0.55           # minimum win rate to include PolymarketScan wallets
INSIDER_MAX_ENRICH_WORKERS = 8        # parallel threads for slug→market enrichment

# Seed roster: wallets always scanned even if absent from the current leaderboard.
#
# The curated roster is NOT published — it is the result of the discovery pipeline
# in whale_watcher.py / insider_detector.py and is the actual research output of this
# project. Reproduce your own by running the discovery pass; the selection criteria
# (min resolved trades, min win rate, P/L ratio, volume floor) are all above.
#
# Format: data/roster.json -> {"wallets": ["0x...", ...]}  (gitignored)
def _load_seed_roster() -> list[str]:
    import json, pathlib
    p = pathlib.Path(__file__).resolve().parents[2] / "data" / "roster.json"
    if not p.exists():
        return []
    return json.loads(p.read_text()).get("wallets", [])


SEED_INSIDER_WALLETS = _load_seed_roster()
# --- Overnight self-improvement (Karpathy loop) ---
RESEARCHER_MAX_EXPERIMENTS = 60       # max Bayesian experiments per night
RESEARCHER_DURATION_HOURS = 8         # stop after this many hours
RESEARCHER_MIN_SHARPE = 0.5           # only promote configs that beat this Sharpe
RESEARCHER_TRAIN_DAYS = 21            # walk-forward: optimise on days 1–21
RESEARCHER_VALIDATE_DAYS = 9          # walk-forward: validate on days 22–30

# --- Copy subscription defaults ---
EXECUTION_MODE = "auto"            # "auto" or "confirm"
EXIT_BEHAVIOR = "mirror_sells"     # "mirror_sells", "notify_me", "manual"

# --- Market scanner (proactive underpriced market discovery) ---
SCANNER_ENABLED      = True
SCANNER_INTERVAL     = 900         # LEARNING MODE: scan every 15 min (was 1800). Restore to 1800 before going live.
SCANNER_PRICE_MIN    = 0.05        # LEARNING MODE: widened from 0.10 to capture more data
SCANNER_PRICE_MAX    = 0.90        # LEARNING MODE: widened from 0.45 to capture both sides. Restore to 0.10/0.45 before going live.
SCANNER_MIN_VOLUME   = 5000        # LEARNING MODE: lowered from 50000 to find more candidates. Restore to 50000 before going live.
SCANNER_MAX_MARKETS  = 200         # LEARNING MODE: raised from 50 to scan more markets. Restore to 50 before going live.
SCANNER_FINANCE_ONLY = True        # Only trade crypto/finance/macro markets — blocks sports/entertainment/celebrity
# ULTRA-REVIEW BOND+SCANNER AUDIT 2026-04-25: bond gets its own finance-only flag
# defaulting OFF. Bond thesis (95%+ YES near resolution) doesn't depend on category.
# 10-day darkness was caused by SCANNER_FINANCE_ONLY over-correction onto bond.
BOND_FINANCE_ONLY    = False       # default OFF — bond accepts sports/all categories at $0.92+ price
SCANNER_ARB_ENABLED  = False       # Dutch book arb requires Kalshi (2nd platform). Disabled until cross-platform arb is built.
SCANNER_USE_CLAUDE_PROBABILITY = False  # 2026-04-20: disabled — API key returning 401. Re-enable after key rotation.
SCANNER_CLAUDE_MAX_CALLS = 10           # Max Claude API calls per scanner cycle (cost control)
SCANNER_CLAUDE_CACHE_TTL = 1800         # Cache Claude estimates for 30 minutes per market

# --- Polling intervals (seconds) ---
DISCOVERY_INTERVAL = 3600          # re-discover traders every 1 hour
MONITOR_INTERVAL = 300             # check copy-trade health every 5 min
LEARNING_INTERVAL = 86400          # run learning engine every 24 hours

# --- Learning engine ---
ANTHROPIC_MODEL = "claude-sonnet-4-6"
LEARNING_LOOKBACK_DAYS = 30
MAX_LEARNING_TOKENS = 4096         # keep Claude API calls efficient
LEARNING_MIN_RESOLVED_TRADES = 5   # LEARNING MODE: set by learning-loop safe fix 2026-04-17 (was 3). Restore to 10 before going live.

# MARATHON SHIP-12 (2026-04-26 IB-002 P0 subset): Karpathy 6-contract gates.
# Used by `learning_eval.py.score_candidate`. Stricter than the legacy
# RESEARCHER_MIN_SHARPE because Sharpe alone is overfit-prone (Bailey-LdP).
LEARNING_NIGHT_BUDGET_USD     = 1.00   # Anthropic spend cap per autoresearch night
RESEARCHER_MIN_DEFLATED_SHARPE = 0.0   # DSR ∈ [0, 1] (probability genuine SR > 0); promote requires > 0
RESEARCHER_MIN_WFE             = 0.5   # Pardo walk-forward efficiency floor: oos/is must be ≥ 0.5
RESEARCHER_MAX_DD_MULTIPLE     = 2.0   # OOS max-drawdown must not exceed 2× IS max-drawdown

# --- Phase 3: Capital allocation (Kelly + EV + per-strategy budgets) ---
# Updated 2026-04-17 per capital_allocation_research.md recommendations:
# - 1/N equal-weight (DeMiguel/Garlappi/Uppal 2009 — beats 14 optimized methods out-of-sample)
# - Sharpe-weighting is fragile with <30 samples per strategy (López de Prado 2016)
# - Restore Sharpe-tilting later once each strategy has 50+ resolved trades
STRATEGY_ALLOCATION_MODE = "equal_weight"  # options: "equal_weight" | "sharpe_weighted" | "inverse_vol"
STRATEGY_BUDGETS = {
    "copy":     0.20,   # 1/N equal — also covers scanner, arb, conviction
    "insider":  0.20,
    "btc_5min": 0.20,
    "weather":  0.20,
    "bond":     0.20,
    # NOTE: arb + scanner + conviction still draw from the "copy" bucket.
    # Total = 1.00 (5 strategies × 0.20 each)
}
# Research-backed insider signal weights (signal_weights_research.md, 2026-04-17)
# Anchored to arXiv 2411.13564 / 2212.05912 / 2603.03136 — sizing & concentration
# outrank win_rate in ML feature importance for informed-trader detection.
INSIDER_WEIGHTS = {
    "win_rate":      0.25,   # was 0.35 — overweighted, easy to game via cherry-picking
    "early_entry":   0.25,   # unchanged — highest-EV signal per Laika Labs / Bullpen guides
    "consistency":   0.15,   # unchanged
    "unusual_sizing": 0.20,  # was 0.15 — top permutation-importance feature in ML studies
    "concentration": 0.15,   # was 0.10 — Polymarket-specific: specialization = edge
}  # sums to 1.00
KELLY_FRACTION     = 0.25    # quarter-Kelly multiplier (industry standard, MacLean/Thorp/Ziemba)
KELLY_MAX_MULTIPLIER = 1.25  # safety cap per capital_allocation_research.md — effective Kelly ≤ 0.3125.
                             # Prevents over-betting when confidence is noisy. Thorp: full Kelly already too aggressive.
KELLY_CONFIDENCE_DAMPEN = True  # BUG H3: dampen confidence toward 0.5 before Kelly/EV (reduces overbetting)
KELLY_PRICE_SCALE_ENABLED = True  # Scale Kelly allocation by market price: longshots (10c) → 20% of normal, 50c+ → 100%
EV_MIN_THRESHOLD   = 0.00    # LEARNING MODE: require non-negative EV (was -0.02). Even dry-run should aim for profit. Restore to 0.02 before going live.
MAX_COST_PER_TRADE = 50.0    # LEARNING MODE: raised from 5 to match bigger budget. Restore to 5 before going live.

# --- NO prior (73.3% of Polymarket markets resolve NO — official polymarket.com/accuracy data) ---
NO_PRIOR_PENALTY = 0.03      # subtract from YES bet confidence
NO_PRIOR_BONUS   = 0.03      # add to NO bet confidence

# --- High-probability bond scanner (YES at $0.92–$0.99 near-certain outcomes) ---
BOND_SCANNER_ENABLED   = True
BOND_PRICE_MIN         = 0.85   # LEARNING MODE: was 0.92 (too narrow — bonds rarely exist above 92¢). Restore to 0.92 before going live.
BOND_PRICE_MAX         = 0.99   # stay below resolution threshold
BOND_MIN_VOLUME        = 5000   # $5k minimum daily volume
BOND_MAX_HOURS         = 1440   # LEARNING MODE: 60 days (was 48h — no bond market resolves that fast). Restore to 168 (7 days) before going live.
BOND_MAX_POSITION_PCT  = 0.20   # Max 20% of bond budget in a single market (black swan diversification)

# --- Dutch book arbitrage (YES+NO < $1.00 = guaranteed profit) ---
ARB_MIN_PROFIT_PCT = 0.02       # only enter arb if guaranteed profit >= 2%

# --- Correlation / logical arbitrage detector (arb_detector.py) ---
ARB_ENABLED       = True        # enable correlation arb scanning
ARB_MIN_EDGE_PCT  = 0.02        # minimum 2% edge after fees to trade

# --- Correlation arb scanner (correlation_arb.py — complement + mutual exclusion + subset/superset) ---
CORRELATION_ARB_ENABLED  = True
CORRELATION_ARB_INTERVAL = 900   # scan every 15 min
CORRELATION_ARB_MIN_PROFIT = 0.02  # min 2c profit after fees to trade

# --- Market regime detector (5-state: CRASH/BEAR/NEUTRAL/BULL/EUPHORIA) ---
REGIME_INTERVAL     = 1800      # recompute regime every 30 min
REGIME_SAMPLE_SIZE  = 100       # markets to sample for regime features

# --- Weather bot (NOAA/Open-Meteo vs Polymarket retail pricing) ---
WEATHER_ENABLED          = False  # PAUSED 2026-04-25: known bugs (clamping, EV formula, no station offsets) — IB-028 weather rebuild required before re-enable. See state/phase4_research/13_weather_strategy_kevin_badi.md.
WEATHER_INTERVAL         = 1800        # LEARNING MODE: check every 30 min (was 3600). Restore to 3600 before going live.
WEATHER_MIN_MISMATCH     = 1.05        # LEARNING MODE: was 1.2 (too tight for noisy NOAA data). Accept small edges. Restore to 2.0 before going live.
WEATHER_MAX_MISMATCH     = 3.0         # Safety cap added 2026-04-17 — weather model produced 20-50x edges in audit (overconfident). Reject anything above 3x until calibration verified.
WEATHER_MIN_VOLUME_USD   = 2000
WEATHER_MAX_ENTRY_PRICE  = 0.85        # LEARNING MODE: widened from 0.45. Restore to 0.45 before going live.
WEATHER_CITIES = [
    # US cities (NYC uses slug "nyc" -- see CITY_SLUG_OVERRIDES in weather_bot.py)
    "New York", "Los Angeles", "Chicago", "Miami", "Houston",
    "Denver", "Austin",
    # Europe
    "London", "Paris", "Munich", "Moscow",
    # China (highest volume on Polymarket)
    "Wuhan", "Shanghai", "Hong Kong", "Shenzhen", "Guangzhou", "Chongqing",
    "Beijing",
    # Asia-Pacific
    "Tokyo", "Seoul", "Taipei", "Singapore", "Jakarta", "Kuala Lumpur",
    # Oceania
    "Sydney", "Wellington",
]

# --- Whale watcher (Polymarket WebSocket $500+ trades) ---
WHALE_WS_ENABLED         = True
WHALE_HTTP_POLL_ENABLED  = True         # 2026-04-17: WS still timing out every 30s. HTTP poll every 60s as belt-and-suspenders.
WHALE_MIN_TRADE_USD      = 500          # minimum trade size to track
WHALE_LOOKBACK_HOURS     = 2            # rolling window for whale signal
WHALE_SIGNAL_THRESHOLD   = 0.7         # whale_signal value when activity detected

# --- LLM consensus swarm (Claude + Groq for high-confidence trades) ---
SWARM_ENABLED                = True      # Re-enabled 2026-04-17: only fires on uncertain band (see SWARM_CONFIDENCE_BAND)
SWARM_CONFIDENCE_THRESHOLD   = 0.5     # Lowered 2026-04-17 from 0.62 — was pre-gating the swarm band. Now aligned with SWARM_CONFIDENCE_BAND low bound so entire band fires.
SWARM_CONFIDENCE_BAND        = (0.5, 0.7)  # (low, high) — swarm only activates when low <= confidence <= high. Outside band = auto-pass to save tokens.
SWARM_DAILY_COST_CAP_USD     = 5.0     # safety ceiling: if running daily cost exceeds this, log warning (does not hard-stop)
SWARM_TIMEOUT_SECONDS        = 30      # max wait for model responses
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# --- Polymarket fee model ---
# Polymarket uses dynamic taker fees: fee = base_rate * price * (1 - price)
# Fee peaks at ~0.55c on 50c contracts, drops as price moves toward 0 or 1.
POLYMARKET_BASE_FEE_RATE = 0.022   # 2.2% base rate (as of 2026-04)

# --- Phase 4B: BTC 15-minute strategy (Polymarket "Bitcoin Up or Down") ---
# Markets are 15-min slots: slug = btc-updown-15m-{unix_ts // 900 * 900}
# Outcomes: "Up" / "Down" — resolves via Chainlink BTC/USD oracle
BTC_5MIN_ENABLED      = False     # DISABLED 2026-04-19: Apr18 backtest shows -19% ROI + inverted calibration [0.6-0.7 conf→33% HR]. BTC_ARB still runs.
BTC_5MIN_MIN_VOLUME   = 500       # 2026-04-18 self-improve: lowered from 750 — IST daytime markets (early AM ET) consistently $247-643, 750 blocked all. 500 captures $500+ while blocking ultra-thin.
BTC_5MIN_INTERVAL     = 60        # check for active markets every 60 seconds
BTC_5MIN_MIN_SCORE    = 0.3       # LEARNING MODE: was 1.0→2.0. Most neutral-market scores are 0.08-0.60, need 0.3 to fire any trades. Restore to 2.0 before going live.
BTC_5MIN_SNIPE_WINDOW = 60        # start monitoring T-60s before market close
BTC_5MIN_MIN_PRICE    = 0.10      # skip BTC trades where market price is below this (blocks fake $0.025 trades)
BTC_5MIN_FEE_BASE_RATE = 0.02    # base rate for dynamic fee calc: fee = base_rate * price * (1 - price)
BTC_5MIN_SNIPE_POLL_INTERVAL = 0.5  # snipe loop poll interval in seconds (was 2s — 500ms gives better edge)

# ULTRA-REVIEW CONFIG-DRIFT AUDIT 2026-04-25: promote 3 IB-shipped constants from
# `getattr(config, "...", default)` fallbacks to explicit module-level constants.
# Was invisible to drift audits. Now first-class.
# IB-047 calibration cap (btc_5min.py:_fire_btc_trade refuses 0.6-0.7 conf band):
BTC_5MIN_BROKEN_CONF_LO = 0.6     # lower bound (inclusive) of broken-calibration band
BTC_5MIN_BROKEN_CONF_HI = 0.7     # upper bound (exclusive). 54-trade backtest N=15 in this band: predicted 65.6%, actual 33.3%, Wilson [15.2, 58.3], -$7,862 P&L.
# IB-048 news_agent worker (cap markets per cycle):
# (NEWS_MAX_MARKETS_PER_CYCLE is defined further down in NEWS section)
# ULTRA-REVIEW POSITION-SIZING AUDIT 2026-04-25: longshot price floor.
# capital_allocator's KELLY_PRICE_SCALE damps but never to zero — at price=$0.025
# even scaled Kelly produces $8.55 bets. Hard floor refuses anything below this.
MIN_PRICE_FOR_BET     = 0.05      # absolute price floor; capital_allocator returns Kelly=0 if below
BINANCE_WS_URL        = "wss://stream.binance.com:9443/ws/btcusdt@kline_1m"

# --- BTC binary arbitrage (paired Up/Down contracts priced under $1.00 in aggregate) ---
# When BTC moves sharply, the losing side crashes before the winning side reprices.
# e.g. Up=$0.52 + Down=$0.04 = $0.56 → buy both → collect $1.00 → profit=$0.44 guaranteed.
BTC_ARB_ENABLED     = True
BTC_ARB_MIN_PROFIT  = 0.05   # Only enter if guaranteed ≥5% profit (gap must be real, not fees)
BTC_ARB_MAX_COST    = 50.0   # Max $ per side ($100 total both sides combined)

# --- BTC 5-min position sizing: scale bet by market price ---
# Longshots ($0.10 market) get 20% of what a fair-odds ($0.50) market gets.
# Formula: scale = min(1.0, price * 2.0) — preserves trades, sizes them proportionally.
BTC_ALLOCATION_SCALE_BY_PRICE = True  # scale Kelly bet by market price

# --- Phase 4C: Polymarket API signals ---
POLYMARKET_GAMMA_URL  = "https://gamma-api.polymarket.com"
POLYMARKET_CLOB_URL   = "https://clob.polymarket.com"
API_SIGNAL_CACHE_TTL  = 120       # cache API signals for 2 minutes

# --- Phase 4D: News / RAG pipeline ---
NEWS_REFRESH_INTERVAL = 1800      # refresh news every 30 minutes
# ULTRA-REVIEW CONFIG-DRIFT AUDIT 2026-04-25: promote IB-048 constant from
# getattr fallback to explicit module-level (was invisible to audits).
NEWS_MAX_MARKETS_PER_CYCLE = 50   # IB-048: cap markets per news cycle (50 × 10s timeout = 8.3min worst case)
NEWS_ARTICLES_PER_QUERY = 5       # top articles per market query
NEWS_CHROMA_PATH = os.path.join(os.path.dirname(__file__), "data", "chroma")
# Optional: set OPENAI_API_KEY in ~/.profile to enable OpenAI embeddings
# Without it, falls back to Claude-based keyword analysis
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
NEWS_API_KEY      = os.getenv("NEWS_API_KEY", "")    # newsapi.org — free tier, optional

# --- Paths ---
DB_PATH = os.path.join(os.path.dirname(__file__), "data", "trades.db")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "data", "reports")

# --- Logging ---
LOG_LEVEL = os.getenv("POLYBOT_LOG_LEVEL", "INFO")

# --- Research fleet ---
RESEARCH_ENABLED = True
RESEARCH_INGEST_INTERVAL = 21600  # 6h — how often research.main.run_all_monitors() should fire
# GEMINI_API_KEY already defined above in swarm section; research fleet reads it for video analysis.

# ── LIVE_CONSTRAINTS ──────────────────────────────────────────────────────────
# Single source of truth for what "live mode" means.
# Backtest adapters import from here, NOT from module-level vars (which are in
# LEARNING MODE with widened thresholds). Everything is a fraction or ratio —
# bankroll size is a separate variable and should never appear here.
LIVE_CONSTRAINTS = {
    # ── Signal gates (dimensionless) ──────────────────────────────────
    "btc_min_score":           3.5,    # 15-min markets — backtest-validated ✅ | score<3.5 → conf<0.5 → 20-30% win rate (anti-predictive)
    "btc_5min_min_score":      0.3,    # 5-min markets — LEARNING MODE (pending backtest validation). Range: 0.3=collect all data, validated=TBD

    # ── BTC Regime gate (2026-04-18 — backtest finding) ───────────────
    # Sideways regime (14d BTC return ±8%): 70.8% win rate, +57.6% ROI — REAL EDGE ✅
    # Trending regime (bull/bear)          : 37-38% win rate, -11 to -14% ROI — NO EDGE ❌
    # Gate prevents trading when BTC is trending. Set False to disable (learning mode).
    "btc_regime_filter_enabled": True,
    "btc_regime_threshold":      0.08,  # ±8% 14-day BTC return = sideways boundary
    "ev_min_threshold":        0.02,   # LEARNING MODE = 0.00
    "conf_base":               0.35,   # LEARNING MODE = 0.25
    "min_entry_price":         0.10,
    "max_entry_price":         0.95,
    "max_spread_pct":          0.02,   # LEARNING MODE = 0.05
    "weather_min_mismatch":    2.0,    # LEARNING MODE = 1.05
    "weather_max_entry_price": 0.45,   # LEARNING MODE = 0.85
    "bond_price_min":          0.92,   # LEARNING MODE = 0.85
    "max_resolve_days":        7,      # LEARNING MODE = 90
    "scanner_price_min":       0.10,   # LEARNING MODE = 0.05
    "scanner_price_max":       0.45,   # LEARNING MODE = 0.90
    "scanner_min_volume_usd":  50000,  # LEARNING MODE = 5000

    # ── Capital rules (ALL as fractions of bankroll) ──────────────────
    "kelly_fraction":           0.25,  # quarter-Kelly — MacLean/Thorp/Ziemba
    "kelly_max_multiplier":     1.25,  # effective cap = 0.3125 Kelly
    "kelly_confidence_dampen":  True,
    "kelly_price_scale":        True,

    "per_strategy_budget_pct":  0.20,  # 1/N equal weight (5 strategies)
    "per_market_cap_pct":       0.05,  # 5% of bankroll max per market
    "max_cost_per_trade_pct":   0.10,  # safety ceiling; Kelly is the real constraint
    "min_cost_per_trade_pct":   0.005, # 0.5% minimum — skip below this

    # ── Risk halt rules (MoonDev spec, scaled to percentages) ─────────
    "max_daily_loss_pct":       0.10,  # halt if daily P&L < -10% of bankroll
                                       # MoonDev original was $10/$25=40% — too loose at scale
    "min_balance_pct":          0.20,  # never trade if bankroll < 20% of start
    "max_open_positions":       5,

    # ── Market quality (absolute — market characteristic, not bankroll-relative) ──
    "min_market_volume_usd":    50000, # live (LEARNING = 5000)
    "btc_min_volume_usd":       1000,  # live BTC-specific threshold
    "min_copy_trade_size_usd":  500,   # live (LEARNING = 50)

    # ── Fees ──────────────────────────────────────────────────────────
    "polymarket_fee_base_rate": 0.022, # 2.2% dynamic base rate
}


# MARATHON SHIP-4 (2026-04-26 audit Patch B): champion_overrides import.
# Auto-applied at end of config.py module load. The autoresearch / researcher
# self-improvement loop writes to data/champion_overrides.json and these
# values OVERRIDE module-level config. Atomic + rollback-able + schema-checked.
# Replaces the prior fragile regex re-write of config.py source code.
# See champion_overrides.py for safety properties + rollback API.
try:
    import champion_overrides as _champion_overrides
    _CHAMPION_OVERRIDES_APPLIED = _champion_overrides.apply_overrides(globals())
except ImportError:
    _CHAMPION_OVERRIDES_APPLIED = 0
except Exception as _ex:
    import logging as _logging
    _logging.getLogger("polybot.config").warning(
        "champion_overrides apply failed: %s — using base config", _ex,
    )
    _CHAMPION_OVERRIDES_APPLIED = 0
