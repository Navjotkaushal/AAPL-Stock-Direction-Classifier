import os
from dotenv import load_dotenv

# Load .env file automatically — no manual sourcing needed
load_dotenv()

# ── Database ──────────────────────────────────────────────────────────────────
DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "user":     os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "apple_stock_prices"),
}

# ── Stock ─────────────────────────────────────────────────────────────────────
TICKER           = "AAPL"
HISTORICAL_START = "2010-01-01"

# ── ML ────────────────────────────────────────────────────────────────────────
TEST_SIZE    = 0.2
RANDOM_STATE = 42

FEATURE_COLS = [
    # ── Returns
    "return_1d", "return_3d", "return_5d", "return_10d",
 
    # ── Trend (ratios only — raw SMAs dropped)
    "sma_5_ratio", "sma_10_ratio", "sma_20_ratio", "sma_50_ratio",
 
    # ── Momentum
    "macd", "macd_signal", "macd_hist",
    "rsi_14",
 
    # ── Volatility (ratios only — raw BB bands / raw ATR dropped)
    "bb_width", "bb_pct",
    "atr_ratio",
 
    # ── Volume (ratio / change only — raw vol_sma_10 dropped)
    "vol_ratio", "vol_change",
 
    # ── Candle structure
    "body", "upper_shadow", "lower_shadow",
 
    # ── Market context  (NEW)
    "spy_return_1d", "spy_return_5d",
    "aapl_vs_spy_5d",          # AAPL relative strength vs SPY
    "vix_level",
    "vix_change_1d",
 
    # ── Regime flags  (NEW)
    "trend_regime",            # 1 = price above 20d SMA
    "vol_regime",              # 1 = ATR above its 60d median (high-vol environment)
    "rsi_zone",                # 0=oversold(<30), 1=neutral, 2=overbought(>70)
 
    # ── Lag features  (NEW) — gives model short-term memory
    "return_lag_1", "return_lag_2", "return_lag_3",
    "rsi_lag_1",    "rsi_lag_2",    "rsi_lag_3",
    "macd_hist_lag_1", "macd_hist_lag_2",
]
# ── SQL ───────────────────────────────────────────────────────────────────────
UPSERT_SQL = """
    INSERT INTO stock_data (ticker, date, open, high, low, close, volume)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
        open   = VALUES(open),
        high   = VALUES(high),
        low    = VALUES(low),
        close  = VALUES(close),
        volume = VALUES(volume)
"""