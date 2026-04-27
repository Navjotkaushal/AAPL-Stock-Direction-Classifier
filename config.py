import os 
from dotenv import load_dotenv


load_dotenv()


# ─── Database ─────────────────────────────────────────────────────────────────

DB_CONFIG = {
    "host":     os.getenv("DB_HOST",     "localhost"),
    "user":     os.getenv("DB_USER",     "root"),
    "password": os.getenv("DB_PASSWORD", ""),          # empty fallback, never hardcode
    "database": os.getenv("DB_NAME",     "apple_stock_prices"),
}

# ─── Stock ────────────────────────────────────────────────────────────────────

TICKER          = "AAPL"
HISTORICAL_START = "2010-01-01"

# ─── ML ───────────────────────────────────────────────────────────────────────

TEST_SIZE    = 0.2
RANDOM_STATE = 42

FEATURE_COLS = [
    "return_1d", "return_3d", "return_5d", "return_10d",
    "sma_5_ratio", "sma_10_ratio", "sma_20_ratio", "sma_50_ratio",
    "macd", "macd_signal", "macd_hist",
    "rsi_14",
    "bb_width", "bb_pct",
    "atr_ratio",
    "vol_ratio", "vol_change",
    "body", "upper_shadow", "lower_shadow",
]

# ─── SQL ──────────────────────────────────────────────────────────────────────

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