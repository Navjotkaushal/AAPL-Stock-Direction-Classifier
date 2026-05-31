import sys 
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd 
import numpy as np 
import yfinance as yf

from config import FEATURE_COLS, TEST_SIZE


# External data helpers 

def fetch_external(start: str, end: str) -> tuple:
    """
    Download SPY, VIX, and TLT for the given date range.
    Returns three timezone-naive Series aligned to trading days.
    """
    tickers = {"SPY": "SPY", "VIX": "^VIX", "TLT": "TLT"}
    out = {}
 
    for key, ticker in tickers.items():
        raw = yf.download(ticker, start=start, end=end,
                          progress=False, auto_adjust=True)
 
        # yfinance ≥0.2 returns MultiIndex columns (field, ticker)
        if isinstance(raw.columns, pd.MultiIndex):
            s = raw["Close"].iloc[:, 0]
        else:
            s = raw["Close"].squeeze()
 
        s.index = pd.to_datetime(s.index).tz_localize(None)
        out[key] = s
 
    return out["SPY"], out["VIX"], out["TLT"]

# Function designed for feature engineering (20+ indicators)



def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)

    # Shorthand aliases to keep lines readable 

    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]
    
    df["return_1d"] = c.pct_change(1)       # yerstday
    df["return_3d"] = c.pct_change(3)       # 3 days ago 
    df["return_5d"] = c.pct_change(5)       # 5 days ago
    df["return_10d"] = c.pct_change(10)     # 10 days ago
     
     
    # Simple moving average ratios 
    
    for w in [5, 10, 20, 50]:
        sma = c.rolling(w).mean()
        df[f"sma_{w}_ratio"] = c / sma
    sma_20 = c.rolling(20).mean() 
        
    
    # Exponenetial moving average  
    
    # df["ema_12"] = c.ewm(span=12, adjust=False).mean()   # fast EMA
    # df["ema_26"] = c.ewm(span=26, adjust=False).mean()   # slow EMA
    ema_12 = c.ewm(span=12, adjust=False).mean()
    ema_26 = c.ewm(span=26, adjust=False).mean()
    
    
    
    # MACD (Moving Average Convergence Divergence)
    
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    
    
    
    # RSI (Relative Strength Index, 14-day)
    delta        = c.diff()                                       # daily change
    gain         = delta.clip(lower=0).rolling(14).mean()         # average of up-days
    loss         = (-delta.clip(upper=0)).rolling(14).mean()      # average of down-days
    df["rsi_14"] = 100 - (100 / (1 + gain / (loss + 1e-9)))
    
    
    
    # BOLLINGER BANDS 
    
    mid             = c.rolling(20).mean()
    std             = c.rolling(20).std()
    bb_upper  = mid + 2 * std
    bb_lower  = mid - 2 * std
    df["bb_width"]  = (bb_upper - bb_lower) / sma_20   # normalised band width
    df["bb_pct"]    = (c - bb_lower) / (bb_upper - bb_lower + 1e-9)
    
    
    
    # ATR ( Average True Range, 14-day)
    
    tr = pd.concat([
                          h - l,                     # intraday range
                          (h - c.shift()).abs(),     # gap up scenario
                          (l - c.shift()).abs()      # gap down scenario
                      ], axis=1).max(axis=1)         # true range = max of the three
    atr_14    = tr.rolling(14).mean()
    df["atr_ratio"] = atr_14 / c               # % of price
    
    
    
    # Volume features
    
    vol_sma_10 = v.rolling(10).mean()
    df["vol_ratio"] = v / vol_sma_10       # 1.0 = normal, >1.5 = spike
    df["vol_change"] = v.pct_change()             # sudden volume surge day-over-day
    
    
    # Candle Structure
    
    body_top         = pd.concat([df["close"], df["open"]], axis=1).max(axis=1)
    body_bottom      = pd.concat([df["close"], df["open"]], axis=1).min(axis=1)
    df["body"]         = (df["close"] - df["open"]).abs() / df["open"]
    df["upper_shadow"] = (h - body_top)    / df["open"]   # rejection above
    df["lower_shadow"] = (body_bottom - l) / df["open"]   # rejection below 
    
    # External market context ( SPY + VIX )
    
    start = str(df.index.min().date())
    end = str((df.index.max() + pd.Timedelta(days = 1)).date())
    spy, vix, tlt = fetch_external(start, end)
    
    # Align to AAPL trading days (forward-fill any missing VIX dates)
    spy = spy.reindex(df.index, method="ffill")
    vix = vix.reindex(df.index, method="ffill")
    tlt = tlt.reindex(df.index, method="ffill")
 
    df["spy_return_1d"]   = spy.pct_change(1)
    df["spy_return_5d"]   = spy.pct_change(5)
    df["aapl_vs_spy_5d"]  = df["return_5d"] - df["spy_return_5d"]  # relative strength
    
    df["vix_level"]       = vix
    df["vix_change_1d"]   = vix.pct_change(1)
    
    tlt_ret = tlt.pct_change(5)
    df["tlt_return_5d"] = tlt_ret 
    df["tlt_trend"] = (tlt > tlt.rolling(20).mean()).astype(int)
    
    # ── 10. Regime flags ─────────────────────────────────────────────────────
    
    
    # trend_regime: is AAPL in an uptrend right now?
    df["trend_regime"] = (c > sma_20).astype(int)
 
    # vol_regime: is current volatility above its own 60-day median?
    df["vol_regime"] = (df["atr_ratio"] > df["atr_ratio"].rolling(60).median()).astype(int)
 
    # rsi_zone: 0 = oversold, 1 = neutral, 2 = overbought
    df["rsi_zone"] = pd.cut(
        df["rsi_14"],
        bins=[0, 30, 70, 100],
        labels=[0, 1, 2],
        include_lowest=True,
    ).astype(float)
 
 
 
     # ── 11. Earnings proximity ───────────────────────────────────────────────
    # AAPL reports roughly at end of Jan, Apr, Jul, Oct.
    # We approximate by measuring days into each fiscal quarter (0-90).
    # Exponential decay gives the model a soft signal around earnings windows.
    
    def _days_into_quarter(date):
        # Quarter boundaries (approximate): Jan, Apr, Jul, Oct = month 1,4,7,10
        quarter_start_month = ((date.month - 1) // 3) * 3 + 1
        quarter_start = pd.Timestamp(date.year, quarter_start_month, 1)
        return (date - quarter_start).days
 
    days_into_qtr          = pd.Series(
        [_days_into_quarter(d) for d in df.index], index=df.index
    )
    
    # Peak near 0 (start of quarter = earnings just released)
    # and near 90 (end of quarter = earnings approaching)
    
    df["earnings_proximity"] = np.exp(-np.minimum(days_into_qtr, 90 - days_into_qtr) / 10)
    
    
    # ── 11. Lag features (short-term memory) ────────────────────────────────
    
    
    for lag in [1, 2, 3]:
        df[f"return_lag_{lag}"]   = df["return_1d"].shift(lag)
        df[f"rsi_lag_{lag}"]      = df["rsi_14"].shift(lag)
    for lag in [1, 2]:
        df[f"macd_hist_lag_{lag}"] = df["macd_hist"].shift(lag)
 
    # ── 12. Target: 5-day forward direction with 0.5% threshold ─────────────
    #   1 = price rises more than 0.5% over next 5 trading days
    #   0 = flat or down
    #   Why 5-day: smooths out daily noise; still actionable
    #   Why 0.5% threshold: filters out near-zero moves that are essentially noise
    df["target"] = (c.shift(-1) > c).astype(int)
 
    return df



def prepare_Xy(df : pd.DataFrame):
    
    df = df.dropna(subset=FEATURE_COLS + ["target"])
    X = df[FEATURE_COLS]
    y = df["target"]
    
    return X, y, df


def time_split(X, y, test_size = TEST_SIZE):
    
    n = len(X)
    cutoff = int(n * (1 - test_size))
    
    X_train, X_test = X.iloc[:cutoff], X.iloc[cutoff:]
    y_train, y_test = y.iloc[:cutoff], y.iloc[cutoff:]
    
    return X_train, X_test, y_train, y_test