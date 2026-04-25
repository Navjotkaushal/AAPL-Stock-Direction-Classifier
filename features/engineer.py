import sys
from pathlib import Path 

sys.path.append(str(Path(__file__).resolve().parent.parent))


import pandas as pd 
import numpy as np 

from config import FEATURE_COLS, TEST_SIZE
 
 
# Shorthand aliases to keep lines readable 

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    
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
        df[f"sma_{w}"] = c.rolling(w).mean()
        df[f"sma_{w}_ratio"] = c / df[f"sma_{w}"]
        
        
        
    # Exponenetial moving average  
    
    df["ema_12"] = c.ewm(span=12, adjust=False).mean()   # fast EMA
    df["ema_26"] = c.ewm(span=26, adjust=False).mean()   # slow EMA
    
    
    
    # MACD (Moving Average Convergence Divergence)
    
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] = df["macd_signal"]
    
    
    
    # RSI (Relative Strength Index, 14-day)
    delta        = c.diff()                                       # daily change
    gain         = delta.clip(lower=0).rolling(14).mean()         # average of up-days
    loss         = (-delta.clip(upper=0)).rolling(14).mean()      # average of down-days
    df["rsi_14"] = 100 - (100 / (1 + gain / (loss + 1e-9)))
    
    
    
    # BOLLINGER BANDS 
    
    mid             = c.rolling(20).mean()
    std             = c.rolling(20).std()
    df["bb_upper"]  = mid + 2 * std
    df["bb_lower"]  = mid - 2 * std
    df["bb_width"]  = (df["bb_upper"] - df["bb_lower"]) / mid   # normalised band width
    df["bb_pct"]    = (c - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-9)
    
    
    
    # ATR ( Average True Range, 14-day)
    
    tr = pd.concat([
                          h - l,                     # intraday range
                          (h - c.shift()).abs(),     # gap up scenario
                          (l - c.shift()).abs()      # gap down scenario
                      ], axis=1).max(axis=1)         # true range = max of the three
    df["atr_14"]    = tr.rolling(14).mean()
    df["atr_ratio"] = df["atr_14"] / c               # % of price
    
    
    
    # Volume features
    
    df["vol_sma_10"] = v.rolling(10).mean()
    df["vol_ratio"] = v / df["vol_sma_10"]        # 1.0 = normal, >1.5 = spike
    df["vol_change"] = v.pct_change()             # sudden volume surge day-over-day
    
    
    # Candle Structure
    
    body_top         = pd.concat([df["close"], df["open"]], axis=1).max(axis=1)
    body_bottom      = pd.concat([df["close"], df["open"]], axis=1).min(axis=1)
    df["body"]         = (df["close"] - df["open"]).abs() / df["open"]
    df["upper_shadow"] = (h - body_top)    / df["open"]   # rejection above
    df["lower_shadow"] = (body_bottom - l) / df["open"]   # rejection below 
    
    
    # Target Variable 
    
    df["target"]  = (c.shift(-1) > c).astype(int)        # 1 = up, 0 = down 
    
    return df 


# Prepairing X and y 

def prepare_Xy(df: pd.DataFrame):
    
    df = df.dropna(subset=FEATURE_COLS + ["target"])
    
    X = df[FEATURE_COLS]        # model inputs  - shape: (rows, 20)
    y = df["target"]              # model output - 0 or 1 
    
    return X, y, df    # Returning df too so predict_tomarrow() can grab the last row later



# TRAIN / TEST SPLIT 

def time_split(X, y, test_size = TEST_SIZE):
    
    n = len(X)
    cutoff = int(n* (1- test_size))    # Eg: 80%
    
    X_train, X_test = X.iloc[:cutoff], X.iloc[cutoff:]
    y_train, y_test = y.iloc[:cutoff], y.iloc[cutoff:]
    
    return X_train, X_test, y_train, y_test