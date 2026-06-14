"""
data/loader.py
--------------
Fetches AAPL OHLCV data from Yahoo Finance via yfinance.
The get_connection() / conn parameter is kept as a no-op so that
nothing else in the codebase needs to change.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

STORE_DIR  = Path(__file__).resolve().parent.parent / "data/raw_data"
STORE_PATH = STORE_DIR / "raw_stock_data.parquet"


def load_data(ticker: str = "AAPL", start: str = "2000-01-01") -> pd.DataFrame:
    """
    Download daily OHLCV data for `ticker` from Yahoo Finance.

    Returns a DataFrame with columns: open, high, low, close, volume
    and a DatetimeIndex named 'date'.
    """
    df = yf.download(
        ticker,
        start=start,
        end = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=True,   # adjusts for splits/dividends automatically
        progress=False,
    )

    if df.empty:
        raise ValueError(f"yfinance returned no data for {ticker}. Check ticker or internet connection.")

    # Flatten MultiIndex columns if present (yfinance >= 0.2.x sometimes returns them)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].lower() for col in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]

    df.index.name = "date"
    df.index = pd.to_datetime(df.index)

    # Keep only the columns the rest of the pipeline expects
    expected = ["open", "high", "low", "close", "volume"]
    df = df[[c for c in expected if c in df.columns]]

    return df


def store_data():
    
    df = load_data()
    
    if df.empty:
        raise ValueError("No data. Run ingest.py before running the pipeline.")
    
    if not STORE_PATH.exists():
        raise FileNotFoundError(
            f"Feature store not found t {STORE_PATH}."
            "Run the pipeline first to generate features."
        )
    df.to_parquet(STORE_PATH, index="date")
    df.index.name = "date"
    print(f"Data succesfully stored , Loaded {len(df)} rows ← {STORE_PATH}")
    
store_data()
    
    

