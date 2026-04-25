import yfinance as yf
import pandas as pd
import mysql.connector
from datetime import timedelta

from config import DB_CONFIG, TICKER, UPSERT_SQL          # ← was inline in your script


# ─── DB helpers ───────────────────────────────────────────────────────────────

def get_connection():
    return mysql.connector.connect(**DB_CONFIG)


def get_last_date(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT MAX(date) FROM stock_data WHERE ticker = %s", (TICKER,))
        return cur.fetchone()[0]   # datetime or None


def load_from_db(conn) -> pd.DataFrame:
    """Load full history into a DataFrame — used by main.py for ML."""
    df = pd.read_sql(
        "SELECT date, open, high, low, close, volume "
        "FROM stock_data WHERE ticker = %s ORDER BY date ASC",
        conn, params=(TICKER,)
    )
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    return df


# ─── Fetch ────────────────────────────────────────────────────────────────────

def flatten_columns(df):
    """yfinance >= 0.2 returns MultiIndex columns — flatten to simple strings."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    return df


def fetch_from_yfinance(start_date):             # ← was fetch_data() in your script
    df = yf.download(TICKER, start=start_date, progress=False, auto_adjust=True)
    if df.empty:
        return pd.DataFrame()

    df = flatten_columns(df)
    df.reset_index(inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    return df[["Date", "Open", "High", "Low", "Close", "Volume"]]


# ─── Insert ───────────────────────────────────────────────────────────────────

def insert_data(conn, df):
    rows = list(zip(
        [TICKER] * len(df),
        df["Date"].tolist(),
        df["Open"].astype(float).tolist(),
        df["High"].astype(float).tolist(),
        df["Low"].astype(float).tolist(),
        df["Close"].astype(float).tolist(),
        df["Volume"].astype(int).tolist(),
    ))
    with conn.cursor() as cur:
        cur.executemany(UPSERT_SQL, rows)
    conn.commit()


# ─── Main ─────────────────────────────────────────────────────────────────────

def update_db():                                 # ← was main() in your script
    conn      = get_connection()
    last_date = get_last_date(conn)

    if last_date:
        start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        start = "2010-01-01"

    df = fetch_from_yfinance(start)

    if not df.empty:
        insert_data(conn, df)
        print(f"Inserted {len(df)} new row(s) from {df['Date'].min().date()} to {df['Date'].max().date()}.")
    else:
        print("No new data to insert.")

    conn.close()