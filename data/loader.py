# Importing required libraires
import os 
import sys 
import sqlalchemy
import yfinance as yf
import pandas as pd 
import pymysql 
from datetime import timedelta 

from config import DB_CONFIG, TICKER, UPSERT_SQL


# DB Helpers 

def get_connection():
    return pymysql.connect(**DB_CONFIG)


def get_last_date(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT MAX(date) FROM stock_data WHERE ticker = %s", (TICKER,))
        return cur.fetchone()[0] # datetime or None 
    

def load_from_db(conn) -> pd.DataFrame:
    # Load full history from database
    df = pd.read_sql(
        """
        SELECT Date, open, high, low, close, volume
        FROM stock_data WHERE ticker = %s ORDER BY date ASC
        """,
        conn, params=(TICKER,)
    )
    df["date"] = pd.to_datetime(df["Date"])
    df.set_index("date", inplace=True)
    return df 


# fetching

def flatten_columns(df): 
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    return df 


def fetch_from_yfinance(start_date):
    df = yf.download(TICKER, start = start_date, 
                     progress=False, 
                     auto_adjust=True)
    if df.empty:
        return pd.DataFrame()
    
    df = flatten_columns(df)
    df.reset_index(inplace=True)
    
    df.columns = [col.lower() for col in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "open", "high", "low", "close", "volume"]]


# Inserting 

def insert_data(conn, df):
    rows = list(zip(
        [TICKER] * len(df),
        df["date"].tolist(),
        df["open"].astype(float).tolist(),
        df["high"].astype(float).tolist(),
        df["low"].astype(float).tolist(),
        df["close"].astype(float).tolist(),
        df["volume"].astype(float).tolist(),
    ))
    
    with conn.cursor() as cur:
        cur.executemany(UPSERT_SQL, rows)
    conn.commit()
    

# Main

def update_db():
    conn = get_connection()
    last_date = get_last_date(conn)
    
    if last_date:
        start = (last_date + timedelta(days = 1)).strftime("%Y-%m-%d")
        
    else:
        start = "2010-01-01"
        
    df = fetch_from_yfinance(start)
    
    if not df.empty:
        insert_data(conn, df)
        print(f"Inserted {len(df)} new row(s) from {df['Date'].min().date()} to {df['Date'].max().date()}")
    else:
        print("No new data to insert.")
        
    conn.close()