# AAPL Stock Direction Classifier

End-to-end ML pipeline that ingests daily AAPL stock data from Yahoo Finance into MySQL via a task scheduler, engineers 20+ technical indicators (MACD, RSI, Bollinger Bands, ATR), and trains Random Forest + XGBoost models to classify next-day price direction (Up/Down) using a modular, production-style project structure.

---

## Project Structure

```
aapl_project/
│
├── config.py              # All constants — DB credentials, ticker, features, SQL
│
├── ingest.py              # Entry point for task scheduler (daily data pull)
├── main.py                # Entry point for model training and evaluation
│
├── data/
│   └── loader.py          # DB connection, yfinance fetch, insert, load for ML
│
├── features/
│   └── engineer.py        # 20+ technical indicators, target column, train/test split
│
└── models/
    ├── train.py           # Random Forest + XGBoost model definitions and fitting
    └── evaluate.py        # Metrics, confusion matrix, feature importance, prediction
```

---

## How It Works

```
Task Scheduler (daily)
       ↓
  ingest.py  →  data/loader.py  →  Fetches from yfinance  →  Inserts into MySQL
  
       ↓  (when you want to train)
  
   main.py
       ↓
  data/loader.py       loads history from MySQL
       ↓
  features/engineer.py engineers 20+ features + target
       ↓
  models/train.py      trains Random Forest + XGBoost
       ↓
  models/evaluate.py   metrics + plots + tomorrow's prediction
```

---

## Features Engineered

| Group | Features |
|---|---|
| Returns | 1d, 3d, 5d, 10d % change |
| Trend | SMA 5/10/20/50 ratio to price |
| Momentum | MACD, MACD signal, MACD histogram, RSI-14 |
| Volatility | Bollinger Band width & %B, ATR ratio |
| Volume | Volume spike ratio, volume % change |
| Candle | Body size, upper shadow, lower shadow |

**Target:** `1` if next day close > today close, else `0`

---

## Setup

### 1. Install dependencies
```bash
pip install yfinance pandas mysql-connector-python scikit-learn xgboost matplotlib
```

### 2. Configure credentials
Open `config.py` and update your MySQL details:
```python
DB_CONFIG = {
    "host":     "localhost",
    "user":     "root",
    "password": "your_password",
    "database": "apple_stock_prices",
}
```

### 3. Make sure your MySQL table exists
```sql
CREATE TABLE stock_data (
    id      BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
    ticker  VARCHAR(10) NOT NULL,
    date    DATE NOT NULL,
    open    DECIMAL(12,4),
    high    DECIMAL(12,4),
    low     DECIMAL(12,4),
    close   DECIMAL(12,4),
    volume  BIGINT,
    UNIQUE KEY uq_ticker_date (ticker, date)
);
```

### 4. Run the ingestion script (or let the task scheduler handle it)
```bash
python ingest.py
```
On first run this backfills all data from 2010. On every subsequent run it only fetches new rows since the last stored date.

### 5. Train and evaluate
```bash
python main.py
```

---

## Output

- **Console** — accuracy, ROC-AUC, classification report for both models
- **model_results.png** — confusion matrix, probability distribution, top 15 features per model
- **Tomorrow's prediction** — Up/Down call with confidence % from both models

---

## Task Scheduler (Windows)

Point Windows Task Scheduler to run daily after market close:
```
Program:   python
Arguments: C:\path\to\aapl_project\ingest.py
Trigger:   Daily at 6:00 PM
```

---

## Tech Stack

- **Data** — yfinance, MySQL, mysql-connector-python
- **ML** — scikit-learn, XGBoost
- **Visualization** — matplotlib

---

## Current Status

- [x] Automated daily data ingestion
- [x] 20+ technical indicator features
- [x] Random Forest classifier
- [x] XGBoost classifier
- [x] Evaluation plots + metrics
- [x] Next-day prediction
- [ ] Hyperparameter tuning with TimeSeriesSplit
- [ ] Walk-forward (rolling window) backtesting
- [ ] REST API to serve predictions
