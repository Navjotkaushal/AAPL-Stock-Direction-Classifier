import pandas as pd 



    
def data_validation(df: pd.DataFrame) -> dict:  
    
    """
    Runs validation checks on stock OHLCV data
    Returns a summary dict pass/fail results
    """  
    results = {}
    
    # Checking shape
    results["row_count"] = len(df)
    results["col_count"] = len(df.columns)
    
    # Missing values
    missing = df.isnull().sum()
    results["missing_values"] = missing[missing > 0].to_dict()
    results["has_nulls"] = bool(results["missing_values"])
    
    
    # Duplicate dates 
    dupes = df.index.duplicated().sum()
    results["duplicate_dates"] = int(dupes)
    
    
    # OHLC Logic - these must always hold 
    ohlc_violations = {
        "high_lt_low": int((df["high"] < df["low"]).sum()),
        "high_lt_open": int((df["high"] < df["open"]).sum()),
        "high_lt_close": int((df["high"] < df["close"]).sum()),
        "low_gt_open": int((df["low"] > df["open"]).sum()),
        "low_gt_close": int((df["low"] > df["close"]).sum())
    }
    results["ohlc_violations"] = {k: v for k, v in ohlc_violations.items() if v > 0}
    results["ohlc_clean"] = len(results["ohlc_violations"]) == 0
    
    # Checking non-positive prices, well prices cannot be zero or negative
    for col in ["open", "close", "high", "low"]:
        bad = int((df[col] <= 0).sum())
        if bad:
            results[f"non_positive_{col}"] = bad
            
    # Negative volume 
    results["negative_volume"] = int((df["volume"] < 0).sum())
    
    # Outlier Detection 
    daily_return = df["close"].pct_change().abs()
    outliers = df[daily_return > 0.5]
    results["suspicious_price_jumps"] = len(outliers)
    if not outliers.empty:
        results["suspicious_dates"] = outliers.index.strftime("%Y-%m-%d").tolist()

    # 8. Date range
    results["date_from"] = str(df.index.min().date())
    results["date_to"]   = str(df.index.max().date())

    return results

def print_validation_report(results: dict):
    print("\n========== VALIDATION REPORT ==========")
    print(f"Rows         : {results['row_count']}")
    print(f"Date range   : {results['date_from']} → {results['date_to']}")
    print(f"Nulls        : {results['missing_values'] or 'None'}")
    print(f"Duplicate dates : {results['duplicate_dates']}")
    print(f"OHLC clean   : {results['ohlc_clean']}")
    if not results["ohlc_clean"]:
        print(f"  Violations : {results['ohlc_violations']}")
    print(f"Negative volume : {results['negative_volume']}")
    print(f"Suspicious jumps: {results['suspicious_price_jumps']}")
    if results.get("suspicious_dates"):
        print(f"  On dates   : {results['suspicious_dates']}")
    print("========================================\n")


if __name__ == "__main__":
    df = pd.read_csv("data/raw_data/raw_stock_data.csv")
    results = data_validation(df)
    print_validation_report(results)

