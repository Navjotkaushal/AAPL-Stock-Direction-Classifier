from data.loader import get_connection, load_from_db 
from features.engineer import add_features, prepare_Xy, time_split 
from models.train import build_models, train_all 
from models.evaluate import evaluate_all, plot_results, predict_tomorrow


def main():
    
    # loading 
    
    print("Loading Data from DB...")
    conn = get_connection()
    df = load_from_db(conn)
    conn.close()
    print(f" {len(df)} rows ({df.index[0].date()} -> {df.index[-1].date()})")
    
    # Features + Target 
    
    print("\nEngineering Features...")
    df = add_features(df)
    X, y, df = prepare_Xy(df)
    
    # 3. Split
    
    X_train, X_test, y_train, y_test = time_split(X, y)
    print(f"  Train: {len(X_train)} rows  |  Test: {len(X_test)} rows")
    print(f"  Class balance (test) — Up: {y_test.mean():.1%}  Down: {1 - y_test.mean():.1%}")
 
    # 4. Train
    
    print("\nTraining models …")
    models = build_models()
    models = train_all(models, X_train, y_train)
 
    # 5. Evaluate
    
    results = evaluate_all(models, X_test, y_test)
 
    # 6. Plot
    
    plot_results(results, models)
 
    # 7. Predict tomorrow
    
    predict_tomorrow(models, df)
    
if __name__ == "__main__":
    main()