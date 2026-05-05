from flask import Flask, jsonify, request
import joblib, os
from data.loader import get_connection, load_from_db
from features.engineer import add_features, prepare_Xy

app = Flask(__name__)
MODEL_DIR = "saved_models/"

def load_models():
    models = {}
    for f in os.listdir(MODEL_DIR):
        if f.endswith(".pkl"):
            name = f.replace(".pkl", "")
            models[name] = joblib.load(f"{MODEL_DIR}{f}")
    return models

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["GET"])
def predict():
    models = load_models()
    conn = get_connection()
    df = load_from_db(conn); conn.close()
    df_feat = add_features(df.copy())
    X, _, df_feat = prepare_Xy(df_feat)
    latest = df_feat[X.columns].dropna().iloc[[-1]]
    
    result = {}
    for name, model in models.items():
        prob = model.predict_proba(latest)[0, 1]
        result[name] = {
            "direction": "UP" if prob >= 0.5 else "DOWN",
            "confidence": round(prob, 4)
        }
    return jsonify(result)

@app.route("/retrain", methods=["POST"])
def retrain():
    from models.train import build_models, train_all
    from features.engineer import time_split
    conn = get_connection()
    df = load_from_db(conn); conn.close()
    df_feat = add_features(df.copy())
    X, y, _ = prepare_Xy(df_feat)
    X_train, _, y_train, _ = time_split(X, y)
    models = build_models()
    models = train_all(models, X_train, y_train)
    save_models(models)
    return jsonify({"status": "retrained", "rows_used": len(X_train)})

if __name__ == "__main__":
    app.run(debug=True, port=5000)