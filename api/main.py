import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix, classification_report
)
import time
import os

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="AAPL Direction Classifier API")

# NOTE: Change this to your actual deployed frontend URL before going live.
# Using "*" is fine for local dev but exposes the API to any origin in production.
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── State ──────────────────────────────────────────────────────────────────────
pipeline_state = {
    "status": "idle",          # idle | running | done | error
    "step": "",
    "steps_done": [],
    "error": None,
    "trained_at": None,
}

_TMP = "/tmp"
MODEL_PATH       = f"{_TMP}/trained_models.pkl"
FEATURE_COL_PATH = f"{_TMP}/feature_columns.pkl"
FEAT_DF_PATH     = f"{_TMP}/last_features.pkl"
METRICS_PATH     = f"{_TMP}/last_metrics.pkl"
SPLIT_PATH       = f"{_TMP}/train_test_split.pkl"

# ── Pipeline runner ────────────────────────────────────────────────────────────
def run_pipeline_task(test_size: float):
    global pipeline_state
    pipeline_state.update({"status": "running", "steps_done": [], "error": None})

    try:
        # Step 1 – load data
        pipeline_state["step"] = "Loading data from MySQL…"
        from data.loader import get_connection, load_from_db
        conn = get_connection()
        df_raw = load_from_db(conn)
        conn.close()
        pipeline_state["steps_done"].append({
            "label": f"Data loaded — {len(df_raw):,} rows "
                     f"({df_raw.index[0].date()} → {df_raw.index[-1].date()})"
        })

        # Step 2 – feature engineering
        pipeline_state["step"] = "Engineering features…"
        from features.engineer import add_features, prepare_Xy, time_split
        df_feat = add_features(df_raw.copy())
        X, y, df_feat = prepare_Xy(df_feat)
        X_train, X_test, y_train, y_test = time_split(X, y, test_size=test_size)
        pipeline_state["steps_done"].append({
            "label": f"Features ready — {X.shape[1]} features | "
                     f"Train: {len(X_train):,}  Test: {len(X_test):,}"
        })

        # Step 3 – train
        pipeline_state["step"] = "Training models…"
        from models.train import build_models, train_all
        models = build_models()
        models = train_all(models, X_train, y_train)
        pipeline_state["steps_done"].append({"label": "Models trained (RF + XGBoost)"})

        # Step 4 – evaluate
        pipeline_state["step"] = "Evaluating models…"
        metrics = {}
        for name, model in models.items():
            preds = model.predict(X_test)
            proba = model.predict_proba(X_test)[:, 1]
            cm    = confusion_matrix(y_test, preds).tolist()
            report = classification_report(
                y_test, preds,
                target_names=["Down", "Up"],
                output_dict=True
            )
            clf = model.named_steps["clf"]
            feat_imp = sorted(
                zip(list(X.columns), clf.feature_importances_.tolist()),
                key=lambda x: x[1], reverse=True
            )[:15]

            metrics[name] = {
                "accuracy":           round(accuracy_score(y_test, preds), 4),
                "roc_auc":            round(roc_auc_score(y_test, proba), 4),
                "confusion_matrix":   cm,
                "report":             report,
                "feature_importance": feat_imp,
                "proba_hist":         proba.tolist(),
            }

        pipeline_state["steps_done"].append({"label": "Evaluation complete"})

        # Step 5 – save
        pipeline_state["step"] = "Saving models to disk…"
        os.makedirs("models", exist_ok=True)
        joblib.dump(models,          MODEL_PATH)
        joblib.dump(list(X.columns), FEATURE_COL_PATH)
        joblib.dump(df_feat,         FEAT_DF_PATH)
        joblib.dump(metrics,         METRICS_PATH)

        split_idx = int(len(df_feat) * (1 - test_size))
        joblib.dump({
            "train_dates": [str(d.date()) for d in df_raw.index[:split_idx]],
            "test_dates":  [str(d.date()) for d in df_raw.index[split_idx:]],
            "train_close": df_raw["close"].iloc[:split_idx].tolist(),
            "test_close":  df_raw["close"].iloc[split_idx:].tolist(),
        }, SPLIT_PATH)

        pipeline_state["steps_done"].append({"label": "Models saved to disk"})
        pipeline_state.update({
            "status":     "done",
            "step":       "Pipeline complete",
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "error":      None,
        })

    except Exception as e:
        pipeline_state.update({
            "status": "error",
            "step":   "Pipeline failed",
            "error":  str(e),
        })

# ── Endpoints ──────────────────────────────────────────────────────────────────

class PipelineRequest(BaseModel):
    test_size: float = 0.2

@app.post("/run-pipeline")
def run_pipeline(req: PipelineRequest, background_tasks: BackgroundTasks):
    if pipeline_state["status"] == "running":
        raise HTTPException(status_code=409, detail="Pipeline already running")
    pipeline_state.update({"status": "idle", "steps_done": [], "error": None})
    background_tasks.add_task(run_pipeline_task, req.test_size)
    return {"message": "Pipeline started"}

@app.get("/pipeline-status")
def get_pipeline_status():
    return pipeline_state

@app.get("/predict")
def predict():
    for path in [MODEL_PATH, FEATURE_COL_PATH, FEAT_DF_PATH, METRICS_PATH]:
        if not os.path.exists(path):
            raise HTTPException(
                status_code=404,
                detail="No trained model found. Run the pipeline first."
            )

    models     = joblib.load(MODEL_PATH)
    feat_cols  = joblib.load(FEATURE_COL_PATH)
    df_feat    = joblib.load(FEAT_DF_PATH)
    metrics    = joblib.load(METRICS_PATH)
    split_data = joblib.load(SPLIT_PATH) if os.path.exists(SPLIT_PATH) else {}

    latest    = df_feat[feat_cols].dropna().iloc[[-1]]
    last_date = str(df_feat.index[-1].date())

    predictions = {}
    for name, model in models.items():
        prob      = float(model.predict_proba(latest)[0, 1])
        direction = "UP" if prob >= 0.5 else "DOWN"
        predictions[name] = {
            "direction":  direction,
            "confidence": round(prob * 100, 2),
            "prob_up":    round(prob, 4),
            "prob_down":  round(1 - prob, 4),
        }

    return {
        "as_of_date":  last_date,
        "trained_at":  pipeline_state.get("trained_at"),
        "predictions": predictions,
        "metrics":     metrics,
        "split_data":  split_data,
    }

@app.get("/health")
def health():
    return {
        "status":      "ok",
        "model_ready": os.path.exists(MODEL_PATH),
        "pipeline":    pipeline_state["status"],
    }

# ── Serve frontend ─────────────────────────────────────────────────────────────
# index.html must be in a folder called "static" next to this file.
# Access via: http://localhost:8000/
app.mount("/", StaticFiles(directory="static", html=True), name="static")