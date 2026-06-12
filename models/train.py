import joblib 
import os
from pathlib import Path

from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from config import RANDOM_STATE 

def build_models() -> dict:
    
    rf = Pipeline(steps=[
        ("scaling", StandardScaler()),
        ("clf", RandomForestClassifier(
            
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=20,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])
    
    xgb = Pipeline(steps=[
        ("scaling", StandardScaler()),
        ("clf", XGBClassifier(
            n_estimators = 100,
            max_depth = 3,
            learning_rate = 0.01,
            subsample = 0.6,
            colsample_bytree = 0.5,
            min_child_weight = 10,
            reg_alpha = 0.5,
            reg_lambda = 2.0,
            eval_metric = "logloss",
            random_state = RANDOM_STATE,
            n_jobs = -1,
        )),
    ])
    
    return {"Random Forest": rf, "XGBoost": xgb}


def train_all(models: dict, X_train, y_train) -> dict:
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
    return models 

def save_models(models: dict, path: str = "saved_models/"):
    os.makedirs(path, exist_ok=True)
    joblib.dump(models, "saved_models/trained_models.pkl")
    print("Saved")