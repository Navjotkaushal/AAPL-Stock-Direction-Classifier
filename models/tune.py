import sys 
from pathlib import Path 
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np 
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.preprocessing import StandardScaler 
from sklearn.pipeline import Pipeline 
from xgboost import XGBClassifier 

from config import RANDOM_STATE 



TSS = TimeSeriesSplit(n_splits=5)

RF_PARAMS = {
    
    "clf__n_estimators" : [100, 200, 300, 400, 500],
    
    "clf__max_depth" : [3, 4, 5, 6, 7, None],
    
    "clf__max_features" : ["sqrt", "log2", 0.5, 0.7],
    
    "clf__class_weight" : ["balanced", None],
    
}


XGB_PARAMS = {
    
    "clf__n_estimators" : [100, 200, 300, 400, 500],
    
    "clf__learning_rate" : [0.01, 0.03, 0.05, 0.1, 0.2],
    
    "clf__max_depth" : [3, 4, 5, 6],
    
    "clf__subsample" : [0.6, 0.7, 0.8, 0.9, 1.0],
    
    "clf__colsample_bytree" : [0.5, 0.6, 0.7, 0.8, 1.0],
    
    "clf__reg_alphha" : [0, 0.01, 0.1, 0.5, 1.0],
    
    "clf__reg_lambda" : [0.5, 1.0, 2.0, 5.0],
    
    "clf__min_child_weight" : [1, 3, 5, 10],
    
}


def tune_model(name, base_model, param_grid, X_train, y_train, n_iter = 40):
    
    print(f"\n  Tuning {name}  ({n_iter} iterations × {TSS.n_splits} folds = {n_iter * TSS.n_splits} fits) …")
    
    search = RandomizedSearchCV(
        estimator = base_model,
        param_distributions = param_grid,
        n_iter = n_iter,
        cv = TSS,    # Time aware cross validation 
        scoring = "roc_auc",
        n_jobs = -1 ,  # use all cpu cores 
        random_state = RANDOM_STATE,
        verbose = 1
    )
    
    search.fit(X_train, y_train)
    
    print(f"  ✅ Best ROC-AUC ({name}): {search.best_score_:.4f}")
    print(f"  Best params: {_format_params(search.best_params_)}")
 
    return search.best_estimator_


def _format_params(params: dict) -> str:
    return {k.replace("clf__", ""): v for k, v in params.items()}


def build_base_models() -> dict:
    
    rf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(random_state=RANDOM_STATE))
    ])
    
    xgb = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(
            eval_metric = "logloss",
            random_state = RANDOM_STATE
        )),
    ])
    
    return {"Random Forest" : (rf, RF_PARAMS), "XGBoost" : (xgb, XGB_PARAMS)}


def tune_all(X_train, y_train, n_iter = 40) -> dict:
    
    models = build_base_models()
    tuned = {}
    
    for name, (base_model, param_grid) in models.items():
        tuned[name] = tune_model(name, base_model, param_grid, X_train, y_train, n_iter)
        
        
    return tuned 