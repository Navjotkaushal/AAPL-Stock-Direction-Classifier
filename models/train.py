import sys 
from pathlib import Path 
sys.path.append(str(Path(__file__).resolve().parent.parent))

from sklearn.ensemble import RandomForestClassifier 
import sys 
from pathlib import Path 
sys.path.append(str(Path(__file__).resolve().parent.parent))

from sklearn.preprocessing import StandardScaler
import sys 
from pathlib import Path 
sys.path.append(str(Path(__file__).resolve().parent.parent))
 
from sklearn.pipeline import Pipeline 
from xgboost import XGBClassifier  

from config import RANDOM_STATE 

def build_models() -> dict:
    
    rf = Pipeline([
        ("scaling", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=RANDOM_STATE
        )),
    ])
    
    xgb = Pipeline([
        ("scaling", StandardScaler()),
        ("clf", XGBClassifier(
            n_estimators = 300,
            max_depth = 4,
            learning_rate = 0.05,
            subsample = 0.8,
            colsample_bytree = 0.8,
            eval_metric = "logloss",
            random_state = RANDOM_STATE
        )),
    ])
    
    return {"Random Forest" : rf, "XGBoost" : xgb}

def train_all(models: dict, X_train, y_train) ->dict:
    for name, model in models.items():
        print(f" Training {name}...")
        model.fit(X_train, y_train)
        
    return models