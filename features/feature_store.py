# Goal :
"""
Saves and loads engineered features from dataFarme to disk so the pipeline 
doesn't have to recompute all indicators on every run
"""

import sys 
from pathlib import Path 

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd 

# Store features next to the data folder, not inside the package
STORE_DIR  = Path(__file__).resolve().parent.parent / "features/feature_store"
STORE_PATH = STORE_DIR / "feature_store.csv"


def save_features(df: pd.DataFrame) -> None:
    
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(STORE_PATH, index = True)
    print(f"[feature_store] Saved {len(df)} rows -> {STORE_PATH}")
    
    
def load_features() -> pd.DataFrame:
    
    if not STORE_PATH.exists():
        raise FileNotFoundError(
            f"Feature store not found t {STORE_PATH}."
            "Run the pipeline first to generate features."
        )
        
    df = pd.read_csv(STORE_PATH, index_col=0, parse_dates=True)
    df.index.name = "date"
    print(f"[feature_store] Loaded {len(df)} rows ← {STORE_PATH}")
    return df


def features_exist() -> bool:
    """Return True if a saved feature store is available on disk."""
    return STORE_PATH.exists()
    
    
def delete_features() -> None:
    if STORE_PATH.exists():
        STORE_PATH.unlink()
        print(f"[feature store] Deleted {STORE_PATH}")
    else:
        print("[feature_store] Nothing to delete — store does not exist.")
 
    
