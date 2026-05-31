import sys
import os 
from pathlib import Path 

import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec

from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    classification_report, roc_auc_score
)

from config import FEATURE_COLS 

def evaluate_all(models: dict, X_test, y_test) -> dict:
    """
    Run evaluation for every fitted model
    and returns the results in dict format
    """
    
    results = {}
    
    for name, model in models.items():
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)
        cm = confusion_matrix(y_test, preds)
        
        print(f"\n{'='*50}")
        print(f"  {name}")
        print(f"{'='*50}")
        print(f"  Accuracy : {accuracy_score(y_test, preds):.4f}")
        print(f"  ROC-AUC  : {roc_auc_score(y_test, proba[:, 1]):.4f}")
        print(f"\n{classification_report(y_test, preds, target_names=['Down', 'Up'])}")
        
        _evaluate_with_threshold(name, proba[:, 1], y_test)
        
        results[name] = (preds, proba, cm)
        
    return results 


# Threshold evaluation

def _evaluate_with_threshold(name: str, proba_up: np.ndarray, y_test, 
                              thresholds: list = [0.55, 0.58, 0.60, 0.63, 0.65]):
    """
    Only predict when the model's confidence exceeds a threshold.
    The rest of the days it abstains.
 
    This answers: 'On the days my model IS confident, how accurate is it?'
    A model that is right 60% of the time on 35% of days is genuinely useful.
    A model that is right 54% of the time on 100% of days is not.
    """
    print(f"\n  ── Threshold Analysis: {name} ──")
    print(f"  {'Threshold':>10}  {'Coverage':>10}  {'Days':>6}  {'Accuracy':>10}")
    print(f"  {'-'*46}")
 
    for t in thresholds:
        # Confident = model says UP with prob >= t, or DOWN with prob <= (1-t)
        confident_mask = (proba_up >= t) | (proba_up <= (1 - t))
        n_confident    = confident_mask.sum()
 
        if n_confident == 0:
            print(f"  {t:>10.2f}  {'—':>10}  {'0':>6}  {'no predictions':>10}")
            continue
 
        y_conf    = np.array(y_test)[confident_mask]
        preds_conf = (proba_up[confident_mask] >= 0.5).astype(int)
        acc        = accuracy_score(y_conf, preds_conf)
        coverage   = n_confident / len(proba_up)
 
        print(f"  {t:>10.2f}  {coverage:>9.1%}  {n_confident:>6}  {acc:>10.4f}")
 
    print()


# Walk Forward validation


def walk_forward_score(models: dict, X, y, n_splits: int = 5):
    """
    TimeSeriesSplit cross-validation across the full dataset.
    More honest than a single train/test split because it tests
    the model across multiple market regimes.
 
    Call this BEFORE the train/test split — pass full X and y.
    """
    tss = TimeSeriesSplit(n_splits=n_splits)
 
    print(f"\n{'='*50}")
    print(f"  Walk-Forward Validation  ({n_splits} folds)")
    print(f"{'='*50}")
 
    for name, model in models.items():
        scores = cross_val_score(
            model, X, y,
            cv=tss,
            scoring="roc_auc",
            n_jobs=-1,
        )
        print(f"\n  {name}")
        print(f"  ROC-AUC per fold : {[f'{s:.4f}' for s in scores]}")
        print(f"  Mean ± Std       : {scores.mean():.4f} ± {scores.std():.4f}")
        # Consistent scores across folds = genuinely learned signal
        # Wildly varying scores = unstable model or regime-dependent signal
 
 
def plot_results(results: dict, models: dict):
    
    """
    3-panel chart per model:
    -confusiion matrix
    -predicted probability distribution 
    -top 15 feature importance 
    """
    
    n = len(results)
    fig = plt.figure(figsize=(16, 5* n))
    fig.suptitle(
        "AAPL - Price Direction Classification",
        fontsize = 16,
        fontweight = "bold"
    )
    
    for i , (name, (preds, proba, cm)) in enumerate(results.items()):
        gs = gridspec.GridSpec(n, 3, figure = fig)
        
        # Confusion matrix 
        ax1 = fig.add_subplot(gs[i, 0])
        ax1.imshow(cm, cmap="Blues")
        ax1.set_xticks([0,1]); ax1.set_yticks([0,1])
        ax1.set_xticklabels(["Down","Up"]); ax1.set_yticklabels(["Down","Up"])
        ax1.set_xlabel("Predicted"); ax1.set_ylabel("Actual")
        ax1.set_title(f"{name}\nConfusion Matrix")
        for r in range(2):
            for c in range(2):
                ax1.text(c, r, cm[r, c], ha = "center", va = "center",
                         color = "white" if cm[r, c] > cm.max() / 2 else "black",
                         fontsize = 14)
                
        
        # Probability distribution 
        
        ax2 = fig.add_subplot(gs[i, 1])
        ax2.hist(proba[:, 1], bins = 30, edgecolor = "white", color = "#4C72B0")
        ax2.axvline(0.5, color = "red", linestyle = "--", label = "threshold = 0.5")
        ax2.set_xlabel("Predicted Probability (Up)")
        ax2.set_ylabel("Count")
        ax2.set_title(f"{name} \n Prediction Probability")
        ax2.legend()
        
        
        # Feature Importance 
        
        ax3 = fig.add_subplot(gs[i, 2])
        clf = models[name].named_steps["clf"]
        imps = clf.feature_importances_
        idx = np.argsort(imps)[-15:]
        ax3.barh(np.array(FEATURE_COLS)[idx], imps[idx], color = "#55A868")
        ax3.set_xlabel("Importance")
        ax3.set_title(f"{name}\n Top 15 Features")
        
        
    plt.tight_layout()
    out_dir = Path("outputs/plots")
    out_dir.mkdir(parents=True, exist_ok=True)  # creates folder if it doesn't exist
    save_path = out_dir / "model_results.png"
    
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nPlot saved -> {save_path}")
    
    
def predict_tomorrow(models: dict, df):
    
    """Run the latest available row through every fitted model"""
    
    latest = df[FEATURE_COLS].dropna().iloc[[-1]]
    
    print(f"\n{'=' * 50}")
    print(f"Tomorrow's prediction (based on {df.index[-1].date()})")
    print(f"{'=' * 50}")

    
    for name, model in models.items():
        prob = model.predict_proba(latest)[0,1]
        direction = "⬆  UP" if prob >= 0.5 else "⬇  DOWN"
        confident = "✅ HIGH CONFIDENCE" if (prob >= 0.60 or prob <= 0.40) else "⚠️  LOW CONFIDENCE — consider abstaining"
        print(f"  {name:20s}  {direction}   (confidence: {prob:.2%})")
        