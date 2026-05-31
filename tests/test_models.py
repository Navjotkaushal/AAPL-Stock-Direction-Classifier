"""
tests/test_models.py
Run with: pytest tests/test_models.py -v
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from sklearn.metrics import accuracy_score, roc_auc_score


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_Xy():
    np.random.seed(42)
    n = 300
    X = pd.DataFrame(np.random.randn(n, 10), columns=[f"feature_{i}" for i in range(10)])
    y = pd.Series((X["feature_0"] + np.random.randn(n) * 0.5 > 0).astype(int))
    cutoff  = int(n * 0.8)
    return X.iloc[:cutoff], X.iloc[cutoff:], y.iloc[:cutoff], y.iloc[cutoff:]


@pytest.fixture
def trained_models(synthetic_Xy):
    """
    FIX: function is build_models() not build_base_models().
    Keys are "Random Forest" and "XGBoost" (with spaces, title case)
    not "random_forest" / "xgboost".
    """
    from models.train import train_all, build_models
    X_train, _, y_train, _ = synthetic_Xy
    models = build_models()
    return train_all(models, X_train, y_train)


# ── Tests: train.py ───────────────────────────────────────────────────────────

class TestTrain:

    def test_train_returns_dict(self, synthetic_Xy):
        from models.train import train_all, build_models
        X_train, _, y_train, _ = synthetic_Xy
        result = train_all(build_models(), X_train, y_train)
        assert isinstance(result, dict)

    def test_both_models_present(self, trained_models):
        """
        FIX: actual keys from build_models() are "Random Forest" and "XGBoost"
        not "random_forest" / "xgboost". Tests were checking the wrong key names.
        """
        assert "Random Forest" in trained_models
        assert "XGBoost"       in trained_models

    def test_models_have_predict(self, trained_models):
        for name, model in trained_models.items():
            assert hasattr(model, "predict"), f"{name} has no predict()"

    def test_models_have_predict_proba(self, trained_models):
        for name, model in trained_models.items():
            assert hasattr(model, "predict_proba"), f"{name} has no predict_proba()"

    def test_predictions_are_binary(self, trained_models, synthetic_Xy):
        _, X_test, _, _ = synthetic_Xy
        for name, model in trained_models.items():
            preds = model.predict(X_test)
            assert set(preds) <= {0, 1}, f"{name} predicted non-binary values"

    def test_predict_proba_sums_to_one(self, trained_models, synthetic_Xy):
        _, X_test, _, _ = synthetic_Xy
        for name, model in trained_models.items():
            proba = model.predict_proba(X_test)
            assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_predict_proba_shape(self, trained_models, synthetic_Xy):
        _, X_test, _, _ = synthetic_Xy
        for name, model in trained_models.items():
            proba = model.predict_proba(X_test)
            assert proba.shape == (len(X_test), 2)

    def test_models_beat_coin_flip(self, trained_models, synthetic_Xy):
        _, X_test, _, y_test = synthetic_Xy
        for name, model in trained_models.items():
            acc = accuracy_score(y_test, model.predict(X_test))
            assert acc > 0.50, f"{name} accuracy {acc:.3f} is no better than random"

    def test_save_models_creates_files(self, trained_models, tmp_path):
        """
        FIX 1: monkeypatch cannot be used inside a class method unless declared
                as a parameter. The original used it as a local variable name
                which caused NameError.

        FIX 2: save_models(models, path) takes a path string — not a module
                attribute to monkeypatch. Pass tmp_path directly as the argument.

        FIX 3: save_models saves as "{path}{name}.pkl" (no separator).
                With path="saved_models/" this produces "saved_models/Random Forest.pkl".
                We pass str(tmp_path) + "/" to match that exact behaviour.
        """
        from models.train import save_models

        save_dir = str(tmp_path) + "/"
        save_models(trained_models, path=save_dir)

        for name in trained_models:
            expected = tmp_path / f"{name}.pkl"
            assert expected.exists(), f"Expected saved file not found: {expected}"


# ── Tests: evaluate.py ────────────────────────────────────────────────────────

class TestEvaluate:

    def test_evaluate_all_returns_dict(self, trained_models, synthetic_Xy):
        from models.evaluate import evaluate_all
        _, X_test, _, y_test = synthetic_Xy
        assert isinstance(evaluate_all(trained_models, X_test, y_test), dict)

    def test_all_models_evaluated(self, trained_models, synthetic_Xy):
        from models.evaluate import evaluate_all
        _, X_test, _, y_test = synthetic_Xy
        result = evaluate_all(trained_models, X_test, y_test)
        for name in trained_models:
            assert name in result, f"{name} missing from evaluate_all output"

    def test_eval_result_structure(self, trained_models, synthetic_Xy):
        """Each value must be (preds, proba, cm) — the contract aapl_app.py depends on."""
        from models.evaluate import evaluate_all
        _, X_test, _, y_test = synthetic_Xy
        result = evaluate_all(trained_models, X_test, y_test)
        for name, val in result.items():
            assert isinstance(val, tuple) and len(val) == 3
            preds, proba, cm = val
            assert len(preds)      == len(y_test)
            assert proba.shape     == (len(y_test), 2)
            assert cm.shape        == (2, 2)

    def test_confusion_matrix_sums_correctly(self, trained_models, synthetic_Xy):
        from models.evaluate import evaluate_all
        _, X_test, _, y_test = synthetic_Xy
        result = evaluate_all(trained_models, X_test, y_test)
        for name, (preds, proba, cm) in result.items():
            assert cm.sum() == len(y_test)

    def test_accuracy_matches_confusion_matrix(self, trained_models, synthetic_Xy):
        from models.evaluate import evaluate_all
        _, X_test, _, y_test = synthetic_Xy
        result = evaluate_all(trained_models, X_test, y_test)
        for name, (preds, proba, cm) in result.items():
            manual_acc = accuracy_score(y_test, preds)
            cm_acc     = (cm[0, 0] + cm[1, 1]) / len(y_test)
            assert abs(manual_acc - cm_acc) < 1e-6


# ── Tests: registry.py ────────────────────────────────────────────────────────
