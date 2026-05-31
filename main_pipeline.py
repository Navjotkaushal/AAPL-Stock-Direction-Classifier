import warnings
warnings.filterwarnings(action="ignore")

from data.loader import load_from_db, get_connection
from data.validator import data_validation, print_validation_report
from features.engineer import add_features, prepare_Xy, time_split, FEATURE_COLS
from features.pipeline import FeaturePipeline
from models.train import build_models, train_all, save_models
from models.tune import tune_all, build_base_models
from models.evaluate import evaluate_all, walk_forward_score, plot_results, predict_tomorrow
from config import TEST_SIZE, RANDOM_STATE


def run_pipeline(tune: bool = False, walk_forward: bool = False):
    conn = get_connection()
    try:
        # ── Step 1: Load ──────────────────────────────────────────────────────
        print("========== Layer 1: Loading Data ==========")
        df = load_from_db(conn)
        df = df.copy()
        if df.empty:
            raise ValueError("No data in DB. Run ingest.py before running the pipeline.")
        print(f"Loaded {df.shape[0]} rows, {df.shape[1]} cols\n")

        # ── Step 2: Validate ──────────────────────────────────────────────────
        print("========== Layer 2: Validating Data ==========")
        results = data_validation(df)
        print_validation_report(results)

        errors = []
        if not results["ohlc_clean"]:
            errors.append(f"OHLC violations: {results['ohlc_violations']}")
        if results["has_nulls"]:
            errors.append(f"Nulls found: {results['missing_values']}")
        if results["duplicate_dates"] > 0:
            errors.append(f"{results['duplicate_dates']} duplicate dates")
        if errors:
            raise ValueError(" | ".join(errors))

        if results["suspicious_price_jumps"] > 0:
            print(
                f"WARNING: {results['suspicious_price_jumps']} suspicious jumps on "
                f"{results.get('suspicious_dates')}. Review manually.\n"
            )

        # ── Step 3: Feature Engineering ───────────────────────────────────────
        print("========== Layer 3: Feature Engineering ==========")
        obj = FeaturePipeline()
        X_train, X_test, y_train, y_test, df_feat = obj.full_run(df)

        print(f"\nTrain: {X_train.index[0].date()} to {X_train.index[-1].date()}, rows: {len(X_train)}")
        print(f"Test:  {X_test.index[0].date()}  to {X_test.index[-1].date()},  rows: {len(X_test)}")
        print(f"Target mean — Train: {y_train.mean():.3f} | Test: {y_test.mean():.3f}")
        print(f"TEST_SIZE = {TEST_SIZE}\n")

        # ── Baseline: always-UP accuracy ──────────────────────────────────────
        baseline_acc = float(y_test.mean())
        print(f"Baseline (always predict UP): {baseline_acc:.4f}\n")

        # ── Step 4: Walk-Forward Validation (optional) ────────────────────────
        # Runs BEFORE train/test split on full data.
        # Gives a regime-robust view of model quality across 5 time windows.
        # Use --walk-forward flag to enable (slow — fits models 5x).
        if walk_forward:
            print("========== Layer 4a: Walk-Forward Validation ==========")
            import pandas as pd
            X_full = pd.concat([X_train, X_test])
            y_full = pd.concat([y_train, y_test])

            if tune:
                wf_models = tune_all(X_train, y_train)
            else:
                wf_base   = build_base_models()
                wf_models = {name: model for name, (model, _) in wf_base.items()}

            walk_forward_score(wf_models, X_full, y_full, n_splits=5)
            print()

        # ── Step 5: Train or Tune ─────────────────────────────────────────────
        if tune:
            print("========== Layer 5: Tuning Models ==========")
            print("WARNING: This will take several minutes.\n")
            trained_models = tune_all(X_train, y_train)
        else:
            print("========== Layer 5: Training Models ==========")
            # build_models() uses the tightened hyperparameters in train.py
            models         = build_models()
            trained_models = train_all(models, X_train, y_train)

        # ── Step 6: Evaluate ──────────────────────────────────────────────────
        # evaluate_all() now prints:
        #   - accuracy + ROC-AUC (standard)
        #   - threshold analysis at 0.55 / 0.58 / 0.60 / 0.63 / 0.65
        print("========== Layer 6: Evaluating Models ==========")
        eval_results = evaluate_all(trained_models, X_test, y_test)

        # ── Step 7: Plot ──────────────────────────────────────────────────────
        print("========== Layer 7: Plotting Results ==========")
        plot_results(eval_results, trained_models)

        # ── Step 8: Predict Tomorrow ──────────────────────────────────────────
        # Also flags low-confidence predictions explicitly
        print("========== Layer 8: Tomorrow's Prediction ==========")
        predict_tomorrow(trained_models, df_feat)

        # ── Step 9: Save ──────────────────────────────────────────────────────
        save_models(trained_models)
        print("\nPipeline completed successfully.")

    except ValueError as e:
        print(f"Pipeline stopped: {e}")

    except Exception as e:
        print(f"Pipeline crashed: {e}")
        try:
            conn.rollback()
        except Exception:
            pass
        raise

    finally:
        conn.close()
        print("Connection closed.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run Optuna hyperparameter tuning instead of default params",
    )
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        dest="walk_forward",
        help="Run 5-fold walk-forward validation before final train/test eval",
    )
    args = parser.parse_args()
    run_pipeline(tune=args.tune, walk_forward=args.walk_forward)