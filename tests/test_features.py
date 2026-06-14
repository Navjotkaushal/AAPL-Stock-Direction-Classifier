"""
tests/test_features.py
Run with: pytest tests/test_features.py -v
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from features.engineer import add_features, prepare_Xy, time_split
from config import FEATURE_COLS


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def raw_ohlcv():
    np.random.seed(42)
    n     = 200
    dates = pd.date_range(start="2022-01-01", periods=n, freq="B")
    close = np.abs(150 + np.cumsum(np.random.randn(n) * 1.5))

    df = pd.DataFrame({
        "open":   close * (1 + np.random.uniform(-0.01, 0.01, n)),
        "high":   close * (1 + np.random.uniform(0.00,  0.02, n)),
        "low":    close * (1 - np.random.uniform(0.00,  0.02, n)),
        "close":  close,
        "volume": np.random.randint(1_000_000, 50_000_000, n).astype(float),
    }, index=dates)
    df.index.name = "date"
    return df


@pytest.fixture
def featured_df(raw_ohlcv):
    return add_features(raw_ohlcv)


# ── Tests: add_features() ─────────────────────────────────────────────────────

class TestAddFeatures:

    def test_returns_dataframe(self, raw_ohlcv):
        assert isinstance(add_features(raw_ohlcv), pd.DataFrame)

    def test_does_not_mutate_input(self, raw_ohlcv):
        """
        add_features() must not modify the original DataFrame.
        Fixed by adding df = df.copy() at top of add_features().
        Without it this test fails because pandas writes new columns back
        into the caller's object through the reference.
        """
        original_cols = set(raw_ohlcv.columns)
        add_features(raw_ohlcv)
        assert set(raw_ohlcv.columns) == original_cols

    def test_all_feature_columns_present(self, featured_df):
        missing = [col for col in FEATURE_COLS if col not in featured_df.columns]
        assert missing == [], f"Missing feature columns: {missing}"

    def test_target_column_present(self, featured_df):
        assert "target" in featured_df.columns

    def test_target_is_binary(self, featured_df):
        unique_vals = set(featured_df["target"].dropna().unique())
        assert unique_vals <= {0, 1}

    def test_macd_hist_is_difference(self, featured_df):
        """
        macd_hist must equal macd - macd_signal.
        Original bug: df["macd_hist"] = df["macd"] = df["macd_signal"]
        This is a chained assignment — overwrites macd with macd_signal value,
        then copies that same wrong value into macd_hist.
        """
        df       = featured_df.dropna(subset=["macd", "macd_signal", "macd_hist"])
        expected = df["macd"] - df["macd_signal"]
        diff     = (df["macd_hist"] - expected).abs()
        assert diff.max() < 1e-6

    def test_rsi_range(self, featured_df):
        rsi = featured_df["rsi_14"].dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all()

    def test_bb_pct_finite(self, featured_df):
        bb_pct = featured_df["bb_pct"].replace([np.inf, -np.inf], np.nan).dropna()
        assert np.isfinite(bb_pct).all()

    def test_volume_ratio_positive(self, featured_df):
        vr = featured_df["vol_ratio"].dropna()
        assert (vr > 0).all()

    def test_row_count_preserved(self, raw_ohlcv, featured_df):
        """add_features must not drop any rows — NaNs are expected but rows stay."""
        assert len(featured_df) == len(raw_ohlcv)


# ── Tests: prepare_Xy() ───────────────────────────────────────────────────────

class TestPrepareXy:

    def test_returns_three_objects(self, featured_df):
        assert len(prepare_Xy(featured_df)) == 3

    def test_X_has_correct_columns(self, featured_df):
        X, y, _ = prepare_Xy(featured_df)
        assert list(X.columns) == FEATURE_COLS

    def test_y_is_series(self, featured_df):
        _, y, _ = prepare_Xy(featured_df)
        assert isinstance(y, pd.Series)

    def test_no_nulls_in_X(self, featured_df):
        X, _, _ = prepare_Xy(featured_df)
        assert X.isnull().sum().sum() == 0

    def test_no_nulls_in_y(self, featured_df):
        _, y, _ = prepare_Xy(featured_df)
        assert y.isnull().sum() == 0

    def test_X_y_same_length(self, featured_df):
        X, y, _ = prepare_Xy(featured_df)
        assert len(X) == len(y)

    def test_fewer_rows_than_input(self, featured_df):
        X, _, _ = prepare_Xy(featured_df)
        assert len(X) < len(featured_df)


# ── Tests: time_split() ───────────────────────────────────────────────────────

class TestTimeSplit:

    @pytest.fixture
    def Xy(self, featured_df):
        X, y, _ = prepare_Xy(featured_df)
        return X, y

    def test_returns_four_objects(self, Xy):
        X, y = Xy
        assert len(time_split(X, y)) == 4

    def test_no_overlap_between_train_and_test(self, Xy):
        """
        Most critical time-series test.
        Original bug: X_test = X.iloc[:cutoff] — identical to X_train.
        That means evaluation was measuring training accuracy, not test accuracy.
        All reported accuracy numbers were fake.
        """
        X, y = Xy
        X_train, X_test, _, _ = time_split(X, y)
        overlap = set(X_train.index) & set(X_test.index)
        assert len(overlap) == 0, f"Data leakage: {len(overlap)} overlapping dates"

    def test_train_comes_before_test(self, Xy):
        X, y = Xy
        X_train, X_test, _, _ = time_split(X, y)
        assert X_train.index.max() < X_test.index.min()

    def test_sizes_add_up(self, Xy):
        X, y = Xy
        X_train, X_test, y_train, y_test = time_split(X, y)
        assert len(X_train) + len(X_test) == len(X)
        assert len(y_train) + len(y_test) == len(y)

    def test_test_size_respected(self, Xy):
        X, y = Xy
        X_train, X_test, _, _ = time_split(X, y, test_size=0.2)
        expected = int(len(X) * 0.2)
        assert abs(len(X_test) - expected) <= 1

    def test_X_and_y_splits_are_aligned(self, Xy):
        X, y = Xy
        X_train, X_test, y_train, y_test = time_split(X, y)
        assert list(X_train.index) == list(y_train.index)
        assert list(X_test.index)  == list(y_test.index)