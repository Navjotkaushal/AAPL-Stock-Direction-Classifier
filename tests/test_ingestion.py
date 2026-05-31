"""
tests/test_ingestion.py
Run with: pytest tests/test_ingestion.py -v
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from data.validator import data_validation


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def clean_ohlcv():
    np.random.seed(0)
    n     = 100
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    close = np.abs(150 + np.cumsum(np.random.randn(n)))
    df = pd.DataFrame({
        "open":   close * 0.99,
        "high":   close * 1.02,
        "low":    close * 0.98,
        "close":  close,
        "volume": np.random.randint(1_000_000, 10_000_000, n).astype(float),
    }, index=dates)
    df.index.name = "date"
    return df


@pytest.fixture
def ohlcv_with_nulls(clean_ohlcv):
    df = clean_ohlcv.copy()
    df.loc[df.index[5],  "close"]  = np.nan
    df.loc[df.index[10], "volume"] = np.nan
    return df


@pytest.fixture
def ohlcv_with_ohlc_violations(clean_ohlcv):
    df = clean_ohlcv.copy()
    df.iloc[3, df.columns.get_loc("high")] = 100.0
    df.iloc[3, df.columns.get_loc("low")]  = 120.0
    df.iloc[7, df.columns.get_loc("high")] = 90.0
    df.iloc[7, df.columns.get_loc("low")]  = 110.0
    return df


@pytest.fixture
def ohlcv_with_duplicates(clean_ohlcv):
    return pd.concat([clean_ohlcv, clean_ohlcv.iloc[[0]]]).sort_index()


@pytest.fixture
def ohlcv_with_price_jump(clean_ohlcv):
    df = clean_ohlcv.copy()
    original = df.iloc[50]["close"]
    df.iloc[51, df.columns.get_loc("close")] = original * 3.0
    df.iloc[51, df.columns.get_loc("high")]  = original * 3.1
    return df


@pytest.fixture
def ohlcv_with_negative_volume(clean_ohlcv):
    df = clean_ohlcv.copy()
    df.iloc[20, df.columns.get_loc("volume")] = -5000
    return df


# ── Tests: data_validation() ─────────────────────────────────────────────────

class TestDataValidation:

    def test_returns_dict(self, clean_ohlcv):
        assert isinstance(data_validation(clean_ohlcv), dict)

    def test_required_keys_present(self, clean_ohlcv):
        result   = data_validation(clean_ohlcv)
        required = [
            "row_count", "col_count", "has_nulls", "missing_values",
            "ohlc_clean", "ohlc_violations", "duplicate_dates",
            "suspicious_price_jumps", "negative_volume", "date_from", "date_to",
        ]
        missing = [k for k in required if k not in result]
        assert missing == [], f"Missing keys: {missing}"

    def test_clean_data_passes_all_checks(self, clean_ohlcv):
        r = data_validation(clean_ohlcv)
        assert r["ohlc_clean"]             == True
        assert r["has_nulls"]              == False
        assert r["duplicate_dates"]        == 0
        assert r["suspicious_price_jumps"] == 0
        assert r["negative_volume"]        == 0

    def test_row_count_correct(self, clean_ohlcv):
        assert data_validation(clean_ohlcv)["row_count"] == len(clean_ohlcv)

    def test_detects_nulls(self, ohlcv_with_nulls):
        r = data_validation(ohlcv_with_nulls)
        assert r["has_nulls"] == True
        assert "close"  in r["missing_values"]
        assert "volume" in r["missing_values"]

    def test_clean_has_no_nulls(self, clean_ohlcv):
        r = data_validation(clean_ohlcv)
        assert r["has_nulls"]      == False
        assert r["missing_values"] == {}

    def test_detects_ohlc_violations(self, ohlcv_with_ohlc_violations):
        """
        ohlc_violations is a dict {violation_type: count} — NOT a single integer.
        e.g. {"high_lt_low": 2, "high_lt_open": 1}
        """
        r = data_validation(ohlcv_with_ohlc_violations)
        assert r["ohlc_clean"]      == False
        assert isinstance(r["ohlc_violations"], dict)
        assert "high_lt_low" in r["ohlc_violations"]
        assert r["ohlc_violations"]["high_lt_low"] == 2

    def test_clean_has_no_ohlc_violations(self, clean_ohlcv):
        r = data_validation(clean_ohlcv)
        assert r["ohlc_clean"]      == True
        assert r["ohlc_violations"] == {}

    def test_detects_duplicate_dates(self, ohlcv_with_duplicates):
        assert data_validation(ohlcv_with_duplicates)["duplicate_dates"] > 0

    def test_detects_price_jumps(self, ohlcv_with_price_jump):
        assert data_validation(ohlcv_with_price_jump)["suspicious_price_jumps"] > 0

    def test_suspicious_dates_is_list(self, ohlcv_with_price_jump):
        r = data_validation(ohlcv_with_price_jump)
        assert "suspicious_dates" in r
        assert isinstance(r["suspicious_dates"], list)

    def test_detects_negative_volume(self, ohlcv_with_negative_volume):
        assert data_validation(ohlcv_with_negative_volume)["negative_volume"] > 0

    def test_date_range_correct(self, clean_ohlcv):
        r = data_validation(clean_ohlcv)
        assert r["date_from"] == str(clean_ohlcv.index.min().date())
        assert r["date_to"]   == str(clean_ohlcv.index.max().date())


# ── Tests: loader.py ─────────────────────────────────────────────────────────

class TestLoader:
    """
    loader.py uses pymysql — mock target is "data.loader.pymysql.connect".
    Mocking "mysql.connector" would have zero effect since it's never imported.
    """

    @patch("data.loader.pymysql.connect")
    def test_get_connection_uses_db_config(self, mock_connect):
        from data.loader import get_connection
        from config import DB_CONFIG
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        conn = get_connection()
        mock_connect.assert_called_once_with(**DB_CONFIG)
        assert conn == mock_conn

    @patch("data.loader.pymysql.connect")
    def test_load_from_db_returns_dataframe(self, mock_connect):
        from data.loader import load_from_db
        mock_conn = MagicMock()
        with patch("data.loader.pd.read_sql") as mock_read_sql:
            mock_read_sql.return_value = pd.DataFrame({
                "Date":   pd.to_datetime(["2024-01-02", "2024-01-03"]),
                "open":   [185.0, 186.0], "high": [186.0, 187.0],
                "low":    [184.0, 185.0], "close": [185.5, 186.5],
                "volume": [50_000_000, 48_000_000],
            })
            result = load_from_db(mock_conn)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    @patch("data.loader.pymysql.connect")
    def test_load_from_db_sets_date_index(self, mock_connect):
        from data.loader import load_from_db
        mock_conn = MagicMock()
        with patch("data.loader.pd.read_sql") as mock_read_sql:
            mock_read_sql.return_value = pd.DataFrame({
                "Date":   pd.to_datetime(["2024-01-02", "2024-01-03"]),
                "open":   [185.0, 186.0], "high": [186.0, 187.0],
                "low":    [184.0, 185.0], "close": [185.5, 186.5],
                "volume": [50_000_000, 48_000_000],
            })
            result = load_from_db(mock_conn)
        assert result.index.name == "date"
        assert isinstance(result.index, pd.DatetimeIndex)

    @patch("data.loader.pymysql.connect")
    def test_get_last_date_uses_ticker(self, mock_connect):
        from data.loader import get_last_date
        from config import TICKER
        fake_cursor = MagicMock()
        fake_cursor.fetchone.return_value = ("2024-01-15",)
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=fake_cursor)
        mock_conn.cursor.return_value.__exit__  = MagicMock(return_value=False)
        get_last_date(mock_conn)
        args = fake_cursor.execute.call_args[0]
        assert TICKER in args[1]

    @patch("data.loader.pymysql.connect")
    def test_get_last_date_returns_none_on_empty_table(self, mock_connect):
        from data.loader import get_last_date
        fake_cursor = MagicMock()
        fake_cursor.fetchone.return_value = (None,)
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=fake_cursor)
        mock_conn.cursor.return_value.__exit__  = MagicMock(return_value=False)
        assert get_last_date(mock_conn) is None

    @patch("data.loader.pymysql.connect")
    def test_insert_data_calls_executemany(self, mock_connect):
        from data.loader import insert_data
        fake_cursor = MagicMock()
        mock_conn   = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=fake_cursor)
        mock_conn.cursor.return_value.__exit__  = MagicMock(return_value=False)
        df = pd.DataFrame({
            "Date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "Open": [185.0, 186.0], "High": [186.0, 187.0],
            "Low":  [184.0, 185.0], "Close": [185.5, 186.5],
            "Volume": [50_000_000, 48_000_000],
        })
        insert_data(mock_conn, df)
        fake_cursor.executemany.assert_called_once()

    @patch("data.loader.pymysql.connect")
    def test_insert_data_commits(self, mock_connect):
        from data.loader import insert_data
        fake_cursor = MagicMock()
        mock_conn   = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=fake_cursor)
        mock_conn.cursor.return_value.__exit__  = MagicMock(return_value=False)
        df = pd.DataFrame({
            "Date": pd.to_datetime(["2024-01-02"]),
            "Open": [185.0], "High": [186.0],
            "Low":  [184.0], "Close": [185.5], "Volume": [50_000_000],
        })
        insert_data(mock_conn, df)
        mock_conn.commit.assert_called_once()

    @patch("data.loader.pymysql.connect")
    def test_insert_data_passes_correct_row_count(self, mock_connect):
        from data.loader import insert_data
        fake_cursor = MagicMock()
        mock_conn   = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=fake_cursor)
        mock_conn.cursor.return_value.__exit__  = MagicMock(return_value=False)
        n  = 5
        df = pd.DataFrame({
            "Date":   pd.date_range("2024-01-02", periods=n),
            "Open":   [185.0]*n, "High": [186.0]*n,
            "Low":    [184.0]*n, "Close": [185.5]*n, "Volume": [50_000_000]*n,
        })
        insert_data(mock_conn, df)
        rows_passed = fake_cursor.executemany.call_args[0][1]
        assert len(rows_passed) == n

    @patch("data.loader.yf.download")
    def test_fetch_from_yfinance_returns_dataframe(self, mock_download):
        """
        FIX: yfinance returns Date as the INDEX, not a column.
        The old test built the mock with Date as a column, then
        fetch_from_yfinance did reset_index() which moved it — but our
        mock never had it as an index so reset_index() added a RangeIndex
        and 'Date' stayed as a column that wasn't there. KeyError followed.
        
        Correct mock: Date must be the index, matching real yfinance output.
        """
        from data.loader import fetch_from_yfinance

        fake_index = pd.to_datetime(["2024-01-02", "2024-01-03"])
        fake_index.name = "Date"

        mock_download.return_value = pd.DataFrame({
            "Open":   [185.0, 186.0],
            "High":   [186.0, 187.0],
            "Low":    [184.0, 185.0],
            "Close":  [185.5, 186.5],
            "Volume": [50_000_000, 48_000_000],
        }, index=fake_index)

        result = fetch_from_yfinance("2024-01-02")

        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert set(result.columns) >= {"date", "open", "high", "low", "close", "volume"}

    @patch("data.loader.yf.download")
    def test_fetch_from_yfinance_empty_returns_empty_df(self, mock_download):
        from data.loader import fetch_from_yfinance
        mock_download.return_value = pd.DataFrame()
        result = fetch_from_yfinance("2024-01-02")
        assert isinstance(result, pd.DataFrame)
        assert result.empty