"""
tests/conftest.py
-----------------
Shared pytest configuration and fixtures available to ALL test files.
pytest loads this file automatically — you never import it directly.
 
Fixtures defined here can be used in any test file without importing.
"""


import pytest
import pandas as pd 
import numpy as np 

@pytest.fixture(scope="session")
def base_ohlcv():
    
    np.random.seed(99)
    n = 300
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 150 + np.cumsum(np.random.rand(n) * 1.5)
    close = np.abs(close)
    
    df = pd.DataFrame({
        "open": close * (1 + np.random.uniform(0.005, 0.005, n)),
        "high": close * (1 + np.random.uniform(0.000, 0.015, n)),
        "low": close * (1 - np.random.uniform(0.000, 0.015, n)),
        "close": close,
        "volume": np.random.randint(5_0000_000, 80_000_000, n).astype(float),

    }, index=dates)
    df.index.name = "date"
    return df 