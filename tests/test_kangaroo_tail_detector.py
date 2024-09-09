import pandas as pd
import pytest
import numpy as np
from src.patterns import KangarooTailDetector


@pytest.fixture
def sample_data():
    """Fixture to provide sample data for testing."""
    dates = pd.date_range(start="2022-01-01", periods=100)
    prices = np.random.uniform(100, 200, size=100)
    return pd.DataFrame({"Date": dates, "Price": prices}).set_index("Date")


@pytest.fixture
def candle_data():
    """Fixture to provide sample candle data for KangarooTailDetector."""
    dates = pd.date_range(start="2022-01-01", periods=10)
    data = {
        "Open": [100, 105, 103, 107, 110, 108, 107, 112, 111, 115],
        "High": [105, 110, 107, 110, 115, 112, 113, 116, 117, 120],
        "Low": [95, 100, 98, 102, 105, 104, 103, 108, 110, 112],
        "Close": [101, 108, 105, 109, 113, 109, 111, 115, 116, 119]
    }
    return pd.DataFrame(data, index=dates)


def test_valid_data_initialization(candle_data):
    detector = KangarooTailDetector(data=candle_data)
    assert not detector.data.empty, "Data should be initialized correctly"


def test_calculate_body_size(candle_data):
    detector = KangarooTailDetector(data=candle_data)
    body_size = detector._calculate_candle_body_size()
    expected_body_size = abs(candle_data["Open"] - candle_data["Close"])
    assert body_size.equals(expected_body_size), "Body sizes should match the expected values"


def test_zero_body_size(candle_data):
    data = candle_data.copy()
    data["Close"] = data["Open"]
    detector = KangarooTailDetector(data=data)
    with pytest.raises(ValueError, match="Body length cannot be 0"):
        detector._calculate_candle_body_size()
