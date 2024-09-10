import pytest
from src.patterns import BigShadowDetector
import pandas as pd
import yfinance as yf


@pytest.fixture
def sample_data():
    """Fixture to generate sample candle data for testing."""
    data = pd.DataFrame({
        'Open': [10, 11, 12, 13, 14],
        'High': [11, 12, 13, 14, 15],
        'Low': [9, 10, 11, 12, 13],
        'Close': [11, 12, 11, 12, 8]
    })
    data.index = pd.date_range(start="2023-01-01", periods=len(data), freq='D')
    return data


@pytest.fixture
def uptrend_data():
    """Fixture to generate sample uptrend candle data for testing."""
    data = pd.DataFrame({
        'Open': [10, 11, 12, 13, 14],
        'High': [11, 12, 13, 14, 15],
        'Low': [9, 10, 11, 12, 13],
        'Close': [11, 12, 13, 14, 15]
    })
    data.index = pd.date_range(start="2023-01-01", periods=len(data), freq='D')
    return data


@pytest.fixture
def downtrend_data():
    """Fixture to generate sample downtrend candle data for testing."""
    data = pd.DataFrame({
        'Open': [10, 11, 12, 13, 14],
        'High': [11, 12, 13, 14, 15],
        'Low': [9, 10, 11, 12, 13],
        'Close': [9, 8, 7, 6, 5]
    })
    data.index = pd.date_range(start="2023-01-01", periods=len(data), freq='D')
    return data


@pytest.fixture
def bearish_big_shadow_data():
    data = yf.download(tickers="BTC-USD",
                       start="2024-09-05",
                       end="2024-09-08",
                       interval="1h")
    return data


@pytest.fixture
def bullish_big_shadow_data():
    data = yf.download(tickers="BTC-USD",
                       start="2024-08-29",
                       end="2024-08-31",
                       interval="1h")
    return data


@pytest.fixture(params=["sample_data",
                        "uptrend_data",
                        "downtrend_data",
                        "bearish_big_shadow_data",
                        "bullish_big_shadow_data"])
def detector(request, sample_data, uptrend_data, downtrend_data, bearish_big_shadow_data, bullish_big_shadow_data):
    """Fixture to create a BigShadowDetector instance dynamically based on the dataset."""
    dataset_mapping = {
        "sample_data": sample_data,
        "uptrend_data": uptrend_data,
        "downtrend_data": downtrend_data,
        "bearish_big_shadow_data": bearish_big_shadow_data,
        "bullish_big_shadow_data": bullish_big_shadow_data,
    }
    return BigShadowDetector(dataset_mapping[request.param])


def test_validate_data(detector):
    assert isinstance(detector.data, pd.DataFrame)


def test_missing_columns():
    data = pd.DataFrame({
        'Open': [10, 11],
        'High': [11, 12]
    })
    with pytest.raises(ValueError, match="Low not in data"):
        BigShadowDetector(data)


@pytest.mark.parametrize("detector", ["uptrend_data"], indirect=True)
def test_uptrend_detection(detector):
    detector._get_trend(ma_window=2, method="ema", trend_candles_check=1)
    assert detector.data['trend'].iloc[-1] == 'Uptrend'


@pytest.mark.parametrize("detector", ["downtrend_data"], indirect=True)
def test_downtrend_detection(detector):
    detector._get_trend(ma_window=2, method="ema", trend_candles_check=1)
    assert detector.data['trend'].iloc[-1] == 'Downtrend'


@pytest.mark.parametrize("detector", ["bearish_big_shadow_data"], indirect=True)
def test_bearish_big_shadow(detector):
    """Test that verifies the detection of a bearish big shadow after an uptrend."""
    detector.get_big_shadow(n=7, trend_check_window=10, ma_window=20, method="sma")
    assert 'bearish_big_shadow' in detector.data.columns
    assert detector.data['bearish_big_shadow'].any() == 1


@pytest.mark.parametrize("detector", ["bullish_big_shadow_data"], indirect=True)
def test_bearish_big_shadow(detector):
    """Test that verifies the detection of a bearish big shadow after an uptrend."""
    detector.get_big_shadow(n=7, trend_check_window=10, ma_window=20, method="sma")
    assert 'bullish_big_shadow' in detector.data.columns
    assert detector.data['bullish_big_shadow'].any() == 1
