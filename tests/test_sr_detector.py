import pytest
import src.patterns as patterns
import pandas as pd
import numpy as np
from src.patterns import SRDetector


@pytest.fixture
def mock_data():
    """Fixture to provide sample data for testing."""
    data = pd.DataFrame({
        'Date': ['2021-01-01', '2021-01-02', '2021-01-03', '2021-01-04', '2021-01-05'],
        'Close': [100, 101, 102, 103, 104],
        'Open': [99, 100, 101, 102, 103],
        'High': [101, 102, 103, 104, 105],
        'Low': [98, 99, 100, 101, 102]
    })
    return data


def test_initialization(mock_data):
    """Test the initialization of the SRDetector class."""
    sr_detector = patterns.SRDetector(data=mock_data, col_to_use='Close')
    assert sr_detector.data.equals(mock_data)
    assert sr_detector.col_to_use == 'Close'
    assert sr_detector.bw_method == 0.5
    assert sr_detector.prominence == 0.05

    with pytest.raises(ValueError):
        patterns.SRDetector(data=None, col_to_use='Close')

    with pytest.raises(ValueError):
        patterns.SRDetector(data=mock_data, col_to_use=None)


def test_get_log_prices(mock_data):
    detector = SRDetector(data=mock_data, col_to_use='Close')
    log_prices = detector._get_log_prices()

    expected_log_prices = np.log(mock_data['Close'])
    np.testing.assert_array_almost_equal(log_prices, expected_log_prices)


def test_get_peaks(mock_data):
    detector = SRDetector(data=mock_data, col_to_use='Close')
    peaks = detector._get_peaks()

    assert len(peaks) > 0


def test_get_sr_levels(mock_data):
    detector = SRDetector(data=mock_data, col_to_use='Close')
    sr_levels = detector.get_sr_levels()

    assert len(sr_levels) > 0
    assert all(sr_levels > 0)


def test_plot_peaks(mock_data):
    detector = SRDetector(data=mock_data, col_to_use='Close')
    try:
        detector.plot_peaks()
    except Exception as e:
        pytest.fail(f"plot_peaks raised an exception: {e}")


def test_plot_sr_levels(mock_data):
    detector = SRDetector(data=mock_data, col_to_use='Close')
    try:
        detector.plot_sr_levels()
    except Exception as e:
        pytest.fail(f"plot_sr_levels raised an exception: {e}")
