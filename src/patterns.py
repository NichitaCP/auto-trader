import pandas as pd
from scipy.stats import gaussian_kde
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from typing import Tuple, Literal


class SRDetector:
    def __init__(self,
                 data: pd.DataFrame = None,
                 col_to_use: str = None,
                 bw_method: float = 0.5,
                 prominence: float = 0.05) -> None:

        self.data = data
        self.col_to_use = col_to_use
        self._validate_data()
        self.bw_method = bw_method
        self.prominence = prominence
        self.log_prices = self._get_log_prices()
        self.kde_values = self._compute_kde()
        self.x_axis_values = np.linspace(min(self.log_prices), max(self.log_prices), 500)
        self.data = data.copy()

    def _validate_data(self) -> None:
        if self.col_to_use is None:
            raise ValueError("col_to_use must be specified")
        if self.data is None:
            raise ValueError("data must be specified")
        if self.col_to_use not in self.data.columns:
            raise ValueError(f"{self.col_to_use} not in data")

    def _get_log_prices(self) -> np.ndarray:
        """Get the prices to use for the KDE"""
        prices = self.data[self.col_to_use]
        return np.log(prices)

    def _compute_kde(self) -> np.ndarray:
        """Compute the kernel density estimation of the data"""
        log_prices = self._get_log_prices()
        x = np.linspace(min(log_prices), max(log_prices), 500)
        kde = gaussian_kde(log_prices, bw_method=self.bw_method)
        return kde(x)

    def _get_peaks(self) -> np.ndarray:
        """Find the peaks of the KDE"""
        peaks, _ = find_peaks(self.kde_values, prominence=self.prominence)
        return peaks

    def plot_peaks(self) -> None:
        """Plot the KDE and the peaks"""
        peaks = self._get_peaks()
        plt.plot(self.x_axis_values, self.kde_values)
        plt.scatter(self.x_axis_values[peaks], self.kde_values[peaks], color='red', marker='x')
        plt.xlabel("Log Prices")
        plt.ylabel("Density")
        plt.title("KDE with Peaks")
        plt.legend(["KDE", "Peaks"])
        plt.show()

    def get_sr_levels(self) -> np.ndarray:
        """Get the support and resistance levels"""
        peaks = self._get_peaks()
        return np.exp(self.x_axis_values[peaks])

    def plot_sr_levels(self) -> None:
        """Plot the support and resistance levels"""
        s_r_levels = self.get_sr_levels()
        plt.plot(self.data.index, self.data[self.col_to_use])
        for level in s_r_levels:
            plt.axhline(level, color='red', linestyle='--')
        plt.xticks(rotation=45)
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.title("Support and Resistance Levels")
        plt.legend(["Price", "Support/Resistance"])
        plt.show()


class KangarooTailDetector:
    def __init__(self,
                 data: pd.DataFrame = None) -> None:

        if data is None:
            raise ValueError("data must be specified")
        self.data = data.copy()
        self._validate_data()

    def _validate_data(self):
        """Validate the data"""
        required_columns = ["Open", "High", "Low", "Close"]
        for col in required_columns:
            if col not in self.data.columns:
                raise ValueError(f"{col} not in data")
        if not isinstance(self.data.index, pd.DatetimeIndex):
            raise ValueError("Data index must be a datetime index")

    def _calculate_candle_body_size(self) -> pd.Series:
        """Calculate the body size of the candle"""
        body_lengths = abs(self.data.Open - self.data.Close)
        if (body_lengths == 0).any():
            raise ValueError("Body length cannot be 0")
        return body_lengths

    def _calculate_candle_tail_size(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the tail size of the candle"""
        lower_tail = np.where(self.data.Open < self.data.Close,
                              self.data.Open - self.data.Low,
                              self.data.Close - self.data.Low)
        upper_tail = np.where(self.data.Open < self.data.Close,
                              self.data.High - self.data.Close,
                              self.data.High - self.data.Open)

        return lower_tail, upper_tail

    def _calculate_body_to_shadow_proportions(self) -> None:
        """Calculate the body to shadow proportions"""
        lower_tail, upper_tail = self._calculate_candle_tail_size()
        body_lengths = self._calculate_candle_body_size()
        self.data["lower_tail_proportion"] = lower_tail / body_lengths
        self.data["upper_tail_proportion"] = upper_tail / body_lengths

    def _calculate_green_tail_condition(self) -> pd.Series:
        """Calculate the green kangaroo tail condition"""
        return ((self.data.lower_tail_proportion >= 4) &
                (self.data.lower_tail_proportion > self.data.upper_tail_proportion))

    def _calculate_red_tail_condition(self) -> pd.Series:
        """Calculate the red kangaroo tail condition"""
        return ((self.data.upper_tail_proportion >= 4) &
                (self.data.upper_tail_proportion > self.data.lower_tail_proportion))

    def _identify_conditions(self) -> None:
        """Identify the conditions for a kangaroo tail"""
        self._calculate_body_to_shadow_proportions()
        green_kangaroo_tail_condition = self._calculate_green_tail_condition()
        red_kangaroo_tail_condition = self._calculate_red_tail_condition()
        self.data["green_kangaroo_tail"] = np.where(green_kangaroo_tail_condition, 1, 0)
        self.data["red_kangaroo_tail"] = np.where(red_kangaroo_tail_condition, 1, 0)

    def identify_kangaroo_tails(self) -> pd.DataFrame:
        """Identify the kangaroo tails"""
        self._identify_conditions()
        return self.data


class BigShadowDetector:
    def __init__(self,
                 data: pd.DataFrame = None) -> None:
        self.data = data
        self._validate_data()
        self.data = data.copy()

    def _validate_data(self):
        """Validate the data"""
        if self.data is None:
            raise ValueError("data must be specified")
        required_columns = ["Open", "High", "Low", "Close"]
        for col in required_columns:
            if col not in self.data.columns:
                raise ValueError(f"{col} not in data")
        if not isinstance(self.data.index, pd.DatetimeIndex):
            raise ValueError("Data index must be a datetime index")

    def _bigger_high_low_than_last_candle(self):
        """Check if the current candle has both a higher high and a lower low than the last candle"""
        self.data["engulfing"] = ((self.data.High > self.data.High.shift(1)) &
                                  (self.data.Low < self.data.Low.shift(1)))

    def _calculate_candle_ranges(self):
        """Calculate the range of the candle"""
        self.data["candle_range"] = self.data.High - self.data.Low

    def _bigger_than_previous_n_candles(self, n=10):
        """Check if the candle is bigger than the previous n candles"""
        self._calculate_candle_ranges()
        for index in range(n, len(self.data)):
            current_range = self.data["candle_range"].iloc[index]
            last_n_ranges = self.data["candle_range"].iloc[index - n:index]  # The 10 previous candle's ranges
            last_n_ranges_max = np.max(last_n_ranges)
            self.data.loc[self.data.index[index], f"bigger_than_{n}_prev_candles"] = current_range > last_n_ranges_max

    def _get_sma(self, window=10):
        """Calculate the simple moving average of the candle range. Window size is n"""
        self.data[f"sma_{window}"] = self.data["Close"].rolling(window).mean()

    def _get_ema(self, window=10):
        """Calculate the exponential moving average of the candle range. Window size is n"""
        self.data[f"ema_{window}"] = self.data["Close"].ewm(span=window, adjust=False).mean()

    def _get_trend(self, window=10, method="sma"):
        """Get the trend of the candle range"""
        if method == "sma":
            self._get_sma(window)
            moving_avg_col = f"sma_{window}"
        elif method == "ema":
            self._get_ema(window)
            moving_avg_col = f"ema_{window}"
        else:
            raise ValueError("method must be either 'sma' or 'ema'")

        mask_above_ma = self.data["Close"] > self.data[moving_avg_col]
        mask_below_ma = self.data["Close"] < self.data[moving_avg_col]

        uptrend_condition = mask_above_ma.rolling(window=window).sum() == window
        downtrend_condition = mask_below_ma.rolling(window=window).sum() == window

        self.data["trend"] = "Mixed"
        self.data.loc[uptrend_condition, "trend"] = "Uptrend"
        self.data.loc[downtrend_condition, "trend"] = "Downtrend"

    def _identify_bearish_big_shadow(self, n=7, window=10, method="sma"):
        """Identify the bearish big shadow"""
        self._bigger_high_low_than_last_candle()
        self._bigger_than_previous_n_candles(n)
        self._get_trend(window=window, method=method)
        self.data["bearish_big_shadow"] = ((self.data["engulfing"]) &
                                           (self.data[f"bigger_than_{n}_prev_candles"]) &
                                           (self.data["trend"] == "Downtrend"))

    def _identify_bullish_big_shadow(self, n=7, window=10, method="sma"):
        """Identify the bullish big shadow"""
        self._bigger_high_low_than_last_candle()
        self._bigger_than_previous_n_candles(n)
        self._get_trend(window=window, method=method)
        self.data["bullish_big_shadow"] = ((self.data["engulfing"]) &
                                           (self.data[f"bigger_than_{n}_prev_candles"]) &
                                           (self.data["trend"] == "Uptrend"))

    def get_big_shadow(self, n=7, window=10, method="sma"):
        """Get the big shadow"""
        self._identify_bearish_big_shadow(n=n, window=window, method=method)
        self._identify_bullish_big_shadow(n=n, window=window, method=method)
        return self.data
