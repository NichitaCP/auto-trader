import pandas as pd
from scipy.stats import gaussian_kde
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from typing import Tuple, Literal


def detect_next_candle_bullish(data: pd.DataFrame):
    shifted_open = data['Open'].shift(-1)
    shifted_close = data['Close'].shift(-1)
    shifted_high = data['High'].shift(-1)

    close_near_high = (shifted_high - shifted_close) < ((shifted_close - shifted_open) * 0.5)
    close_above_open = shifted_close > shifted_open
    return close_near_high & close_above_open


def detect_next_candle_bearish(data: pd.DataFrame):
    shifted_open = data['Open'].shift(-1)
    shifted_close = data['Close'].shift(-1)
    shifted_low = data['Low'].shift(-1)

    close_near_low = (shifted_close - shifted_low) < ((shifted_open - shifted_close) * 0.5)
    close_below_open = shifted_close < shifted_open
    return close_near_low & close_below_open


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


class AddSupportResistanceToData:
    def __init__(self,
                 data: pd.DataFrame = None) -> None:
        """Be aware the data is modified in place."""
        if data is None:
            raise ValueError("data must be specified")
        self.data = data

    def _get_sr_levels(self,
                       col_to_use: str = "Close",
                       bw_method: float = 0.5,
                       prominence: float = 0.05) -> np.ndarray:
        """Get the support and resistance levels"""
        sr_detector = SRDetector(data=self.data, col_to_use=col_to_use, bw_method=bw_method, prominence=prominence)
        return sr_detector.get_sr_levels()

    def _get_support_indexes(self,
                             col_to_use: str = "Close",
                             levels: np.ndarray = None):
        """Get the indexes of the support levels"""
        if levels is None:
            raise ValueError("levels must be specified")
        support_indexes = np.argmax(levels > self.data[col_to_use].values[:, None], axis=1) - 1
        return support_indexes

    def _get_resistance_indexes(self,
                                col_to_use: str = "Close",
                                levels: np.ndarray = None):
        """Get the indexes of the resistance levels"""
        if levels is None:
            raise ValueError("levels must be specified")
        resistance_indexes = np.argmax(levels > self.data[col_to_use].values[:, None], axis=1)
        return resistance_indexes

    def add_s_r_levels_to_data(self,
                               col_to_use: str = "Close",
                               bw_method: float = 0.5,
                               prominence: float = 0.05) -> None:
        """Add the support and resistance levels to the data."""
        sr_levels = self._get_sr_levels(col_to_use=col_to_use, bw_method=bw_method, prominence=prominence)
        support_indexes = self._get_support_indexes(col_to_use=col_to_use, levels=sr_levels)
        resistance_indexes = self._get_resistance_indexes(col_to_use=col_to_use, levels=sr_levels)
        self.data["next_support"] = sr_levels[support_indexes]
        self.data["next_resistance"] = sr_levels[resistance_indexes]

        """
        This part avoids the next support to be mapped to index -1, meaning hihgest support.
        Similarly, we want to map the max resistance to the highest resistance if that level wasn't 
        found by our KDE method.
        """
        price_below_support_mask = self.data[col_to_use] < self.data["next_support"]
        price_above_resistance_mask = self.data[col_to_use] > self.data["next_resistance"]
        self.data.loc[price_below_support_mask, "next_support"] = self.data.Close.min()
        self.data.loc[price_above_resistance_mask, "next_resistance"] = self.data.Close.max()


class AverageTrueRange:
    def __init__(self,
                 data: pd.DataFrame = None) -> None:
        if data is None:
            raise ValueError("data must be specified")
        self.data = data

    def _get_true_range(self):
        """Calculate the true range of the data"""
        high_low = self.data.High - self.data.Low
        high_prev_close = abs(self.data.High - self.data.Close.shift(1))
        low_prev_close = abs(self.data.Low - self.data.Close.shift(1))
        self.data["true_range"] = np.max([high_low, high_prev_close, low_prev_close], axis=0)

    def get_atr(self,
                window: int = 14,
                smoothing: Literal["ema", "sma"] = "sma") -> None:
        """Compute the average true range given a window size"""
        self._get_true_range()
        if smoothing == "sma":
            self.data["atr"] = self.data["true_range"].rolling(window).mean()
        elif smoothing == "ema":
            self.data["atr"] = self.data["true_range"].ewm(span=window, adjust=False).mean()
        else:
            raise ValueError("smoothing must be either 'sma' or 'ema'")


class TrendDetector:
    def __init__(self,
                 data: pd.DataFrame = None) -> None:
        if data is None:
            raise ValueError("data must be specified")
        self.data = data  # Copy might be needed, have to analyze behaviour

    def _get_sma(self,
                 ma_window: int = 20) -> None:
        """Calculate the simple moving average of the candle range. Window size is n"""
        self.data[f"sma_{ma_window}"] = self.data["Close"].rolling(ma_window).mean()

    def _get_ema(self,
                 ma_window: int = 20) -> None:
        """Calculate the exponential moving average of the candle range. Window size is n"""
        self.data[f"ema_{ma_window}"] = self.data["Close"].ewm(span=ma_window, adjust=False).mean()

    def get_trend(self,
                  ma_window: int = 20,
                  trend_candles_check: int = 10,
                  method: Literal["sma", "ema"] = "sma") -> None:
        """Get the trend of the candle range"""
        if method == "sma":
            self._get_sma(ma_window)
            moving_avg_col = f"sma_{ma_window}"
        elif method == "ema":
            self._get_ema(ma_window)
            moving_avg_col = f"ema_{ma_window}"
        else:
            raise ValueError("method must be either 'sma' or 'ema'")

        mask_above_ma = self.data["Close"] > self.data[moving_avg_col]
        mask_below_ma = self.data["Close"] < self.data[moving_avg_col]

        """
        For future. Tinkering with this part might give better tren detection
        Currently we are checking that exactly {trend_candles_check} close prices are above 
        or below the MA.
        """

        uptrend_condition = mask_above_ma.rolling(window=trend_candles_check).sum() == trend_candles_check
        downtrend_condition = mask_below_ma.rolling(window=trend_candles_check).sum() == trend_candles_check

        self.data["trend"] = "Mixed"
        self.data.loc[uptrend_condition, "trend"] = "Uptrend"
        self.data.loc[downtrend_condition, "trend"] = "Downtrend"


class KangarooTailDetector:
    def __init__(self,
                 data: pd.DataFrame = None) -> None:

        if data is None:
            raise ValueError("data must be specified")
        self.data = data
        self._validate_data()

    # Will probably move this outside as a static function
    def _validate_data(self) -> None:
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
            body_lengths[body_lengths == 0] = 0.0001
        return body_lengths

    def _calculate_candle_tail_size(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the tail size of the candle"""

        # Depending on the nature of candle (Bearish/Bullish), the tail is calculated differently.
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

    def identify_kangaroo_tails(self) -> None:
        """Identify the kangaroo tails"""
        self._identify_conditions()


class BigShadowDetector:
    def __init__(self,
                 data: pd.DataFrame = None) -> None:
        # Initialization is probably suboptimal lol (the first 3 lines at least)
        self.data = data
        self._validate_data()
        self.trend_detector = TrendDetector(data=self.data)

    def _validate_data(self) -> None:
        """Validate the data"""
        if self.data is None:
            raise ValueError("data must be specified")
        required_columns = ["Open", "High", "Low", "Close"]
        for col in required_columns:
            if col not in self.data.columns:
                raise ValueError(f"{col} not in data")
        if not isinstance(self.data.index, pd.DatetimeIndex):
            raise ValueError("Data index must be a datetime index")

    def _get_trend(self,
                   ma_window: int = 20,
                   trend_candles_check: int = 10,
                   method: Literal["sma", "ema"] = "sma") -> None:
        """Get the trend of the candle range"""
        self.trend_detector.get_trend(ma_window=ma_window,
                                      trend_candles_check=trend_candles_check,
                                      method=method)

    def _bullish_close_higher_than_last_open(self) -> None:
        """Check if the close price is higher than the last open price"""
        self.data["bullish_close_higher_than_last_open"] = self.data.Close > self.data.Open.shift(1)

    def _bearish_close_lower_than_last_open(self) -> None:
        """Check if the close price is lower than the last open price"""
        self.data["bearish_close_lower_than_last_open"] = self.data.Close < self.data.Open.shift(1)

    def _bigger_high_low_than_last_candle(self) -> None:
        """Check if the current candle has both a higher high and a lower low than the last candle"""

        self.data["engulfing"] = ((self.data.High > self.data.High.shift(1)) &
                                  (self.data.Low < self.data.Low.shift(1)))

    def _bigger_body_range_than_last_candle(self) -> None:
        """Check if the body range is bigger than the previous candle"""
        self.data["body_range"] = abs(self.data.Open - self.data.Close)
        self.data["body_range_prev"] = abs(self.data.Open.shift(1) - self.data.Close.shift(1))
        self.data["bigger_body"] = self.data["body_range"] > self.data["body_range_prev"]

    def _calculate_candle_ranges(self) -> None:
        """Calculate the range of the candle"""
        self.data["candle_range"] = self.data.High - self.data.Low

    def _bigger_than_previous_n_candles(self,
                                        n: int = 10) -> None:
        """Check if the candle is bigger than the previous n candles"""
        self._calculate_candle_ranges()
        for index in range(n, len(self.data)):
            current_range = self.data["candle_range"].iloc[index]
            last_n_ranges = self.data["candle_range"].iloc[index - n:index]  # The 10 previous candle's ranges
            last_n_ranges_max = np.max(last_n_ranges)
            self.data.loc[self.data.index[index], f"bigger_than_{n}_prev_candles"] = current_range > last_n_ranges_max

    def _close_price_near_low(self) -> None:
        """Check if the close price is near the low of the candle"""
        self.data["low_to_close"] = abs(self.data.Low - self.data.Close)
        self.data["mid_point_open_close"] = abs(self.data.Open - self.data.Close) / 2
        self.data["close_near_low"] = (self.data["low_to_close"] < self.data["mid_point_open_close"])

    def _close_price_near_high(self) -> None:
        """Check if the close price is near the high of the candle"""
        self.data["high_to_close"] = abs(self.data.High - self.data.Close)
        self.data["mid_point_open_close"] = abs(self.data.Open - self.data.Close) / 2
        self.data["close_near_high"] = (self.data["high_to_close"] < self.data["mid_point_open_close"])

    def _identify_bearish_big_shadow(self,
                                     n: int = 7,  # Number of candles to lookback for the range
                                     ma_window: int = 20,
                                     trend_check_window: int = 10,
                                     method: Literal["sma", "ema"] = "sma") -> None:
        """Identify the bearish big shadow"""
        self._bigger_high_low_than_last_candle()
        self._bigger_body_range_than_last_candle()
        self._bearish_close_lower_than_last_open()
        self._bigger_than_previous_n_candles(n)
        self._close_price_near_low()
        self._get_trend(ma_window=ma_window, method=method, trend_candles_check=trend_check_window)
        self.data["bearish_big_shadow"] = ((self.data["engulfing"])
                                           & (self.data[f"bigger_than_{n}_prev_candles"])
                                           & (self.data["bearish_close_lower_than_last_open"])
                                           & (self.data["trend"] == "Uptrend")
                                           & (self.data["close_near_low"])
                                           & (self.data["bigger_body"]))

    def _identify_bullish_big_shadow(self,
                                     n: int = 7,  # Number of candles to lookback for the range
                                     ma_window: int = 20,
                                     trend_check_window: int = 10,
                                     method: Literal["sma", "ema"] = "sma") -> None:
        """Identify the bullish big shadow"""
        self._get_trend(ma_window=ma_window, method=method, trend_candles_check=trend_check_window)
        self._bigger_high_low_than_last_candle()
        self._bigger_body_range_than_last_candle()
        self._bullish_close_higher_than_last_open()
        self._bigger_than_previous_n_candles(n)
        self._close_price_near_high()
        self.data["bullish_big_shadow"] = ((self.data["engulfing"])
                                           & (self.data[f"bigger_than_{n}_prev_candles"])
                                           & (self.data["bullish_close_higher_than_last_open"])
                                           & (self.data["trend"] == "Downtrend")
                                           & (self.data["close_near_high"])
                                           & (self.data["bigger_body"]))

    def get_big_shadow(self,
                       n: int = 7,  # Number of candles to lookback for the range
                       ma_window: int = 20,
                       trend_check_window: int = 10,
                       method: Literal["sma", "ema"] = "sma") -> None:
        """Get the big shadow"""
        self._identify_bearish_big_shadow(n=n,
                                          ma_window=ma_window,
                                          method=method,
                                          trend_check_window=trend_check_window)

        self._identify_bullish_big_shadow(n=n,
                                          ma_window=ma_window,
                                          method=method,
                                          trend_check_window=trend_check_window)
