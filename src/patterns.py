import pandas as pd
from scipy.stats import gaussian_kde
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


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
