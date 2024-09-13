from backtesting import Backtest, Strategy
from src.patterns import AverageTrueRange, AddSupportResistanceToData, TrendDetector, KangarooTailDetector
from src.patterns import detect_next_candle_bullish, detect_next_candle_bearish
import yfinance
import datetime as dt
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor

# Min winrate to rrr dict


def optimize_winrate_and_expectancy(series):
    if series["# Trades"] < 40:
        return -1
    return series["Win Rate [%]"] * series["Expectancy [%]"]


def optimize_return_expectancy(series):
    if series["# Trades"] < 40:
        return -1
    return series["Return [%]"] * series["Expectancy [%]"]


class KangarooTailStrategy(Strategy):

    long_entry = True
    size = 0.1
    entry_atr_factor = 0.15
    atr_factor = 1.5
    rrr = 2.0
    atr = None
    # buy_stop, sell_stop = None, None

    def init(self):

        self.atr = self.data.atr

    def next(self):

        i = len(self.data) - 1
        green_kt = self.data.green_kangaroo_tail[i]
        red_kt = self.data.red_kangaroo_tail[i]
        uptrend = self.data.trend[i] == "Uptrend"
        downtrend = self.data.trend[i] == "Downtrend"
        mixed = self.data.trend[i] == "Mixed"
        bullish_next = self.data.next_candle_bullish[i]
        bearish_next = self.data.next_candle_bearish[i]

        next_high = self.data.High[i]
        next_low = self.data.Low[i]

        buy_stop = next_high + self.atr[i] * self.entry_atr_factor
        sell_stop = next_low - self.atr[i] * self.entry_atr_factor

        stop_loss_long = self.data.Low[i] - self.atr[i] * self.atr_factor
        take_profit_long = buy_stop + (buy_stop - stop_loss_long) * self.rrr
        stop_loss_short = self.data.High[i] + self.atr[i] * self.atr_factor
        take_profit_short = sell_stop - (stop_loss_short - sell_stop) * self.rrr

        vars_to_check = np.array([buy_stop, sell_stop, stop_loss_long,
                                  take_profit_long, stop_loss_short, take_profit_short])

        if not all(np.isfinite(var) and var > 0 for var in vars_to_check):
            return

        if not self.position:
            if self.long_entry:
                if bullish_next and uptrend:
                    if stop_loss_long < buy_stop < take_profit_long:
                        self.buy(sl=stop_loss_long, tp=take_profit_long, stop=buy_stop, size=self.size)
            else:
                if bearish_next and not uptrend:
                    if stop_loss_short > sell_stop > take_profit_short:
                        self.sell(sl=stop_loss_short, tp=take_profit_short, stop=sell_stop, size=self.size)


class SwingTradingStrategy(Strategy):
    size = 0.1
    entry_atr_factor = 0.15
    atr_factor = 1.5
    rrr = 2.0
    def init(self):
        pass

    def next(self):

        i = len(data)