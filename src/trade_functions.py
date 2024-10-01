import pandas as pd
from src.patterns import AverageTrueRange, AddSupportResistanceToData, TrendDetector, KangarooTailDetector
from src.patterns import detect_next_candle_bullish, detect_next_candle_bearish
import MetaTrader5 as mt5
import talib
import numpy as np


def connect_to_mt5(login, password, server):
    if not mt5.initialize(login=login, password=password, server=server):
        print(f"Failed to connect to MetaTrader 5 with login {login}, error: {mt5.last_error()}")
        mt5.shutdown()
    # else:
    #     print(f"Connected to MetaTrader 5, login: {login}")


def get_mt5_data(time_frame, symbol, start_pos, bar_count):
    if not mt5.symbol_select(symbol, True):
        print(f"Failed to select {symbol}")
    rates = mt5.copy_rates_from_pos(symbol, time_frame, start_pos, bar_count)
    if rates is not None:
        # Convert to DataFrame for better handling and readability
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')  # Convert time to readable format
        df.index = df.time
    else:
        print(f"Could not retrieve data for {symbol} in the given time frame")
        df = None
    return df[["high", "low", "close", "open"]].rename(columns={
        "high": "High",
        "low": "Low",
        "close": "Close",
        "open": "Open"})


def prepare_data_for_backtest_optimization(data):
    AverageTrueRange(data).get_atr()
    AddSupportResistanceToData(data).add_s_r_levels_to_data("Close", bw_method=0.05, prominence=0.1)
    KangarooTailDetector(data).identify_kangaroo_tails()
    TrendDetector(data).get_trend(ma_window=50, trend_candles_check=10)
    data["next_candle_bullish"] = detect_next_candle_bullish(data)
    data["next_candle_bearish"] = detect_next_candle_bearish(data)
    data["next_high"] = data.High.shift(-1)
    data["next_low"] = data.Low.shift(-1)
    data["bullish_gap"] = data.Low - data.High.shift(2)
    data["bearish_gap"] = data.Low.shift(2) - data.High
    data["swing_high"] = np.where(data.High >= data.High.rolling(20).max(), data.High, np.nan)
    data["swing_low"] = np.where(data.Low <= data.Low.rolling(20).min(), data.Low, np.nan)
    data['swing_high'] = data['swing_high'].ffill().bfill()
    data['swing_low'] = data['swing_low'].ffill().bfill()
    data["change_of_character"] = np.where(
        (data.trend == "Downtrend") & (data.High >= data.swing_high), 1,
        np.where(
            (data.trend == "Uptrend") & (data.Low <= data.swing_low), -1,
            0
        )
    )
    data["break_of_structure"] = np.where(
        (data.trend == "Downtrend") & (data.Low <= data.swing_low), 1,
        np.where(
            (data.trend == "Uptrend") & (data.High >= data.swing_high), -1,
            0
        )
    )
    last_10_bos_or_choch = (data.break_of_structure.rolling(window=10, min_periods=1).max() == 1) | \
                           (data.change_of_character.rolling(window=10, min_periods=1).max() == 1)
    next_10_bos_or_choch = (data.break_of_structure.shift(-1).rolling(window=10, min_periods=1).max() == 1) | \
                           (data.change_of_character.shift(-1).rolling(window=10, min_periods=1).max() == 1)

    valid_bullish_fvg_condition = (last_10_bos_or_choch | next_10_bos_or_choch) & (data.bullish_gap > 0)
    valid_bearish_fvg_condition = (last_10_bos_or_choch | next_10_bos_or_choch) & (data.bearish_gap > 0)
    data["fvg"] = np.where(valid_bullish_fvg_condition, 1,
                           np.where(valid_bearish_fvg_condition, -1, 0))
    data["bullish_fvg_upper_bound"] = data.Low
    data["bullish_fvg_lower_bound"] = data.High.shift(2)

    return data
