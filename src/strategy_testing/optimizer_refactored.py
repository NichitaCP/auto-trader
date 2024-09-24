import pandas as pd
from backtesting import Backtest
from src.patterns import AverageTrueRange, AddSupportResistanceToData, TrendDetector, KangarooTailDetector
from src.patterns import detect_next_candle_bullish, detect_next_candle_bearish
import datetime as dt
import numpy as np
from src.strategy_testing.strategy import KangarooTailStrategy
import warnings
import os
import MetaTrader5 as mt5
from typing import Iterable


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
    return data


def optimize_strategy(
        strategy,
        data: pd.DataFrame,
        cash: float = 10000,
        commission: float = 0.0002,
        margin: float = 1,
        size: Iterable = None,
        entry_atr_factor: Iterable = None,
        atr_factor: Iterable = None,
        rrr: Iterable = None,
        maximize: str = "Sharpe Ratio",
        max_tries: int = None,
        trade_on_close: bool = False):

    bt = Backtest(data, strategy, cash=cash, commission=commission, margin=margin, hedging=False, trade_on_close=trade_on_close)
    stats = bt.optimize(
        size=size,
        entry_atr_factor=entry_atr_factor,
        atr_factor=atr_factor,
        rrr=rrr,
        maximize=maximize,
        max_tries=max_tries
    )
    return stats


def log_best_params_and_stats(ticker, stats, filename):
    best_params = stats._strategy
    log_entry = f"{ticker},{best_params.size},{best_params.atr_factor},{best_params.entry_atr_factor}," \
                f"{round(best_params.rrr, 2)},{stats['Return [%]']},{stats['Max. Drawdown [%]']}," \
                f"{stats['Avg. Drawdown [%]']},{stats['Sharpe Ratio']}," \
                f"{stats['Win Rate [%]']},{stats['Expectancy [%]']}\n"
    with open(filename, "a+") as f:
        f.write(log_entry)

    print(f"Best parameters for {ticker}: {best_params}")
    print(stats)


def main():
    warnings.filterwarnings("ignore", category=UserWarning)

    # MT5 credentials
    login = 1510009878
    password = "825$tnr$DJ"
    server = "FTMO-Demo"
    file_name = f"../strategy_results/best_params_and_stats_xauusd.csv"
    if not os.path.exists(file_name):
        with open(file_name, "w") as f:
            f.write("ticker,size,atr_factor,entry_atr_factor,rrr,return,max_dd,avg_dd,sharpe,win_rate,expectancy\n")

    # Define the pairs to optimize
    crypto = ["ETHUSD", "BTCUSD"]
    eur_low_spread_pairs = ["NZDCAD", "NZDUSD"]
    commodities = ["XAUUSD", "XAUUSD"]

    connect_to_mt5(login, password, server)

    for ticker in crypto:
        # now = dt.datetime.now().replace(tzinfo=None)
        # start = now - dt.timedelta(days=30)
        start_pos = 1
        count_bars = 15000

        data = get_mt5_data(mt5.TIMEFRAME_M15, ticker, start_pos, count_bars)
        if data is None:
            continue

        data = prepare_data_for_backtest_optimization(data)
        stats = optimize_strategy(
            strategy=KangarooTailStrategy,
            data=data,
            size=list(np.arange(1, 100, 1)),
            entry_atr_factor=list(np.arange(0.15, 1.25, 0.025)),
            atr_factor=list(np.arange(0.2, 2, 0.05)),
            rrr=list(np.arange(1.1, 3, 0.05)),
            maximize="Expectancy [%]",
            max_tries=200,
            cash=200000,
            commission=0.00035,
            margin=0.05
        )
        log_best_params_and_stats(ticker, stats, file_name)
    mt5.shutdown()


if __name__ == "__main__":
    main()
