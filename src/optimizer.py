import pandas as pd
from backtesting import Backtest
from src.patterns import AverageTrueRange, AddSupportResistanceToData, TrendDetector, KangarooTailDetector
from src.patterns import detect_next_candle_bullish, detect_next_candle_bearish
import datetime as dt
import numpy as np
from src.strategy import KangarooTailStrategy
import warnings
import os
import MetaTrader5 as mt5

warnings.filterwarnings("ignore", category=UserWarning)


def connect_to_mt5(login, password, server):
    if not mt5.initialize(login=login, password=password, server=server):
        print(f"Failed to connect to MetaTrader 5 with login {login}, error: {mt5.last_error()}")
        mt5.shutdown()
    else:
        print(f"Connected to MetaTrader 5, login: {login}")
    return mt5


def get_mt5_data(time_frame, symbol, start, end):
    login = 5030037673
    password = "Vc!0XyWd"
    server = "MetaQuotes-Demo"
    mt5 = connect_to_mt5(login, password, server)
    if not mt5.symbol_select(symbol, True):
        print(f"Failed to select {symbol}")
    else:
        print(f"Selected {symbol}")
    rates = mt5.copy_rates_range(symbol, time_frame, start, end)
    if rates is not None:
        # Convert to DataFrame for better handling and readability
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')  # Convert time to readable format
        df.index = df.time
    else:
        print(f"Could not retrieve data for {symbol} in the given time frame")
        df = None
    return df[["high", "low", "close", "open"]].rename(columns={"high": "High",
                                                                "low": "Low",
                                                                "close": "Close",
                                                                "open": "Open"})


def main():
    if not os.path.exists("strategy_results/best_params.csv"):
        with open("strategy_results/best_params_and_stats.csv", "w") as f:
            f.write("ticker,size,atr_factor,entry_atr_factor,rrr,return,max_dd,avg_dd,sharpe,win_rate,expectancy\n")

    key_metrics = ["Equity Final [$]", "Sharpe Ratio", "Win Rate [%]",
                   "Expectancy [%]", "Return [%]", "Buy & Hold Return [%]",
                   "Max Drawdown [%]", "Avg Drawdown [%]"]

    crypto_tickers = ["ETH-USD", "LTC-USD", "DASH-USD", "XMR-USD", "NEO-USD",
                      "XRP-USD", "DOT-USD", "ADA-USD", "DOGE-USD"]

    fx_low_pairs = ["EURUSD", "GBPUSD", "NZDUSD", "AUDUSD", "USDCAD",
                    "USDCHF", "USDZAR", "USDNOK", "USDSEK"]

    fx_high_pairs = ["USDJPY", "EURJPY", "GBPJPY"]

    for ticker in fx_low_pairs:
        now = dt.datetime.now().replace(tzinfo=None)
        start = now - dt.timedelta(days=720)
        data = get_mt5_data(mt5.TIMEFRAME_M15, ticker, start, now)
        AverageTrueRange(data).get_atr()
        AddSupportResistanceToData(data).add_s_r_levels_to_data("Close", bw_method=0.05, prominence=0.1)
        KangarooTailDetector(data).identify_kangaroo_tails()
        TrendDetector(data).get_trend(ma_window=20, trend_candles_check=7)
        data["next_candle_bullish"] = detect_next_candle_bullish(data)
        data["next_candle_bearish"] = detect_next_candle_bearish(data)
        data["next_high"] = data.High.shift(-1)
        data["next_low"] = data.Low.shift(-1)
        bt = Backtest(data, KangarooTailStrategy, commission=0.0002, cash=10000, margin=0.02)

        stats = bt.optimize(
            # size=list(np.arange(0.05, 0.5, 0.05)),  # For crypto
            size=range(500, 10000, 250),  # Low fx pairs
            #size=range(10, 500, 10),
            entry_atr_factor=list(np.arange(0.01, 0.5, 0.01)),
            atr_factor=list(np.arange(0.02, 2.0, 0.02)),
            rrr=list(np.arange(1.0, 3.0, 0.1)),
            maximize="Sharpe Ratio",
            max_tries=250
        )

        best_params = stats._strategy
        print(stats)
        print(f"Best parameters for {ticker}: {best_params}")
        with open("strategy_results/best_params_and_stats.csv", "a+") as f:
            f.write(f"{ticker},{best_params.size},{best_params.atr_factor},{best_params.entry_atr_factor},"
                    f"{best_params.rrr},{stats['Return [%]']},{stats['Max. Drawdown [%]']},{stats['Avg. Drawdown [%]']},"
                    f"{stats['Sharpe Ratio']},{stats['Win Rate [%]']},{stats['Expectancy [%]']}\n")


if __name__ == "__main__":
    main()
