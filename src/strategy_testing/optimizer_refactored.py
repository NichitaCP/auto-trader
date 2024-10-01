import pandas as pd
from backtesting import Backtest
from src.trade_functions import connect_to_mt5, get_mt5_data, prepare_data_for_backtest_optimization
import numpy as np
from src.strategy_testing.strategy import KangarooTailStrategy, FVGStrategy
import warnings
import os
import MetaTrader5 as mt5
from typing import Iterable


def optimize_strategy(
        strategy,
        data: pd.DataFrame,
        cash: float = 10000,
        commission: float = 0.0002,
        margin: float = 1,
        size: Iterable = None,
        # entry_atr_factor: Iterable = None,
        fvg_gap_factor: Iterable = None,
        limit_factor: Iterable = None,
        atr_factor: Iterable = None,
        rrr: Iterable = None,
        maximize: str = "Sharpe Ratio",
        max_tries: int = None,
        trade_on_close: bool = False):

    bt = Backtest(data, strategy, cash=cash, commission=commission, margin=margin, hedging=False, trade_on_close=trade_on_close)
    stats = bt.optimize(
        size=size,
        # entry_atr_factor=entry_atr_factor,
        limit_factor=limit_factor,
        fvg_gap_factor=fvg_gap_factor,
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
    eur_low_spread_pairs = ["AUDJPY", "USDJPY"]
    commodities = ["XAUUSD", "XAUUSD"]

    connect_to_mt5(login, password, server)

    for ticker in eur_low_spread_pairs:
        # now = dt.datetime.now().replace(tzinfo=None)
        # start = now - dt.timedelta(days=30)
        start_pos = 1
        count_bars = 40000

        data = get_mt5_data(mt5.TIMEFRAME_M15, ticker, start_pos, count_bars)
        if data is None:
            continue

        data = prepare_data_for_backtest_optimization(data)
        stats = optimize_strategy(
            strategy=FVGStrategy,
            data=data,
            size=list(range(100, 500, 25)),
            # entry_atr_factor=list(np.arange(0.15, 1.25, 0.025)),
            limit_factor=list(np.arange(0.05, 1, 0.05)),
            fvg_gap_factor=list(np.arange(0.01, 0.75, 0.01)),
            atr_factor=list(np.arange(0.05, 2, 0.025)),
            rrr=list(np.arange(1.1, 3, 0.05)),
            maximize="Expectancy [%]",
            max_tries=500,
            cash=20000,
            commission=3e-5,
            margin=0.01
        )
        print(stats)
        print("Best parameters:")
        print(stats._strategy)
        # log_best_params_and_stats(ticker, stats, file_name)
    mt5.shutdown()


if __name__ == "__main__":
    main()
