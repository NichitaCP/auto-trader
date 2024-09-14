import datetime as dt
from src.patterns import AverageTrueRange, TrendDetector, detect_next_candle_bullish
from src.patterns import detect_next_candle_bearish
import pandas as pd
import os
from src.strategy_testing.optimizer_refactored import get_mt5_data, connect_to_mt5
import MetaTrader5 as mt5


def get_position_size(account_size, risk_per_trade, stop_loss_distance):
    return (account_size * risk_per_trade) / stop_loss_distance


def prepare_data_for_signal(data):
    AverageTrueRange(data).get_atr(window=14)
    TrendDetector(data).get_trend(ma_window=50, trend_candles_check=7)
    data["next_bullish"] = detect_next_candle_bullish(data)
    data["next_bearish"] = detect_next_candle_bearish(data)
    data["next_high"] = data.High.shift(-1)
    data["next_low"] = data.Low.shift(-1)


def calculate_long_entry_prices(candle, entry_atr_factor, rrr, atr_factor, atr):
    buy_stop = candle.High + atr * entry_atr_factor
    stop_loss = candle.Low - atr * atr_factor
    take_profit = buy_stop + (buy_stop - stop_loss) * rrr
    return buy_stop, stop_loss, take_profit


def get_buy_signal(prev_candle):
    if (prev_candle["next_bullish"] == 1) & (prev_candle["trend"] == "Uptrend"):
        return True


def run_strategy(data, ticker, entry_atr_factor, rrr, atr_factor):
    prepare_data_for_signal(data)
    prev_candle = data.iloc[-2]
    atr = prev_candle["atr"]
    buy_stop, stop_loss, take_profit, position_size = None, None, None, None
    open_order_now = False
    if get_buy_signal(prev_candle):
        buy_stop, stop_loss, take_profit = calculate_long_entry_prices(prev_candle, entry_atr_factor, rrr, atr_factor,
                                                                       atr)
        stop_loss_distance = buy_stop - stop_loss
        position_size = get_position_size(account_size=100, risk_per_trade=0.01, stop_loss_distance=stop_loss_distance)
        print(f"Long signal detected for {ticker}\n{'#' * 50}")
        print(f"Buy stop: {buy_stop}\nStop loss: {stop_loss}\nTake profit: {take_profit}")
        print(f"Position size: {position_size}")
        open_order_now = True
    else:
        print(f"No signal detected")
    return open_order_now, ticker, buy_stop, stop_loss, take_profit, position_size


def get_signals_for_tickers_15m(csv_path, login, password, server):
    csv_path = os.path.abspath(csv_path)
    results = pd.read_csv(csv_path)
    connect_to_mt5(login, password, server)
    signal_results_dict = {}

    for ticker, entry_atr_factor, atr_factor, rrr in zip(results["ticker"].values, results["entry_atr_factor"].values,
                                                         results["atr_factor"].values, results["rrr"].values):
        print(ticker)
        # df = yfinance.download(ticker,
        #                        start=dt.datetime.now() - dt.timedelta(days=50),
        #                        end=dt.datetime.now(),
        #                        interval='15m')
        start, end = dt.datetime.now() - dt.timedelta(days=5), dt.datetime.now()
        data = get_mt5_data(mt5.TIMEFRAME_M15, ticker, start, end)

        """
        T = ticker; BS = buy_stop; SL = stop_loss; TP = take_profit; PS = position_size
        """
        enter_position, T, BS, SL, TP, PS = run_strategy(data,
                                                         ticker=ticker,
                                                         entry_atr_factor=entry_atr_factor,
                                                         atr_factor=atr_factor,
                                                         rrr=rrr)
        signal_results_dict[ticker] = {
            "enter_position": enter_position,
            "T": T,
            "BS": BS,
            "SL": SL,
            "TP": TP,
            "PS": PS
        }
    return signal_results_dict


if __name__ == "__main__":
    results = get_signals_for_tickers_15m(
        csv_path="../livetesting/crypto/crypto_strategy_params.csv",
        login=51958387,
        password="W3A&9BJbtX3&NX",
        server="ICMarketsEU-Demo")

    for ticker in results:
        for key, value in results[ticker].items():
            print(f"{key}: {value}")
