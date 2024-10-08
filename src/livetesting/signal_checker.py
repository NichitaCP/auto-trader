import datetime as dt
from src.patterns import AverageTrueRange, TrendDetector, detect_next_candle_bullish
from src.patterns import detect_next_candle_bearish
import pandas as pd
import os
from src.trade_functions import connect_to_mt5, get_mt5_data
import MetaTrader5 as mt5


def get_position_size(account_size, risk_per_trade, stop_loss_distance, one_lot_value=100_000):
    size = (account_size * risk_per_trade) / stop_loss_distance
    size_in_lots = size / one_lot_value
    return size_in_lots


def get_position_size_test(account_size, risk_per_trade, stop_loss_distance, pip_value_per_lot, pip_size=0.0001):
    max_risk = account_size * risk_per_trade
    stop_loss_distance_pips = stop_loss_distance / pip_size
    position_size_lots = max_risk / (pip_value_per_lot * stop_loss_distance_pips)
    return position_size_lots


def prepare_data_for_signal(data):
    AverageTrueRange(data).get_atr(window=14)
    TrendDetector(data).get_trend(ma_window=50, trend_candles_check=7)
    data["next_bullish"] = detect_next_candle_bullish(data)
    data["next_bearish"] = detect_next_candle_bearish(data)
    data["next_high"] = data.High.shift(-1)
    data["next_low"] = data.Low.shift(-1)


def calculate_long_entry_prices(candle, entry_atr_factor, rrr, atr_factor, atr, round_factor=5):
    buy_stop = round(candle.High + atr * entry_atr_factor, round_factor)
    stop_loss = round(candle.Low - atr * atr_factor, round_factor)
    take_profit = round(buy_stop + (buy_stop - stop_loss) * rrr, round_factor)
    return buy_stop, stop_loss, take_profit


def calculate_short_entry_prices(candle, entry_atr_factor, rrr, atr_factor, atr, round_factor=5):
    sell_stop = round(candle.Low - atr * entry_atr_factor, round_factor)
    stop_loss = round(candle.High + atr * atr_factor, round_factor)
    take_profit = round(sell_stop - (stop_loss - sell_stop) * rrr, round_factor)

    return sell_stop, stop_loss, take_profit


def get_buy_signal(prev_candle):
    if (prev_candle["next_bullish"] == 1) & (prev_candle["trend"] != "Downtrend"):
        return True


def get_sell_signal(prev_candle):
    if (prev_candle["next_bearish"] == 1) & (prev_candle["trend"] != "Uptrend"):
        return True


def run_strategy(data, ticker, entry_atr_factor, rrr, atr_factor, account_size=20000,
                 risk_per_trade=0.01, round_factor=5, pip_value_per_lot=10, pip_size=0.0001):

    prepare_data_for_signal(data)
    prev_candle = data.iloc[-2]
    current_candle = data.iloc[-1]
    atr = current_candle["atr"]
    stop_price, stop_loss, take_profit, position_size, position_type = None, None, None, None, None
    open_order_now = False
    if get_buy_signal(prev_candle):
        stop_price, stop_loss, take_profit = calculate_long_entry_prices(current_candle,
                                                                         entry_atr_factor,
                                                                         rrr,
                                                                         atr_factor,
                                                                         atr,
                                                                         round_factor=round_factor)
        stop_loss_distance = stop_price - stop_loss
        position_size = get_position_size_test(account_size=account_size,
                                               risk_per_trade=risk_per_trade,
                                               stop_loss_distance=stop_loss_distance,
                                               pip_value_per_lot=pip_value_per_lot,
                                               pip_size=pip_size)
        print(f"Long signal detected for {ticker}\n{'#' * 50}")
        print(f"Buy stop: {stop_price}\nStop loss: {stop_loss}\nTake profit: {take_profit}")
        print(f"Position size: {position_size}")
        open_order_now = True
        position_type = "Long"

    elif get_sell_signal(prev_candle):
        stop_price, stop_loss, take_profit = calculate_short_entry_prices(current_candle,
                                                                          entry_atr_factor,
                                                                          rrr,
                                                                          atr_factor,
                                                                          atr,
                                                                          round_factor=round_factor)
        stop_loss_distance = stop_loss - stop_price
        position_size = get_position_size_test(account_size=account_size,
                                               risk_per_trade=risk_per_trade,
                                               stop_loss_distance=stop_loss_distance,
                                               pip_value_per_lot=pip_value_per_lot,
                                               pip_size=pip_size)

        print(f"Short signal detected for {ticker}\n{'#' * 50}")
        print(f"Sell stop: {stop_price}\nStop loss: {stop_loss}\nTake profit: {take_profit}")
        print(f"Position size: {position_size}")
        open_order_now = True
        position_type = "Short"
    else:
        print(f"No signal detected for {ticker}\n{'#' * 50}")

    return open_order_now, ticker, stop_price, stop_loss, take_profit, position_size, position_type


def get_signals_for_tickers(csv_path,
                            login,
                            password,
                            server,
                            timeframe,
                            start_pos=0,
                            bar_count=500,
                            account_size=20000,
                            risk_per_trade=0.01,
                            round_factor=5,
                            pip_value_per_lot=10,
                            pip_size=0.0001):
    csv_path = os.path.abspath(csv_path)
    results = pd.read_csv(csv_path)
    connect_to_mt5(login, password, server)
    signal_results_dict = {}

    for ticker, entry_atr_factor, atr_factor, rrr in zip(results["ticker"].values, results["entry_atr_factor"].values,
                                                         results["atr_factor"].values, results["rrr"].values):

        data = get_mt5_data(timeframe, ticker, start_pos, bar_count)

        """
        T = ticker; SP = stop_price; SL = stop_loss; TP = take_profit; PS = position_size, PT = position_type
        """
        enter_position, T, SP, SL, TP, PS, PT = run_strategy(data,
                                                             ticker=ticker,
                                                             entry_atr_factor=entry_atr_factor,
                                                             atr_factor=atr_factor,
                                                             rrr=rrr,
                                                             account_size=account_size,
                                                             risk_per_trade=risk_per_trade,
                                                             round_factor=round_factor,
                                                             pip_value_per_lot=pip_value_per_lot,
                                                             pip_size=pip_size)
        signal_results_dict[ticker] = {
            "enter_position": enter_position,
            "T": T,
            "SP": SP,
            "SL": SL,
            "TP": TP,
            "PS": PS,
            "PT": PT
        }
    return signal_results_dict


if __name__ == "__main__":
    results_ = get_signals_for_tickers(
        csv_path="../livetesting/FTMO/JPY.csv",
        login=1510009878,
        password="825$tnr$DJ",
        server="FTMO-Demo",
        timeframe=mt5.TIMEFRAME_M15,
        start_pos=0,
        account_size=20000,
        risk_per_trade=0.01,
        pip_value_per_lot=7,
        pip_size=0.01)

    for ticker_ in results_:
        for key, value in results_[ticker_].items():
            print(f"{key}: {value}")
