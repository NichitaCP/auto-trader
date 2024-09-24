from src.livetesting.signal_checker import get_signals_for_tickers
import MetaTrader5 as mt5
from schedule import repeat, every
import schedule
import time as tm
from src.livetesting.bots.functions_for_bots import (check_if_positions_are_open, check_if_orders_are_open,
                                                     open_long_trade_for_symbol, open_short_trade_for_symbol)


def run_bot(params_csv_path, timeframe, start_pos, risk_per_trade,
            account_size, round_factor, pip_value_per_lot, pip_size):
    results = get_signals_for_tickers(
            csv_path=params_csv_path,
            login=1510009878,
            password="825$tnr$DJ",
            server="FTMO-Demo",
            timeframe=timeframe,
            start_pos=start_pos,
            risk_per_trade=risk_per_trade,
            account_size=account_size,
            round_factor=round_factor,
            pip_value_per_lot=pip_value_per_lot,
            pip_size=pip_size)

    for ticker in results:
        open_orders = check_if_orders_are_open(symbol=ticker)
        open_positions = check_if_positions_are_open(symbol=ticker)
        enter_position = results[ticker]["enter_position"]
        entry_price = results[ticker]["SP"]
        stop_loss_price = results[ticker]["SL"]
        take_profit_price = results[ticker]["TP"]
        pos_size = max(results[ticker]["PS"], 0.05) if results[ticker]["PS"] else 0.0
        pos_type = results[ticker]["PT"]

        if not open_orders and not open_positions:
            if enter_position:
                if pos_type == "Long":
                    open_long_trade_for_symbol(symbol=ticker,
                                               stop_price=entry_price,
                                               stop_loss=stop_loss_price,
                                               take_profit=take_profit_price,
                                               position_size=round(pos_size, 2),
                                               round_factor=round_factor)
                elif pos_type == "Short":
                    open_short_trade_for_symbol(symbol=ticker,
                                                stop_price=entry_price,
                                                stop_loss=stop_loss_price,
                                                take_profit=take_profit_price,
                                                position_size=round(pos_size, 2),
                                                round_factor=round_factor)


@repeat(every(15).minutes)
def main():
    eur_fx_path = "../FTMO/EURUSD.csv"
    xag_path = "../FTMO/XAGUSD.csv"
    xau_path = "../FTMO/XAUUSD.csv"
    jpy_path = "../FTMO/JPY.csv"
    run_bot(params_csv_path=eur_fx_path,
            timeframe=mt5.TIMEFRAME_M15,
            start_pos=1,
            risk_per_trade=0.0075,
            account_size=20000,
            round_factor=5,
            pip_value_per_lot=9.5,
            pip_size=0.0001)

    run_bot(params_csv_path=xag_path,
            timeframe=mt5.TIMEFRAME_M15,
            start_pos=1,
            risk_per_trade=0.0075,
            account_size=20000,
            round_factor=3,
            pip_value_per_lot=50,
            pip_size=0.01)

    run_bot(params_csv_path=jpy_path,
            timeframe=mt5.TIMEFRAME_M15,
            start_pos=1,
            risk_per_trade=0.0075,
            account_size=20000,
            round_factor=3,
            pip_value_per_lot=7,
            pip_size=0.01)


if __name__ == "__main__":
    main()
    while True:
        schedule.run_pending()
        tm.sleep(1)
