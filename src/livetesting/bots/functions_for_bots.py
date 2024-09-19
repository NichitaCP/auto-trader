from src.livetesting.open_order import request_open_order
import MetaTrader5 as mt5
import datetime as dt
from src.log_tool import log_trade


def check_if_orders_are_open(symbol):
    orders = mt5.orders_get(symbol=symbol)
    if orders:
        print(f"Orders not closed yet for {symbol}")
        return True
    else:
        print(f"No open orders for {symbol}")
        return False


def check_if_positions_are_open(symbol):
    positions = mt5.positions_get(symbol=symbol)
    if positions:
        print(f"Positions not closed yet for {symbol}")
        return True
    else:
        print(f"No open positions for {symbol}\n{'#' * 50}\n")
        return False


@log_trade
def open_long_trade_for_symbol(symbol, stop_price, stop_loss, take_profit, position_size, round_factor):
    expiration = mt5.symbol_info_tick(symbol).time + 45*60
    request_open_order(symbol=symbol,
                       lot_size=position_size,
                       order_type=mt5.ORDER_TYPE_BUY_STOP,
                       price=stop_price,
                       stop_loss=stop_loss,
                       take_profit=take_profit,
                       round_factor=round_factor,
                       expiration=expiration)


@log_trade
def open_short_trade_for_symbol(symbol, stop_price, stop_loss, take_profit, position_size, round_factor):
    expiration = mt5.symbol_info_tick(symbol).time + 45*60
    request_open_order(symbol=symbol,
                       lot_size=position_size,
                       order_type=mt5.ORDER_TYPE_SELL_STOP,
                       price=stop_price,
                       stop_loss=stop_loss,
                       take_profit=take_profit,
                       round_factor=round_factor,
                       expiration=expiration)
