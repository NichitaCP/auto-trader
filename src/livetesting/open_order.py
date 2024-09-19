import time
import datetime as dt
import MetaTrader5 as mt5


def request_open_order(symbol, **kwargs):

    mt5.initialize()
    if not mt5.symbol_select(symbol, True):
        print(f"Failed to select symbol {symbol}, error:", mt5.last_error())
        mt5.shutdown()
        quit()
    else:
        pass

    lot_size = kwargs.get("lot_size", 0.01)
    expiration = kwargs.get("expiration", 0)
    round_factor = kwargs.get("round_fractions", 5)
    order_type = kwargs.get("order_type", mt5.ORDER_TYPE_BUY_STOP)
    price = round(kwargs.get("price", 0.0000), round_factor)
    stop_loss = round(kwargs.get("stop_loss", 0.0), round_factor)
    take_profit = round(kwargs.get("take_profit", 0.0), round_factor)
    deviation = kwargs.get("deviation", 25)
    request = {
        "action": mt5.TRADE_ACTION_PENDING,
        "symbol": symbol,    #
        "volume": lot_size,  #
        "type": order_type,
        "price": price,  #
        "sl": stop_loss,  #
        "tp": take_profit,  #
        "deviation": deviation,
        "expiration": expiration,
        "magic": 25122017,
        "comment": "Opened trade with Python bot",
        "type_time": mt5.ORDER_TIME_SPECIFIED,
        "type_filling": mt5.ORDER_FILLING_RETURN,
    }
    result = mt5.order_send(request)
    if result is None:
        print(f"Order send failed, no result. Last error: {mt5.last_error()}")
    else:
        print(f"Order send response: {result}")
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Order send failed, retcode: {result.retcode}")
            print(f"Error description: {mt5.last_error()}")
        else:
            print("Order successfully sent!")
