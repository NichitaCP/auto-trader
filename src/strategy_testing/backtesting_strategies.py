from backtesting import Backtest
from src.strategy_testing.strategy import KangarooTailStrategy
import warnings
import MetaTrader5 as mt5
from src.strategy_testing.optimizer_refactored import connect_to_mt5, get_mt5_data, \
     prepare_data_for_backtest_optimization



# warnings.filterwarnings("ignore", category=UserWarning)

login = 1510009878
password = "825$tnr$DJ"
server = "FTMO-Demo"

connect_to_mt5(login, password, server)
start_pos = 1
bar_count = 20000
data = get_mt5_data(mt5.TIMEFRAME_M15, "EURAUD", start_pos=start_pos, bar_count=bar_count)
data = prepare_data_for_backtest_optimization(data)


bt = Backtest(data,
              KangarooTailStrategy,
              cash=20000,
              commission=7e-5,
              margin=0.01,
              exclusive_orders=True,
              trade_on_close=True,
              hedging=False)
stats = bt.run()
print(stats)
