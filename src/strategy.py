from backtesting import Backtest, Strategy
from src.patterns import AverageTrueRange, AddSupportResistanceToData, TrendDetector, KangarooTailDetector
from src.patterns import detect_bullish_candle, detect_bearish_candle
import yfinance
import datetime as dt


class KTStrategy(Strategy):
    def init(self):
        pass

    def next(self):
        current_candle_index = len(self.data) - 1
        prev_candle_index = len(self.data) - 2

        O, H, L, C = (self.data.Open[current_candle_index],
                      self.data.High[current_candle_index],
                      self.data.Low[current_candle_index],
                      self.data.Close[current_candle_index])

        prev_O, prev_H, prev_L, prev_C = (self.data.Open[prev_candle_index],
                                          self.data.High[prev_candle_index],
                                          self.data.Low[prev_candle_index],
                                          self.data.Close[prev_candle_index])

        if self.data.green_kangaroo_tail[prev_candle_index] == 1 & self.data.prints_on_support[prev_candle_index]:
            if self.data.trend[prev_candle_index] == "Downtrend":
                if detect_bullish_candle(open_price=O, close_price=C, high_price=H):
                    buy_stop = H + 0.01
                    stop_loss = prev_L - self.data.atr[prev_candle_index] * 0.25
                    take_profit = buy_stop + 2.25 * (buy_stop - stop_loss)

                    self.buy(limit=float(buy_stop), sl=stop_loss, tp=take_profit)


def prepare_data_for_kt_strategy():
    now = dt.datetime.now()
    start = now - dt.timedelta(days=100)
    data = yfinance.download(tickers="EURUSD=X",
                             start=start,
                             end=now,
                             interval="1h")
    atr = AverageTrueRange(data)
    data = atr.get_atr(window=7)
    sr = AddSupportResistanceToData(data)
    sr.add_s_r_levels_to_data("Close", bw_method=0.05, prominence=0.1)
    kt_detector = KangarooTailDetector(data)
    data = kt_detector.identify_kangaroo_tails()
    data["prints_on_support"] = False
    data.loc[data["green_kangaroo_tail"] == 1, "prints_on_support"] = (
            (data["High"].shift(1) > data["next_support"]) &
            (data["Low"] > data["next_support"] - data["atr"]) &
            (data["Low"] < data["next_support"] + data["atr"]))
    TrendDetector(data).get_trend(ma_window=20, trend_candles_check=10)
    return data


df = prepare_data_for_kt_strategy()

bt = Backtest(df, KTStrategy, cash=100000, commission=.001)
results = bt.run()
print(results)
