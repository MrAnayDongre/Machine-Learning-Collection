import backtrader as bt
import datetime

class TestStrategy(bt.Strategy):
    params = (
        ('sma_period', 20),   # Period for the simple moving average
    )

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None
        self.sma = bt.indicators.SimpleMovingAverage(self.datas[0], period=self.params.sma_period)

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status == order.Completed:
            if order.isbuy():
                self.log('BUY EXECUTED, Price: %.2f' % order.executed.price)
            elif order.issell():
                self.log('SELL EXECUTED, Price: %.2f' % order.executed.price)

        self.order = None

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def next(self):
        if self.order:
            return

        if not self.position:
            if self.dataclose[0] > self.sma[0] and self.dataclose[-1] < self.sma[-1]:
                self.log('BUY CREATE, Price: %.2f' % self.dataclose[0])
                self.order = self.buy()
        else:
            if self.dataclose[0] < self.sma[0] and self.dataclose[-1] > self.sma[-1]:
                self.log('SELL CREATE, Price: %.2f' % self.dataclose[0])
                self.order = self.sell()

