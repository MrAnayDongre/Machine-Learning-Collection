import backtrader
import datetime
from strategies import TestStrategy
import matplotlib

import warnings

cerebro = backtrader.Cerebro()

cerebro.broker.set_cash(10000)

# Create a Data Feed
data = backtrader.feeds.YahooFinanceCSVData(
        dataname='aapl.csv',
        # Do not pass values before this date
        fromdate=datetime.datetime(1981, 1, 1),
        # Do not pass values after this date
        todate=datetime.datetime(2021, 12, 31),
        reverse=False)

cerebro.adddata(data)

cerebro.addstrategy(TestStrategy)

cerebro.addsizer(backtrader.sizers.FixedSize, stake=100)

print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

cerebro.run()

print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

cerebro.plot()