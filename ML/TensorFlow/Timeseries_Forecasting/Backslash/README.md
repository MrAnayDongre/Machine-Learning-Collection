# Testing Trading Strategies using backtrader

This repository contains the implementation of two trading strategies using the backtrader library. The strategies are tested on historical stock data of Apple from 1980 to 2023.

## backtrader

[backtrader](https://www.backtrader.com/) is a popular Python library for backtesting trading strategies. It provides a flexible framework to develop, test, and analyze trading algorithms using historical market data.

## Strategy 1: 

- Stock: Apple
- Timeline: 1980 to 2023
- Initial Stake: $10,000
- Final Value: $18,591

Strategy 1 is implemented in the `strategies.py` file. It is a simple strategy that aims to identify potential buying opportunities based on the closing prices of the stock.

## Strategy 2: Simple Moving Average (SMA)

- Stock: Apple
- Timeline: 1980 to 2023
- Initial Stake: $10,000
- Final Value: $25,725

Strategy 2 is implemented in the `strategy_sma.py` file. It is based on a Simple Moving Average (SMA) with a period of 20. The strategy filters trades based on the SMA indicator. Here's how the strategy works:

1. On each `next()` iteration, the strategy checks if there is a pending order. If so, it returns and does not take any action.
2. If there is no pending order, the strategy checks if it is currently not in the market (i.e., does not hold any position).
3. If not in the market, the strategy checks two conditions:
   - If the current closing price is greater than the current SMA value.
   - If the previous closing price is less than the previous SMA value.
4. If both conditions are met, the strategy generates a "BUY" order at the current closing price and logs the event.
5. If already in the market, the strategy checks two conditions:
   - If the current closing price is less than the current SMA value.
   - If the previous closing price is greater than the previous SMA value.
6. If both conditions are met, the strategy generates a "SELL" order at the current closing price and logs the event.

Strategy 2 was applied to the historical stock data of Apple from 1980 to 2023 with an initial stake of $10,000. The final value achieved with this strategy was $25,725.

## Usage

To run the strategies and reproduce the results, follow the instructions below:

1. Clone this repository to your local machine.
2. Install the required dependencies by running `pip install -r requirements.txt`.
3. Run the Python script(s) for each strategy, e.g., `python trader.py` and `python trader_sma.py`.
4. The scripts will execute the strategies on the provided historical data and display the final values achieved.

Feel free to modify the strategies, explore different parameters, or test them on other stocks or timeframes.

## Disclaimer

Please note that trading strategies, backtesting results, and historical data are for educational and informational purposes only. They do not constitute financial advice or guarantee future performance. Always do your own research and seek professional advice before making any investment decisions.



