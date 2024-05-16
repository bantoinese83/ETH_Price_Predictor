# Ethereum Price Predictor

This project uses a Random Forest Regressor model to predict the price of Ethereum.

## Features

- Fetches historical price data for Ethereum from Yahoo Finance.
- Trains a Random Forest Regressor model on the historical data.
- Uses the trained model to predict future prices.
- Visualizes the true and predicted prices on a plot.
- Evaluates the model's predictions using Mean Squared Error and Mean Absolute Error.
- Displays the true and predicted prices in a table.

## Usage

1. Ensure that you have the necessary Python packages installed. These include `yfinance`, `sklearn`, `matplotlib`, `numpy`, `loguru`, and `rich`.

2. Run the script with the command `python eth_price_predictor.py`.

## Customization

You can customize the ticker and period for fetching price data by modifying the `DEFAULT_TICKER` and `DEFAULT_PERIOD` variables in `eth_price_predictor.py`.

## Note

This project is for educational purposes and should not be used for financial advice.