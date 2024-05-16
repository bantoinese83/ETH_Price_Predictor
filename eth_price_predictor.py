import logging
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from loguru import logger
from rich.progress import track
from rich.table import Table
from rich.console import Console
from rich.markdown import Markdown

DEFAULT_TICKER = 'ETH-USD'
DEFAULT_PERIOD = '9y'


class PriceDataFetcher:
    def __init__(self, ticker=DEFAULT_TICKER, period=DEFAULT_PERIOD):
        self.ticker = ticker
        self.period = period
        self.yf_ticker = yf.Ticker(self.ticker)

    def fetch_price_data(self):
        try:
            data = self.yf_ticker.history(period=self.period)
            if data is None or data.empty:
                logging.error("Failed to fetch price data. Empty or None returned.")
                return None, None
            return data['Close'].values.reshape(-1, 1), data.index
        except Exception as e:
            logging.error(f"An error occurred while fetching data: {e}")
            return None, None


class EthPricePredictor:
    def __init__(self, model):
        self.model = model

    @staticmethod
    def prepare_price_data(price_data):
        if price_data is None or len(price_data) == 0:
            logging.error("Price data is empty or None.")
            return None, None
        data_X = np.arange(len(price_data)).reshape(-1, 1)
        data_y = price_data.ravel()  # reshape y to a 1D array
        return data_X, data_y

    def fit_model(self, training_data, training_labels):
        try:
            self.model.fit(training_data, training_labels)
        except Exception as e:
            logging.error(f"Failed to fit model: {e}")

    def predict_prices(self, testing_data):
        try:
            return self.model.predict(testing_data)
        except Exception as e:
            logging.error(f"Failed to predict prices: {e}")
            return None

    @staticmethod
    def evaluate_predictions(true_labels, predicted_labels):
        if true_labels is None or predicted_labels is None:
            logger.error("True or predicted labels are None.")
            return
        mean_squared_error_value = mean_squared_error(true_labels, predicted_labels)
        mean_absolute_error_value = mean_absolute_error(true_labels, predicted_labels)
        logger.info(f"Mean Squared Error: {mean_squared_error_value}, Mean Absolute Error: {mean_absolute_error_value}")
        logger.info(f"True Prices: {true_labels}")
        logger.info(f"Predicted Prices: {predicted_labels}")

    @staticmethod
    def visualize_predictions(true_labels, predicted_labels, prediction_dates):
        if true_labels is None or predicted_labels is None or prediction_dates is None:
            logging.error("True or predicted labels or dates are None.")
            return
        plt.figure(figsize=(10, 6))
        plt.plot(prediction_dates, true_labels, label='True Price', color='blue')
        plt.plot(prediction_dates, predicted_labels, label='Predicted Price', color='red')
        plt.title('Ethereum Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()
        logging.info(f"True Prices: {true_labels}")
        logging.info(f"Predicted Prices: {predicted_labels}")
        logging.info(f"Dates: {prediction_dates}")
        logging.info("Visualization completed.")


if __name__ == "__main__":
    fetcher = PriceDataFetcher()
    prices, dates = fetcher.fetch_price_data()

    results = []  # List to store the results

    if prices is not None and dates is not None:
        predictor = EthPricePredictor(RandomForestRegressor(n_estimators=100, random_state=42))
        X, y = predictor.prepare_price_data(prices)
        if X is not None and y is not None:
            tscv = TimeSeriesSplit(n_splits=5)
            for train_index, test_index in track(tscv.split(X), description="Processing..."):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                dates_train, dates_test = dates[train_index], dates[test_index]
                predictor.fit_model(X_train, y_train)
                y_pred = predictor.predict_prices(X_test)
                if y_pred is not None:
                    predictor.evaluate_predictions(y_test, y_pred)
                    predictor.visualize_predictions(y_test, y_pred, dates_test)
                    results.extend(zip(dates_test, y_test, y_pred))
                else:
                    logger.error("Failed to predict prices.")
        else:
            logger.error("Failed to prepare price data.")
    else:
        logger.error("Failed to fetch price data.")

    # Display a table with rich
    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Date")
    table.add_column("True Price")
    table.add_column("Predicted Price")
    for date, true, pred in results:
        table.add_row(str(date), str(true), str(pred))
    console.print(table)

    # Display a Markdown text with rich
    markdown = Markdown("# Ethereum Price Prediction")
    console.print(markdown)
