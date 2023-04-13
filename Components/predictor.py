import datetime as dt
import numpy as np
import pandas as pd
from data_collector import DataCollector
from Models.lstm_model import LSTMModel
from Models.gru_model import GRUModel
from Models.arima_model import ARIMAModel
from data_handler import DataHandler
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from stock_visualizer import StockVisualizer
import json


class BasePredictor:
    def __init__(self, model_class, tickers, start, end):
        self.model_class = model_class
        self.tickers = tickers
        self.start = start
        self.end = end
        self.data_collector = DataCollector()
        self.data_handler = DataHandler(self.data_collector)

    def download_data(self):
        print("Downloading data...")
        self.data = self.data_collector.get_data(tickers=self.tickers, start=self.start, end=self.end)
        print(self.data.shape)

    def time_series_split(self, n_splits):
        return self.data_handler.time_series_split(self.data, n_splits)

    def train_and_evaluate(self, n_splits=2):
        self.download_data()
        tscv = self.time_series_split(n_splits)
        best_hyperparameters = []

        for train_index, test_index in tscv.split(self.data):
            train_data, test_data = self.data.iloc[train_index], self.data.iloc[test_index]
            X_train, y_train = self.data_handler.create_sequences(train_data, 60)
            X_test, y_test = self.data_handler.create_sequences(test_data, 60)

            result = self.optimize_model((X_train, y_train)) if isinstance(self, ARIMAPredictor) else self.optimize_model((X_train, y_train), (X_test, y_test))

            best_params = result.x
            print(f"Best hyperparameters found for this split: {best_params}")

            best_hyperparameters.append(best_params)

        save_best_params(best_hyperparameters, f'{self.model_class.__name__}_best_hyperparameters.json')
        result = self.optimize_model((X_train, y_train)) if isinstance(self, ARIMAPredictor) else self.optimize_model((X_train, y_train), (X_test, y_test))
        print(f"Best hyperparameters found: {result.x}")

        model_instance = self.model_class(input_shape=(60, 7), **result.x)
        model_instance.train(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, patience=5)

        predictions = model_instance.predict(X_test)

        stock_visualizer = StockVisualizer(test_data, y_test, predictions)
        stock_visualizer.visualize_predictions()


class LSTMPredictor(BasePredictor):
    def __init__(self, tickers, start, end):
        super().__init__(LSTMModel, tickers, start, end)
        self.search_space = [
            Integer(10, 200, name='units'),
            Real(0.1, 0.5, name='dropout_rate'),
            Categorical(['adam', 'rmsprop'], name='optimizer')
        ]

    def optimize_model(self, train_data, test_data):
        results_list = []

        @use_named_args(self.search_space)
        def evaluate_model(**params):
            model_instance = self.model_class(input_shape=(60, 7), **params)

            X_train, y_train = train_data
            X_test, y_test = test_data

            model_instance.train(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, patience=5)

            predictions = model_instance.predict(X_test)
            mse = np.mean((predictions - y_test) ** 2)
            results_list.append((params, mse))
            return mse

        result = gp_minimize(evaluate_model, self.search_space, n_calls=10, random_state=0, verbose=True, n_jobs=-1)

        # Sort the results list by MSE
        results_list.sort(key=lambda x: x[1])

        # Print the best parameters
        print("Best hyperparameters found: ", results_list[0][0])

        return result

class ARIMAPredictor(BasePredictor):
    def __init__(self, tickers, start, end):
        super().__init__(ARIMAModel, tickers, start, end)
        self.search_space = [
            Integer(0, 5, name='p'),
            Integer(0, 5, name='q')
        ]

    def optimize_model(self, train_data):
        results_list = []

        @use_named_args(self.search_space)
        def evaluate_model(p, q):
            order = (p, 1, q)
            arima_model = ARIMAModel(order=order)
            arima_model.train(train_data)

            mse = arima_model.model.mse
            results_list.append((order, mse))
            return mse

        result = gp_minimize(evaluate_model, self.search_space, n_calls=10, random_state=0, verbose=True, n_jobs=-1)

        # Sort the results list by MSE
        results_list.sort(key=lambda x: x[1])

        # Print the best parameters
        print("Best hyperparameters found: ", results_list[0][0])

        return result

class GRUPredictor(BasePredictor):
    def __init__(self, tickers, start, end):
        super().__init__(GRUModel, tickers, start, end)
        self.search_space = [
            Integer(10, 200, name='units'),
            Real(0.1, 0.5, name='dropout_rate'),
            Categorical(['adam', 'rmsprop'], name='optimizer')
        ]

    def optimize_model(self, train_data, test_data):
        results_list = []

        @use_named_args(self.search_space)  # Replace search_space with self.search_space
        def evaluate_model(**params):
            gru_model = GRUModel(input_shape=(60, 7), **params)

            X_train, y_train = train_data
            X_test, y_test = test_data

            gru_model.train(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, patience=5)

            predictions = gru_model.predict(X_test)
            mse = np.mean((predictions - y_test) ** 2)
            results_list.append((params, mse))
            return mse

        result = gp_minimize(evaluate_model, self.search_space, n_calls=10, random_state=0, verbose=True, n_jobs=-1)

        # Sort the results list by MSE
        results_list.sort(key=lambda x: x[1])

        # Print the best parameters
        print("Best hyperparameters found: ", results_list[0][0])

        return result


def save_best_params(best_hyperparameters, filename):
    pass
    with open(filename, 'w') as outfile:
        json.dump(best_hyperparameters, outfile)


if __name__ == "__main__":
    tickers = ["TSLA"]
    start = dt.datetime(2021, 1, 1)
    end = dt.datetime(2023, 1, 1)
    n_splits = 2

    print("Training LSTM model...")
    # lstm_predictor = LSTMPredictor(tickers, start, end)
    # lstm_predictor.train_and_evaluate(n_splits=n_splits)
    
    print("Training GRU model...")
    # gru_predictor = GRUPredictor(tickers=tickers, start=start, end=end)
    # gru_predictor.train_and_evaluate(n_splits=n_splits)

    print("Training ARIMA model...")
    arima_predictor = ARIMAPredictor(tickers=tickers, start=start, end=end)
    arima_predictor.train_and_evaluate(n_splits=n_splits)
