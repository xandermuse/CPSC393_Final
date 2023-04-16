import datetime as dt
import numpy as np
import pandas as pd
from .data_collector import DataCollector
from .Models.lstm_model import LSTMModel
from .Models.gru_model import GRUModel
from .Models.arima_model import ARIMAModel
from .data_handler import DataHandler
from .stock_visualizer import StockVisualizer
from skopt import gp_minimize
from skopt.utils import use_named_args
import skopt.space
from functools import partial


from sklearn.metrics import mean_squared_error
from skopt.space import Integer, Real, Categorical
import dill



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
        print(f"Data downloaded. Shape: {self.data.shape}")
        
        # Convert the 'Date' column to a datetime object and set it as the index
        self.data['Date'] = pd.to_datetime(self.data['Date'], unit='s')
        self.data.set_index('Date', inplace=True)

        # Print the first 3 rows of the DataFrame
        print(self.data.head(3))


    def time_series_split(self, n_splits):
        return self.data_handler.time_series_split(self.data, n_splits)


    def train_and_evaluate(self, n_splits=2):
        self.download_data()
        tscv = self.time_series_split(n_splits)
        best_hyperparameters = []

        for train_index, test_index in tscv.split(self.data):
            train_data, test_data = self.data.iloc[train_index], self.data.iloc[test_index]
            X_train, y_train = self.data_handler.create_sequences(train_data[:-60], 60)
            X_test, y_test = self.data_handler.create_sequences(test_data[:-60], 60)


            if isinstance(self, ARIMAPredictor):
                best_params = self.optimize_model((X_train, y_train))
            else:
                best_params = self.optimize_model((X_train, y_train), (X_test, y_test))

            best_hyperparameters.append(best_params)

        save_best_params(best_hyperparameters, f'{self.model_class.__name__}_best_hyperparameters.pickle')
        return best_hyperparameters



class ARIMAPredictor(BasePredictor):
    def __init__(self, tickers, start, end): 
        super().__init__(ARIMAModel, tickers, start, end)
        self.search_space = [
            Integer(0, 5, name="p"),
            Integer(0, 2, name="d"),
            Integer(0, 5, name="q"),
        ]

    def optimize_model(self, train_data):
        results_list = []

        @use_named_args(self.search_space)
        def partial_evaluate_model(p, d, q):
            p, d, q = int(p), int(d), int(q)
            arima = ARIMAModel(p=p, d=d, q=q)
            arima.train(train_data)
            predictions = arima.predict(train_data)
            return mean_squared_error(train_data, predictions)

        result = gp_minimize(partial_evaluate_model, self.search_space, n_calls=10, random_state=0, verbose=True, n_jobs=-1)

        # Sort the results list by MSE
        results_list.sort(key=lambda x: x[1])

        # Print the best parameters
        print("Best hyperparameters found: ", results_list[0][0])

        return result


def evaluate_model(model_class, train_data, test_data, **params):
    model = model_class(**params)
    X_train, y_train = train_data
    X_test, y_test = test_data

    model.train(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, patience=5)
    mse = model.evaluate(X_test, y_test)
    
    return mse


class GRUPredictor(BasePredictor):
    def __init__(self, tickers, start, end):
        super().__init__(GRUModel, tickers, start, end)
        self.search_space = [
            Integer(10, 200, name='units'),
            Real(0.1, 0.5, name='dropout_rate'),
            Categorical(['adam', 'rmsprop'], name='optimizer')
        ]

    def optimize_model(self, train_data, test_data=None):
        results_list = []

        named_evaluate_model = use_named_args(self.search_space)(lambda **params: evaluate_model(self.model_class, train_data, test_data, **params))

        result = gp_minimize(named_evaluate_model, self.search_space, n_calls=10, random_state=0, verbose=True, n_jobs=-1)
        results_list.append(result)
        
        return results_list

class LSTMPredictor(BasePredictor):
    def __init__(self, tickers, start, end):
        super().__init__(LSTMModel, tickers, start, end)
        self.search_space = [
            Integer(10, 200, name='units'),
            Real(0.1, 0.5, name='dropout_rate'),
            Categorical(['adam', 'rmsprop'], name='optimizer')
        ]

    def optimize_model(self, train_data, test_data=None):
        @use_named_args(self.search_space)
        def evaluate_model(**params):
            model = self.model_class(**params)
            X_train, y_train = train_data
            X_test, y_test = test_data

            print("X_train shape:", X_train.shape)
            print("y_train shape:", y_train.shape)

            model.train(X_train, y_train, epochs=1, batch_size=32, validation_split=0.2, patience=5)
            mse = model.evaluate(X_test, y_test)
            return mse

        result = gp_minimize(evaluate_model, self.search_space, n_calls=10, random_state=0, verbose=True, n_jobs=-1)

        best_hyperparameters = result.x
        return best_hyperparameters


def save_best_params(best_hyperparameters, filename):
    with open(filename, 'wb') as outfile:
        for params in best_hyperparameters:
            print(params)
            dill.dump(params, outfile)

def load_best_params(filename):
    with open(filename, 'rb') as infile:
        best_hyperparameters_list = dill.load(infile)

    # Find the best hyperparameters with the lowest MSE
    best_hyperparameters = min(best_hyperparameters_list, key=lambda x: x[1])
    return best_hyperparameters[0]



# if __name__ == '__main__':
#     tickers = 'AAPL'
#     start = '2010-01-01'
#     end = '2020-01-01'
#     n_splits = 2

# print("Training ARIMA model...")
# arima_predictor = ARIMAPredictor(tickers=tickers, start=start, end=end)
# arima_predictor.train_and_evaluate(n_splits=n_splits)