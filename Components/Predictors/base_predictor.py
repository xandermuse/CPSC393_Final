import pandas as pd
import numpy as np
from skopt import gp_minimize
from skopt.utils import use_named_args
import skopt.space
from functools import partial
from sklearn.metrics import mean_squared_error
from skopt.space import Integer, Real, Categorical
from sklearn.model_selection import TimeSeriesSplit
import dill
import os
# MSE, MAE, R2
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from Components.Models.lstm_model import LSTMModel
from Components.Models.gru_model import GRUModel
from Components.Models.arima_model import ARIMAModel
from Components.Models.transformer_model import TransformerModel
from Components.Data.data_collector import DataCollector
from Components.Data.data_handler import DataHandler

from typing import TYPE_CHECKING

class BasePredictor:
    """Base class for stock price predictors.
    """
    def __init__(self, model_class, tickers, start_date, end_date):
        """Initializes the BasePredictor with the given model class, tickers, and date range.

        Args:
            model_class (object): Model class to be used for prediction.
            tickers (list): List of stock tickers.
            start_date (str): Start date for the stock data in 'YYYY-MM-DD' format.
            end_date (str): End date for the stock data in 'YYYY-MM-DD' format.
        """
        self.model_class = model_class
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.model_class = model_class
        self.data_handler = DataHandler(tickers, start_date, end_date)

    def train_and_evaluate(self, n_splits=2):
        """Trains and evaluates the model using time series cross-validation.

        Args:
            n_splits (int, optional): Number of splits for time series cross-validation. Defaults to 2.

        Returns:
            tuple: Tuple containing true_values, predictions, mse_scores, mae_scores, and r2_scores.
        """
        self.data_handler.preprocess_data()
        tscv = TimeSeriesSplit(n_splits=n_splits)

        true_values = []
        predictions = []
        train_losses = []
        val_losses = []
        mse_scores = []
        mae_scores = []
        r2_scores = []

        for train_index, test_index in tscv.split(np.arange(len(self.data_handler.X))):
            X_train, X_test = self.data_handler.X[train_index], self.data_handler.X[test_index]
            y_train, y_test = self.data_handler.y[train_index], self.data_handler.y[test_index]

            self.model = self.model_class()

            self.model.train(X_train, y_train)
            y_pred = self.model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            mse_scores.append(mse)
            mae_scores.append(mae)
            r2_scores.append(r2)

            true_values.extend(y_test)
            predictions.extend(y_pred)

            train_losses.append(self.model.get_train_loss())
            val_losses.append(self.model.get_val_loss())

        return true_values, predictions, mse_scores, mae_scores, r2_scores




    def save_best_params(self, best_hyperparameters, file_name):
        """Saves the best hyperparameters to a file.

        Args:
            best_hyperparameters (dict): Dictionary containing the best hyperparameters.
            file_name (str): File name to save the best hyperparameters.
        """
        with open(file_name, 'wb') as f:
            dill.dump(best_hyperparameters, f)

    def load_best_params(self, file_name):
        """Loads the best hyperparameters from a file.

        Args:
            file_name (str): File name to load the best hyperparameters from.

        Returns:
            dict: Dictionary containing the best hyperparameters.
        """
        with open(file_name, 'rb') as f:
            best_hyperparameters = dill.load(f)
        return best_hyperparameters

def save_best_params(best_hyperparameters, file_name):
    """Saves the best hyperparameters to a file.

    Args:
        best_hyperparameters (dict): Dictionary containing the best hyperparameters.
        file_name (str): File name to save the best hyperparameters.
    """
    with open(file_name, 'wb') as f:
        dill.dump(best_hyperparameters, f)


