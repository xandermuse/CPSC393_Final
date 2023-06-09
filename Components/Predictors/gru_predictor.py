from skopt.utils import use_named_args
from skopt.space import Integer, Real, Categorical
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from skopt import gp_minimize
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

from Components.Models.gru_model import GRUModel
from Components.Data.data_handler import DataHandler

class GRUPredictor:
    """A class for stock price prediction using Gated Recurrent Unit (GRU) models.
    """
    def __init__(self, tickers, start, end, model=None):
        """Initializes the GRUPredictor with the given tickers and date range.

        Args:
            tickers (list): List of stock tickers.
            start (str): Start date for the stock data in 'YYYY-MM-DD' format.
            end (str): End date for the stock data in 'YYYY-MM-DD' format.
            model (GRUModel, optional): Pre-trained GRUModel instance. Defaults to None.
        """
        self.tickers = tickers
        self.start = start
        self.end = end
        self.data = DataHandler(tickers, start, end)
        self.model = model if model else GRUModel()
        self.test_data = None
        self.current_call = 0

        self.search_space = [
            Integer(10, 200, name="units"),
            Integer(1, 5, name="num_layers"),
            Real(0.0, 0.9, name="dropout_rate"),
            Real(0.0001, 0.01, name="learning_rate"),
            Categorical(['adam', 'rmsprop'], name="optimizer")
        ]

    def train_and_evaluate(self, n_splits=5, sequence_length=60):
        """Trains and evaluates the model using time series cross-validation.

        Args:
            n_splits (int, optional): Number of splits for time series cross-validation. Defaults to 5.
            sequence_length (int, optional): Length of the input sequence. Defaults to 60.

        Returns:
            tuple: Tuple containing true_values, predictions, mse_scores, mae_scores, and r2_scores.
        """
        self.data.preprocess_data(sequence_length)
        X, y = self.data.X, self.data.y

        tscv = TimeSeriesSplit(n_splits=n_splits)

        mse_scores = []
        mae_scores = []
        r2_scores = []

        all_true_values = np.empty((0, 6))
        all_predictions = np.empty((0, 6))

        for train_index, test_index in tscv.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            self.model.train(X_train, y_train)
            y_pred = self.model.predict(X_test)

            scaler = self.data.scaler
            y_pred_transformed = scaler.inverse_transform(y_pred)
            y_test_transformed = scaler.inverse_transform(y_test)

            mse = mean_squared_error(y_test_transformed, y_pred_transformed)
            mae = mean_absolute_error(y_test_transformed, y_pred_transformed)
            r2 = r2_score(y_test_transformed, y_pred_transformed)

            mse_scores.append(mse)
            mae_scores.append(mae)
            r2_scores.append(r2)

            all_true_values = np.vstack((all_true_values, y_test_transformed))
            all_predictions = np.vstack((all_predictions, y_pred_transformed))

            self.test_data = (X_test, y_test)

        return all_true_values, all_predictions, mse_scores, mae_scores, r2_scores


    def optimize_model(self, train_data, test_data=None):
        """Optimizes the GRU model by searching for the best hyperparameters using Gaussian process optimization.

        Args:
            train_data (tuple): Tuple containing the training data (X_train, y_train).
            test_data (tuple, optional): Tuple containing the test data (X_test, y_test). Defaults to None.

        Returns:
            tuple: Tuple containing the best_hyperparameters and the best_model.
        """
        @use_named_args(self.search_space)
        def evaluate_model(**params):
            """Evaluates the GRU model with given hyperparameters.

            Returns:
                float: Mean squared error for the model with given hyperparameters.
            """
            self.current_call += 1
            print(f"Current call: {self.current_call}")
            model = GRUModel(**params)
            X_train, y_train = train_data
            X_test, y_test = test_data
            model.train(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2, patience=5)
            mse = model.evaluate(X_test, y_test)
            return mse

        # Adjust n_calls to control the number of times the model will be run during optimization
        result = gp_minimize(evaluate_model, self.search_space, n_calls=10, random_state=0, verbose=False, n_jobs=-1)
        best_hyperparameters = result.x
        best_model = GRUModel(**dict(zip([param.name for param in self.search_space], best_hyperparameters)))
        best_model.train(train_data[0], train_data[1], epochs=10, batch_size=32, validation_split=0.2, patience=5)

        # Update the current model in the LSTMPredictor class
        self.model = best_model

        return best_hyperparameters, best_model

    def predict_future(self, days_to_predict=7):
        """Predicts future stock prices for the given number of days using the trained GRU model.

        Args:
            days_to_predict (int, optional): The number of days to predict stock prices for. Defaults to 7.

        Returns:
            numpy.ndarray: Array of predicted stock prices for the given number of days.
        """
        X_test, _ = self.test_data  
        input_data = X_test[-1] 

        future_predictions = [] 

        for _ in range(days_to_predict):
            prediction = self.model.predict(np.expand_dims(input_data, axis=0))
            future_predictions.append(prediction[0])
            input_data = np.vstack((input_data[1:], prediction))

        future_predictions_transformed = self.data.scaler.inverse_transform(future_predictions)
        return future_predictions_transformed