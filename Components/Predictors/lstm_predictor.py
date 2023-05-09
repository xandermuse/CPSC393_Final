from skopt.utils import use_named_args
from skopt.space import Integer, Real, Categorical
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from skopt import gp_minimize
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

from Components.Models.lstm_model import LSTMModel
from Components.Data.data_handler import DataHandler

from tensorflow.keras.callbacks import EarlyStopping
from kerastuner.tuners import RandomSearch


class LSTMPredictor:
    """A class for training, evaluating and optimizing LSTM models for stock price prediction.
    """
    def __init__(self, tickers, start, end, model=None):
        """Initializes the LSTMPredictor with given stock tickers, date range, and an optional model.

        Args:
            tickers (list): List of stock tickers to be used for prediction.
            start (str): Start date for the stock data in the format 'YYYY-MM-DD'.
            end (str): End date for the stock data in the format 'YYYY-MM-DD'.
            model (LSTMModel, optional): Pre-trained LSTMModel instance. Defaults to None.
        """
        self.tickers = tickers
        self.start = start
        self.end = end
        self.data = DataHandler(tickers, start, end)
        self.model = model if model else LSTMModel()
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
        """Trains and evaluates the LSTM model using TimeSeriesSplit cross-validation.

        Args:
            n_splits (int, optional): Number of splits for time series cross-validation. Defaults to 5.
            sequence_length (int, optional): Length of the input sequence for the LSTM model. Defaults to 60.

        Returns:
            tuple: Tuple containing all_true_values, all_predictions, mse_scores, mae_scores, and r2_scores.
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

    def optimize_model_with_tuner(self, train_data, test_data=None, n_trials=100):
        """Optimizes the LSTM model using the Keras Tuner.

        Args:
            train_data (tuple): Tuple containing the training data (X_train, y_train).
            test_data (tuple, optional): Tuple containing the test data (X_test, y_test). Defaults to None.
            n_trials (int, optional): Number of trials for the tuner. Defaults to 10.
        """
        X_train, y_train = train_data
        X_test, y_test = test_data

        def build_model(hp):
            model = LSTMModel(
                units=hp.Int("units", 10, 200),
                num_layers=hp.Int("num_layers", 1, 5),
                dropout_rate=hp.Float("dropout_rate", 0.0, 0.9),
                learning_rate=hp.Float("learning_rate", 0.0001, 0.01, log=True),
                optimizer=hp.Choice("optimizer", ["adam", "rmsprop"]),
            )
            return model.model

        tuner = RandomSearch(
            build_model,
            objective="val_loss",
            max_trials=n_trials,
            executions_per_trial=1,
            directory="my_dir",
            project_name="lstm_hoard",
        )

        early_stopping = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)

        tuner.search(
            X_train,
            y_train,
            epochs=200,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
        )

        # Train the best model
        best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]
        best_hp = best_trial.hyperparameters
        best_model = build_model(best_hp)
        best_model.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test), callbacks=[early_stopping])

        # Save the best model
        best_model.save('model.h5')

        # Update the current model in the LSTMPredictor class
        self.model = LSTMModel(model=best_model)


    def optimize_model(self, train_data, test_data=None):
        """Optimizes the LSTM model using Bayesian optimization with Gaussian processes.

        Args:
            train_data (tuple): Tuple containing the training data (X_train, y_train).
            test_data (tuple, optional): Tuple containing the test data (X_test, y_test). Defaults to None.

        Returns:
            tuple: Tuple containing the best_hyperparameters and the best_model.
        """
        @use_named_args(self.search_space)
        def evaluate_model(**params):
            """Evaluates the LSTM model with given hyperparameters.

            Returns:
                float: Mean squared error for the model with given hyperparameters.
            """
            self.current_call += 1
            print(f"Current call: {self.current_call}")
            model = LSTMModel(**params)
            X_train, y_train = train_data
            X_test, y_test = test_data
            model.train(X_train, y_train, epochs=400, batch_size=32, validation_split=0.2, patience=20)
            mse = model.evaluate(X_test, y_test)
            return mse

        # Adjust n_calls to control the number of times the model will be run during optimization
        result = gp_minimize(evaluate_model, self.search_space, n_calls=10, random_state=0, verbose=False, n_jobs=-1)
        best_hyperparameters = result.x
        best_model = LSTMModel(**dict(zip([param.name for param in self.search_space], best_hyperparameters)))
        best_model.train(train_data[0], train_data[1], epochs=10, batch_size=32, validation_split=0.2, patience=5)

        # Update the current model in the LSTMPredictor class
        self.model = best_model

        return best_hyperparameters, best_model

    def predict_future(self, days_to_predict=7):
        """Predicts future stock prices for the given number of days.

        Args:
            days_to_predict (int, optional): Number of days to predict stock prices for. Defaults to 7.

        Returns:
            numpy.ndarray: Array containing the predicted stock prices for the given number of days.
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