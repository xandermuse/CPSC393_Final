from skopt.utils import use_named_args
from skopt.space import Integer, Real, Categorical
from sklearn.metrics import mean_squared_error
from skopt import gp_minimize
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

from Components.Models.lstm_model import LSTMModel
from Components.Data.data_handler import DataHandler

class LSTMPredictor:
    def __init__(self, tickers, start, end, model=None):
        self.tickers = tickers
        self.start = start
        self.end = end
        self.data = DataHandler(tickers, start, end)
        self.model = model if model else LSTMModel()
        
        self.search_space = [
            Integer(10, 200, name="units"),
            Real(0.0, 0.9, name="dropout_rate"),
            Categorical(['adam', 'rmsprop'], name="optimizer")
        ]

    def train_and_evaluate(self, n_splits=5, sequence_length=60):
        self.data.preprocess_data(sequence_length)
        X, y = self.data.X, self.data.y

        tscv = TimeSeriesSplit(n_splits=n_splits)

        mse_scores = []
        mae_scores = []
        r2_scores = []
        all_predictions = []
        all_true_values = []

        for train_index, test_index in tscv.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            self.model.train(X_train, y_train)
            y_pred = self.model.predict(X_test)

            # Reverse the normalization for predictions
            scaler = self.data.scaler
            y_pred_transformed = scaler.inverse_transform(y_pred)
            y_test_transformed = scaler.inverse_transform(y_test)

            # Calculate the scores
            mse = mean_squared_error(y_test_transformed, y_pred_transformed)
            mae = mean_absolute_error(y_test_transformed, y_pred_transformed)
            r2 = r2_score(y_test_transformed, y_pred_transformed)

            mse_scores.append(mse)
            mae_scores.append(mae)
            r2_scores.append(r2)

            all_predictions.append(y_pred_transformed)
            all_true_values.append(y_test_transformed)

        return all_true_values, all_predictions, mse_scores, mae_scores, r2_scores


    def optimize_model(self, train_data, test_data=None):
        @use_named_args(self.search_space)
        def evaluate_model(**params):
            model = self.model_class(**params)
            X_train, y_train = train_data
            X_test, y_test = test_data
            model.train(X_train, y_train, epochs=1, batch_size=32, validation_split=0.2, patience=5)
            mse = model.evaluate(X_test, y_test)
            return mse

        result = gp_minimize(evaluate_model, self.search_space, n_calls=10, random_state=0, verbose=False, n_jobs=-1)

        best_hyperparameters = result.x
        best_model = self.model_class(**dict(zip([param.name for param in self.search_space], best_hyperparameters)))
        best_model.train(train_data[0], train_data[1], epochs=1, batch_size=32, validation_split=0.2, patience=5)
        return best_hyperparameters  # Changed this line to return best_hyperparameters

    def make_predictions(model, X_test):
        predictions = model.predict(X_test)
        return predictions

    def predict_future(self, days_to_predict=30):
        future_predictions = []
        input_data = self.data.X[-1]  # Use the last sequence in the dataset as input
        
        for _ in range(days_to_predict):
            # Make a prediction for the next day
            prediction = self.model.predict(np.expand_dims(input_data, axis=0))
            
            # Add the prediction to the list of future predictions
            future_predictions.append(prediction[0])

            # Remove the oldest data point in the input_data and add the prediction
            input_data = np.concatenate((input_data[1:], prediction), axis=0)

        # Reverse the normalization for future predictions
        future_predictions_transformed = self.data.scaler.inverse_transform(future_predictions)
        return future_predictions_transformed
