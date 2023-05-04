from .base_predictor import BasePredictor
from skopt.utils import use_named_args
from skopt.space import Integer
from Models.arima_model import ARIMAModel
from sklearn.metrics import mean_squared_error
# bayesian optimization
from skopt import gp_minimize


class ARIMAPredictor(BasePredictor):
    """A stock price predictor based on the ARIMA model.

    Args:
        BasePredictor (BasePredictor): Base class for predictors.
    """
    def __init__(self, tickers, start, end): 
        """Initializes the ARIMAPredictor with given tickers and date range.

        Args:
            tickers (list): List of stock tickers.
            start (str): Start date for the stock data in 'YYYY-MM-DD' format.
            end (str): End date for the stock data in 'YYYY-MM-DD' format.
        """
        super().__init__(ARIMAModel, tickers, start, end)
        self.search_space = [
            Integer(0, 5, name="p"),
            Integer(0, 2, name="d"),
            Integer(0, 5, name="q"),
        ]
        self.models = {feature: ARIMAModel() for feature in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']}

    def train_all_models(self, train_data):
        """Trains ARIMA models for all features.

        Args:
            train_data (pd.DataFrame): Training data for the ARIMA models.
        """
        for feature in self.models:
            print(f"Training ARIMA model for {feature}...")
            self.models[feature].train(train_data, feature)

    def predict_all_models(self, steps):
        """Predicts future values for all features using the ARIMA models.

        Args:
            steps (int): Number of time steps to predict.

        Returns:
            dict: Dictionary containing predicted values for each feature.
        """
        predictions = {}
        for feature in self.models:
            print(f"Predicting {feature} with ARIMA model...")
            predictions[feature] = self.models[feature].predict(steps)
        return predictions

    def optimize_model(self, train_data):
        """Optimizes the ARIMA model using Bayesian optimization.

        Args:
            train_data (pd.DataFrame): Training data for optimization.

        Returns:
            skopt.OptimizeResult: Optimization result object.
        """
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
