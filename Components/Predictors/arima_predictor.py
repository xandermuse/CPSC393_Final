from .base_predictor import BasePredictor
from skopt.utils import use_named_args
from skopt.space import Integer
from Models.arima_model import ARIMAModel


class ARIMAPredictor(BasePredictor):
    def __init__(self, tickers, start, end): 
        super().__init__(ARIMAModel, tickers, start, end)
        self.search_space = [
            Integer(0, 5, name="p"),
            Integer(0, 2, name="d"),
            Integer(0, 5, name="q"),
        ]
        self.models = {feature: ARIMAModel() for feature in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']}

    def train_all_models(self, train_data):
        for feature in self.models:
            print(f"Training ARIMA model for {feature}...")
            self.models[feature].train(train_data, feature)

    def predict_all_models(self, steps):
        predictions = {}
        for feature in self.models:
            print(f"Predicting {feature} with ARIMA model...")
            predictions[feature] = self.models[feature].predict(steps)
        return predictions

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
