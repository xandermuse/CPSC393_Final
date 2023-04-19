from .base_predictor import BasePredictor
from skopt.utils import use_named_args
from skopt.space import Integer, Real, Categorical
from skopt import gp_minimize
from Components.Models.gru_model import GRUModel

class GRUPredictor(BasePredictor):
    def __init__(self, tickers, start, end):
        super().__init__(GRUModel, tickers, start, end)
        self.search_space = [
            Integer(10, 50, name='units'),
            Real(0.1, 0.5, name='dropout_rate'),
            Categorical(['adam'], name='optimizer')
        ]

    def optimize_model(self, train_data, test_data=None):
        results_list = []

        named_evaluate_model = use_named_args(self.search_space)(lambda **params: evaluate_model(self.model_class, train_data, test_data, **params))

        result = gp_minimize(named_evaluate_model, self.search_space, n_calls=10, random_state=0, verbose=True, n_jobs=-1)  # Reduced n_calls to 5
        results_list.append(result)
        
        return results_list
