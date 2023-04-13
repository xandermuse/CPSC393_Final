import datetime as dt
import numpy as np
from skopt.space import Real, Integer, Categorical
import json
from lstm_predictor import LSTMPredictor


today = dt.datetime.today()
two_years_ago = today - dt.timedelta(days=730)

search_space = [
    Integer(10, 200, name='units'),
    Real(0.1, 0.5, name='dropout_rate'),
    Categorical(['adam', 'rmsprop'], name='optimizer')
]

def save_data(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)

def load_data(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def save_best_params(best_params, filename):
    with open(filename, 'w') as f:
        json.dump(best_params, f, default=lambda x: x.item() if isinstance(x, np.generic) else x)

if __name__ == "__main__":
    today = dt.datetime.today()
    two_years_ago = today - dt.timedelta(days=730)

    lstm_predictor = LSTMPredictor(tickers=["TSLA"], start=two_years_ago, end=today)
    lstm_predictor.train_and_evaluate(n_splits=2)

