import sys
import matplotlib
matplotlib.use('TkAgg')

# Add the path to the folder containing the predictor classes
sys.path.append("/Final")

import datetime as dt
from Components.Predictors.lstm_predictor import LSTMPredictor
from Components.Predictors.gru_predictor import GRUPredictor
from Components.stock_visualizer import StockVisualizer

import warnings
warnings.filterwarnings("ignore")

tickers = ["TSLA"]
start = dt.datetime(2021, 1, 1)
end = dt.datetime(2023, 1, 1)
n_splits = 2
days_to_predict = 7
sys.path.insert(0, 'Components')


if __name__ == "__main__":
    lstm_predictor = LSTMPredictor(tickers, start, end)
    true_values, predictions, mse_scores, mae_scores, r2_scores = lstm_predictor.train_and_evaluate(n_splits=n_splits)
    lstm_future_predictions = lstm_predictor.predict_future(days_to_predict=days_to_predict)
    visualizer = StockVisualizer(lstm_predictor.data.stock_data, true_values, predictions, lstm_future_predictions)
    visualizer.visualize_predictions()

    # gru_predictor = GRUPredictor(tickers, start, end)
    # true_values, predictions, mse_scores, mae_scores, r2_scores = gru_predictor.train_and_evaluate(n_splits=n_splits)
    
    # gru_future_predictions = gru_predictor.predict_future(days_to_predict=days_to_predict)
    # gru_visualizer = StockVisualizer(gru_predictor.data.stock_data, true_values, predictions, gru_future_predictions)
    # gru_visualizer.visualize_predictions()
