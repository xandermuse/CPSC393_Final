import sys
import pandas as pd
import datetime as dt
from Components.Predictors.lstm_predictor import LSTMPredictor
from Components.stock_visualizer import StockVisualizer
from Components.Data.data_handler import DataHandler

import warnings
warnings.filterwarnings("ignore")

tickers = ["TSLA"]
start = dt.datetime(2021, 1, 1)
end = dt.datetime(2023, 5, 7)
n_splits = 2  # no larger than 6
days_to_predict = 7
sys.path.insert(0, 'Components')

if __name__ == "__main__":
    # Create DataHandler instance
    data_handler = DataHandler(tickers=tickers, start_date=start, end_date=end)
    data_handler.preprocess_data(sequence_length=60)
    
    '''Updated'''
    # Create LSTMPredictor instance
    lstm_predictor = LSTMPredictor(tickers, start, end)
    true_values, predictions, mse_scores, mae_scores, r2_scores = lstm_predictor.train_and_evaluate(n_splits=n_splits)
    print("here\n\n\n\n\n\n\n\n\n\n")
    # Optimize the LSTM model
    lstm_predictor.optimize_model_with_tuner((lstm_predictor.data.X, lstm_predictor.data.y), lstm_predictor.test_data, n_trials=100)
    lstm_future_predictions = lstm_predictor.predict_future(days_to_predict=days_to_predict)
    
    combined_data_lstm = data_handler.create_future_dataframe(lstm_predictor, days_to_predict=days_to_predict)
    print("LSTM Combined Data:")
    print(combined_data_lstm)
