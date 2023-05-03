import sys
sys.path.append("/Final")
import pandas as pd


import datetime as dt
from Components.Predictors.lstm_predictor import LSTMPredictor
from Components.Predictors.gru_predictor import GRUPredictor
from Components.stock_visualizer import StockVisualizer
from Components.Data.data_handler import DataHandler

import warnings
warnings.filterwarnings("ignore")
    
tickers = ["TSLA"]
start = dt.datetime(2021, 1, 1)
end = dt.datetime(2023, 5, 2)
n_splits = 2            # no larger than 6
days_to_predict = 7
sys.path.insert(0, 'Components')


if __name__ == "__main__":
    lstm_predictor = LSTMPredictor(tickers, start, end)
    true_values, predictions, mse_scores, mae_scores, r2_scores = lstm_predictor.train_and_evaluate(n_splits=n_splits)
    
    # Optimize the LSTM model
    best_hyperparameters = lstm_predictor.optimize_model((lstm_predictor.data.X, lstm_predictor.data.y), lstm_predictor.test_data)
    print("Best hyperparameters:", best_hyperparameters)

    # Make future predictions with the optimized model
    lstm_future_predictions = lstm_predictor.predict_future(days_to_predict=days_to_predict)
    
    # Create DataHandler instance
    data_handler = DataHandler(tickers=tickers, start_date=start, end_date=end)
    data_handler.preprocess_data(sequence_length=60)
    
    last_5_days = data_handler.stock_data.tail(5)
    
    # Create a DataFrame for the 7 predicted days
    last_date = pd.to_datetime(last_5_days.index[-1])
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=lstm_future_predictions.shape[0])

    future_predictions_df = pd.DataFrame(lstm_future_predictions, columns=last_5_days.columns, index=future_dates)
    for i in future_predictions_df.columns:
        future_predictions_df[i] = future_predictions_df[i].astype(float)
    # Combine the original data and the predictions
    combined_data = pd.concat([last_5_days, future_predictions_df])
    
    # Print the DataFrame with the last 5 days of the original data and the 7 predicted days
    print(combined_data)


    # stock_visualizer = StockVisualizer(
    #     original_data=data_handler.stock_data,
    #     true_values=true_values,
    #     predictions=predictions,
    #     future_predictions=lstm_future_predictions,
    #     dates=data_handler.dates,
    #     sequence_length=60
    # )
    # stock_visualizer.visualize_predictions()

    # Get the last 5 days of the original data

    # gru_predictor = GRUPredictor(tickers, start, end)
    # true_values, predictions, mse_scores, mae_scores, r2_scores = gru_predictor.train_and_evaluate(n_splits=n_splits)
    
    # gru_future_predictions = gru_predictor.predict_future(days_to_predict=days_to_predict)
    # gru_visualizer = StockVisualizer(gru_predictor.data.stock_data, true_values, predictions, gru_future_predictions)
    # gru_visualizer.visualize_predictions()
