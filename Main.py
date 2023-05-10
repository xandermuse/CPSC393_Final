import sys
import pandas as pd
import datetime as dt
from Components.Predictors.lstm_predictor import LSTMPredictor
from Components.Predictors.gru_predictor import GRUPredictor
from Components.Models.prophet_model import ProphetModel
from Components.stock_visualizer import StockVisualizer
from Components.Data.data_handler import DataHandler

import warnings
warnings.filterwarnings("ignore")

tickers = ["TSLA"]
start = dt.datetime(2021, 1, 1)
end = dt.datetime(2023, 5, 9)
n_splits = 2  # no larger than 6
days_to_predict = 7
sys.path.insert(0, 'Components')

if __name__ == "__main__":
    # Create DataHandler instance
    data_handler = DataHandler(tickers=tickers, start_date=start, end_date=end)
    data_handler.preprocess_data(sequence_length=60)

    # Create ProphetModel instance
    prophet_model = ProphetModel(tickers, start, end)
    # Assuming data_handler.stock_data has the columns you want to predict
    prophet_model.train(data_handler.stock_data, target_cols=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])

    # Prepare the data for Prophet
    prophet_data = data_handler.stock_data[['Close']].reset_index()
    prophet_data.columns = ['ds', 'y']
    prophet_data['ds'] = pd.to_datetime(prophet_data['ds'])  # Convert 'ds' column to datetime objects

    # Define the target columns
    target_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

    # Optimize hyperparameters for each target column
    for col in target_cols:
        # Prepare the data for Prophet
        df = data_handler.stock_data[[col]].reset_index()
        df.columns = ['ds', 'y']
        df['ds'] = pd.to_datetime(df['ds'])  # Convert 'ds' column to datetime objects

        # Optimize hyperparameters and train the model with the best parameters
        best_params, best_model = prophet_model.optimize_hyperparameters(df, col)
        print(f"Best parameters for {col} (additive):", best_params['additive'])
        print(f"Best parameters for {col} (multiplicative):", best_params['multiplicative'])
        prophet_model.best_params[col] = best_params

    # Create the future dataframe with predictions for the specified target columns
    prophet_future_df = prophet_model.create_future_dataframe(data_handler.stock_data, future_periods=days_to_predict, target_cols=target_cols)

    # Print the future dataframe with predictions
    print("\nProphet future predictions:")
    print(prophet_future_df)


    # # Create GRUPredictor instance
    gru_predictor = GRUPredictor(tickers, start, end)
    true_values_gru, predictions_gru, mse_scores_gru, mae_scores_gru, r2_scores_gru = gru_predictor.train_and_evaluate(n_splits=n_splits)

    # Optimize the GRU model
    best_hyperparameters_gru, best_model_gru = gru_predictor.optimize_model((gru_predictor.data.X, gru_predictor.data.y), gru_predictor.test_data)
    gru_future_predictions = gru_predictor.predict_future(days_to_predict=days_to_predict)

    # Make future predictions with the optimized GRU model and create a combined DataFrame
    combined_data_gru = data_handler.create_future_dataframe(gru_predictor, days_to_predict=days_to_predict)

    # Create LSTMPredictor instance
    lstm_predictor = LSTMPredictor(tickers, start, end)
    true_values, predictions, mse_scores, mae_scores, r2_scores = lstm_predictor.train_and_evaluate(n_splits=n_splits)
    
    # Optimize the LSTM model
    best_hyperparameters = lstm_predictor.optimize_model((lstm_predictor.data.X, lstm_predictor.data.y), lstm_predictor.test_data)
    lstm_future_predictions = lstm_predictor.predict_future(days_to_predict=days_to_predict)

    print("GRU Combined Data:")
    print(combined_data_gru)

    combined_data_lstm = data_handler.create_future_dataframe(lstm_predictor, days_to_predict=days_to_predict)
    print("LSTM Combined Data:")
    print(combined_data_lstm)
    
    # Extracting relevant predictions from the different models
    lstm_open, lstm_high, lstm_low, lstm_close, lstm_adj, lstm_vol = lstm_future_predictions.T
    gru_open, gru_high, gru_low, gru_close, gru_adj, gru_vol = gru_future_predictions.T
    prophet_open = prophet_future_df['Open']
    prophet_high = prophet_future_df['High']
    prophet_low = prophet_future_df['Low']
    prophet_close = prophet_future_df['Close']
    prophet_adj = prophet_future_df['Adj Close']
    prophet_vol = prophet_future_df['Volume']

    # prophet_open, prophet_high, prophet_low, prophet_close, prophet_adj, prophet_vol = prophet_future_df.T
    print("Prophet Open:")
    print(prophet_open)

    

    # Create the DataFrame with the given structure
    predictions_df = pd.DataFrame({
        'lstm_close': lstm_close, 'gru_close': gru_close, 'prophet_close': prophet_close, 
        'lstm_open': lstm_open, 'gru_open': gru_open, 'prophet_open': prophet_open, 
        'lstm_high': lstm_high, 'gru_high': gru_high, 'prophet_high': prophet_high, 
        'lstm_low': lstm_low, 'gru_low': gru_low, 'prophet_low': prophet_low, 
        'lstm_adj': lstm_adj, 'gru_adj': gru_adj, 'prophet_adj': prophet_adj, 
        'lstm_vol': lstm_vol, 'gru_vol': gru_vol, 'prophet_vol': prophet_vol
    }, index=combined_data_gru.index[-days_to_predict:])

    # Calculate the mean for each row
    predictions_df['mean_close'] = predictions_df[['lstm_close', 'gru_close', 'prophet_close']].mean(axis=1)
    predictions_df['mean_open'] = predictions_df[['lstm_open', 'gru_open', 'prophet_open']].mean(axis=1)
    predictions_df['mean_high'] = predictions_df[['lstm_high', 'gru_high', 'prophet_high']].mean(axis=1)
    predictions_df['mean_low'] = predictions_df[['lstm_low', 'gru_low', 'prophet_low']].mean(axis=1)
    predictions_df['mean_adj'] = predictions_df[['lstm_adj', 'gru_adj', 'prophet_adj']].mean(axis=1)
    predictions_df['mean_vol'] = predictions_df[['lstm_vol', 'gru_vol', 'prophet_vol']].mean(axis=1)

    print("Predictions DataFrame:")
    print(predictions_df)

    
    close_df = pd.DataFrame({
        'lstm_close': lstm_close, 'gru_close': gru_close, 'prophet_close': prophet_close,
        'mean_close': predictions_df['mean_close']
    }, index=combined_data_gru.index[-days_to_predict:])
    
    open_df = pd.DataFrame({
        'lstm_open': lstm_open, 'gru_open': gru_open, 'prophet_open': prophet_open,
        'mean_open': predictions_df['mean_open']
    }, index=combined_data_gru.index[-days_to_predict:])

    high_df = pd.DataFrame({
        'lstm_high': lstm_high, 'gru_high': gru_high, 'prophet_high': prophet_high,
        'mean_high': predictions_df['mean_high']
    }, index=combined_data_gru.index[-days_to_predict:])

    low_df = pd.DataFrame({
        'lstm_low': lstm_low, 'gru_low': gru_low, 'prophet_low': prophet_low,
        'mean_low': predictions_df['mean_low']
    }, index=combined_data_gru.index[-days_to_predict:])

    adj_df = pd.DataFrame({
        'lstm_adj': lstm_adj, 'gru_adj': gru_adj, 'prophet_adj': prophet_adj,
        'mean_adj': predictions_df['mean_adj']
    }, index=combined_data_gru.index[-days_to_predict:])

    vol_df = pd.DataFrame({
        'lstm_vol': lstm_vol, 'gru_vol': gru_vol, 'prophet_vol': prophet_vol,
        'mean_vol': predictions_df['mean_vol']
    }, index=combined_data_gru.index[-days_to_predict:])

    print("Close DataFrame:")
    print(close_df)
    print("Open DataFrame:")
    print(open_df)
    print("High DataFrame:")
    print(high_df)
    print("Low DataFrame:")
    print(low_df)
    print("Adj DataFrame:")
    print(adj_df)
    print("Vol DataFrame:")
    print(vol_df)
    # save the dataframe to csv use {ticker}_close.csv
    close_df.to_csv(f'{tickers[0]}_close.csv')
    open_df.to_csv(f'{tickers[0]}_open.csv')
    high_df.to_csv(f'{tickers[0]}_high.csv')
    low_df.to_csv(f'{tickers[0]}_low.csv')
    adj_df.to_csv(f'{tickers[0]}_adj.csv')
    vol_df.to_csv(f'{tickers[0]}_vol.csv')
    
    # dataframe for mean predictions
    mean_df = pd.DataFrame({
            'mean_close': predictions_df['mean_close'],
            'mean_open': predictions_df['mean_open'],
            'mean_high': predictions_df['mean_high'],
            'mean_low': predictions_df['mean_low'],
            'mean_adj': predictions_df['mean_adj'],
            'mean_vol': predictions_df['mean_vol']
        }, index=combined_data_gru.index[-days_to_predict:])

    print("Mean DataFrame:")
    print(mean_df)

    mean_df.to_csv(f'{tickers[0]}_mean.csv')

    # create dataframe for original data
    original_df = pd.DataFrame({
            'open': data_handler.stock_data['Open'],
            'high': data_handler.stock_data['High'],
            'low': data_handler.stock_data['Low'],
            'close': data_handler.stock_data['Close'],
            'adj': data_handler.stock_data['Adj Close'],
            'vol': data_handler.stock_data['Volume']
        }, index=data_handler.stock_data.index)

    print("Original DataFrame:")
    print(original_df)

    original_df.to_csv(f'{tickers[0]}_original.csv')