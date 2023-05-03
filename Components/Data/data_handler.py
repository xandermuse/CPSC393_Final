import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from Components.Data.data_collector import DataCollector



class DataHandler:
    def __init__(self, tickers, start_date, end_date):
        self.data_collector = DataCollector()
        self.stock_data = self.data_collector.get_data(tickers, start_date, end_date)
        self.stock_data['Date'] = pd.to_datetime(self.stock_data['Date']).dt.strftime('%Y-%m-%d')
        self.stock_data = self.stock_data.set_index('Date')
        self.stock_data = self.stock_data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]  # Keep only specified columns
        self.dates = self.stock_data.index
        self.X = None
        self.y = None

    def inverse_transform_data(self, data):
        return self.scaler.inverse_transform(data)

    def create_sequences(self, data, sequence_length):
        n_sequences = len(data) // sequence_length
        n_data_points = n_sequences * sequence_length
        inputs = data.iloc[:n_data_points - sequence_length].values.reshape(-1, sequence_length, data.shape[1])
        outputs = data.iloc[sequence_length: n_data_points].values

        return inputs, outputs


    def normalize_data(self, data):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        return pd.DataFrame(self.scaler.fit_transform(data), columns=data.columns)

    def preprocess_data(self, sequence_length=60):
        normalized_data = self.normalize_data(self.stock_data)
        self.X, self.y = self.create_sequences(normalized_data, sequence_length)
        self.dates = self.dates[sequence_length:]

    def split_data(self, n_splits):
        tscv = TimeSeriesSplit(n_splits=n_splits)
        return tscv


