import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit

from Components.Data.data_collector import DataCollector

class DataHandler:
    def __init__(self, tickers, start_date, end_date):
        self.data_collector = DataCollector()
        self.stock_data = self.data_collector.get_data(tickers, start_date, end_date)
        self.stock_data = self.stock_data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]  # Keep only specified columns
        self.X = None
        self.y = None

    def create_sequences(self, data, sequence_length):
        inputs, outputs = [], []
        for i in range(len(data) - sequence_length):
            inputs.append(data.iloc[i : i + sequence_length].values)
            outputs.append(data.iloc[i + sequence_length].values)

        inputs, outputs = np.array(inputs), np.array(outputs)

        return inputs, outputs
    
    def normalize_data(self, data):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        return pd.DataFrame(self.scaler.fit_transform(data), columns=data.columns)

    def time_series_split(self, data, n_splits):
        tscv = TimeSeriesSplit(n_splits=n_splits)
        return tscv

    def preprocess_data(self, sequence_length=60):
        normalized_data = self.normalize_data(self.stock_data)
        self.X, self.y = self.create_sequences(normalized_data, sequence_length)
