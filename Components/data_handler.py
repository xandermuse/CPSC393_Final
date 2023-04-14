import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit

class DataHandler:
    def __init__(self, data_collector):
        self.data_collector = data_collector

    def create_sequences(self, data, seq_length):
        num_features = data.shape[1]
        temp = data.copy()

        sequences = np.zeros((len(temp) - seq_length, seq_length, num_features))
        targets = np.zeros(len(temp) - seq_length)

        for i in range(len(temp) - seq_length):
            sequences[i] = temp.iloc[i:i + seq_length].values
            targets[i] = temp.iloc[i + seq_length]['Close']

        return sequences, targets
    
    def normalize_data(self, data):
        scaler = MinMaxScaler(feature_range=(0, 1))
        return pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    
    def time_series_split(self, data, n_splits):
        tscv = TimeSeriesSplit(n_splits=n_splits)
        return tscv
