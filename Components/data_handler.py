import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit

class DataHandler:
    def __init__(self, data_collector):
        self.data_collector = data_collector

    def create_sequences(self, data, sequence_length):
        inputs, outputs = [], []
        for i in range(len(data) - sequence_length):
            inputs.append(data.iloc[i : i + sequence_length].values)
            outputs.append(data.iloc[i + sequence_length].values)

        inputs, outputs = np.array(inputs), np.array(outputs)

        return inputs, outputs
    
    def normalize_data(self, data):
        scaler = MinMaxScaler(feature_range=(0, 1))
        return pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    
    def time_series_split(self, data, n_splits):
        tscv = TimeSeriesSplit(n_splits=n_splits)
        return tscv
