import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit

class DataHandler:
    def __init__(self, data_collector):
        self.data_collector = data_collector

    def create_sequences(self, data, window):
        X, y = [], []
        data = self.normalize_data(data)  # Normalize the data
        for i in range(len(data) - window):
            X.append(data[i : (i + window)].values)
            y.append(data.iloc[i + window, 1])  # Assuming 'Close' price is in the second column
        return np.array(X), np.array(y)
    
    def normalize_data(self, data):
        scaler = MinMaxScaler(feature_range=(0, 1))
        return pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    
    def time_series_split(self, data, n_splits):
        tscv = TimeSeriesSplit(n_splits=n_splits)
        return tscv
