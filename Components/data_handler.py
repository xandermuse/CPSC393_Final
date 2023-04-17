import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit

class DataHandler:
    """A class to handle data preprocessing tasks for time series data, such as normalization and sequence creation.
    """
    def __init__(self, data_collector):
        """Initializes the DataHandler with a DataCollector instance.

        Args:
            data_collector (DataCollector): An instance of the DataCollector class to fetch stock data.
        """
        self.data_collector = data_collector

    def create_sequences(self, data, sequence_length):
        """Creates sequences of data points of a given length from the input data.

        Args:
            data (pandas.DataFrame): The input data to create sequences from.
            sequence_length (int): The length of the sequences to create.

        Returns:
            tuple: A tuple of two numpy arrays, the first containing the input sequences and the second containing the corresponding output data points.
        """
        inputs, outputs = [], []
        for i in range(len(data) - sequence_length):
            inputs.append(data.iloc[i : i + sequence_length].values)
            outputs.append(data.iloc[i + sequence_length].values)

        inputs, outputs = np.array(inputs), np.array(outputs)

        return inputs, outputs
    
    def normalize_data(self, data):
        """Normalizes the input data using MinMaxScaler.

        Args:
            data (pandas.DataFrame): The input data to normalize.

        Returns:
            pandas.DataFrame: A dataframe containing the normalized data.
        """
        scaler = MinMaxScaler(feature_range=(0, 1))
        return pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    
    def time_series_split(self, data, n_splits):
        """Performs time series cross-validation on the input data.

        Args:
            data (pandas.DataFrame): The input data to split.
            n_splits (int): The number of splits to create in the data.

        Returns:
            TimeSeriesSplit: A TimeSeriesSplit object representing the cross-validation splits.
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        return tscv
