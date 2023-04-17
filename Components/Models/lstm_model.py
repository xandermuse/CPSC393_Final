import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error


class LSTMModel:
    """Long Short-Term Memory (LSTM) model for time series forecasting.
    """
    def __init__(self, units=50, dropout_rate=0.2, optimizer='adam'):
        """Initializes the LSTMModel with the given parameters.

        Args:
            units (int, optional): Number of LSTM units in each layer. Defaults to 50.
            dropout_rate (float, optional): Dropout rate to apply between layers. Defaults to 0.2.
            optimizer (str, optional): Optimizer to use for training. Defaults to 'adam'.
        """
        input_shape = (60, 6)  # Set the input_shape manually
        self.model = Sequential()
        self.model.add(LSTM(units, return_sequences=True, input_shape=input_shape))
        self.model.add(Dropout(dropout_rate))
        self.model.add(LSTM(units, return_sequences=True))
        self.model.add(Dropout(dropout_rate))
        self.model.add(LSTM(units))
        self.model.add(Dense(6, activation='linear'))
        self.model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    def train(self, X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, patience=5):
        """Trains the LSTM model using the given input data and labels.

        Args:
            X_train (array-like): Input training data.
            y_train (array-like): Training labels.
            epochs (int, optional): Number of epochs for training. Defaults to 10.
            batch_size (int, optional): Batch size for training. Defaults to 32.
            validation_split (float, optional): Fraction of the training data to be used as validation data. Defaults to 0.2.
            patience (int, optional): Number of epochs with no improvement after which training will be stopped. Defaults to 5.
        """
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        self.model.fit(X_train, y_train, epochs=epochs,
                       batch_size=batch_size, validation_split=validation_split,
                       callbacks=[early_stopping], shuffle=False)

    def predict(self, data):
        """Predicts future values using the trained LSTM model.

        Args:
            data (array-like): Input data for prediction.

        Returns:
            array-like: The predicted future values.
        """
        predictions = self.model.predict(data)
        return predictions.flatten()

    def evaluate(self, X_test, y_test):
        """Evaluates the performance of the LSTM model using the test data and labels.

        Args:
            X_test (array-like): Input test data.
            y_test (array-like): Test labels.

        Returns:
            float: Mean squared error of the predictions.
        """
        predictions = self.model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        return mse

