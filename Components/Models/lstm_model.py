import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error


class LSTMModel:
    def __init__(self, units=50, dropout_rate=0.2, optimizer='adam'):
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
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        self.model.fit(X_train, y_train, epochs=epochs,
                       batch_size=batch_size, validation_split=validation_split,
                       callbacks=[early_stopping], shuffle=False)

    def predict(self, data):
        predictions = self.model.predict(data)
        return predictions.flatten()

    def evaluate(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        return mse

