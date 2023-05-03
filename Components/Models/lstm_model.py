from sklearn.metrics import mean_squared_error
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau

class LSTMModel:
    def __init__(self, units=50, num_layers=2, dropout_rate=0.2, optimizer='adam', learning_rate=0.001):
        self.model = Sequential()
        for i in range(num_layers):
            if i == num_layers - 1:
                self.model.add(LSTM(units=units, input_shape=(None, 6)))
            else:
                self.model.add(LSTM(units=units, return_sequences=True, input_shape=(None, 6)))
            self.model.add(Dropout(dropout_rate))
        self.model.add(Dense(6))
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.model.compile(optimizer=optimizer, loss='mean_squared_error')

    def train(self, X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, patience=15):
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=10, verbose=1)

        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, callbacks=[early_stopping, lr_schedule])
        self.train_loss = history.history['loss']
        self.val_loss = history.history['val_loss']

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)
