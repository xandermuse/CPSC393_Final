import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error

class LSTMModel:
    def __init__(self, units=50, dropout_rate=0.2, optimizer='adam'):
        self.model = Sequential()
        self.model.add(LSTM(units=units, return_sequences=True, input_shape=(None, 6)))
        self.model.add(Dropout(dropout_rate))
        self.model.add(LSTM(units=units))
        self.model.add(Dropout(dropout_rate))
        self.model.add(Dense(6))
        self.model.compile(optimizer=optimizer, loss='mean_squared_error')

    def train(self, X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, patience=5):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, callbacks=[early_stopping])
        self.train_loss = history.history['loss']
        self.val_loss = history.history['val_loss']

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)

    def get_train_loss(self):
        return self.train_loss

    def get_val_loss(self):
        return self.val_loss
