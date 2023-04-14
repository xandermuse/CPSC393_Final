import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Dropout

class GRUModel:
    def __init__(self, input_shape=(60, 7), units=50, dropout_rate=0.2, optimizer='adam'):
        self.model = Sequential()
        self.model.add(GRU(units, return_sequences=True, input_shape=input_shape, dropout=dropout_rate))
        self.model.add(Dropout(dropout_rate))
        self.model.add(GRU(units, return_sequences=True, dropout=dropout_rate))
        self.model.add(Dropout(dropout_rate))
        self.model.add(GRU(units))
        self.model.add(Dense(1))
        self.model.compile(optimizer=optimizer, loss='mean_squared_error')

    def train(self, X_train, y_train, epochs=1, batch_size=32, validation_split=0.2, patience=5):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, callbacks=[early_stopping])

    def predict(self, data):
        predictions = self.model.predict(data)
        return predictions.flatten()
