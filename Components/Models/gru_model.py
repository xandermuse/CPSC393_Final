import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Dropout
from tensorflow.keras.callbacks import EarlyStopping


class GRUModel:
    def __init__(self, units=50, dropout_rate=0.2, optimizer='adam'):
        self.model = Sequential()
        self.model.add(GRU(units, return_sequences=True, input_shape=(60, 6)))
        self.model.add(Dropout(dropout_rate))
        self.model.add(GRU(units))
        self.model.add(Dropout(dropout_rate))
        self.model.add(Dense(6))

        self.model.compile(optimizer=optimizer, loss='mean_squared_error')

    def train(self, X_train, y_train, epochs=1, batch_size=32, validation_split=0.2, patience=5):
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience)
        self.model.fit(X_train, y_train, epochs=epochs,
                        batch_size=batch_size, validation_split=validation_split,
                        callbacks=[early_stopping], shuffle=False)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        mse = np.mean((y_pred - y_test) ** 2)
        return mse
