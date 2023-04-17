import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Dropout
from tensorflow.keras.callbacks import EarlyStopping


class GRUModel:
    """A Gated Recurrent Unit (GRU) model for stock price prediction.
    """
    def __init__(self, units=50, dropout_rate=0.2, optimizer='adam'):
        """Initializes the GRUModel with the given parameters.

        Args:
            units (int, optional): The number of recurrent units in the GRU layers. Defaults to 50.
            dropout_rate (float, optional): The dropout rate for Dropout layers. Defaults to 0.2.
            optimizer (str, optional): The optimizer to use for model training. Defaults to 'adam'.
        """
        self.model = Sequential()

        self.model.add(GRU(units, return_sequences=True, input_shape=(60, 6)))
        self.model.add(Dropout(dropout_rate))
        self.model.add(GRU(units))
        self.model.add(Dropout(dropout_rate))
        self.model.add(Dense(6))

        self.model.compile(optimizer=optimizer, loss='mean_squared_error')

    def train(self, X_train, y_train, epochs=1, batch_size=32, validation_split=0.2, patience=5):
        """Trains the GRU model using the given training data.

        Args:
            X_train (array-like): The input features for training.
            y_train (array-like): The target output for training.
            epochs (int, optional): The number of epochs to train the model. Defaults to 1.
            batch_size (int, optional): The batch size for training. Defaults to 32.
            validation_split (float, optional): The fraction of the training data to be used as validation data. Defaults to 0.2.
            patience (int, optional): The number of epochs with no improvement after which training will be stopped. Defaults to 5.
        """
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience)
        self.model.fit(X_train, y_train, epochs=epochs,
                        batch_size=batch_size, validation_split=validation_split,
                        callbacks=[early_stopping], shuffle=False)

    def predict(self, X_test):
        """Predicts stock prices using the trained GRU model.

        Args:
            X_test (array-like): The input features for prediction.

        Returns:
            array-like: The predicted stock prices.
        """
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        """Evaluates the GRU model using the given test data.

        Args:
            X_test (array-like): The input features for testing.
            y_test (array-like): The true target output for testing.

        Returns:
            float: The mean squared error (MSE) of the model's predictions.
        """
        y_pred = self.predict(X_test)
        mse = np.mean((y_pred - y_test) ** 2)
        return mse
