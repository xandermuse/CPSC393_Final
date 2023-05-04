from sklearn.metrics import mean_squared_error
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau


EPOCHS = 1

class GRUModel:
    """A Gated Recurrent Unit (GRU) model for time series prediction.
    """
    def __init__(self, units=50, num_layers=2, dropout_rate=0.2, optimizer='adam', learning_rate=0.001):
        """
        Initializes a GRU model with the specified parameters.

        Args:
            units (int, optional): Number of recurrent units. Defaults to 50.
            num_layers (int, optional): Number of GRU layers. Defaults to 2.
            dropout_rate (float, optional): Dropout rate for regularization. Defaults to 0.2.
            optimizer (str, optional): Optimizer for training the model. Defaults to 'adam'.
            learning_rate (float, optional): Learning rate for the optimizer. Defaults to 0.001.
        """
        self.model = Sequential()
        for i in range(num_layers):
            if i == num_layers - 1:
                self.model.add(GRU(units=units, input_shape=(None, 6)))
            else:
                self.model.add(GRU(units=units, return_sequences=True, input_shape=(None, 6)))
            self.model.add(Dropout(dropout_rate))
        self.model.add(Dense(6))
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.model.compile(optimizer=optimizer, loss='mean_squared_error')

    def train(self, X_train, y_train, epochs=EPOCHS, batch_size=32, validation_split=0.2, patience=15):
        """
        Trains the GRU model on the given data.

        Args:
            X_train (numpy.ndarray): Input data for training.
            y_train (numpy.ndarray): Target data for training.
            epochs (int, optional): Number of training epochs. Defaults to EPOCHS.
            batch_size (int, optional): Batch size for training. Defaults to 32.
            validation_split (float, optional): Fraction of the training data to be used as validation data. Defaults to 0.2.
            patience (int, optional): Number of epochs with no improvement after which training will be stopped. Defaults to 15.
        """
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=10, verbose=1)

        history = self.model.fit(X_train, y_train, epochs=EPOCHS, batch_size=batch_size, validation_split=validation_split, callbacks=[early_stopping, lr_schedule])
        self.train_loss = history.history['loss']
        self.val_loss = history.history['val_loss']

    def predict(self, X_test):
        """
        Generates predictions for the given input data.

        Args:
            X_test (numpy.ndarray): Input data for prediction.

        Returns:
            numpy.ndarray: Predicted values.
        """
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        """
        Evaluates the model on the given test data.

        Args:
            X_test (numpy.ndarray): Input data for evaluation.
            y_test (numpy.ndarray): Target data for evaluation.

        Returns:
            tuple: A tuple containing the evaluation loss and metric values.
        """
        return self.model.evaluate(X_test, y_test)
    
    def get_train_loss(self):
        """Retrieves the training loss history.

        Returns:
            list: List of training loss values for each epoch.
        """
        return self.train_loss

    def get_val_loss(self):
        """Retrieves the validation loss history.

        Returns:
            list: List of validation loss values for each epoch.
        """
        return self.val_loss