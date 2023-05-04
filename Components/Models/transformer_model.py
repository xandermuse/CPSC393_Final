import numpy as np
import pandas as pd
from .model import Model
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, TimeDistributed
from tensorflow.keras.optimizers import Adam
from transformers import TFDistilBertModel
from tensorflow.keras.callbacks import EarlyStopping


class TransformerModel(Model):
    """A class for the Transformer-based forecasting model, inheriting from the Model abstract base class.
    """
    def __init__(self, sequence_length, units, dropout_rate, learning_rate):
        """Initializes an instance of the TransformerModel class.

        Args:
            sequence_length (int): The length of the input sequence.
            units (int): The number of units in the LSTM layers.
            dropout_rate (float): The dropout rate for the Dropout layers.
            learning_rate (float): The learning rate for the optimizer.
        """
        self.sequence_length = sequence_length
        self.units = units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = self.build_model()

    def build_model(self):
        """Builds and compiles the Transformer-based model.

        Returns:
            tf.keras.Model: The compiled Transformer-based model.
        """
        input_layer = Input(shape=(self.sequence_length, 1))
        transformer = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
        transformer.trainable = False
        x = transformer(input_layer)[0]
        x = LSTM(self.units, return_sequences=True)(x)
        x = Dropout(self.dropout_rate)(x)
        x = LSTM(self.units)(x)
        x = Dropout(self.dropout_rate)(x)
        output_layer = Dense(6)(x)  # Updated to predict 6 values

        model = KerasModel(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mean_squared_error', metrics=['mse'])

        return model

    def train(self, X_train, y_train, epochs, batch_size, validation_split, patience):
        """Trains the TransformerModel using the given input data.

        Args:
            X_train (np.array): The input training data.
            y_train (np.array): The target values for the training data.
            epochs (int): The number of epochs to train the model.
            batch_size (int): The batch size for the training process.
            validation_split (float): The fraction of the training data to be used for validation.
            patience (int): The number of epochs with no improvement after which training will be stopped.
        """
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            shuffle=False,
            verbose=1
        )

    def predict(self, X):
        """Predicts future closing prices using the trained Transformer model.

        Args:
            X (np.array): The input data for making predictions.

        Returns:
            np.array: The predicted closing prices.
        """
        return self.model.predict(X)