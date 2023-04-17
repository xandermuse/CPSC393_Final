import numpy as np
import pandas as pd
from .model import Model
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, TimeDistributed
from tensorflow.keras.optimizers import Adam
from transformers import TFDistilBertModel
from tensorflow.keras.callbacks import EarlyStopping


class TransformerModel(Model):
    def __init__(self, sequence_length, units, dropout_rate, learning_rate):
        self.sequence_length = sequence_length
        self.units = units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = self.build_model()

    def build_model(self):
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
        return self.model.predict(X)
