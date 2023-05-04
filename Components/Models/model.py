import pandas as pd
from typing import Union
from abc import ABC, abstractmethod

class Model(ABC):
    """Abstract base class for implementing custom models for forecasting.
    """
    @abstractmethod
    def train(self, data: pd.DataFrame) -> None:
        """Trains the model using the given input data.

        Args:
            data (pd.DataFrame): Input training data containing features and labels.
        """
        pass

    @abstractmethod
    def predict(self, input: pd.DataFrame) -> pd.DataFrame:
        """Predicts future values using the trained model.

        Args:
            input (pd.DataFrame): Input data for prediction containing features.

        Returns:
            pd.DataFrame: A DataFrame containing the predicted future values.
        """
        pass