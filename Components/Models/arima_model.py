import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

class ARIMAModel:
    """An Autoregressive Integrated Moving Average (ARIMA) model for time series forecasting.
    """
    def __init__(self, order=(1, 1, 1)):
        """Initializes the ARIMAModel with the given ARIMA order.
        Args:
            order (tuple, optional): A tuple of three integers representing the (p, d, q) order of the ARIMA model. Defaults to (1, 1, 1).
        """
        self.order = order
        self.model = None

    def train(self, series, feature):
        """Trains the ARIMA model using the given time series data.
        Args:
            series (array-like): The input time series data for training.
        """
        self.model = ARIMA(series[feature], order=self.order)
        self.model = self.model.fit()


    def predict(self, steps):
        """Predicts future values using the trained ARIMA model.
        Args:
            steps (int): The number of future time steps to predict.
        Raises:
            ValueError: Raised if the model has not been trained before making predictions.
        Returns:
            array-like: The predicted future values.
        """
        if self.model is None:
            raise ValueError("The model must be trained before making predictions")

        forecast, _, _ = self.model.forecast(steps=steps)
        return forecast
