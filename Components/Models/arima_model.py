import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

class ARIMAModel:
    def __init__(self, order=(1, 1, 1)):
        self.order = order
        self.model = None

    def train(self, series, feature):
        self.model = ARIMA(series[feature], order=self.order)
        self.model = self.model.fit()


    def predict(self, steps):
        if self.model is None:
            raise ValueError("The model must be trained before making predictions")

        forecast, _, _ = self.model.forecast(steps=steps)
        return forecast
