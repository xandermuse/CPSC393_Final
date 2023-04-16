from model import Model
from .data_collector import DataCollector
import pandas as pd
from fbprophet import Prophet

class ProphetModel(Model):
    def __init__(self):
        self.model = None

    def train(self, data: pd.DataFrame) -> None:
        # Rename columns to 'ds' and 'y' as required by Prophet
        data = data.rename(columns={'Date': 'ds', 'Close': 'y'})

        # Create and fit the Prophet model
        self.model = Prophet()
        self.model.fit(data)

    def predict(self, periods: int) -> pd.DataFrame:
        # Create a DataFrame with future dates for prediction
        future = self.model.make_future_dataframe(periods=periods)

        # Predict using the Prophet model
        forecast = self.model.predict(future)

        # Return the forecast DataFrame
        return forecast


if __name__ == '__main__':
    data_collector = DataCollector()
    model = ProphetModel()
    ticker = 'AAPL'
    start = '2019-01-01'
    end = '2020-01-01'
    # Download the data
    data = data_collector.get_data(ticker, start, end)
    # Train the model with your dataset
    model.train(data)

    # Predict future values (e.g., for the next 30 days)
    forecast = model.predict(periods=30)

    # Print the forecast DataFrame
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())