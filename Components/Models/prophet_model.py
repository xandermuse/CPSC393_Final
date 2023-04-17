from model import Model
from .data_collector import DataCollector
import pandas as pd
from fbprophet import Prophet

class ProphetModel(Model):
    """A class for the Prophet forecasting model, inheriting from the Model abstract base class.
    """
    def __init__(self):
        """Initializes an instance of the ProphetModel class.
        """
        self.model = None

    def train(self, data: pd.DataFrame) -> None:
        """Trains the Prophet model using the given input data.

        Args:
            data (pd.DataFrame): Input training data containing dates and closing prices.
        """
        # Rename columns to 'ds' and 'y' as required by Prophet
        data = data.rename(columns={'Date': 'ds', 'Close': 'y'})

        # Create and fit the Prophet model
        self.model = Prophet()
        self.model.fit(data)

    def predict(self, periods: int) -> pd.DataFrame:
        """Predicts future closing prices using the trained Prophet model.

        Args:
            periods (int): The number of periods (days) in the future to predict.

        Returns:
            pd.DataFrame: A DataFrame containing the predicted future closing prices.
        """
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