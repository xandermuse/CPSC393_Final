import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np

class StockVisualizer:
    """StockVisualizer is a class for visualizing stock price data and predictions.
    """
    def __init__(self, original_data, true_values, predictions, future_predictions, dates, sequence_length=60):
        """Initializes the StockVisualizer with given data.

        Args:
            original_data (pd.DataFrame): Original stock price data.
            true_values (np.array): True stock prices for the test data.
            predictions (np.array): Predicted stock prices for the test data.
            future_predictions (np.array): Predicted stock prices for future days.
            dates (pd.Series): Dates corresponding to the original_data.
            sequence_length (int, optional): Sequence length used in data preprocessing. Defaults to 60.
        """
        self.original_data = original_data
        self.true_values = true_values
        self.predictions = predictions
        self.future_predictions = future_predictions
        self.dates = dates
        self.sequence_length = sequence_length
        self.original_data = self.original_data.iloc[sequence_length:]
## i can do merge function tojoin pred and og

## change axis ticks to limit the number of displayed x axis ticks 


    def visualize_predictions(self):
        """Visualizes the stock price predictions against the true values.
        """
        plt.figure(figsize=(14, 6))
        plt.clf() 

        plt.locator_params(axis='x', nbins=25)
        # Plot original data
        plt.plot(self.dates, self.original_data['Close'], label='Original Data', color='blue')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.title('Stock Price Predictions vs True Values')
        # angle the x-axis labels
        
        plt.xticks(rotation=45)
        # every 5th label is kept'
        plt.legend()
        plt.show()