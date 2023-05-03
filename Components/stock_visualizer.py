import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np

class StockVisualizer:
    def __init__(self, original_data, true_values, predictions, future_predictions, dates, sequence_length=60):
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
