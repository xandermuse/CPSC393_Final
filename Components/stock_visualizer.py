import matplotlib.pyplot as plt

class StockVisualizer:
    def __init__(self, original_data, true_values, predictions):
        self.original_data = original_data
        self.true_values = true_values
        self.predictions = predictions

    def plot(self):
        plt.figure(figsize=(14, 6))
        plt.plot(self.original_data['Date'], self.original_data['Close'], label='Actual Prices', color='blue')
        plt.plot(self.original_data['Date'].iloc[-len(self.true_values):], self.true_values, label='True Values', color='green')
        plt.plot(self.original_data['Date'].iloc[-len(self.predictions):], self.predictions, label='Predictions', color='red')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.title('Stock Price Prediction')
        plt.legend()
        plt.savefig('stock_prediction_plot.jpg', format='jpg', dpi=300)
        plt.show()
