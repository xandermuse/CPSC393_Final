import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class StockVisualizer:
    def __init__(self, original_data, true_values, predictions):
        self.original_data = original_data
        self.true_values = true_values
        self.predictions = predictions

    def visualize_predictions(self, output_file='stock_prediction_plot.jpg'):
        plt.figure(figsize=(14, 6))
        plt.plot(self.original_data['Date'], self.original_data['Close'], label='Actual Prices', color='blue')
        plt.plot(self.original_data['Date'].iloc[-len(self.true_values):], self.true_values, label='True Values', color='green')
        plt.plot(self.original_data['Date'].iloc[-len(self.predictions):], self.predictions, label='Predictions', color='red')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.title('Stock Price Prediction')
        plt.legend()
        plt.savefig(output_file, format='jpg', dpi=300)
        plt.show()

    def plot_true_vs_predicted(self, output_file='true_vs_predicted_plot.jpg'):
        plt.figure(figsize=(14, 6))
        plt.scatter(self.true_values, self.predictions, alpha=0.7)
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title('True Values vs Predictions')
        
        mse = mean_squared_error(self.true_values, self.predictions)
        mae = mean_absolute_error(self.true_values, self.predictions)
        r2 = r2_score(self.true_values, self.predictions)

        plt.text(0.05, 0.95, f'MSE: {mse:.2f}\nMAE: {mae:.2f}\nR2: {r2:.2f}', 
                 transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

        plt.savefig(output_file, format='jpg', dpi=300)
        plt.show()
