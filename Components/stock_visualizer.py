import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np

class StockVisualizer:
    def __init__(self, original_data, true_values, predictions, future_predictions):
        self.original_data = original_data
        self.true_values = true_values
        
        # Convert predictions to a NumPy array and reshape it
        predictions_array = np.array(predictions)
        predictions_reshaped = predictions_array.reshape(-1, predictions_array.shape[2])
        self.predictions = pd.DataFrame(predictions_reshaped, columns=original_data.columns, index=original_data.index[-len(predictions_reshaped):])
        
        self.future_predictions = future_predictions
        self.future_predictions_df = self.prepare_future_predictions_df()
    
    def prepare_future_predictions_df(self):
        columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        future_predictions_df = pd.DataFrame(self.future_predictions, columns=columns)

        last_date = self.original_data.index[-1]
        num_future_dates = len(self.future_predictions)
        future_dates = pd.date_range(start=last_date, periods=num_future_dates + 1, closed='right', freq='B')

        future_predictions_df['Date'] = future_dates
        future_predictions_df.set_index('Date', inplace=True)

        return future_predictions_df

    def visualize_predictions(self):
        plt.figure(figsize=(14, 6))
        plt.title('Stock Price Predictions')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.plot(self.original_data.index, self.original_data['Close'], label='Original Data', color='blue')
        plt.plot(self.predictions.index, self.predictions['Close'], label='Predictions', color='red')
        plt.plot(self.future_predictions_df.index, self.future_predictions_df['Close'], label='Future Predictions', color='orange')
        plt.legend()
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

    def plot_loss(self, train_losses, val_losses, output_file='loss_plot.jpg'):
        train_losses = [item for sublist in train_losses for item in sublist]
        val_losses = [item for sublist in val_losses for item in sublist]

        plt.figure(figsize=(14, 6))
        plt.plot(train_losses, label='Training Loss', color='blue')
        plt.plot(val_losses, label='Validation Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss vs Epoch')
        plt.legend()
        plt.savefig(output_file, format='jpg', dpi=300)
        plt.show()
