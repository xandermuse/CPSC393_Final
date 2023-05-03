import datetime
import matplotlib.pyplot as plt
from Components.Predictors.lstm_predictor import LSTMPredictor

if __name__ == "__main__":
    tickers = ["AAPL"]
    start_date = "2018-01-01"
    end_date = "2021-09-01"

    predictor = LSTMPredictor(tickers, start_date, end_date)
    true_values, predictions, mse_scores, mae_scores, r2_scores = predictor.train_and_evaluate()

    print("Mean MSE:", sum(mse_scores) / len(mse_scores))
    print("Mean MAE:", sum(mae_scores) / len(mae_scores))
    print("Mean R2 Score:", sum(r2_scores) / len(r2_scores))

    days_to_predict = 7
    future_predictions = predictor.predict_future(days_to_predict)

    print("Future Predictions (Open, High, Low, Close, Adj Close, Volume):")
    for i, prediction in enumerate(future_predictions):
        print(f"Day {i + 1}: {prediction}")

    # Plot the true values and predictions
    plt.plot(predictor.data.dates, true_values[:, 3], label="True Close Prices")
    plt.plot(predictor.data.dates, predictions[:, 3], label="Predicted Close Prices")
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title("True vs. Predicted Close Prices")
    plt.show()
