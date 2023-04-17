import sys

# Add the path to the folder containing the predictor classes
sys.path.append("c:/Users/xande/Documents/Chapman/Spring2023/CPSC 393 Parlett/CPSC_393/HomeWork/Final")


from Components.predictor import LSTMPredictor, GRUPredictor, ARIMAPredictor, TransformerPredictor
import datetime as dt

import warnings
warnings.filterwarnings("ignore")

tickers = ["TSLA"]
start = dt.datetime(2021, 1, 1)
end = dt.datetime(2023, 1, 1)
n_splits = 2

if __name__ == "__main__":
    transformer_predictor = TransformerPredictor(tickers=tickers, start=start, end=end)
    lstm_predictor = LSTMPredictor(tickers, start, end)
    gru_predictor = GRUPredictor(tickers=tickers, start=start, end=end)
    arima_predictor = ARIMAPredictor(tickers=tickers, start=start, end=end)

    
    transformer_predictor.train_and_evaluate(n_splits=n_splits)
    lstm_predictor.train_and_evaluate(n_splits=n_splits)
    gru_predictor.train_and_evaluate(n_splits=n_splits)
    arima_predictor.train_and_evaluate(n_splits=n_splits)