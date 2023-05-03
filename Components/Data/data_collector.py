import yfinance as yf
import pandas as pd

class DataCollector:
    def get_data(self, tickers, start, end):
        data = pd.DataFrame()
        for ticker in tickers:
            stock_data = yf.download(ticker, start=start, end=end)
            stock_data = stock_data.reset_index()
            data = data.append(stock_data)
        return data
