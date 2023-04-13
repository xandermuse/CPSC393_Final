import yfinance as yf
import pandas as pd



class DataCollector:
    def get_data(self, tickers, start, end):
        data = pd.DataFrame()
        for ticker in tickers:
            stock_data = yf.download(ticker, start=start, end=end)
            stock_data = stock_data.reset_index()
            stock_data["Date"] = stock_data["Date"].apply(lambda x: x.timestamp())  # Convert Timestamp to Unix timestamp
            data = data.append(stock_data)
        return data

