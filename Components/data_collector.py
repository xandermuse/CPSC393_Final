import yfinance as yf
import pandas as pd

class DataCollector:
    """A class for collecting financial data of stocks from Yahoo Finance using yfinance library.
    
    tickers: the unique series of letters assigned to a publicly traded company's shares that represent the company's stock on a stock exchange. i.e. TSLA: Tesla, Inc.
    """
    def get_data(self, tickers, start, end):
        """Fetches historical stock data for the given tickers within the specified date range.

        Args:
            tickers (list of str): A list of stock symbols/tickers to fetch data for.
            start (str): The start date of the data range in the format 'YYYY-MM-DD'.
            end (str): The end date of the data range in the format 'YYYY-MM-DD'.

        Returns:
            pandas.DataFrame: A dataframe containing the historical stock data for the given tickers within the specified date range.
        """
        data = pd.DataFrame()
        for ticker in tickers:
            stock_data = yf.download(ticker, start=start, end=end)
            stock_data = stock_data.reset_index()
            stock_data["Date"] = stock_data["Date"].apply(lambda x: x.timestamp())  # Convert Timestamp to Unix timestamp
            data = data.append(stock_data)
        return data

