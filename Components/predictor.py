import pandas as pd
from skopt import gp_minimize
from skopt.utils import use_named_args
import skopt.space
from functools import partial
from sklearn.metrics import mean_squared_error
from skopt.space import Integer, Real, Categorical
import dill

from .data_collector import DataCollector
from .Models.lstm_model import LSTMModel
from .Models.gru_model import GRUModel
from .Models.arima_model import ARIMAModel
from .Models.transformer_model import TransformerModel
from .data_handler import DataHandler


class BasePredictor:
    """Base class for creating predictors for different types of time series models.
    """
    def __init__(self, model_class, tickers, start, end):
        """Initializes the BasePredictor class.

        Args:
            model_class (class): The class of the model to be used for prediction.
            tickers (list): List of stock tickers to be used for collecting data.
            start (str): Start date for data collection in the format 'YYYY-MM-DD'.
            end (str): End date for data collection in the format 'YYYY-MM-DD'.
        """
        self.model_class = model_class
        self.tickers = tickers
        self.start = start
        self.end = end
        self.data_collector = DataCollector()
        self.data_handler = DataHandler(self.data_collector)
    
    def download_data(self):
        """Downloads stock data for the specified tickers and date range.
        """
        print("Downloading data...")
        self.data = self.data_collector.get_data(tickers=self.tickers, start=self.start, end=self.end)
        print(f"Data downloaded. Shape: {self.data.shape}")
        
        # Convert the 'Date' column to a datetime object and set it as the index
        self.data['Date'] = pd.to_datetime(self.data['Date'], unit='s')
        self.data.set_index('Date', inplace=True)

        # Print the first 3 rows of the DataFrame
        print(self.data.head(3))


    def time_series_split(self, n_splits):
        """Performs a time series split on the data using the specified number of splits.

        Args:
            n_splits (int): Number of splits for time series cross-validation.

        Returns:
            TimeSeriesSplit: A TimeSeriesSplit object with the specified number of splits.
        """
        return self.data_handler.time_series_split(self.data, n_splits)


    def train_and_evaluate(self, n_splits=2):
        """Trains and evaluates the model using time series cross-validation.

        Args:
            n_splits (int, optional): Number of splits for time series cross-validation. Defaults to 2.

        Returns:
            list: List of best hyperparameters for each split.
        """
        self.download_data()
        tscv = self.time_series_split(n_splits)
        best_hyperparameters = []

        for train_index, test_index in tscv.split(self.data):
            train_data, test_data = self.data.iloc[train_index], self.data.iloc[test_index]
            X_train, y_train = self.data_handler.create_sequences(train_data[:-60], 60)
            X_test, y_test = self.data_handler.create_sequences(test_data[:-60], 60)


            if isinstance(self, ARIMAPredictor):
                best_params = self.optimize_model((X_train, y_train))
            else:
                best_params = self.optimize_model((X_train, y_train), (X_test, y_test))

            best_hyperparameters.append(best_params)

        save_best_params(best_hyperparameters, f'{self.model_class.__name__}_best_hyperparameters.pickle')
        return best_hyperparameters

class TransformerPredictor(BasePredictor):
    """Predictor class for the Transformer model.

    Args:
        BasePredictor (class): Inherits from the BasePredictor class.
    """
    def __init__(self, tickers, start, end):
        """Initializes the TransformerPredictor class.

        Args:
            tickers (list): List of stock tickers to be used for collecting data.
            start (str): Start date for data collection in the format 'YYYY-MM-DD'.
            end (str): End date for data collection in the format 'YYYY-MM-DD'.
        """
        super().__init__(TransformerModel, tickers, start, end)
        # Define the search space for hyperparameters
        self.search_space = [
            # Add hyperparameters here
        ]


    def optimize_model(self, train_data, test_data=None):
        """Optimizes the Transformer model using the search space.

        Args:
            train_data (tuple): Tuple containing the training data (X_train, y_train).
            test_data (tuple, optional): Tuple containing the test data (X_test, y_test). Defaults to None.
        """
        # Optimize the Transformer model using the search space
        pass



class ARIMAPredictor(BasePredictor):
    """Predictor class for the ARIMA model.

    Args:
        BasePredictor (class): Inherits from the BasePredictor class.
    """
    def __init__(self, tickers, start, end): 
        super().__init__(ARIMAModel, tickers, start, end)
        self.search_space = [
            Integer(0, 5, name="p"),
            Integer(0, 2, name="d"),
            Integer(0, 5, name="q"),
        ]

    def optimize_model(self, train_data):
        """Optimizes the ARIMA model using the search space.

        Args:
            train_data (tuple): Tuple containing the training data (X_train, y_train).

        Returns:
            list: List of best hyperparameters for the ARIMA model.
        """
        results_list = []

        @use_named_args(self.search_space)
        def partial_evaluate_model(p, d, q):
            p, d, q = int(p), int(d), int(q)
            arima = ARIMAModel(p=p, d=d, q=q)
            arima.train(train_data)
            predictions = arima.predict(train_data)
            return mean_squared_error(train_data, predictions)

        result = gp_minimize(partial_evaluate_model, self.search_space, n_calls=10, random_state=0, verbose=True, n_jobs=-1)

        # Sort the results list by MSE
        results_list.sort(key=lambda x: x[1])

        # Print the best parameters
        print("Best hyperparameters found: ", results_list[0][0])

        return result


def evaluate_model(model_class, train_data, test_data, **params):
    """Evaluates a given model with the provided data and parameters.

    Args:
        model_class (class): The class of the model to be evaluated.
        train_data (tuple): Tuple containing the training data (X_train, y_train).
        test_data (tuple): Tuple containing the test data (X_test, y_test).

    Returns:
        float: Mean squared error of the model evaluated on the test data.
    """
    model = model_class(**params)
    X_train, y_train = train_data
    X_test, y_test = test_data

    model.train(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, patience=5)
    mse = model.evaluate(X_test, y_test)
    
    return mse


class GRUPredictor(BasePredictor):
    """Predictor class for the GRU model.

    Args:
        BasePredictor (class): Inherits from the BasePredictor class.
    """
    def __init__(self, tickers, start, end):
        """Initializes the GRUPredictor with the given tickers and date range.

        Args:
            tickers (list): List of stock tickers to predict.
            start (str): Start date for fetching stock data in the format "YYYY-MM-DD".
            end (str): End date for fetching stock data in the format "YYYY-MM-DD".
        """
        super().__init__(GRUModel, tickers, start, end)
        self.search_space = [
            Integer(10, 200, name='units'),
            Real(0.1, 0.5, name='dropout_rate'),
            Categorical(['adam', 'rmsprop'], name='optimizer')
        ]

    def optimize_model(self, train_data, test_data=None):
        """Optimizes the GRU model hyperparameters using the given data.

        Args:
            train_data (tuple): Tuple containing the training data (X_train, y_train).
            test_data (tuple, optional): Tuple containing the test data (X_test, y_test). Defaults to None.

        Returns:
            list: A list containing the optimization results.
        """
        results_list = []

        named_evaluate_model = use_named_args(self.search_space)(lambda **params: evaluate_model(self.model_class, train_data, test_data, **params))

        result = gp_minimize(named_evaluate_model, self.search_space, n_calls=10, random_state=0, verbose=True, n_jobs=-1)
        results_list.append(result)
        
        return results_list

class LSTMPredictor(BasePredictor):
    """Predictor class for the LSTM model.

    Args:
        BasePredictor (class): Inherits from the BasePredictor class.
    """
    def __init__(self, tickers, start, end):
        """Initializes the LSTMPredictor with the given tickers and date range.

        Args:
            tickers (list): List of stock tickers to predict.
            start (str): Start date for fetching stock data in the format "YYYY-MM-DD".
            end (str): End date for fetching stock data in the format "YYYY-MM-DD".
        """
        super().__init__(LSTMModel, tickers, start, end)
        self.search_space = [
            Integer(10, 200, name='units'),
            Real(0.1, 0.5, name='dropout_rate'),
            Categorical(['adam', 'rmsprop'], name='optimizer')
        ]

    def optimize_model(self, train_data, test_data=None):
        """Optimizes the LSTM model hyperparameters using the given data.

        Args:
            train_data (tuple): Tuple containing the training data (X_train, y_train).
            test_data (tuple, optional): Tuple containing the test data (X_test, y_test). Defaults to None.

        Returns:
            list: List of best hyperparameters for the model.
        """
        @use_named_args(self.search_space)
        def evaluate_model(**params):
            """Evaluates the LSTM model using the provided hyperparameters.

            Args:
                **params (dict): Dictionary containing the hyperparameters for the LSTM model.

            Returns:
                float: Mean squared error (MSE) for the model on the test data.
            """
            model = self.model_class(**params)
            X_train, y_train = train_data
            X_test, y_test = test_data

            print("X_train shape:", X_train.shape)
            print("y_train shape:", y_train.shape)

            model.train(X_train, y_train, epochs=1, batch_size=32, validation_split=0.2, patience=5)
            mse = model.evaluate(X_test, y_test)
            return mse

        result = gp_minimize(evaluate_model, self.search_space, n_calls=10, random_state=0, verbose=True, n_jobs=-1)

        best_hyperparameters = result.x
        return best_hyperparameters


def save_best_params(best_hyperparameters, filename):
    """Saves the best hyperparameters to a file.

    Args:
        best_hyperparameters (list): List of best hyperparameters for each model.
        filename (str): The name of the file to save the hyperparameters in.
    """
    with open(filename, 'wb') as outfile:
        for params in best_hyperparameters:
            print(params)
            dill.dump(params, outfile)

def load_best_params(filename):
    """Loads the best hyperparameters from a file.

    Args:
        filename (str): The name of the file to load the hyperparameters from.

    Returns:
        list: List of best hyperparameters for each model.
    """
    with open(filename, 'rb') as infile:
        best_hyperparameters_list = dill.load(infile)

    # Find the best hyperparameters with the lowest MSE
    best_hyperparameters = min(best_hyperparameters_list, key=lambda x: x[1])
    return best_hyperparameters[0]



# if __name__ == '__main__':
#     tickers = 'AAPL'
#     start = '2010-01-01'
#     end = '2020-01-01'
#     n_splits = 2

# print("Training Transformer model...")
# transformer_predictor = TransformerPredictor(tickers=tickers, start=start, end=end)
# transformer_predictor.train_and_evaluate(n_splits=n_splits)