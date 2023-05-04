from Models.lstm_model import LSTMModel
from Models.gru_model import GRUModel
from Models.transformer_model import TransformerModel
from Models.arima_model import ARIMAModel
from Models.prophet_model import ProphetModel
from Models.model import Model

class ModelFactory:
    """A factory class for creating instances of different time series forecasting models.
    """
    def create_model(self, model_type: str) -> Model:
        """Creates an instance of the specified model type.

        Args:
            model_type (str): The type of the model to create. Supported types are "lstm", "gru", "transformer", "arima", and "prophet".

        Raises:
            Exception: If the provided model_type is not one of the supported types.

        Returns:
            Model: An instance of the specified model type, inheriting from the Model base class.
        """
        if model_type == "lstm":
            return LSTMModel()
        elif model_type == "gru":
            return GRUModel()
        elif model_type == "transformer":
            return TransformerModel()
        elif model_type == "arima":
            return ARIMAModel()
        elif model_type == "prophet":
            return ProphetModel()
        else:
            raise Exception("Invalid model type")