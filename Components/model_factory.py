from Models.lstm_model import LSTMModel
from Models.gru_model import GRUModel
from Models.transformer_model import TransformerModel
from Models.arima_model import ARIMAModel
from Models.prophet_model import ProphetModel
from Models.model import Model

class ModelFactory:
    def create_model(self, model_type: str) -> Model:
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