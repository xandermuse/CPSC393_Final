from .model import Model
import pandas as pd

class ProphetModel(Model):
    def train(self, data: pd.DataFrame) -> None:
        # Train the Prophet model
        pass

    def predict(self, input: pd.DataFrame) -> pd.DataFrame:
        # Predict using the Prophet model
        pass