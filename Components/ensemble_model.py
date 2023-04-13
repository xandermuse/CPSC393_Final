from Models.model import Model
import pandas as pd

class EnsembleModel:
    def __init__(self):
        self.models = []
        self.weights = []

    def add_model(self, model: Model, weight: float) -> None:
        self.models.append(model)
        self.weights.append(weight)

    def predict(self, input: pd.DataFrame) -> pd.DataFrame:
        # Ensemble prediction logic
        pass