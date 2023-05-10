from Models.model import Model
import pandas as pd

class EnsembleModel:
    """A class for creating an ensemble model by combining multiple models with specified weights.
    """
    def __init__(self):
        """Initializes the EnsembleModel with an empty list of models and weights.
        """
        self.models = []
        self.weights = []

    def add_model(self, model: Model, weight: float) -> None:
        """Adds a model to the ensemble along with its corresponding weight.

        Args:
            model (Model): An instance of a model that inherits from the Model base class.
            weight (float): The weight assigned to the model in the ensemble, representing its contribution to the final prediction.
        """
        self.models.append(model)
        self.weights.append(weight)

    def predict(self, input: pd.DataFrame) -> pd.DataFrame:
        """Generates predictions from the ensemble model using the weighted average of predictions from its individual models.

        Args:
            input (pd.DataFrame): The input data for which predictions are to be generated.

        Returns:
            pd.DataFrame: A dataframe containing the ensemble model's predictions.
        """
        # Ensemble prediction logic
        pass
