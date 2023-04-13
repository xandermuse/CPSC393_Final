import pandas as pd
from typing import Union
from abc import ABC, abstractmethod

class Model(ABC):
    @abstractmethod
    def train(self, data: pd.DataFrame) -> None:
        pass

    @abstractmethod
    def predict(self, input: pd.DataFrame) -> pd.DataFrame:
        pass