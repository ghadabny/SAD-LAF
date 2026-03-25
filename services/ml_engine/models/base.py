from abc import ABC, abstractmethod
import pandas as pd
from pathlib import Path

class BaseScorer(ABC):
    """
    Contrat abstrait pour tous les modèles prédictifs du projet SAD-LAF.
    Garantit que tout modèle (LightGBM, XGBoost, etc.) exposera les mêmes méthodes.
    """

    @abstractmethod
    def train(self, df: pd.DataFrame, target_col: str = 'fraud_score') -> 'BaseScorer':
        """Entraîne le modèle sur les features et la cible fournies."""
        pass

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Prédit le score (ex: taux de fraude) pour les données fournies."""
        pass

    @abstractmethod
    def save(self, filename: str) -> Path:
        """Sauvegarde le modèle sur le disque."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, filename: str) -> 'BaseScorer':
        """Charge un modèle depuis le disque."""
        pass