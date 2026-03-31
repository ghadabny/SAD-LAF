# services/ml_engine/models/base.py
from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd


class BaseScorer(ABC):
    """
    Contrat abstrait pour tous les modèles prédictifs du projet SAD-LAF.

    Garantit que tout modèle (LightGBM, XGBoost, règles métier...)
    expose la même interface — train.py, l'API et les tests n'ont
    jamais besoin de connaître le type concret du scorer.

    SOLID — principe I (refactoring) :
        evaluate() a été ajouté à l'interface car train.py l'appelle via
        l'abstraction scorer.evaluate(df_test). Sans cette déclaration,
        un futur XGBoostScorer pourrait omettre evaluate() et causer
        un AttributeError au runtime plutôt qu'une TypeError à l'instanciation.

    SOLID — principe L :
        Toute implémentation concrète (LGBMScorer, etc.) doit pouvoir
        remplacer BaseScorer sans modifier le comportement de l'appelant.
        evaluate() doit retourner un dict de métriques avec au minimum "rmse".
    """

    @abstractmethod
    def train(self, df: pd.DataFrame, target_col: str = "fraud_score") -> "BaseScorer":
        """
        Entraîne le modèle sur les features et la cible fournies.
        Retourne self pour le chaînage.
        """
        pass

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Prédit le score de fraude pour les données fournies.
        Retourne une Series de float ∈ [0.0, 1.0].
        """
        pass

    @abstractmethod
    def evaluate(self, df: pd.DataFrame, target_col: str = "fraud_score") -> dict:
        """
        Calcule les métriques de performance sur le DataFrame fourni.

        Retourne un dict avec au minimum :
            {
                "rmse":         float,
                "mae":          float,
                "r2":           float,
                "spearman_rho": float,
            }

        Pourquoi dans l'interface et pas seulement dans LGBMScorer ?
            train.py appelle scorer.evaluate() via l'abstraction.
            Sans cette déclaration, un futur scorer qui omet evaluate()
            lève AttributeError au runtime — pas à l'instanciation.
            La déclarer ici transforme l'erreur en TypeError explicite
            dès la création de l'objet.
        """
        pass

    @abstractmethod
    def save(self, filename: str) -> Path:
        """Sauvegarde le modèle entraîné sur le disque."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, filename: str) -> "BaseScorer":
        """Charge un modèle depuis le disque."""
        pass