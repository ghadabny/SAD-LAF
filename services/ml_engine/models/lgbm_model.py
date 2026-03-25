from services.ml_engine.models.base import BaseScorer
import lightgbm as lgb
import pandas as pd
from pathlib import Path
import joblib
from shared.config import config

class LGBMScorer(BaseScorer):
        """
        Encapsule le modèle LightGBM pour la prédiction du taux de fraude par tronçon.
        """

        def __init__(self, **kwargs):
            # Hyperparamètres par défaut pour la régression (prédire un taux entre 0 et 1)
            self.params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'learning_rate': 0.05,
                'num_leaves': 31,
                'verbose': -1
            }
            # Permet d'écraser les paramètres par défaut si besoin
            self.params.update(kwargs)
            self.model = None

            # Liste des features que le modèle va utiliser
            self.feature_cols = [
                'dep_hour', 'day_of_week', 'is_weekend',
                'is_vacances', 'is_jour_ferie', 'is_peak_hour',
                'hist_fraud_score_segment'
            ]

        def train(self, df: pd.DataFrame, target_col: str = 'fraud_score') -> 'LGBMScorer':
            """
            Entraîne le modèle LightGBM sur les features extraites.
            """
            # Vérification Fail-Fast
            missing_cols = set(self.feature_cols + [target_col]) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Colonnes manquantes pour l'entraînement : {missing_cols}")

            X = df[self.feature_cols]
            y = df[target_col]

            # Création du dataset LightGBM
            lgb_data = lgb.Dataset(X, label=y)

            # Entraînement
            self.model = lgb.train(
                self.params,
                lgb_data,
                num_boost_round=100
            )
            return self

        def predict(self, df: pd.DataFrame) -> pd.Series:
            """
            Prédit le score de fraude pour de nouveaux tronçons.
            Retourne une Series de scores bornés entre 0.0 et 1.0.
            """
            if self.model is None:
                raise RuntimeError("Le modèle doit être entraîné (train) ou chargé (load) avant de prédire.")

            missing_cols = set(self.feature_cols) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Colonnes manquantes pour la prédiction : {missing_cols}")

            X = df[self.feature_cols]

            # Prédiction brute
            predictions = self.model.predict(X)

            # On s'assure que les prédictions restent entre 0 et 1 (taux de fraude)
            predictions = pd.Series(predictions).clip(lower=0.0, upper=1.0)

            return predictions

        def save(self, filename: str = 'lgbm_scorer.joblib') -> Path:
            """Sauvegarde le modèle entraîné sur le disque."""
            if self.model is None:
                raise RuntimeError("Impossible de sauvegarder un modèle non entraîné.")

            save_path = config.MODELS_DIR / filename
            save_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self, save_path)
            return save_path

        @classmethod
        def load(cls, filename: str = 'lgbm_scorer.joblib') -> 'LGBMScorer':
            """Charge un modèle depuis le disque."""
            load_path = config.MODELS_DIR / filename
            if not load_path.exists():
                raise FileNotFoundError(f"Fichier modèle introuvable : {load_path}")
            return joblib.load(load_path)