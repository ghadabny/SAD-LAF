import pandas as pd
from services.ml_engine.features.base import BaseFeatureTransformer


class HistoricalFeatureTransformer(BaseFeatureTransformer):
    """
    Transformateur qui ajoute des statistiques historiques de fraude
    (ex: taux de fraude moyen par tronçon) aux données GTFS du jour.
    """

    def __init__(self):
        # Dictionnaire pour un accès en O(1) lors du transform
        self.historical_segment_stats: dict[tuple[str, str], float] = {}
        self.global_mean: float = 0.0

    def fit(self, df: pd.DataFrame) -> 'HistoricalFeatureTransformer':
        """
        Apprend les statistiques historiques sur les données d'entraînement.
        Le DataFrame d'entrée DOIT contenir la colonne 'fraud_score' (Vérité terrain LAF).
        """
        if 'fraud_score' not in df.columns:
            # Si on n'a pas de cible (ex: on est en production sur les données de demain),
            # on lève une erreur claire (principe Fail-Fast)
            raise ValueError("La colonne 'fraud_score' est requise pour fiter l'historique.")

        # 1. Calcul de la moyenne globale (fallback pour les tronçons jamais vus)
        self.global_mean = df['fraud_score'].mean()

        # 2. Calcul de la moyenne par tronçon (Origine -> Destination)
        # On groupe par gares de départ et d'arrivée et on fait la moyenne du taux de fraude
        stats = df.groupby(['stop_id_dep', 'stop_id_arr'])['fraud_score'].mean().reset_index()

        # 3. Stockage sous forme de dictionnaire pour des performances optimales
        # Clé : (stop_id_dep, stop_id_arr), Valeur : moyenne de fraude
        self.historical_segment_stats = stats.set_index(['stop_id_dep', 'stop_id_arr'])['fraud_score'].to_dict()

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applique les statistiques apprises au DataFrame fourni.
        """
        # Vérification Fail-fast des colonnes requises
        required_cols = {'stop_id_dep', 'stop_id_arr'}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Colonnes manquantes pour transform(). Requis : {required_cols}")

        # Copie défensive : on ne modifie JAMAIS le DataFrame d'entrée [cite: 662]
        result = df.copy()

        # Fonction de mapping rapide
        def get_historical_score(row) -> float:
            key = (row['stop_id_dep'], row['stop_id_arr'])
            # Retourne la stat du tronçon, ou la moyenne globale si le tronçon est inconnu
            return self.historical_segment_stats.get(key, self.global_mean)

        # Création de la nouvelle feature : 'hist_fraud_score_segment'
        result['hist_fraud_score_segment'] = result.apply(get_historical_score, axis=1)

        return result