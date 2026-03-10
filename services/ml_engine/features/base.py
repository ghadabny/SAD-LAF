from abc import ABC, abstractmethod
import pandas as pd


class BaseFeatureTransformer(ABC):
    """
    Contrat commun pour tous les transformateurs de features du projet.

    Toutes les classes de features (TemporalFeatures, HistoricalFeatures...)
    héritent de cette classe et implémentent obligatoirement fit() et transform().

    Pourquoi fit() et transform() séparément ?
        - fit()       : apprend des paramètres sur les données d'entraînement
                        (ex: moyenne, min/max, liste des jours fériés vus...)
        - transform() : applique la transformation sur n'importe quel DataFrame
                        (entraînement OU production)

    Cette séparation est la convention scikit-learn. Elle garantit qu'on
    n'apprend jamais de paramètres sur les données de test (data leakage).

    Pour TemporalFeatures spécifiquement, fit() ne fait rien (les features
    temporelles ne nécessitent pas d'apprentissage), mais on respecte
    l'interface pour que le pipeline.py puisse traiter tous les transformateurs
    de façon uniforme.
    """

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "BaseFeatureTransformer":
        """
        Apprend les paramètres nécessaires à la transformation.
        Doit toujours retourner self pour permettre le chaînage :
            transformer.fit(df_train).transform(df_test)
        """
        ...

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applique la transformation et retourne un nouveau DataFrame
        enrichi des features calculées.
        Ne modifie jamais le DataFrame d'entrée (toujours .copy()).
        """
        ...

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Raccourci : fit() puis transform() sur le même DataFrame.
        Implémenté ici une fois pour toutes — les sous-classes n'ont
        pas besoin de le redéfinir.
        """
        return self.fit(df).transform(df)