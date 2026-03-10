import joblib
import pandas as pd
from pathlib import Path

from services.ml_engine.features.base import BaseFeatureTransformer
from shared.config import config


class FeaturePipeline:
    """
    Orchestre l'application séquentielle de plusieurs transformateurs
    de features sur un DataFrame de tronçons.

    Responsabilité unique : appliquer les transformateurs dans l'ordre
    et gérer la persistance sur disque.
    Ne sait pas ce que font les transformateurs — il les appelle juste.

    Principe O de SOLID :
        Pour ajouter HistoricalFeatureTransformer demain, on passe
        juste une liste plus longue au constructeur.
        pipeline.py lui-même ne change jamais.

    Usage :
        # Entraînement
        pipeline = FeaturePipeline([
            TemporalFeatureTransformer(),
            HistoricalFeatureTransformer(),  # quand les données LAF arrivent
        ])
        df_enrichi = pipeline.fit_transform(df_troncons)
        pipeline.save()

        # Production (recharge depuis disque)
        pipeline = FeaturePipeline.load()
        df_enrichi = pipeline.transform(df_troncons)
    """

    def __init__(self, transformers: list[BaseFeatureTransformer]):
        """
        Paramètres :
            transformers : liste ordonnée de transformateurs.
                           L'ordre est important — chaque transformateur
                           reçoit le DataFrame enrichi par le précédent.
                           Ex: TemporalFeatureTransformer doit être avant
                           HistoricalFeatureTransformer si ce dernier
                           utilise des features temporelles.
        """
        if not transformers:
            raise ValueError(
                "[FeaturePipeline] La liste de transformateurs ne peut pas "
                "être vide."
            )

        # Vérifie que tous les éléments sont bien des BaseFeatureTransformer
        for i, t in enumerate(transformers):
            if not isinstance(t, BaseFeatureTransformer):
                raise TypeError(
                    f"[FeaturePipeline] Le transformateur à l'index {i} "
                    f"({type(t).__name__}) ne hérite pas de "
                    f"BaseFeatureTransformer."
                )

        self.transformers = transformers
        self._is_fitted = False

    # ── Interface principale ──────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "FeaturePipeline":
        """
        Appelle fit() sur chaque transformateur dans l'ordre.

        Pourquoi on passe le DataFrame transformé au fit suivant ?
            Certains transformateurs futurs pourraient avoir besoin
            de statistiques calculées sur les features déjà produites.
            Ex: un transformateur qui normalise dep_hour aurait besoin
            que dep_hour existe déjà dans le DataFrame au moment du fit.
        """
        df_courant = df.copy()
        for transformer in self.transformers:
            transformer.fit(df_courant)
            # On transforme pour que le prochain transformateur
            # voie les features déjà produites
            df_courant = transformer.transform(df_courant)

        self._is_fitted = True
        print(
            f"[FeaturePipeline] fit() terminé sur {len(self.transformers)} "
            f"transformateur(s) : "
            f"{[type(t).__name__ for t in self.transformers]}"
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applique chaque transformateur dans l'ordre sur le DataFrame.
        La sortie de l'un devient l'entrée du suivant.

        Ne modifie pas le DataFrame d'entrée (chaque transformateur
        travaille sur une copie interne).
        """
        self._check_fitted()

        df_courant = df
        for transformer in self.transformers:
            df_courant = transformer.transform(df_courant)

        print(
            f"[FeaturePipeline] transform() : "
            f"{len(df)} lignes → "
            f"{len(df_courant.columns)} colonnes "
            f"(+{len(df_courant.columns) - len(df.columns)} features ajoutées)"
        )
        return df_courant

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Raccourci : fit() puis transform() sur le même DataFrame.
        Utilisé pendant l'entraînement.
        """
        return self.fit(df).transform(df)

    # ── Persistance ───────────────────────────────────────────────────────────

    def save(self, filename: str = "feature_pipeline.joblib") -> Path:
        """
        Sauvegarde le pipeline fitté sur le disque avec joblib.

        Pourquoi joblib plutôt que pickle ?
            joblib est optimisé pour les objets numpy/pandas — il compresse
            mieux et est plus rapide pour les gros tableaux numériques.
            C'est la convention scikit-learn.

        Paramètres :
            filename : nom du fichier de sauvegarde.
                       Par défaut dans config.MODELS_DIR.

        Retourne :
            Path vers le fichier sauvegardé.
        """
        self._check_fitted()

        save_path = config.MODELS_DIR / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self, save_path)
        print(f"[FeaturePipeline] Pipeline sauvegardé → {save_path}")
        return save_path

    @classmethod
    def load(cls, filename: str = "feature_pipeline.joblib") -> "FeaturePipeline":
        """
        Charge un pipeline depuis le disque.

        @classmethod : cette méthode est appelée sur la classe, pas sur
        une instance. On peut écrire FeaturePipeline.load() sans avoir
        d'objet existant — c'est exactement ce qu'on veut en production.

        Lève FileNotFoundError si le fichier n'existe pas.
        """
        load_path = config.MODELS_DIR / filename

        if not load_path.exists():
            raise FileNotFoundError(
                f"[FeaturePipeline] Fichier introuvable : {load_path}\n"
                f"Lance d'abord pipeline.fit_transform(df).save() "
                f"pour créer le fichier."
            )

        pipeline = joblib.load(load_path)
        print(f"[FeaturePipeline] Pipeline chargé depuis → {load_path}")
        return pipeline

    # ── Méthodes utilitaires ──────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        """
        Vérifie que fit() a bien été appelé avant transform() ou save().
        Fail-fast avec un message clair.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "[FeaturePipeline] Le pipeline n'a pas encore été fitté.\n"
                "Appelle fit() ou fit_transform() avant transform() ou save()."
            )

    @property
    def n_transformers(self) -> int:
        """Nombre de transformateurs dans le pipeline."""
        return len(self.transformers)

    @property
    def feature_names(self) -> list[str]:
        """
        Retourne les noms des transformateurs dans l'ordre.
        Utile pour les logs et le débogage.
        """
        return [type(t).__name__ for t in self.transformers]

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        steps = " → ".join(self.feature_names)
        return f"FeaturePipeline({steps}) [{status}]"