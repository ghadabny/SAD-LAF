# services/ml_engine/features/pipeline.py
import joblib
import pandas as pd
from pathlib import Path
from typing import Optional

from services.ml_engine.features.base import BaseFeatureTransformer
from shared.config import config


class FeaturePipeline:
    """
    Orchestre l'application séquentielle de plusieurs transformateurs
    de features sur un DataFrame de tronçons.

    Responsabilité unique : appliquer les transformateurs dans l'ordre
    et gérer la persistance sur disque.
    Ne sait pas ce que font les transformateurs — il les appelle juste.

    SOLID — principe O :
        Pour ajouter un transformateur, on passe une liste plus longue au
        constructeur. pipeline.py lui-même ne change jamais.

    SOLID — principe D (refactoring) :
        models_dir est maintenant injectable via le constructeur.
        Avant ce refactoring, config.MODELS_DIR était importé directement,
        ce qui obligeait à monkeypatching dans les tests.
        Avec l'injection, les tests passent simplement tmp_path sans modifier
        l'état global de config.

    Usage :
        # Production (chemin par défaut depuis config)
        pipeline = FeaturePipeline([TemporalFeatureTransformer(), ...])

        # Tests (chemin temporaire injecté)
        pipeline = FeaturePipeline([...], models_dir=tmp_path)
    """

    def __init__(
        self,
        transformers: list[BaseFeatureTransformer],
        models_dir: Optional[Path] = None,
    ):
        """
        Paramètres :
            transformers : liste ordonnée de transformateurs.
                           L'ordre est important — chaque transformateur
                           reçoit le DataFrame enrichi par le précédent.
            models_dir   : répertoire de sauvegarde/chargement des .joblib.
                           Par défaut : config.MODELS_DIR.
                           Injectez un Path différent pour les tests ou
                           pour supporter plusieurs environnements.
        """
        if not transformers:
            raise ValueError(
                "[FeaturePipeline] La liste de transformateurs ne peut pas "
                "être vide."
            )

        for i, t in enumerate(transformers):
            if not isinstance(t, BaseFeatureTransformer):
                raise TypeError(
                    f"[FeaturePipeline] Le transformateur à l'index {i} "
                    f"({type(t).__name__}) ne hérite pas de "
                    f"BaseFeatureTransformer."
                )

        self.transformers  = transformers
        self._models_dir   = models_dir  # None = utilise config.MODELS_DIR au moment du save/load
        self._is_fitted    = False

    # ── Propriété : résolution tardive du chemin ──────────────────────────────

    @property
    def models_dir(self) -> Path:
        """
        Résout le répertoire de modèles au moment de l'appel.

        Résolution tardive (lazy) : si models_dir n'a pas été injecté,
        on lit config.MODELS_DIR au moment du save() ou load(), pas au
        moment de la construction. Cela permet de modifier config.MODELS_DIR
        après instanciation sans recréer le pipeline.
        """
        return self._models_dir if self._models_dir is not None else config.MODELS_DIR

    # ── Interface principale ──────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "FeaturePipeline":
        """
        Appelle fit() sur chaque transformateur dans l'ordre.

        On passe le DataFrame transformé au fit suivant pour que chaque
        transformateur puisse utiliser les features déjà produites.
        """
        df_courant = df.copy()
        for transformer in self.transformers:
            transformer.fit(df_courant)
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
        """Raccourci : fit() puis transform() sur le même DataFrame."""
        return self.fit(df).transform(df)

    # ── Persistance ───────────────────────────────────────────────────────────

    def save(self, filename: str = "feature_pipeline.joblib") -> Path:
        """
        Sauvegarde le pipeline fitté sur le disque avec joblib.

        Utilise self.models_dir (injecté ou config.MODELS_DIR).
        """
        self._check_fitted()

        save_path = self.models_dir / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self, save_path)
        print(f"[FeaturePipeline] Pipeline sauvegardé → {save_path}")
        return save_path

    @classmethod
    def load(
        cls,
        filename: str = "feature_pipeline.joblib",
        models_dir: Optional[Path] = None,
    ) -> "FeaturePipeline":
        """
        Charge un pipeline depuis le disque.

        Paramètres :
            filename   : nom du fichier .joblib
            models_dir : répertoire source. Par défaut : config.MODELS_DIR.
                         Injectez un Path pour les tests ou environnements alternatifs.

        Lève FileNotFoundError si le fichier n'existe pas.
        """
        load_dir  = models_dir if models_dir is not None else config.MODELS_DIR
        load_path = load_dir / filename

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
        if not self._is_fitted:
            raise RuntimeError(
                "[FeaturePipeline] Le pipeline n'a pas encore été fitté.\n"
                "Appelle fit() ou fit_transform() avant transform() ou save()."
            )

    @property
    def n_transformers(self) -> int:
        return len(self.transformers)

    @property
    def feature_names(self) -> list[str]:
        return [type(t).__name__ for t in self.transformers]

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        steps  = " → ".join(self.feature_names)
        return f"FeaturePipeline({steps}) [{status}]"