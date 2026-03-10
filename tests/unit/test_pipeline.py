import pytest
import pandas as pd
from datetime import date
from pathlib import Path

from services.ml_engine.features.pipeline import FeaturePipeline
from services.ml_engine.features.temporal import TemporalFeatureTransformer
from services.ml_engine.features.base import BaseFeatureTransformer


# ─────────────────────────────────────────────────────────────────────────────
# Transformateur factice pour les tests
#
# Pourquoi ne pas utiliser TemporalFeatureTransformer directement ?
#   On teste le pipeline, pas le transformateur. En utilisant un
#   transformateur factice, on isole complètement le comportement
#   du pipeline. Si TemporalFeatureTransformer a un bug, les tests
#   du pipeline ne sont pas affectés.
#   C'est le principe d'isolation des tests unitaires.
# ─────────────────────────────────────────────────────────────────────────────

class AddColumnTransformer(BaseFeatureTransformer):
    """
    Transformateur factice : ajoute une colonne avec une valeur fixe.
    Permet de vérifier que le pipeline enchaîne bien les transformateurs.
    """
    def __init__(self, col_name: str, value):
        self.col_name = col_name
        self.value = value

    def fit(self, df: pd.DataFrame) -> "AddColumnTransformer":
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        result[self.col_name] = self.value
        return result


class FailTransformer(BaseFeatureTransformer):
    """
    Transformateur factice qui plante toujours.
    Permet de vérifier que le pipeline fail-fast correctement.
    """
    def fit(self, df: pd.DataFrame) -> "FailTransformer":
        raise RuntimeError("FailTransformer explose volontairement")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        raise RuntimeError("FailTransformer explose volontairement")


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def df_simple():
    """DataFrame minimal pour les tests."""
    return pd.DataFrame([
        {"dep_minutes": 503, "service_date": date(2024, 9, 2)},
        {"dep_minutes": 600, "service_date": date(2024, 9, 7)},
    ])


@pytest.fixture
def pipeline_un_transformateur():
    return FeaturePipeline([AddColumnTransformer("feature_A", 1.0)])


@pytest.fixture
def pipeline_deux_transformateurs():
    return FeaturePipeline([
        AddColumnTransformer("feature_A", 1.0),
        AddColumnTransformer("feature_B", 2.0),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFeaturePipeline:

    # ── Construction ─────────────────────────────────────────────────────────

    def test_init_valide(self, pipeline_un_transformateur):
        """Un pipeline avec un transformateur valide doit s'instancier."""
        assert pipeline_un_transformateur.n_transformers == 1

    def test_init_liste_vide_leve_erreur(self):
        """Une liste vide de transformateurs doit lever ValueError."""
        with pytest.raises(ValueError, match="ne peut pas être vide"):
            FeaturePipeline([])

    def test_init_mauvais_type_leve_erreur(self):
        """Un objet qui n'hérite pas de BaseFeatureTransformer doit lever TypeError."""
        with pytest.raises(TypeError, match="ne hérite pas de BaseFeatureTransformer"):
            FeaturePipeline(["pas_un_transformateur"])

    # ── fit() ────────────────────────────────────────────────────────────────

    def test_fit_retourne_self(self, pipeline_un_transformateur, df_simple):
        """fit() doit retourner self pour le chaînage."""
        result = pipeline_un_transformateur.fit(df_simple)
        assert result is pipeline_un_transformateur

    def test_fit_marque_pipeline_comme_fitte(self, pipeline_un_transformateur, df_simple):
        """Après fit(), _is_fitted doit être True."""
        assert pipeline_un_transformateur._is_fitted == False
        pipeline_un_transformateur.fit(df_simple)
        assert pipeline_un_transformateur._is_fitted == True

    # ── transform() ──────────────────────────────────────────────────────────

    def test_transform_sans_fit_leve_erreur(self, pipeline_un_transformateur, df_simple):
        """transform() sans fit() préalable doit lever RuntimeError."""
        with pytest.raises(RuntimeError, match="n'a pas encore été fitté"):
            pipeline_un_transformateur.transform(df_simple)

    def test_transform_ajoute_colonne(self, pipeline_un_transformateur, df_simple):
        """Le transformateur doit ajouter sa colonne au DataFrame."""
        result = pipeline_un_transformateur.fit_transform(df_simple)
        assert "feature_A" in result.columns

    def test_transform_valeur_correcte(self, pipeline_un_transformateur, df_simple):
        """La valeur ajoutée doit être celle configurée dans le transformateur."""
        result = pipeline_un_transformateur.fit_transform(df_simple)
        assert (result["feature_A"] == 1.0).all()

    def test_input_non_modifie(self, pipeline_un_transformateur, df_simple):
        """transform() ne doit pas modifier le DataFrame d'entrée."""
        colonnes_avant = list(df_simple.columns)
        pipeline_un_transformateur.fit_transform(df_simple)
        assert list(df_simple.columns) == colonnes_avant

    # ── Enchaînement des transformateurs ─────────────────────────────────────

    def test_deux_transformateurs_deux_colonnes(self, pipeline_deux_transformateurs, df_simple):
        """Deux transformateurs doivent produire deux nouvelles colonnes."""
        result = pipeline_deux_transformateurs.fit_transform(df_simple)
        assert "feature_A" in result.columns
        assert "feature_B" in result.columns

    def test_ordre_transformateurs_respecte(self, df_simple):
        """
        Le second transformateur doit voir les colonnes produites
        par le premier.
        On vérifie ça en créant un transformateur qui lit feature_A
        pour produire feature_B.
        """
        class DoubleTransformer(BaseFeatureTransformer):
            """Multiplie feature_A par 2 pour produire feature_B."""
            def fit(self, df): return self
            def transform(self, df):
                result = df.copy()
                result["feature_B"] = result["feature_A"] * 2
                return result

        pipeline = FeaturePipeline([
            AddColumnTransformer("feature_A", 3.0),
            DoubleTransformer(),
        ])
        result = pipeline.fit_transform(df_simple)
        assert (result["feature_B"] == 6.0).all()

    # ── Propriétés utilitaires ────────────────────────────────────────────────

    def test_n_transformers(self, pipeline_deux_transformateurs):
        assert pipeline_deux_transformateurs.n_transformers == 2

    def test_feature_names(self, pipeline_deux_transformateurs):
        names = pipeline_deux_transformateurs.feature_names
        assert names == ["AddColumnTransformer", "AddColumnTransformer"]

    def test_repr_not_fitted(self, pipeline_un_transformateur):
        assert "not fitted" in repr(pipeline_un_transformateur)

    def test_repr_fitted(self, pipeline_un_transformateur, df_simple):
        pipeline_un_transformateur.fit(df_simple)
        assert "fitted" in repr(pipeline_un_transformateur)

    # ── Intégration avec TemporalFeatureTransformer ───────────────────────────

    def test_avec_temporal_transformer(self):
        """
        Test d'intégration : le pipeline doit fonctionner avec le vrai
        TemporalFeatureTransformer sans erreur.
        """
        df = pd.DataFrame([
            {"dep_minutes": 503, "service_date": date(2024, 9, 2)},
            {"dep_minutes": 600, "service_date": date(2024, 9, 7)},
        ])
        pipeline = FeaturePipeline([TemporalFeatureTransformer()])
        result = pipeline.fit_transform(df)

        # Toutes les features temporelles doivent être présentes
        assert "dep_hour" in result.columns
        assert "is_weekend" in result.columns
        assert "is_vacances" in result.columns
        assert "is_peak_hour" in result.columns

    # ── Persistance joblib ────────────────────────────────────────────────────

    def test_save_cree_fichier(self, pipeline_un_transformateur, df_simple, tmp_path, monkeypatch):
        """
        save() doit créer un fichier .joblib sur le disque.

        tmp_path : fixture pytest qui fournit un dossier temporaire
                   propre pour chaque test. Supprimé automatiquement après.
        monkeypatch : fixture pytest qui permet de remplacer temporairement
                      config.MODELS_DIR par tmp_path pour ce test uniquement.
        """
        from shared import config as cfg
        monkeypatch.setattr(cfg.config, "MODELS_DIR", tmp_path)

        pipeline_un_transformateur.fit(df_simple)
        save_path = pipeline_un_transformateur.save("test_pipeline.joblib")

        assert save_path.exists()
        assert save_path.suffix == ".joblib"

    def test_load_recharge_pipeline(self, pipeline_un_transformateur, df_simple, tmp_path, monkeypatch):
        """
        Un pipeline sauvegardé puis rechargé doit produire les mêmes
        résultats que l'original.
        """
        from shared import config as cfg
        monkeypatch.setattr(cfg.config, "MODELS_DIR", tmp_path)

        pipeline_un_transformateur.fit(df_simple)
        pipeline_un_transformateur.save("test_pipeline.joblib")

        pipeline_rechargé = FeaturePipeline.load("test_pipeline.joblib")
        result = pipeline_rechargé.transform(df_simple)

        assert "feature_A" in result.columns

    def test_load_fichier_inexistant_leve_erreur(self, tmp_path, monkeypatch):
        """Charger un fichier inexistant doit lever FileNotFoundError."""
        from shared import config as cfg
        monkeypatch.setattr(cfg.config, "MODELS_DIR", tmp_path)

        with pytest.raises(FileNotFoundError, match="Fichier introuvable"):
            FeaturePipeline.load("inexistant.joblib")