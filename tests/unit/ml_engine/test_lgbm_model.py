# tests/unit/test_lgbm_model.py
"""
Tests unitaires de LGBMScorer.

Corrections apportées :
  1. test_train_missing_columns_raises :
       Avant : le test supprimait 'dep_hour' et attendait ValueError.
               MAIS LGBMScorer sélectionne automatiquement les features numériques
               disponibles → il entraîne sur les colonnes restantes sans lever d'erreur.
       Après : on supprime TOUTES les colonnes numériques utiles pour forcer
               le cas "Aucune feature numérique trouvée après exclusion".
               Message attendu : "Aucune feature numérique trouvée".

  2. test_predict_before_train_raises :
       Avant regex : "Le modèle doit être entraîné"
       Message réel : "[LGBMScorer] Le modèle n'est pas entraîné."
       Après regex  : "n'est pas entraîné" (sous-chaîne robuste)
"""
import pytest
import pandas as pd
import numpy as np
from services.ml_engine.models.lgbm_model import LGBMScorer


class TestLGBMScorer:

    def setup_method(self):
        """Initialise le scorer avant chaque test."""
        self.scorer = LGBMScorer(verbose=-1)

    @pytest.fixture
    def sample_data(self):
        """Génère un faux jeu de données avec toutes les colonnes requises."""
        np.random.seed(42)
        n_samples = 100
        return pd.DataFrame({
            "dep_hour":               np.random.randint(0, 24, n_samples),
            "day_of_week":            np.random.randint(0, 7, n_samples),
            "is_weekend":             np.random.choice([True, False], n_samples),
            "is_vacances":            np.random.choice([True, False], n_samples),
            "is_jour_ferie":          np.random.choice([True, False], n_samples),
            "is_peak_hour":           np.random.choice([True, False], n_samples),
            "hist_nb_controles":      np.random.uniform(10, 300, n_samples),
            "hist_log_controles":     np.random.uniform(2.0, 6.0, n_samples),
            "hist_pct_pv_tariff":     np.random.uniform(0.0, 1.0, n_samples),
            "fraud_score":            np.random.uniform(0, 1, n_samples),  # cible
        })

    # ── Tests d'entraînement ──────────────────────────────────────────────────

    def test_train_success(self, sample_data):
        """Vérifie que l'entraînement se passe bien et retourne self."""
        result = self.scorer.train(sample_data)
        assert result is self.scorer
        assert self.scorer.model is not None

    def test_train_missing_columns_raises(self, sample_data):
        """
        Vérifie le Fail-Fast quand il n'y a plus aucune feature numérique exploitable.

        CORRECTION : LGBMScorer sélectionne automatiquement les features
        numériques disponibles (il ne vérifie pas une liste fixe de features
        obligatoires). Pour forcer le ValueError, on supprime TOUTES les colonnes
        numériques utiles, ne laissant que la cible 'fraud_score'.

        Message attendu : "Aucune feature numérique trouvée après exclusion."
        """
        # On ne garde que la cible — LGBMScorer ne peut pas entraîner sans features
        df_sans_features = sample_data[["fraud_score"]].copy()
        with pytest.raises(ValueError, match="Aucune feature numérique trouvée"):
            self.scorer.train(df_sans_features)

    def test_train_missing_target_raises(self, sample_data):
        """Vérifie le Fail-Fast si la colonne cible 'fraud_score' est absente."""
        df_sans_cible = sample_data.drop(columns=["fraud_score"])
        with pytest.raises(ValueError, match="Colonne cible 'fraud_score' absente"):
            self.scorer.train(df_sans_cible)

    def test_train_selects_features_automatically(self, sample_data):
        """
        Vérifie que feature_cols est bien rempli après entraînement
        et ne contient pas la cible ni les colonnes exclues.
        """
        self.scorer.train(sample_data)
        assert len(self.scorer.feature_cols) > 0
        assert "fraud_score" not in self.scorer.feature_cols

    # ── Tests de prédiction ───────────────────────────────────────────────────

    def test_predict_before_train_raises(self, sample_data):
        """
        On ne peut pas prédire sans modèle entraîné.

        CORRECTION : le message d'erreur réel est
        "[LGBMScorer] Le modèle n'est pas entraîné. Appelle train() ou charge..."
        Regex corrigée pour correspondre à ce message.
        """
        df_test = sample_data.drop(columns=["fraud_score"])
        with pytest.raises(RuntimeError, match="n'est pas entraîné"):
            self.scorer.predict(df_test)

    def test_predict_success_and_clipping(self, sample_data):
        """Vérifie que la prédiction fonctionne et reste bornée entre 0 et 1."""
        self.scorer.train(sample_data)
        df_test = sample_data.drop(columns=["fraud_score"]).head(10)

        predictions = self.scorer.predict(df_test)

        assert isinstance(predictions, pd.Series)
        assert len(predictions) == 10
        assert (predictions >= 0.0).all()
        assert (predictions <= 1.0).all()

    def test_predict_missing_feature_raises(self, sample_data):
        """
        Vérifie que predict() lève ValueError si une feature du train
        est absente au moment de l'inférence.
        """
        self.scorer.train(sample_data)
        # Supprimer une feature que le modèle a utilisée
        feature_utilisee = self.scorer.feature_cols[0]
        df_incomplet = sample_data.drop(columns=["fraud_score", feature_utilisee])
        with pytest.raises(ValueError, match="Colonnes manquantes pour la prédiction"):
            self.scorer.predict(df_incomplet)

    # ── Tests de sauvegarde et chargement ────────────────────────────────────

    def test_save_and_load(self, sample_data, tmp_path, monkeypatch):
        """
        Vérifie la persistance joblib en isolant le test dans un dossier temp.
        """
        monkeypatch.setattr(
            "services.ml_engine.models.lgbm_model.config.MODELS_DIR", tmp_path
        )

        self.scorer.train(sample_data)
        save_path = self.scorer.save("test_model.joblib")
        assert save_path.exists()

        loaded_scorer = LGBMScorer.load("test_model.joblib")

        df_test = sample_data.drop(columns=["fraud_score"]).head(5)
        orig_preds   = self.scorer.predict(df_test)
        loaded_preds = loaded_scorer.predict(df_test)

        pd.testing.assert_series_equal(orig_preds, loaded_preds)