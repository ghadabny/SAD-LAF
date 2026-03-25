import pytest
import pandas as pd
import numpy as np
from services.ml_engine.models.lgbm_model import LGBMScorer


class TestLGBMScorer:

    def setup_method(self):
        """Initialise le scorer avant chaque test."""
        # On peut passer des kwargs pour écraser les paramètres si besoin
        self.scorer = LGBMScorer(verbose=-1)

    @pytest.fixture
    def sample_data(self):
        """Génère un faux jeu de données avec toutes les colonnes requises."""
        np.random.seed(42)  # Pour la reproductibilité
        n_samples = 100
        return pd.DataFrame({
            'dep_hour': np.random.randint(0, 24, n_samples),
            'day_of_week': np.random.randint(0, 7, n_samples),
            'is_weekend': np.random.choice([True, False], n_samples),
            'is_vacances': np.random.choice([True, False], n_samples),
            'is_jour_ferie': np.random.choice([True, False], n_samples),
            'is_peak_hour': np.random.choice([True, False], n_samples),
            'hist_fraud_score_segment': np.random.uniform(0, 1, n_samples),
            'fraud_score': np.random.uniform(0, 1, n_samples)  # La cible
        })

    # --- TESTS D'ENTRAÎNEMENT (TRAIN) ---

    def test_train_success(self, sample_data):
        """Vérifie que l'entraînement se passe bien et retourne self."""
        result = self.scorer.train(sample_data)
        assert result is self.scorer
        assert self.scorer.model is not None

    def test_train_missing_columns_raises(self, sample_data):
        """Vérifie le Fail-Fast s'il manque des features pour entraîner."""
        df_incomplet = sample_data.drop(columns=['dep_hour'])
        with pytest.raises(ValueError, match="Colonnes manquantes pour l'entraînement"):
            self.scorer.train(df_incomplet)

    # --- TESTS DE PRÉDICTION (PREDICT) ---

    def test_predict_before_train_raises(self, sample_data):
        """On ne peut pas prédire sans modèle."""
        df_test = sample_data.drop(columns=['fraud_score'])
        with pytest.raises(RuntimeError, match="Le modèle doit être entraîné"):
            self.scorer.predict(df_test)

    def test_predict_success_and_clipping(self, sample_data):
        """Vérifie que la prédiction fonctionne et reste bornée entre 0 et 1."""
        self.scorer.train(sample_data)
        df_test = sample_data.drop(columns=['fraud_score']).head(10)

        predictions = self.scorer.predict(df_test)

        assert isinstance(predictions, pd.Series)
        assert len(predictions) == 10
        # Vérification vitale : les scores doivent être des probabilités/taux
        assert (predictions >= 0.0).all()
        assert (predictions <= 1.0).all()

    # --- TESTS DE SAUVEGARDE ET CHARGEMENT (SAVE / LOAD) ---

    def test_save_and_load(self, sample_data, tmp_path, monkeypatch):
        """
        Vérifie la persistance joblib en isolant le test dans un dossier temp.
        """
        # On remplace config.MODELS_DIR par le dossier temporaire de pytest
        # Assure-toi que le chemin d'import correspond bien à l'endroit où tu utilises config
        monkeypatch.setattr('services.ml_engine.models.lgbm_model.config.MODELS_DIR', tmp_path)

        # 1. Entraînement et sauvegarde
        self.scorer.train(sample_data)
        save_path = self.scorer.save('test_model.joblib')
        assert save_path.exists()

        # 2. Chargement dans une nouvelle instance
        loaded_scorer = LGBMScorer.load('test_model.joblib')

        # 3. Vérification : le modèle rechargé fait les mêmes prédictions
        df_test = sample_data.drop(columns=['fraud_score']).head(5)
        orig_preds = self.scorer.predict(df_test)
        loaded_preds = loaded_scorer.predict(df_test)

        pd.testing.assert_series_equal(orig_preds, loaded_preds)