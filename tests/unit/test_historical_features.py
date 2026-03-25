import pytest
import pandas as pd
from services.ml_engine.features.historical import HistoricalFeatureTransformer


class TestHistoricalFeatureTransformer:

    def setup_method(self):
        """Initialise le transformateur avant chaque test."""
        self.transformer = HistoricalFeatureTransformer()

    @pytest.fixture
    def sample_train_data(self):
        """Données d'entraînement fictives (LAF historique) avec la cible."""
        return pd.DataFrame([
            {'stop_id_dep': '87212027', 'stop_id_arr': '87214007', 'fraud_score': 0.8},
            {'stop_id_dep': '87212027', 'stop_id_arr': '87214007', 'fraud_score': 0.4},
            {'stop_id_dep': '87214007', 'stop_id_arr': '87214080', 'fraud_score': 0.2},
            {'stop_id_dep': '87214080', 'stop_id_arr': '87214288', 'fraud_score': 0.4},
        ])

    @pytest.fixture
    def sample_test_data(self):
        """Données de production fictives (GTFS du jour) SANS la cible."""
        return pd.DataFrame([
            {'stop_id_dep': '87212027', 'stop_id_arr': '87214007'},
            {'stop_id_dep': '87214007', 'stop_id_arr': '87214080'},
            {'stop_id_dep': '99999999', 'stop_id_arr': '88888888'},
        ])

    # --- TESTS DE LA MÉTHODE FIT ---

    def test_fit_returns_self(self, sample_train_data):
        result = self.transformer.fit(sample_train_data)
        assert result is self.transformer

    def test_fit_missing_target_raises(self, sample_test_data):
        with pytest.raises(ValueError, match="La colonne 'fraud_score' est requise"):
            self.transformer.fit(sample_test_data)

    def test_fit_calculates_means_correctly(self, sample_train_data):
        self.transformer.fit(sample_train_data)

        # CORRECTION ICI : utilisation de pytest.approx
        assert self.transformer.global_mean == pytest.approx(0.45)

        stats = self.transformer.historical_segment_stats
        # CORRECTION ICI
        assert stats[('87212027', '87214007')] == pytest.approx(0.6)
        assert stats[('87214007', '87214080')] == pytest.approx(0.2)
        assert stats[('87214080', '87214288')] == pytest.approx(0.4)

    # --- TESTS DE LA MÉTHODE TRANSFORM ---

    def test_transform_missing_columns_raises(self):
        df_incomplet = pd.DataFrame([{'stop_id_dep': '87212027'}])
        with pytest.raises(ValueError, match="Colonnes manquantes pour transform"):
            self.transformer.transform(df_incomplet)

    def test_input_not_modified(self, sample_train_data, sample_test_data):
        self.transformer.fit(sample_train_data)
        original_cols = list(sample_test_data.columns)
        self.transformer.transform(sample_test_data)
        assert list(sample_test_data.columns) == original_cols

    def test_transform_applies_correct_scores_and_fallback(self, sample_train_data, sample_test_data):
        self.transformer.fit(sample_train_data)
        result = self.transformer.transform(sample_test_data)

        # CORRECTION ICI : utilisation de pytest.approx
        assert result.iloc[0]['hist_fraud_score_segment'] == pytest.approx(0.6)
        assert result.iloc[1]['hist_fraud_score_segment'] == pytest.approx(0.2)
        assert result.iloc[2]['hist_fraud_score_segment'] == pytest.approx(0.45)