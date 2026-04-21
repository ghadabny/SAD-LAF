# tests/unit/test_historical_features.py
"""
Tests unitaires de HistoricalFeatureTransformer (v3 anti-leakage).

Corrections apportées par rapport à la version précédente :
    - L'ancienne API attendait fraud_score comme colonne cible dans fit().
      La v3 attend nb_controles + pct_pv_tariff (features de VOLUME, pas la cible).
    - Les features produites sont maintenant hist_nb_controles, hist_log_controles,
      hist_pct_pv_tariff (et non plus hist_fraud_score_segment).
    - Les attributs internes sont _stats (dict) et _fallback (dict),
      et non plus global_mean / historical_segment_stats.
"""
import math

import pytest
import pandas as pd

from services.ml_engine.features.historical import HistoricalFeatureTransformer


class TestHistoricalFeatureTransformer:

    def setup_method(self):
        """Initialise le transformateur avant chaque test."""
        self.transformer = HistoricalFeatureTransformer()

    # ─────────────────────────────────────────────────────────────────────────
    # Fixtures
    # ─────────────────────────────────────────────────────────────────────────

    @pytest.fixture
    def sample_train_data(self):
        """
        Données d'entraînement fictives (LAF historique) dans le FORMAT v3.
        Colonnes sources : nb_controles, pct_pv_tariff (proxies de volume,
        pas la cible fraud_score pour éviter le leakage).
        """
        return pd.DataFrame([
            {
                "stop_id_dep": "87212027",
                "stop_id_arr": "87214007",
                "nb_controles": 100,
                "pct_pv_tariff": 0.6,
                "fraud_score": 0.8,   # cible présente dans le dataset d'entrainement
            },
            {
                "stop_id_dep": "87212027",
                "stop_id_arr": "87214007",
                "nb_controles": 200,
                "pct_pv_tariff": 0.4,
                "fraud_score": 0.4,
            },
            {
                "stop_id_dep": "87214007",
                "stop_id_arr": "87214080",
                "nb_controles": 50,
                "pct_pv_tariff": 0.2,
                "fraud_score": 0.2,
            },
            {
                "stop_id_dep": "87214080",
                "stop_id_arr": "87214288",
                "nb_controles": 80,
                "pct_pv_tariff": 0.5,
                "fraud_score": 0.4,
            },
        ])

    @pytest.fixture
    def sample_test_data(self):
        """
        Données de production fictives (GTFS du jour) SANS les colonnes LAF.
        Le transformateur doit utiliser le lookup ou le fallback.
        """
        return pd.DataFrame([
            {"stop_id_dep": "87212027", "stop_id_arr": "87214007"},
            {"stop_id_dep": "87214007", "stop_id_arr": "87214080"},
            {"stop_id_dep": "99999999", "stop_id_arr": "88888888"},  # tronçon inconnu → fallback
        ])

    # ─────────────────────────────────────────────────────────────────────────
    # Tests fit()
    # ─────────────────────────────────────────────────────────────────────────

    def test_fit_returns_self(self, sample_train_data):
        """fit() doit retourner self pour le chaînage."""
        result = self.transformer.fit(sample_train_data)
        assert result is self.transformer

    def test_fit_missing_laf_columns_raises(self, sample_test_data):
        """
        fit() doit lever ValueError si nb_controles ET pct_pv_tariff
        sont absents — les données LAF n'ont pas été jointes.

        CORRECTION : l'ancienne version testait l'absence de fraud_score.
        La v3 exige les colonnes sources LAF (pas la cible).
        """
        with pytest.raises(ValueError, match="Aucune colonne source LAF trouvée"):
            self.transformer.fit(sample_test_data)

    def test_fit_missing_od_columns_raises(self):
        """fit() doit lever ValueError si stop_id_dep ou stop_id_arr est absent."""
        df_sans_od = pd.DataFrame([
            {"nb_controles": 100, "pct_pv_tariff": 0.5, "fraud_score": 0.3}
        ])
        with pytest.raises(ValueError, match="stop_id_dep.*stop_id_arr"):
            self.transformer.fit(df_sans_od)

    def test_fit_calculates_means_correctly(self, sample_train_data):
        """
        fit() doit calculer les moyennes par paire O/D correctement.

        CORRECTION : on vérifie _stats (dict interne), pas global_mean.
        Pour O/D (87212027 → 87214007) :
            nb_controles mean = (100 + 200) / 2 = 150
            pct_pv_tariff mean = (0.6 + 0.4) / 2 = 0.5
        """
        self.transformer.fit(sample_train_data)

        key = ("87212027", "87214007")
        assert key in self.transformer._stats

        stats = self.transformer._stats[key]
        assert stats["hist_nb_controles"] == pytest.approx(150.0)
        assert stats["hist_pct_pv_tariff"] == pytest.approx(0.5)
        # log1p(150) ≈ 5.011
        assert stats["hist_log_controles"] == pytest.approx(math.log1p(150.0))

    def test_fit_single_od_no_aggregation(self):
        """
        Pour un O/D avec une seule ligne, la valeur doit être conservée telle quelle.
        """
        df_single = pd.DataFrame([{
            "stop_id_dep": "87214007",
            "stop_id_arr": "87214080",
            "nb_controles": 50,
            "pct_pv_tariff": 0.2,
            "fraud_score": 0.2,
        }])
        self.transformer.fit(df_single)

        key = ("87214007", "87214080")
        assert key in self.transformer._stats
        assert self.transformer._stats[key]["hist_nb_controles"] == pytest.approx(50.0)

    def test_fit_computes_fallback(self, sample_train_data):
        """
        fit() doit calculer les fallbacks depuis les valeurs du lookup
        (et non depuis df_train directement).
        """
        self.transformer.fit(sample_train_data)

        assert "hist_nb_controles" in self.transformer._fallback
        assert "hist_log_controles" in self.transformer._fallback
        assert "hist_pct_pv_tariff" in self.transformer._fallback

        # Les fallbacks doivent être des flottants positifs
        assert self.transformer._fallback["hist_nb_controles"] > 0
        assert self.transformer._fallback["hist_pct_pv_tariff"] >= 0

    # ─────────────────────────────────────────────────────────────────────────
    # Tests transform()
    # ─────────────────────────────────────────────────────────────────────────

    def test_transform_missing_columns_raises(self):
        """transform() avant fit() sans stop_id_dep doit lever ValueError."""
        df_incomplet = pd.DataFrame([{"stop_id_dep": "87212027"}])
        with pytest.raises(ValueError, match="Colonnes manquantes pour transform"):
            self.transformer.transform(df_incomplet)

    def test_input_not_modified(self, sample_train_data, sample_test_data):
        """transform() ne doit pas modifier le DataFrame d'entrée."""
        self.transformer.fit(sample_train_data)
        original_cols = list(sample_test_data.columns)
        self.transformer.transform(sample_test_data)
        assert list(sample_test_data.columns) == original_cols

    def test_transform_adds_feature_columns(self, sample_train_data, sample_test_data):
        """
        transform() doit ajouter les 3 colonnes de features historiques.

        CORRECTION : les features produites sont hist_nb_controles,
        hist_log_controles, hist_pct_pv_tariff — pas hist_fraud_score_segment.
        """
        self.transformer.fit(sample_train_data)
        result = self.transformer.transform(sample_test_data)

        for col in HistoricalFeatureTransformer.FEATURE_COLS:
            assert col in result.columns, f"Colonne attendue absente : {col}"

    def test_transform_applies_correct_scores_for_known_od(self, sample_train_data, sample_test_data):
        """
        Pour un O/D connu, transform() doit appliquer les valeurs du lookup.

        CORRECTION : on vérifie hist_nb_controles (et non hist_fraud_score_segment).
        O/D (87212027 → 87214007) : nb_controles mean = 150
        """
        self.transformer.fit(sample_train_data)
        result = self.transformer.transform(sample_test_data)

        # Première ligne = O/D connu
        assert result.iloc[0]["hist_nb_controles"] == pytest.approx(150.0)
        assert result.iloc[0]["hist_pct_pv_tariff"] == pytest.approx(0.5)
        assert result.iloc[0]["hist_log_controles"] == pytest.approx(math.log1p(150.0))

    def test_transform_applies_fallback_for_unknown_od(self, sample_train_data, sample_test_data):
        """
        Pour un O/D inconnu, transform() doit appliquer le fallback
        (moyennes calculées depuis le lookup, pas depuis df_train brut).
        """
        self.transformer.fit(sample_train_data)
        result = self.transformer.transform(sample_test_data)

        # Troisième ligne = O/D inconnu (99999999 → 88888888)
        assert result.iloc[2]["hist_nb_controles"] == pytest.approx(
            self.transformer._fallback["hist_nb_controles"]
        )
        assert result.iloc[2]["hist_pct_pv_tariff"] == pytest.approx(
            self.transformer._fallback["hist_pct_pv_tariff"]
        )

    def test_transform_nb_rows_unchanged(self, sample_train_data, sample_test_data):
        """transform() ne doit pas changer le nombre de lignes."""
        self.transformer.fit(sample_train_data)
        result = self.transformer.transform(sample_test_data)
        assert len(result) == len(sample_test_data)

    def test_transform_no_nan_in_features(self, sample_train_data, sample_test_data):
        """Aucune valeur NaN ne doit rester dans les features (fallback appliqué)."""
        self.transformer.fit(sample_train_data)
        result = self.transformer.transform(sample_test_data)

        for col in HistoricalFeatureTransformer.FEATURE_COLS:
            assert result[col].isna().sum() == 0, f"NaN détecté dans {col}"