import pytest
import pandas as pd
from datetime import date

from services.ml_engine.features.temporal import TemporalFeatureTransformer


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# On crée des cas précis dont on connaît le résultat attendu à la main
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_troncons():
    """
    DataFrame minimal avec 5 tronçons couvrant tous les cas à tester :

        lundi matin pointe     : 2024-09-02, 08:23 → jour ouvré, heure de pointe
        samedi                 : 2024-09-07, 10:00 → week-end, pas de pointe
        vacances Toussaint     : 2024-10-21, 08:30 → vacances zone B
        jour férié Armistice   : 2024-11-11, 09:00 → jour férié
        vendredi soir pointe   : 2024-09-06, 17:30 → heure de pointe soir
    """
    return pd.DataFrame([
        {"dep_minutes": 8 * 60 + 23, "service_date": date(2024,  9,  2)},  # lundi matin
        {"dep_minutes": 10 * 60,     "service_date": date(2024,  9,  7)},  # samedi
        {"dep_minutes": 8 * 60 + 30, "service_date": date(2024, 10, 21)},  # vacances Toussaint
        {"dep_minutes": 9 * 60,      "service_date": date(2024, 11, 11)},  # Armistice
        {"dep_minutes": 17 * 60 + 30,"service_date": date(2024,  9,  6)},  # vendredi soir
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTemporalFeatureTransformer:

    def setup_method(self):
        self.transformer = TemporalFeatureTransformer()

    # ── Interface SOLID ───────────────────────────────────────────────────────

    def test_fit_returns_self(self, sample_troncons):
        """fit() doit retourner self pour permettre le chaînage."""
        result = self.transformer.fit(sample_troncons)
        assert result is self.transformer

    def test_fit_transform_coherent(self, sample_troncons):
        """fit_transform() doit donner le même résultat que fit().transform()."""
        result1 = self.transformer.fit_transform(sample_troncons)
        result2 = self.transformer.fit(sample_troncons).transform(sample_troncons)
        pd.testing.assert_frame_equal(result1, result2)

    def test_input_not_modified(self, sample_troncons):
        """transform() ne doit jamais modifier le DataFrame d'entrée."""
        original_cols = list(sample_troncons.columns)
        self.transformer.fit_transform(sample_troncons)
        assert list(sample_troncons.columns) == original_cols

    # ── Colonnes produites ────────────────────────────────────────────────────

    def test_all_columns_present(self, sample_troncons):
        """Toutes les colonnes de features attendues doivent être présentes."""
        result = self.transformer.fit_transform(sample_troncons)
        expected_cols = [
            "dep_hour", "dep_minute_of_day", "day_of_week",
            "is_weekend", "is_vacances", "is_jour_ferie", "is_peak_hour"
        ]
        for col in expected_cols:
            assert col in result.columns, f"Colonne manquante : {col}"

    # ── dep_hour ──────────────────────────────────────────────────────────────

    def test_dep_hour_correct(self, sample_troncons):
        """dep_hour = dep_minutes // 60."""
        result = self.transformer.fit_transform(sample_troncons)
        # lundi 08:23 → dep_hour = 8
        assert result.iloc[0]["dep_hour"] == 8
        # samedi 10:00 → dep_hour = 10
        assert result.iloc[1]["dep_hour"] == 10

    # ── day_of_week ───────────────────────────────────────────────────────────

    def test_day_of_week_monday(self, sample_troncons):
        """2024-09-02 est un lundi → day_of_week = 0."""
        result = self.transformer.fit_transform(sample_troncons)
        assert result.iloc[0]["day_of_week"] == 0

    def test_day_of_week_saturday(self, sample_troncons):
        """2024-09-07 est un samedi → day_of_week = 5."""
        result = self.transformer.fit_transform(sample_troncons)
        assert result.iloc[1]["day_of_week"] == 5

    # ── is_weekend ────────────────────────────────────────────────────────────

    def test_is_weekend_false_on_monday(self, sample_troncons):
        result = self.transformer.fit_transform(sample_troncons)
        assert result.iloc[0]["is_weekend"] == False

    def test_is_weekend_true_on_saturday(self, sample_troncons):
        result = self.transformer.fit_transform(sample_troncons)
        assert result.iloc[1]["is_weekend"] == True

    # ── is_vacances ───────────────────────────────────────────────────────────

    def test_is_vacances_true_toussaint(self, sample_troncons):
        """2024-10-21 est pendant les vacances de Toussaint zone B."""
        result = self.transformer.fit_transform(sample_troncons)
        assert result.iloc[2]["is_vacances"] == True

    def test_is_vacances_false_on_school_day(self, sample_troncons):
        """2024-09-02 est un jour de rentrée scolaire → pas de vacances."""
        result = self.transformer.fit_transform(sample_troncons)
        assert result.iloc[0]["is_vacances"] == False

    # ── is_jour_ferie ─────────────────────────────────────────────────────────

    def test_is_jour_ferie_armistice(self, sample_troncons):
        """2024-11-11 est le jour de l'Armistice."""
        result = self.transformer.fit_transform(sample_troncons)
        assert result.iloc[3]["is_jour_ferie"] == True

    def test_is_jour_ferie_false_on_normal_day(self, sample_troncons):
        """2024-09-02 n'est pas un jour férié."""
        result = self.transformer.fit_transform(sample_troncons)
        assert result.iloc[0]["is_jour_ferie"] == False

    # ── is_peak_hour ──────────────────────────────────────────────────────────

    def test_peak_hour_true_monday_morning(self, sample_troncons):
        """Lundi 08:23 → heure de pointe matin en semaine."""
        result = self.transformer.fit_transform(sample_troncons)
        assert result.iloc[0]["is_peak_hour"] == True

    def test_peak_hour_false_on_weekend(self, sample_troncons):
        """Samedi 10:00 → pas de pointe le week-end même si heure creuse."""
        result = self.transformer.fit_transform(sample_troncons)
        assert result.iloc[1]["is_peak_hour"] == False

    def test_peak_hour_true_friday_evening(self, sample_troncons):
        """Vendredi 17:30 → heure de pointe soir."""
        result = self.transformer.fit_transform(sample_troncons)
        assert result.iloc[4]["is_peak_hour"] == True

    # ── Validation ────────────────────────────────────────────────────────────

    def test_missing_column_raises(self):
        """Un DataFrame sans service_date doit lever une ValueError claire."""
        df_incomplet = pd.DataFrame([{"dep_minutes": 500}])
        with pytest.raises(ValueError, match="Colonnes manquantes"):
            self.transformer.transform(df_incomplet)