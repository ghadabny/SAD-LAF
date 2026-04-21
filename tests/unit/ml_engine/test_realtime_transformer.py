import pytest
import pandas as pd
from unittest.mock import MagicMock

from services.ml_engine.features.realtime import GTFSRealtimeFeatureTransformer


@pytest.fixture
def troncons_minimal():
    """Tronçons avec train_number — requis par le Merger depuis la refonte."""
    return pd.DataFrame([
        {"trip_id": "TRIP_A", "train_number": "1001", "stop_id_dep": "S1", "stop_id_arr": "S2"},
        {"trip_id": "TRIP_B", "train_number": "1002", "stop_id_dep": "S3", "stop_id_arr": "S4"},
    ])


class TestGTFSRealtimeFeatureTransformer:

    def _make_transformer_with_mock_data(self, delay_sec=0, is_cancelled=False):
        """Helper : transformateur avec données RT mockées."""
        stu = pd.DataFrame([
            {
                "train_number": "1001",
                "stop_id": "S1",
                "delay_dep_sec": delay_sec,
                "delay_arr_sec": delay_sec,
            },
        ])
        cancelled = pd.DataFrame({
            "trip_id": ["OCESN1002F_test"] if is_cancelled else []
        })

        transformer = GTFSRealtimeFeatureTransformer()
        transformer._load_realtime_data = MagicMock(return_value=(stu, cancelled))
        return transformer

    def test_fit_retourne_self(self, troncons_minimal):
        """fit() doit retourner self pour le chaînage."""
        transformer = GTFSRealtimeFeatureTransformer()
        result = transformer.fit(troncons_minimal)
        assert result is transformer

    def test_transform_ajoute_colonnes_rt(self, troncons_minimal):
        """Les 5 colonnes RT doivent être présentes après transform."""
        transformer = self._make_transformer_with_mock_data(delay_sec=120)
        result = transformer.transform(troncons_minimal)
        for col in ["delay_dep_sec", "delay_arr_sec", "delay_dep_min", "is_cancelled", "is_delayed"]:
            assert col in result.columns, f"Colonne manquante : {col}"

    def test_is_delayed_vrai_si_retard_superieur_seuil(self, troncons_minimal):
        """is_delayed = True si delay_dep_sec > 300."""
        transformer = self._make_transformer_with_mock_data(delay_sec=400)
        result = transformer.transform(troncons_minimal)
        trip_a = result[result["trip_id"] == "TRIP_A"].iloc[0]
        assert trip_a["is_delayed"]

    def test_is_delayed_faux_si_retard_inferieur_seuil(self, troncons_minimal):
        """is_delayed = False si delay_dep_sec <= 300."""
        transformer = self._make_transformer_with_mock_data(delay_sec=120)
        result = transformer.transform(troncons_minimal)
        trip_a = result[result["trip_id"] == "TRIP_A"].iloc[0]
        assert not trip_a["is_delayed"]

    def test_delay_dep_min_coherent(self, troncons_minimal):
        """delay_dep_min = delay_dep_sec / 60."""
        transformer = self._make_transformer_with_mock_data(delay_sec=180)
        result = transformer.transform(troncons_minimal)
        trip_a = result[result["trip_id"] == "TRIP_A"].iloc[0]
        assert trip_a["delay_dep_min"] == pytest.approx(3.0)

    def test_fallback_si_fetch_echoue(self, troncons_minimal):
        """Si le fetch RT échoue, transform doit retourner un DataFrame valide avec zéros."""
        empty_stu = pd.DataFrame({
            "train_number":  pd.Series([], dtype=str),
            "stop_id":       pd.Series([], dtype=str),
            "delay_dep_sec": pd.Series([], dtype=int),
            "delay_arr_sec": pd.Series([], dtype=int),
        })
        empty_cancelled = pd.DataFrame({"trip_id": pd.Series([], dtype=str)})

        transformer = GTFSRealtimeFeatureTransformer()
        transformer._load_realtime_data = MagicMock(
            return_value=(empty_stu, empty_cancelled)
        )
        result = transformer.transform(troncons_minimal)
        assert "delay_dep_sec" in result.columns
        assert (result["delay_dep_sec"] == 0).all()

    def test_ne_modifie_pas_df_entree(self, troncons_minimal):
        """transform() ne doit pas muter le DataFrame d'entrée."""
        transformer = self._make_transformer_with_mock_data()
        original_cols = set(troncons_minimal.columns)
        transformer.transform(troncons_minimal)
        assert set(troncons_minimal.columns) == original_cols