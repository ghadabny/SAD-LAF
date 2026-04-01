import pytest
import pandas as pd

from services.ml_engine.data.gtfs_rt.merger import GTFSRTMerger


@pytest.fixture
def troncons_sample():
    """
    Tronçons synthétiques avec train_number — clé de jointure avec le RT.
    TRIP_A → train_number '1' (même valeur que dans stop_time_updates)
    TRIP_B → train_number '2' (sera dans les cancelled_trips)
    TRIP_C → train_number '3' (absent du RT → délais à 0)
    """
    return pd.DataFrame([
        {"trip_id": "TRIP_A", "train_number": "1", "stop_id_dep": "S1", "stop_id_arr": "S2"},
        {"trip_id": "TRIP_B", "train_number": "2", "stop_id_dep": "S3", "stop_id_arr": "S4"},
        {"trip_id": "TRIP_C", "train_number": "3", "stop_id_dep": "S5", "stop_id_arr": "S6"},
    ])


@pytest.fixture
def stop_time_updates():
    """STU avec train_number correspondant à TRIP_A."""
    return pd.DataFrame([
        {"train_number": "1", "stop_id": "S1", "delay_dep_sec": 180, "delay_arr_sec": 120},
        {"train_number": "1", "stop_id": "S2", "delay_dep_sec": 200, "delay_arr_sec": 190},
    ])


@pytest.fixture
def cancelled_trips():
    """
    trip_id au format SNCF pour TRIP_B.
    Le Merger extrait '2' via _extract_train_number,
    qui doit matcher train_number='2' dans troncons_sample.
    """
    return pd.DataFrame({"trip_id": ["OCESN2F1187_F:IC:FR:Line::test"]})


class TestGTFSRTMerger:

    def setup_method(self):
        self.merger = GTFSRTMerger()

    def test_merge_ajoute_colonnes_requises(self, troncons_sample, stop_time_updates, cancelled_trips):
        """Le résultat doit contenir les 3 nouvelles colonnes."""
        result = self.merger.merge(troncons_sample, stop_time_updates, cancelled_trips)
        assert "delay_dep_sec" in result.columns
        assert "delay_arr_sec" in result.columns
        assert "is_cancelled" in result.columns

    def test_merge_delai_correct_pour_trip_connu(self, troncons_sample, stop_time_updates, cancelled_trips):
        """TRIP_A / S1 → delay_dep_sec doit être 180."""
        result = self.merger.merge(troncons_sample, stop_time_updates, cancelled_trips)
        trip_a = result[result["trip_id"] == "TRIP_A"].iloc[0]
        assert trip_a["delay_dep_sec"] == 180

    def test_merge_fallback_zero_pour_trip_inconnu(self, troncons_sample, stop_time_updates, cancelled_trips):
        """TRIP_C absent du RT → délais = 0."""
        result = self.merger.merge(troncons_sample, stop_time_updates, cancelled_trips)
        trip_c = result[result["trip_id"] == "TRIP_C"].iloc[0]
        assert trip_c["delay_dep_sec"] == 0
        assert trip_c["delay_arr_sec"] == 0

    def test_merge_is_cancelled_vrai_pour_trip_alerte(self, troncons_sample, stop_time_updates, cancelled_trips):
        """TRIP_B est dans les alertes → is_cancelled=True."""
        result = self.merger.merge(troncons_sample, stop_time_updates, cancelled_trips)
        trip_b = result[result["trip_id"] == "TRIP_B"].iloc[0]
        assert trip_b["is_cancelled"]

    def test_merge_is_cancelled_faux_pour_trip_normal(self, troncons_sample, stop_time_updates, cancelled_trips):
        """TRIP_A n'est pas dans les alertes → is_cancelled=False."""
        result = self.merger.merge(troncons_sample, stop_time_updates, cancelled_trips)
        trip_a = result[result["trip_id"] == "TRIP_A"].iloc[0]
        assert not trip_a["is_cancelled"]

    def test_merge_ne_modifie_pas_troncons_original(self, troncons_sample, stop_time_updates, cancelled_trips):
        """Le DataFrame d'entrée ne doit pas être muté."""
        original_cols = set(troncons_sample.columns)
        self.merger.merge(troncons_sample, stop_time_updates, cancelled_trips)
        assert set(troncons_sample.columns) == original_cols

    def test_merge_stu_vide_remplit_zeros(self, troncons_sample, cancelled_trips):
        """STU vide → tous les délais à 0."""
        empty_stu = pd.DataFrame({
            "train_number":  pd.Series([], dtype=str),
            "stop_id":       pd.Series([], dtype=str),
            "delay_dep_sec": pd.Series([], dtype=int),
            "delay_arr_sec": pd.Series([], dtype=int),
        })
        result = self.merger.merge(troncons_sample, empty_stu, cancelled_trips)
        assert (result["delay_dep_sec"] == 0).all()

    def test_merge_sans_colonne_train_number_ne_plante_pas(self):
        """
        Si train_number est absent des tronçons (ex: données externes),
        le merger doit fonctionner sans KeyError.
        """
        troncons_sans_train_number = pd.DataFrame([
            {"trip_id": "TRIP_X", "stop_id_dep": "S1", "stop_id_arr": "S2"},
        ])
        empty_stu = pd.DataFrame({
            "train_number":  pd.Series([], dtype=str),
            "stop_id":       pd.Series([], dtype=str),
            "delay_dep_sec": pd.Series([], dtype=int),
            "delay_arr_sec": pd.Series([], dtype=int),
        })
        empty_cancelled = pd.DataFrame({"trip_id": pd.Series([], dtype=str)})
        result = self.merger.merge(troncons_sans_train_number, empty_stu, empty_cancelled)
        assert "delay_dep_sec" in result.columns
        assert (result["delay_dep_sec"] == 0).all()