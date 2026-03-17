import pytest
import pandas as pd
from services.ml_engine.data.gtfs.preprocessor import GTFSPreprocessor


class TestCleanStopId:
    """Tests unitaires pour la méthode clean_stop_id."""

    def setup_method(self):
        self.preprocessor = GTFSPreprocessor()

    def test_clean_ter_prefix(self):
        result = self.preprocessor.clean_stop_id("StopPoint:OCETrain TER-87723197")
        assert result == "87723197"

    def test_clean_tgv_prefix(self):
        result = self.preprocessor.clean_stop_id("StopPoint:OCETGV INOUI-87723197")
        assert result == "87723197"

    def test_already_clean(self):
        """Un stop_id sans préfixe ne doit pas être modifié."""
        result = self.preprocessor.clean_stop_id("87723197")
        assert result == "87723197"

    def test_none_value(self):
        """Une valeur non-string doit être retournée telle quelle."""
        result = self.preprocessor.clean_stop_id(None)
        assert result is None


class TestParseGtfsTime:
    """Tests unitaires pour la méthode parse_gtfs_time."""

    def setup_method(self):
        self.preprocessor = GTFSPreprocessor()

    def test_normal_time(self):
        series = pd.Series(["09:16:00"])
        result = self.preprocessor.parse_gtfs_time(series)
        assert result.iloc[0] == 9 * 60 + 16  # 556 minutes

    def test_midnight(self):
        series = pd.Series(["00:00:00"])
        result = self.preprocessor.parse_gtfs_time(series)
        assert result.iloc[0] == 0

    def test_over_24h(self):
        """Le GTFS peut avoir des heures > 24h pour les trains de nuit."""
        series = pd.Series(["25:30:00"])
        result = self.preprocessor.parse_gtfs_time(series)
        assert result.iloc[0] == 25 * 60 + 30  # 1530 minutes

    def test_invalid_value(self):
        """Une valeur invalide doit retourner -1 (sentinelle)."""
        series = pd.Series(["invalide"])
        result = self.preprocessor.parse_gtfs_time(series)
        assert result.iloc[0] == -1



class TestBuildTroncons:
    """Tests d'intégration pour la méthode build_troncons."""

    def setup_method(self):
        self.preprocessor = GTFSPreprocessor()

    def test_troncons_count(self, synthetic_gtfs, synthetic_routes):
        """
        Avec N arrêts par trip, on doit obtenir N-1 tronçons par trip.
        """
        stop_times, trips, stops = synthetic_gtfs
        routes = synthetic_routes

        troncons = self.preprocessor.build_troncons(
            stop_times, trips, stops, routes, ter_only=False
        )

        # Chaque trip a 3 arrêts → 2 tronçons par trip
        n_trips = trips["trip_id"].nunique()
        assert len(troncons) == n_trips * 2

    def test_colonnes_presentes(self, synthetic_gtfs, synthetic_routes):
        """Toutes les colonnes attendues doivent être présentes."""
        stop_times, trips, stops = synthetic_gtfs
        routes = synthetic_routes

        troncons = self.preprocessor.build_troncons(
            stop_times, trips, stops, routes, ter_only=False
        )

        colonnes_attendues = [
            "trip_id", "train_number", "service_id", "stop_sequence",
            "stop_id_dep", "stop_name_dep", "dep_minutes",
            "stop_id_arr", "stop_name_arr", "arr_minutes", "duration_min"
        ]
        for col in colonnes_attendues:
            assert col in troncons.columns, f"Colonne manquante : {col}"

    def test_duration_positive(self, synthetic_gtfs, synthetic_routes):
        """Tous les tronçons doivent avoir une durée positive."""
        stop_times, trips, stops = synthetic_gtfs
        routes = synthetic_routes

        troncons = self.preprocessor.build_troncons(
            stop_times, trips, stops, routes, ter_only=False
        )
        assert (troncons["duration_min"] > 0).all()

    def test_no_same_dep_arr(self, synthetic_gtfs, synthetic_routes):
        """Un tronçon ne peut pas avoir la même gare de départ et d'arrivée."""
        stop_times, trips, stops = synthetic_gtfs
        routes = synthetic_routes

        troncons = self.preprocessor.build_troncons(
            stop_times, trips, stops, routes, ter_only=False
        )
        assert (troncons["stop_id_dep"] != troncons["stop_id_arr"]).all()