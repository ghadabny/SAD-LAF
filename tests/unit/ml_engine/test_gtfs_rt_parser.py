import pytest
from unittest.mock import MagicMock, patch
from google.transit import gtfs_realtime_pb2

from services.ml_engine.data.gtfs_rt.parser import GTFSRTParser


def _build_trip_update_bytes(trip_id: str, stop_id: str, dep_delay: int) -> bytes:
    """Construit un message protobuf Trip Update minimal pour les tests."""
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.header.gtfs_realtime_version = "2.0"
    entity = feed.entity.add()
    entity.id = "test-entity-1"
    entity.trip_update.trip.trip_id = trip_id
    stu = entity.trip_update.stop_time_update.add()
    stu.stop_id = stop_id
    stu.departure.delay = dep_delay
    stu.arrival.delay = dep_delay - 30
    return feed.SerializeToString()


def _build_service_alert_bytes(trip_id: str) -> bytes:
    """Construit un message protobuf Service Alert minimal."""
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.header.gtfs_realtime_version = "2.0"
    entity = feed.entity.add()
    entity.id = "test-alert-1"
    informed = entity.alert.informed_entity.add()
    informed.trip.trip_id = trip_id
    return feed.SerializeToString()


class TestGTFSRTParser:

    def setup_method(self):
        self.parser = GTFSRTParser()

    def test_parse_trip_update_colonnes_attendues(self):
        """Le DataFrame produit doit avoir les bonnes colonnes."""
        raw = _build_trip_update_bytes("TRIP_001", "STOP_A", 120)
        result = self.parser.parse(raw, b"")  # SA vide
        assert set(["trip_id", "stop_id", "delay_dep_sec", "delay_arr_sec"]).issubset(
            result.stop_time_updates.columns
        )

    def test_parse_trip_update_valeurs_correctes(self):
        """Les valeurs de délai sont correctement extraites du protobuf."""
        raw = _build_trip_update_bytes("TRIP_001", "STOP_A", 300)
        result = self.parser.parse(raw, b"")
        row = result.stop_time_updates.iloc[0]
        assert row["trip_id"] == "TRIP_001"
        assert row["stop_id"] == "STOP_A"
        assert row["delay_dep_sec"] == 300

    def test_parse_service_alert_trip_id_extrait(self):
        """Les trip_id en alerte sont correctement extraits."""
        raw_sa = _build_service_alert_bytes("TRIP_ANNULE")
        result = self.parser.parse(b"", raw_sa)
        assert "TRIP_ANNULE" in result.cancelled_trips["trip_id"].values

    def test_parse_bytes_vides_retourne_dataframes_vides(self):
        """Des bytes vides ne doivent pas lever d'exception."""
        # Un FeedMessage vide est valide en protobuf
        empty_feed = gtfs_realtime_pb2.FeedMessage()
        empty_feed.header.gtfs_realtime_version = "2.0"
        empty_bytes = empty_feed.SerializeToString()
        result = self.parser.parse(empty_bytes, empty_bytes)
        assert result.stop_time_updates.empty
        assert result.cancelled_trips.empty

    def test_parse_types_colonnes(self):
        """Les colonnes doivent avoir les bons types (pas de float parasite)."""
        raw = _build_trip_update_bytes("TRIP_001", "STOP_A", 60)
        result = self.parser.parse(raw, b"")
        df = result.stop_time_updates
        assert df["delay_dep_sec"].dtype == int
        assert df["trip_id"].dtype in (object, "string")