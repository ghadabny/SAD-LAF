import pytest
import pandas as pd
from google.transit import gtfs_realtime_pb2
from unittest.mock import MagicMock

from services.ml_engine.data.gtfs_rt import GTFSRTParser, GTFSRTMerger
from services.ml_engine.features.realtime import GTFSRealtimeFeatureTransformer


def _build_feed_bytes(trip_id: str, stop_id: str, delay: int) -> bytes:
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.header.gtfs_realtime_version = "2.0"
    entity = feed.entity.add()
    entity.id = "e1"
    entity.trip_update.trip.trip_id = trip_id
    stu = entity.trip_update.stop_time_update.add()
    stu.stop_id = stop_id
    stu.departure.delay = delay
    return feed.SerializeToString()


@pytest.fixture
def troncons():
    """
    Tronçons avec train_number au format numérique SNCF.
    Les valeurs concrètes sont surchargées dans chaque test si besoin.
    """
    return pd.DataFrame([
        {"trip_id": "TRIP_RT_01", "train_number": "1001", "stop_id_dep": "STOP_1", "stop_id_arr": "STOP_2"},
        {"trip_id": "TRIP_RT_02", "train_number": "1002", "stop_id_dep": "STOP_3", "stop_id_arr": "STOP_4"},
    ])


class TestGTFSRTPipelineIntegration:

    def test_pipeline_complet_sans_reseau(self, troncons, tmp_path, monkeypatch):
        """
        Valide la chaîne Parser → Merger sur bytes protobuf synthétiques.

        Stratégie : on utilise des trip_id au format SNCF réel pour que
        _extract_train_number produise exactement le train_number attendu.
        OCESN1001F → extrait '1001' → doit matcher train_number='1001' dans troncons.
        """
        from shared import config as cfg
        monkeypatch.setattr(cfg.config, "GTFS_RT_CACHE_DIR", tmp_path)

        # Format réel SNCF : OCESN{numero}F...
        # _extract_train_number("OCESN1001F_test") → '1001'
        tu_bytes = _build_feed_bytes("OCESN1001F_test", "STOP_1", 240)
        empty_feed = gtfs_realtime_pb2.FeedMessage()
        empty_feed.header.gtfs_realtime_version = "2.0"
        sa_bytes = empty_feed.SerializeToString()

        parser = GTFSRTParser()
        merger = GTFSRTMerger()
        rt_data = parser.parse(tu_bytes, sa_bytes)

        # Les tronçons doivent avoir train_number='1001' pour matcher
        troncons_adaptes = troncons.copy()
        troncons_adaptes["train_number"] = ["1001", "1002"]

        result = merger.merge(troncons_adaptes, rt_data.stop_time_updates, rt_data.cancelled_trips)

        trip_01 = result[result["trip_id"] == "TRIP_RT_01"].iloc[0]
        assert trip_01["delay_dep_sec"] == 240
        assert not trip_01["is_cancelled"]

        trip_02 = result[result["trip_id"] == "TRIP_RT_02"].iloc[0]
        assert trip_02["delay_dep_sec"] == 0

    def test_transformer_dans_pipeline_feature(self, troncons, tmp_path, monkeypatch):
        """
        Vérifie que GTFSRealtimeFeatureTransformer s'insère dans FeaturePipeline
        sans conflit avec TemporalFeatureTransformer.
        """
        from shared import config as cfg
        from services.ml_engine.features.pipeline import FeaturePipeline
        from services.ml_engine.features.temporal import TemporalFeatureTransformer
        from datetime import date
        monkeypatch.setattr(cfg.config, "GTFS_RT_CACHE_DIR", tmp_path)

        troncons_complets = troncons.copy()
        troncons_complets["dep_minutes"] = 480
        troncons_complets["service_date"] = date(2024, 9, 2)

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

        pipeline = FeaturePipeline([TemporalFeatureTransformer(), transformer])
        result = pipeline.fit_transform(troncons_complets)

        assert "dep_hour" in result.columns
        assert "delay_dep_sec" in result.columns
        assert "is_delayed" in result.columns