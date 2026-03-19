import json
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

from services.scheduler.daily.daily_fetcher import DailyGTFSFetcher
from services.scheduler.realtime.realtime_fetcher import RealtimeFetcher, DELAY_THRESHOLD_SECONDS
from services.scheduler.realtime.realtime_update import RealtimeUpdate


# ─────────────────────────────────────────────────────────────────────────────
# Tests DailyGTFSFetcher — inchangés
# ─────────────────────────────────────────────────────────────────────────────

class TestDailyGTFSFetcher:

    def test_run_appelle_download_force_true(self):
        with patch(
            "services.scheduler.daily.daily_fetcher.GTFSDownloader"
        ) as MockDownloader:
            mock_instance = MockDownloader.return_value
            fetcher = DailyGTFSFetcher()
            fetcher.run()
            mock_instance.download.assert_called_once_with(force=True)

    def test_run_ne_propage_pas_exception(self):
        with patch(
            "services.scheduler.daily.daily_fetcher.GTFSDownloader"
        ) as MockDownloader:
            mock_instance = MockDownloader.return_value
            mock_instance.download.side_effect = ConnectionError("réseau indisponible")
            fetcher = DailyGTFSFetcher()
            fetcher.run()


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures communes
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def fetcher(tmp_path, monkeypatch):
    from shared import config as cfg
    monkeypatch.setattr(cfg.config, "GTFS_RT_DIR", tmp_path)
    return RealtimeFetcher()


@pytest.fixture
def trip_updates_sans_retard():
    return {
        "fetched_at": "2024-09-02T08:00:00",
        "source": "gtfs-rt-trip-updates",
        "trip_updates": [
            {
                "trip_id": "TRIP_001",
                "train_number": "117756",
                "route_id": "FR:Line::TER_ALSACE:",
                "stop_time_updates": [
                    {"stop_id": "87212027", "stop_sequence": 0,
                     "arrival_delay_seconds": 60,
                     "departure_delay_seconds": 60},
                ]
            }
        ]
    }


@pytest.fixture
def trip_updates_avec_retard():
    return {
        "fetched_at": "2024-09-02T08:02:00",
        "source": "gtfs-rt-trip-updates",
        "trip_updates": [
            {
                "trip_id": "TRIP_001",
                "train_number": "117756",
                "route_id": "FR:Line::TER_ALSACE:",
                "stop_time_updates": [
                    {"stop_id": "87212027", "stop_sequence": 0,
                     "arrival_delay_seconds": 600,
                     "departure_delay_seconds": 600},
                ]
            }
        ]
    }


@pytest.fixture
def service_alerts_vide():
    return {
        "fetched_at": "2024-09-02T08:00:00",
        "source": "gtfs-rt-service-alerts",
        "alerts": []
    }


@pytest.fixture
def service_alerts_avec_suppression():
    return {
        "fetched_at": "2024-09-02T08:02:00",
        "source": "gtfs-rt-service-alerts",
        "alerts": [
            {
                "alert_id": "ALERT_001",
                "cause": "STRIKE",
                "effect": "NO_SERVICE",
                "affected_trips": ["TRIP_002"],
                "affected_stops": [],
                "active_periods": []
            }
        ]
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tests RealtimeUpdate
# ─────────────────────────────────────────────────────────────────────────────

class TestRealtimeUpdate:

    def test_trips_impacted_union(self):
        """trips_impacted = union des retards et suppressions."""
        update = RealtimeUpdate(
            has_changed=True,
            trips_delayed={"TRIP_001", "TRIP_002"},
            trips_cancelled={"TRIP_003"},
        )
        assert update.trips_impacted == {"TRIP_001", "TRIP_002", "TRIP_003"}

    def test_nb_impacted_correct(self):
        update = RealtimeUpdate(
            has_changed=True,
            trips_delayed={"TRIP_001"},
            trips_cancelled={"TRIP_002"},
        )
        assert update.nb_impacted == 2

    def test_has_changed_false_par_defaut(self):
        update = RealtimeUpdate()
        assert update.has_changed == False
        assert update.trips_impacted == set()

    def test_trips_impacted_sans_doublon(self):
        """Un trip à la fois retardé et supprimé ne compte qu'une fois."""
        update = RealtimeUpdate(
            has_changed=True,
            trips_delayed={"TRIP_001"},
            trips_cancelled={"TRIP_001"},
        )
        assert update.nb_impacted == 1


# ─────────────────────────────────────────────────────────────────────────────
# Tests RealtimeFetcher — sauvegarde (inchangés)
# ─────────────────────────────────────────────────────────────────────────────

class TestRealtimeFetcherSave:

    def test_save_cree_fichier_json(self, fetcher, tmp_path):
        data = {"fetched_at": "2024-09-02T08:23:00", "test": True}
        fetcher._save(data, "test.json")
        assert (tmp_path / "test.json").exists()

    def test_save_contenu_correct(self, fetcher, tmp_path):
        data = {"fetched_at": "2024-09-02T08:23:00", "alerts": []}
        fetcher._save(data, "alerts.json")
        with open(tmp_path / "alerts.json", encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded["fetched_at"] == "2024-09-02T08:23:00"
        assert loaded["alerts"] == []

    def test_save_ecrase_fichier_existant(self, fetcher, tmp_path):
        fetcher._save({"version": 1}, "test.json")
        fetcher._save({"version": 2}, "test.json")
        with open(tmp_path / "test.json", encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded["version"] == 2


# ─────────────────────────────────────────────────────────────────────────────
# Tests RealtimeFetcher — détection de changement
# ─────────────────────────────────────────────────────────────────────────────

class TestDetectionChangement:

    def test_extract_delayed_trips_sous_seuil(
        self, fetcher, trip_updates_sans_retard
    ):
        """Un retard sous le seuil ne doit pas être extrait."""
        delayed = fetcher._extract_delayed_trips(trip_updates_sans_retard)
        assert delayed == set()

    def test_extract_delayed_trips_au_dessus_seuil(
        self, fetcher, trip_updates_avec_retard
    ):
        """Un retard > seuil doit être extrait."""
        delayed = fetcher._extract_delayed_trips(trip_updates_avec_retard)
        assert "TRIP_001" in delayed

    def test_extract_delayed_trips_exactement_au_seuil(self, fetcher):
        """Un retard exactement égal au seuil doit être extrait."""
        data = {
            "trip_updates": [{
                "trip_id": "TRIP_001",
                "stop_time_updates": [{
                    "arrival_delay_seconds":   DELAY_THRESHOLD_SECONDS,
                    "departure_delay_seconds": 0,
                }]
            }]
        }
        delayed = fetcher._extract_delayed_trips(data)
        assert "TRIP_001" in delayed

    def test_extract_cancelled_trips(
        self, fetcher, service_alerts_avec_suppression
    ):
        """Les trips NO_SERVICE doivent être extraits."""
        cancelled = fetcher._extract_cancelled_trips(
            service_alerts_avec_suppression
        )
        assert "TRIP_002" in cancelled

    def test_extract_cancelled_trips_vide(
        self, fetcher, service_alerts_vide
    ):
        """Pas de suppression → set vide."""
        cancelled = fetcher._extract_cancelled_trips(service_alerts_vide)
        assert cancelled == set()

    def test_load_previous_absent(self, fetcher):
        """Premier démarrage — fichier absent → retourne None."""
        result = fetcher._load_previous("trip_updates.json")
        assert result is None

    def test_load_previous_existant(self, fetcher, tmp_path):
        """Un fichier existant doit être chargé correctement."""
        data = {"fetched_at": "2024-09-02T08:00:00", "trip_updates": []}
        with open(tmp_path / "trip_updates.json", "w") as f:
            json.dump(data, f)
        result = fetcher._load_previous("trip_updates.json")
        assert result is not None
        assert result["trip_updates"] == []


# ─────────────────────────────────────────────────────────────────────────────
# Tests RealtimeFetcher — run() avec RealtimeUpdate
# ─────────────────────────────────────────────────────────────────────────────

class TestRealtimeFetcherRun:

    def test_run_retourne_realtime_update(self, fetcher):
        """run() doit retourner un RealtimeUpdate."""
        with patch.object(fetcher, "_run_trip_updates", return_value=set()), \
             patch.object(fetcher, "_run_service_alerts", return_value=set()):
            result = fetcher.run()
        assert isinstance(result, RealtimeUpdate)

    def test_run_has_changed_false_si_rien(self, fetcher):
        """Pas de changement → has_changed = False."""
        with patch.object(fetcher, "_run_trip_updates", return_value=set()), \
             patch.object(fetcher, "_run_service_alerts", return_value=set()):
            result = fetcher.run()
        assert result.has_changed == False

    def test_run_has_changed_true_si_retard(self, fetcher):
        """Retard détecté → has_changed = True."""
        with patch.object(fetcher, "_run_trip_updates", return_value={"TRIP_001"}), \
             patch.object(fetcher, "_run_service_alerts", return_value=set()):
            result = fetcher.run()
        assert result.has_changed == True
        assert "TRIP_001" in result.trips_delayed

    def test_run_has_changed_true_si_suppression(self, fetcher):
        """Suppression détectée → has_changed = True."""
        with patch.object(fetcher, "_run_trip_updates", return_value=set()), \
             patch.object(fetcher, "_run_service_alerts", return_value={"TRIP_002"}):
            result = fetcher.run()
        assert result.has_changed == True
        assert "TRIP_002" in result.trips_cancelled

    def test_run_ne_propage_pas_exception(self, fetcher):
        """Si le réseau est indisponible, run() ne propage pas l'exception."""
        with patch.object(
            fetcher, "_fetch_protobuf", side_effect=ConnectionError("réseau")
        ):
            result = fetcher.run()
        assert isinstance(result, RealtimeUpdate)
        assert result.has_changed == False

    def test_run_appelle_les_deux_fetch(self, fetcher):
        """run() doit appeler _run_trip_updates ET _run_service_alerts."""
        with patch.object(fetcher, "_run_trip_updates", return_value=set()) as mock_tu, \
             patch.object(fetcher, "_run_service_alerts", return_value=set()) as mock_sa:
            fetcher.run()
        mock_tu.assert_called_once()
        mock_sa.assert_called_once()

    def test_get_translated_text_french(self, fetcher):
        mock_translation = MagicMock()
        mock_translation.language = "fr"
        mock_translation.text = "Train supprimé"
        mock_ts = MagicMock()
        mock_ts.translation = [mock_translation]
        assert fetcher._get_translated_text(mock_ts) == "Train supprimé"

    def test_get_translated_text_fallback(self, fetcher):
        mock_translation = MagicMock()
        mock_translation.language = "en"
        mock_translation.text = "Train cancelled"
        mock_ts = MagicMock()
        mock_ts.translation = [mock_translation]
        assert fetcher._get_translated_text(mock_ts) == "Train cancelled"

    def test_get_translated_text_empty(self, fetcher):
        mock_ts = MagicMock()
        mock_ts.translation = []
        assert fetcher._get_translated_text(mock_ts) == ""

    def test_parse_trip_updates_structure(self, fetcher):
        mock_feed = MagicMock()
        mock_feed.entity = []
        result = fetcher._parse_trip_updates(mock_feed)
        assert "fetched_at" in result
        assert result["source"] == "gtfs-rt-trip-updates"
        assert result["trip_updates"] == []

    def test_parse_service_alerts_structure(self, fetcher):
        mock_feed = MagicMock()
        mock_feed.entity = []
        result = fetcher._parse_service_alerts(mock_feed)
        assert "fetched_at" in result
        assert result["source"] == "gtfs-rt-service-alerts"
        assert result["alerts"] == []