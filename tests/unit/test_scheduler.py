import json
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

from services.scheduler.daily.daily_fetcher import DailyGTFSFetcher
from services.scheduler.realtime.realtime_fetcher import RealtimeFetcher


# ─────────────────────────────────────────────────────────────────────────────
# Tests DailyGTFSFetcher
# ─────────────────────────────────────────────────────────────────────────────

class TestDailyGTFSFetcher:

    def test_run_appelle_download_force_true(self):
        """
        run() doit appeler GTFSDownloader.download(force=True).
        On mock GTFSDownloader pour ne pas faire de vrai téléchargement.

        Pourquoi mock ?
            Un test unitaire ne doit jamais faire de requêtes réseau.
            Si le réseau est indisponible, le test échouerait pour la
            mauvaise raison. On teste uniquement que run() appelle
            download(force=True) — pas que le téléchargement fonctionne.
        """
        with patch(
            "services.scheduler.daily.daily_fetcher.GTFSDownloader"
        ) as MockDownloader:
            mock_instance = MockDownloader.return_value
            fetcher = DailyGTFSFetcher()
            fetcher.run()
            mock_instance.download.assert_called_once_with(force=True)

    def test_run_ne_propage_pas_exception(self):
        """
        Si GTFSDownloader lève une exception, run() ne doit pas la propager.
        Le scheduler doit continuer à tourner même si un téléchargement échoue.
        """
        with patch(
            "services.scheduler.daily.daily_fetcher.GTFSDownloader"
        ) as MockDownloader:
            mock_instance = MockDownloader.return_value
            mock_instance.download.side_effect = ConnectionError("réseau indisponible")

            fetcher = DailyGTFSFetcher()
            # Ne doit pas lever d'exception
            fetcher.run()


# ─────────────────────────────────────────────────────────────────────────────
# Tests RealtimeFetcher
# ─────────────────────────────────────────────────────────────────────────────

class TestRealtimeFetcher:

    @pytest.fixture
    def fetcher(self, tmp_path, monkeypatch):
        """
        Crée un RealtimeFetcher avec GTFS_RT_DIR pointant vers tmp_path.
        Évite d'écrire dans le vrai dossier data/raw/gtfs_rt pendant les tests.
        """
        from shared import config as cfg
        monkeypatch.setattr(cfg.config, "GTFS_RT_DIR", tmp_path)
        return RealtimeFetcher()

    # ── _save() ──────────────────────────────────────────────────────────────

    def test_save_cree_fichier_json(self, fetcher, tmp_path):
        """_save() doit créer un fichier JSON dans GTFS_RT_DIR."""
        data = {"fetched_at": "2024-09-02T08:23:00", "test": True}
        fetcher._save(data, "test.json")
        assert (tmp_path / "test.json").exists()

    def test_save_contenu_correct(self, fetcher, tmp_path):
        """Le contenu du fichier JSON doit correspondre aux données sauvegardées."""
        data = {"fetched_at": "2024-09-02T08:23:00", "alerts": []}
        fetcher._save(data, "alerts.json")

        with open(tmp_path / "alerts.json", encoding="utf-8") as f:
            loaded = json.load(f)

        assert loaded["fetched_at"] == "2024-09-02T08:23:00"
        assert loaded["alerts"] == []

    def test_save_ecrase_fichier_existant(self, fetcher, tmp_path):
        """Un second save() doit écraser le fichier précédent."""
        fetcher._save({"version": 1}, "test.json")
        fetcher._save({"version": 2}, "test.json")

        with open(tmp_path / "test.json", encoding="utf-8") as f:
            loaded = json.load(f)

        assert loaded["version"] == 2

    # ── _get_translated_text() ────────────────────────────────────────────────

    def test_get_translated_text_french(self, fetcher):
        """Doit retourner le texte français si disponible."""
        mock_translation = MagicMock()
        mock_translation.language = "fr"
        mock_translation.text = "Train supprimé"

        mock_ts = MagicMock()
        mock_ts.translation = [mock_translation]

        result = fetcher._get_translated_text(mock_ts)
        assert result == "Train supprimé"

    def test_get_translated_text_fallback(self, fetcher):
        """Si pas de français, retourne le premier texte disponible."""
        mock_translation = MagicMock()
        mock_translation.language = "en"
        mock_translation.text = "Train cancelled"

        mock_ts = MagicMock()
        mock_ts.translation = [mock_translation]

        result = fetcher._get_translated_text(mock_ts)
        assert result == "Train cancelled"

    def test_get_translated_text_empty(self, fetcher):
        """Si aucun texte disponible, retourne une chaîne vide."""
        mock_ts = MagicMock()
        mock_ts.translation = []

        result = fetcher._get_translated_text(mock_ts)
        assert result == ""

    # ── fetch_trip_updates() et fetch_service_alerts() ────────────────────────

    def test_fetch_trip_updates_ne_propage_pas_exception(self, fetcher):
        """
        Si le réseau est indisponible, fetch_trip_updates() ne doit pas
        propager l'exception — le scheduler doit continuer.
        """
        with patch.object(fetcher, "_fetch_protobuf", side_effect=ConnectionError("réseau")):
            fetcher.fetch_trip_updates()  # ne doit pas lever d'exception

    def test_fetch_service_alerts_ne_propage_pas_exception(self, fetcher):
        """Même comportement pour fetch_service_alerts()."""
        with patch.object(fetcher, "_fetch_protobuf", side_effect=ConnectionError("réseau")):
            fetcher.fetch_service_alerts()  # ne doit pas lever d'exception

    def test_run_appelle_les_deux_fetch(self, fetcher):
        """run() doit appeler fetch_trip_updates() ET fetch_service_alerts()."""
        with patch.object(fetcher, "fetch_trip_updates") as mock_tu, \
             patch.object(fetcher, "fetch_service_alerts") as mock_sa:
            fetcher.run()
            mock_tu.assert_called_once()
            mock_sa.assert_called_once()

    # ── _parse_trip_updates() ─────────────────────────────────────────────────

    def test_parse_trip_updates_structure(self, fetcher):
        """
        _parse_trip_updates() doit retourner un dict avec les clés
        fetched_at, source et trip_updates.
        On crée un FeedMessage mock pour ne pas dépendre du réseau.
        """
        mock_feed = MagicMock()
        mock_feed.entity = []  # feed vide — on teste juste la structure

        result = fetcher._parse_trip_updates(mock_feed)

        assert "fetched_at" in result
        assert result["source"] == "gtfs-rt-trip-updates"
        assert result["trip_updates"] == []

    def test_parse_service_alerts_structure(self, fetcher):
        """_parse_service_alerts() doit retourner la structure attendue."""
        mock_feed = MagicMock()
        mock_feed.entity = []

        result = fetcher._parse_service_alerts(mock_feed)

        assert "fetched_at" in result
        assert result["source"] == "gtfs-rt-service-alerts"
        assert result["alerts"] == []