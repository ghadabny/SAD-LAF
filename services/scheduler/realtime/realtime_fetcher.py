# services/scheduler/realtime/realtime_fetcher.py
"""
SOLID — SRP (v2) :
    RealtimeFetcher ne gère plus que 2 responsabilités :
        1. Fetch réseau des bytes protobuf bruts
        2. Détection de changements (comparaison ancien/nouveau état)

    La 3e responsabilité (mise à jour cache Parquet ML) est déléguée à
    ParquetCacheUpdater via injection de dépendance.

SOLID — DIP :
    ParquetCacheUpdater est injecté dans le constructeur.
    En test, on passe une doublure sans appel réseau ni écriture disque.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import requests
from google.transit import gtfs_realtime_pb2

from shared.config import config
from services.scheduler.realtime.realtime_update import RealtimeUpdate
from services.scheduler.realtime.parquet_cache_updater import ParquetCacheUpdater


DELAY_THRESHOLD_SECONDS: int = 300


class RealtimeFetcher:
    """
    Responsabilités (après refactoring SRP) :
        1. Télécharger les flux GTFS-RT (bytes protobuf, 1 appel réseau par flux)
        2. Détecter les changements par comparaison JSON ancien/nouveau

    Ce que cette classe NE FAIT PLUS (délégué à ParquetCacheUpdater) :
        - Parser les protobuf en DataFrames pandas
        - Écrire les fichiers .parquet pour le pipeline ML

    SOLID — DIP :
        cache_updater est injecté. En production, ParquetCacheUpdater() par défaut.
        En test, on injecte un FakeCacheUpdater qui ne touche pas au disque.

    Séquence d'un cycle run() :
        1. _fetch_bytes()                    → bytes bruts (1 appel réseau par flux)
        2. _run_detection_trip_updates()     → set[trip_id] retardés
        3. _run_detection_service_alerts()   → set[trip_id] supprimés
        4. cache_updater.update()            → cache Parquet ML (délégué)
        5. RealtimeUpdate                    → retourné au scheduler
    """

    def __init__(
        self,
        cache_updater: ParquetCacheUpdater | None = None,
    ) -> None:
        """
        Paramètre :
            cache_updater : gestionnaire du cache Parquet ML.
                            Par défaut : ParquetCacheUpdater() (production).
                            Injectez un mock en test pour éviter les I/O disque.
        """
        self.trip_updates_url   = config.GTFS_RT_TRIP_UPDATES_URL
        self.service_alerts_url = config.GTFS_RT_SERVICE_ALERTS_URL
        self.output_dir         = config.GTFS_RT_DIR
        self._cache_updater     = cache_updater or ParquetCacheUpdater()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Interface publique ────────────────────────────────────────────────────

    def run(self) -> RealtimeUpdate:
        """
        Cycle complet : fetch → détection → cache Parquet → RealtimeUpdate.

        Resilient par design :
            - Si Trip Updates échoue, on continue avec Service Alerts.
            - Si le cache Parquet échoue, le RealtimeUpdate est quand même retourné.
        """
        print(
            f"[RealtimeFetcher] Fetch GTFS-RT "
            f"— {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        # ── Étape 1 : fetch bytes (1 appel réseau par flux) ───────────────────
        tu_bytes = self._fetch_bytes(self.trip_updates_url, "Trip Updates")
        sa_bytes = self._fetch_bytes(self.service_alerts_url, "Service Alerts")

        # ── Étape 2 : détection de changements ───────────────────────────────
        delayed   = self._run_detection_trip_updates(tu_bytes)
        cancelled = self._run_detection_service_alerts(sa_bytes)

        # ── Étape 3 : mise à jour cache Parquet (délégué) ────────────────────
        # Réutilise les bytes déjà téléchargés — pas de second appel réseau.
        if tu_bytes and sa_bytes:
            self._cache_updater.update(tu_bytes, sa_bytes)

        # ── Étape 4 : retourner le RealtimeUpdate ─────────────────────────────
        has_changed = bool(delayed or cancelled)
        update = RealtimeUpdate(
            has_changed=has_changed,
            trips_delayed=delayed,
            trips_cancelled=cancelled,
        )

        if has_changed:
            print(
                f"[RealtimeFetcher] Changement détecté : "
                f"{len(delayed)} retards, {len(cancelled)} suppressions "
                f"→ {update.nb_impacted} trains à recalculer."
            )
        else:
            print("[RealtimeFetcher] Aucun changement détecté.")

        return update

    # ── Pipeline détection — Trip Updates ────────────────────────────────────

    def _run_detection_trip_updates(self, raw_bytes: bytes | None) -> set[str]:
        """
        Détecte les changements dans le flux Trip Updates.
        Retourne les trip_ids en retard > DELAY_THRESHOLD_SECONDS.
        Retourne set vide si bytes absents ou pas de changement.
        """
        if not raw_bytes:
            return set()
        try:
            feed    = gtfs_realtime_pb2.FeedMessage()
            feed.ParseFromString(raw_bytes)
            nouveau = self._parse_trip_updates(feed)
            ancien  = self._load_previous("trip_updates.json")

            delayed        = self._extract_delayed_trips(nouveau)
            ancien_delayed = self._extract_delayed_trips(ancien) if ancien else set()

            if delayed != ancien_delayed:
                self._save(nouveau, "trip_updates.json")
                print(
                    f"[RealtimeFetcher] Trip Updates : "
                    f"{len(nouveau['trip_updates'])} trains, "
                    f"{len(delayed)} en retard significatif."
                )
                return delayed

            return set()

        except Exception as exc:
            print(f"[RealtimeFetcher] Trip Updates erreur : {exc}")
            return set()

    def _run_detection_service_alerts(self, raw_bytes: bytes | None) -> set[str]:
        """
        Détecte les changements dans le flux Service Alerts.
        Retourne les trip_ids supprimés (effect = NO_SERVICE).
        Retourne set vide si bytes absents ou pas de changement.
        """
        if not raw_bytes:
            return set()
        try:
            feed    = gtfs_realtime_pb2.FeedMessage()
            feed.ParseFromString(raw_bytes)
            nouveau = self._parse_service_alerts(feed)
            ancien  = self._load_previous("service_alerts.json")

            cancelled        = self._extract_cancelled_trips(nouveau)
            ancien_cancelled = self._extract_cancelled_trips(ancien) if ancien else set()

            if cancelled != ancien_cancelled:
                self._save(nouveau, "service_alerts.json")
                print(
                    f"[RealtimeFetcher] Service Alerts : "
                    f"{len(nouveau['alerts'])} alertes, "
                    f"{len(cancelled)} suppressions."
                )
                return cancelled

            return set()

        except Exception as exc:
            print(f"[RealtimeFetcher] Service Alerts erreur : {exc}")
            return set()

    # ── Téléchargement réseau ─────────────────────────────────────────────────

    def _fetch_bytes(self, url: str, label: str) -> bytes | None:
        """
        Télécharge les bytes bruts d'un flux RT.
        Retourne None en cas d'erreur réseau sans propager l'exception.
        Les bytes sont réutilisés par détection ET cache (pas de double fetch).
        """
        try:
            proxies = (
                {"http": config.HTTP_PROXY, "https": config.HTTPS_PROXY}
                if config.HTTP_PROXY
                else None
            )
            response = requests.get(url, timeout=30, proxies=proxies)
            response.raise_for_status()
            return response.content
        except Exception as exc:
            print(f"[RealtimeFetcher] {label} fetch erreur : {exc}")
            return None

    # ── Parsing JSON (pipeline détection) ────────────────────────────────────

    def _parse_trip_updates(self, feed) -> dict:
        """Décode un FeedMessage Trip Updates en dict JSON pour la détection."""
        trip_updates = []
        for entity in feed.entity:
            if not entity.HasField("trip_update"):
                continue
            tu   = entity.trip_update
            trip = tu.trip
            stop_time_updates = []
            for stu in tu.stop_time_update:
                stop_time_updates.append({
                    "stop_id":                 stu.stop_id,
                    "stop_sequence":           stu.stop_sequence,
                    "arrival_delay_seconds":   stu.arrival.delay  if stu.HasField("arrival")   else None,
                    "departure_delay_seconds": stu.departure.delay if stu.HasField("departure") else None,
                })
            trip_updates.append({
                "trip_id":           trip.trip_id,
                "train_number":      trip.trip_headsign if trip.trip_headsign else "",
                "route_id":          trip.route_id,
                "stop_time_updates": stop_time_updates,
            })
        return {
            "fetched_at":   datetime.now().isoformat(),
            "source":       "gtfs-rt-trip-updates",
            "trip_updates": trip_updates,
        }

    def _parse_service_alerts(self, feed) -> dict:
        """Décode un FeedMessage Service Alerts en dict JSON pour la détection."""
        alerts = []
        for entity in feed.entity:
            if not entity.HasField("alert"):
                continue
            alert          = entity.alert
            affected_trips = []
            affected_stops = []
            for informed in alert.informed_entity:
                if informed.trip.trip_id:
                    affected_trips.append(informed.trip.trip_id)
                if informed.stop_id:
                    affected_stops.append(informed.stop_id)
            active_periods = []
            for period in alert.active_period:
                active_periods.append({
                    "start": datetime.fromtimestamp(period.start).isoformat() if period.start else None,
                    "end":   datetime.fromtimestamp(period.end).isoformat()   if period.end   else None,
                })
            alerts.append({
                "alert_id":       entity.id,
                "cause":          gtfs_realtime_pb2.Alert.Cause.Name(alert.cause),
                "effect":         gtfs_realtime_pb2.Alert.Effect.Name(alert.effect),
                "header":         self._get_translated_text(alert.header_text),
                "description":    self._get_translated_text(alert.description_text),
                "affected_trips": affected_trips,
                "affected_stops": affected_stops,
                "active_periods": active_periods,
            })
        return {
            "fetched_at": datetime.now().isoformat(),
            "source":     "gtfs-rt-service-alerts",
            "alerts":     alerts,
        }

    # ── Détection de changement ───────────────────────────────────────────────

    def _extract_delayed_trips(self, data: dict) -> set[str]:
        """Extrait les trip_ids en retard > DELAY_THRESHOLD_SECONDS."""
        delayed = set()
        for tu in data.get("trip_updates", []):
            for stu in tu.get("stop_time_updates", []):
                arr_delay = stu.get("arrival_delay_seconds") or 0
                dep_delay = stu.get("departure_delay_seconds") or 0
                if max(arr_delay, dep_delay) >= DELAY_THRESHOLD_SECONDS:
                    delayed.add(tu["trip_id"])
                    break
        return delayed

    def _extract_cancelled_trips(self, data: dict) -> set[str]:
        """Extrait les trip_ids complètement supprimés (effect = NO_SERVICE)."""
        return {
            trip_id
            for alert in data.get("alerts", [])
            if alert.get("effect") == "NO_SERVICE"
            for trip_id in alert.get("affected_trips", [])
        }

    def _load_previous(self, filename: str) -> dict | None:
        """Charge le JSON du fetch précédent pour comparaison."""
        path = self.output_dir / filename
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    # ── Persistance JSON ──────────────────────────────────────────────────────

    def _get_translated_text(self, translated_string) -> str:
        """Extrait le texte français d'un TranslatedString GTFS-RT."""
        for translation in translated_string.translation:
            if translation.language in ("fr", "fr-FR", ""):
                return translation.text
        if translated_string.translation:
            return translated_string.translation[0].text
        return ""

    def _save(self, data: dict, filename: str) -> Path:
        """Sauvegarde JSON de façon atomique (write tmp → rename)."""
        output_path = self.output_dir / filename
        tmp_path    = output_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp_path.replace(output_path)
        return output_path

    # ── Méthodes publiques conservées pour compatibilité tests ───────────────

    def fetch_trip_updates(self) -> None:
        """Conservé pour compatibilité avec les tests existants."""
        raw = self._fetch_bytes(self.trip_updates_url, "Trip Updates")
        if raw:
            feed = gtfs_realtime_pb2.FeedMessage()
            feed.ParseFromString(raw)
            data = self._parse_trip_updates(feed)
            self._save(data, "trip_updates.json")
            print(f"[RealtimeFetcher] Trip Updates : {len(data['trip_updates'])} trains récupérés.")

    def fetch_service_alerts(self) -> None:
        """Conservé pour compatibilité avec les tests existants."""
        raw = self._fetch_bytes(self.service_alerts_url, "Service Alerts")
        if raw:
            feed = gtfs_realtime_pb2.FeedMessage()
            feed.ParseFromString(raw)
            data = self._parse_service_alerts(feed)
            self._save(data, "service_alerts.json")
            print(f"[RealtimeFetcher] Service Alerts : {len(data['alerts'])} alertes récupérées.")