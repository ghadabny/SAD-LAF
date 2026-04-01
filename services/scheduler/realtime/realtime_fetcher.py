import json
from datetime import datetime
from pathlib import Path

import requests
from google.transit import gtfs_realtime_pb2

from shared.config import config
from services.scheduler.realtime.realtime_update import RealtimeUpdate
from services.ml_engine.data.gtfs_rt import GTFSRTParser, GTFSRTCache


DELAY_THRESHOLD_SECONDS: int = 300


class RealtimeFetcher:
    """
    Responsabilité unique : télécharger, décoder et synchroniser les flux GTFS-RT.

    Ce service fait le pont entre les deux pipelines RT du projet :

        Pipeline 1 — Détection (scheduler) :
            Télécharge les flux protobuf, les décode en JSON lisible,
            compare avec le fetch précédent et retourne un RealtimeUpdate
            indiquant quels trains ont changé de statut.
            Écrit dans : data/raw/gtfs_rt/trip_updates.json
                         data/raw/gtfs_rt/service_alerts.json

        Pipeline 2 — Features ML (ml_engine) :
            Alimente le cache Parquet lu par GTFSRealtimeFeatureTransformer
            lors du scoring LightGBM. Sans ce cache, l'API score sans données RT.
            Écrit dans : data/cache/gtfs_rt/stop_time_updates.parquet
                         data/cache/gtfs_rt/cancelled_trips.parquet
                         data/cache/gtfs_rt/meta.txt

    Pourquoi le scheduler alimente le cache Parquet et pas l'API ?
        Le scheduler est le seul service avec accès réseau planifié aux flux RT.
        L'API ne doit jamais fetcher elle-même — elle lit uniquement depuis le cache.
        Un seul appel réseau toutes les 2 minutes alimente les deux pipelines.

    Séquence d'un cycle complet :
        1. _fetch_protobuf()         → bytes bruts (1 seul appel réseau par flux)
        2. _parse_trip_updates()     → dict JSON   → trip_updates.json
        3. _parse_service_alerts()   → dict JSON   → service_alerts.json
        4. _refresh_parquet_cache()  → DataFrames  → *.parquet  (via GTFSRTParser)
        5. Retourne RealtimeUpdate   → scheduler décide si recalcul nécessaire
    """

    def __init__(self):
        self.trip_updates_url   = config.GTFS_RT_TRIP_UPDATES_URL
        self.service_alerts_url = config.GTFS_RT_SERVICE_ALERTS_URL
        self.output_dir         = config.GTFS_RT_DIR
        self._parser_ml         = GTFSRTParser()
        self._cache             = GTFSRTCache()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Interface publique ────────────────────────────────────────────────────

    def run(self) -> RealtimeUpdate:
        """
        Cycle complet : fetch → détection → cache Parquet → RealtimeUpdate.

        On continue même si un flux échoue — si Trip Updates plante,
        on veut quand même récupérer Service Alerts.
        """
        print(
            f"[RealtimeFetcher] Fetch GTFS-RT "
            f"— {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        # ── Étape 1 : fetch des bytes bruts (1 appel réseau par flux) ─────────
        tu_bytes = self._fetch_bytes(self.trip_updates_url, "Trip Updates")
        sa_bytes = self._fetch_bytes(self.service_alerts_url, "Service Alerts")

        # ── Étape 2 : pipeline de détection (JSON pour comparaison) ──────────
        delayed   = self._run_detection_trip_updates(tu_bytes)
        cancelled = self._run_detection_service_alerts(sa_bytes)

        # ── Étape 3 : pipeline ML (Parquet pour GTFSRealtimeFeatureTransformer)
        if tu_bytes and sa_bytes:
            self._refresh_parquet_cache(tu_bytes, sa_bytes)

        # ── Étape 4 : construire et retourner le RealtimeUpdate ───────────────
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

    # ── Pipeline 2 — Cache Parquet pour le ML ────────────────────────────────

    def _refresh_parquet_cache(self, tu_bytes: bytes, sa_bytes: bytes) -> None:
        """
        Alimente le cache Parquet utilisé par GTFSRealtimeFeatureTransformer.

        Réutilise les bytes déjà téléchargés — pas de second appel réseau.
        GTFSRTParser (ml_engine) décode le protobuf en DataFrames pandas,
        GTFSRTCache persiste en Parquet avec horodatage TTL.

        En cas d'erreur : log + continue sans planter le scheduler.
        """
        try:
            rt_data = self._parser_ml.parse(tu_bytes, sa_bytes)
            self._cache.save(rt_data.stop_time_updates, rt_data.cancelled_trips)
            print(
                f"[RealtimeFetcher] Cache Parquet mis a jour — "
                f"{len(rt_data.stop_time_updates):,} stop updates, "
                f"{len(rt_data.cancelled_trips):,} trips annules."
            )
        except Exception as exc:
            print(f"[RealtimeFetcher] Cache Parquet erreur (non bloquant) : {exc}")

    # ── Pipeline 1 — Détection de changements (JSON) ─────────────────────────

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

        Retourne None en cas d'erreur réseau sans propager l'exception —
        le scheduler doit continuer même si un flux est indisponible.
        Les bytes sont réutilisés par les deux pipelines (pas de double fetch).
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