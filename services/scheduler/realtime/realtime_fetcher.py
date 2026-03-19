import json
from datetime import datetime
from pathlib import Path

import requests
from google.transit import gtfs_realtime_pb2

from shared.config import config
from services.scheduler.realtime.realtime_update import RealtimeUpdate


# Seuil de retard en secondes au-delà duquel on considère un train comme impacté.
# 5 minutes = 300 secondes — en dessous c'est une variation normale,
# pas assez significative pour recalculer une tournée.
DELAY_THRESHOLD_SECONDS: int = 300


class RealtimeFetcher:
    """
    Responsabilité unique : télécharger, décoder et comparer les flux GTFS-RT.

    Deux flux gérés :
        Trip Updates    : retards sur les trains en circulation
        Service Alerts  : suppressions et alertes sur les circulations

    Nouveauté par rapport à la version initiale :
        run() détecte maintenant les changements entre deux fetches
        et retourne un RealtimeUpdate indiquant quels trains sont impactés.
        Le scheduler décide ensuite si un recalcul est nécessaire.

    Séparation des responsabilités (S de SOLID) :
        RealtimeFetcher → détecte QUOI a changé
        Scheduler       → décide QUOI faire avec ce changement
        ml_engine       → recalcule les scores si demandé

    Usage :
        fetcher = RealtimeFetcher()
        update  = fetcher.run()
        if update.has_changed:
            print(f"{update.nb_impacted} trains impactés")
    """

    def __init__(self):
        self.trip_updates_url   = config.GTFS_RT_TRIP_UPDATES_URL
        self.service_alerts_url = config.GTFS_RT_SERVICE_ALERTS_URL
        self.output_dir         = config.GTFS_RT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Interface publique ────────────────────────────────────────────────────

    def run(self) -> RealtimeUpdate:
        """
        Télécharge les deux flux GTFS-RT et détecte les changements.

        Retourne un RealtimeUpdate indiquant :
            - si quelque chose a changé depuis le dernier fetch
            - quels trip_ids sont en retard ou supprimés

        On continue même si un flux échoue — si Trip Updates plante,
        on veut quand même récupérer Service Alerts.
        """
        print(
            f"[RealtimeFetcher] Fetch GTFS-RT "
            f"— {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        delayed   = self._run_trip_updates()
        cancelled = self._run_service_alerts()

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

    def fetch_trip_updates(self) -> None:
        """
        Télécharge et sauvegarde le flux Trip Updates sans détection de changement.
        Conservé pour la compatibilité avec les tests existants.
        """
        try:
            feed = self._fetch_protobuf(self.trip_updates_url)
            data = self._parse_trip_updates(feed)
            self._save(data, "trip_updates.json")
            print(
                f"[RealtimeFetcher] Trip Updates : "
                f"{len(data['trip_updates'])} trains récupérés."
            )
        except Exception as e:
            print(f"[RealtimeFetcher] Trip Updates : {e}")

    def fetch_service_alerts(self) -> None:
        """
        Télécharge et sauvegarde le flux Service Alerts sans détection de changement.
        Conservé pour la compatibilité avec les tests existants.
        """
        try:
            feed = self._fetch_protobuf(self.service_alerts_url)
            data = self._parse_service_alerts(feed)
            self._save(data, "service_alerts.json")
            print(
                f"[RealtimeFetcher] Service Alerts : "
                f"{len(data['alerts'])} alertes récupérées."
            )
        except Exception as e:
            print(f"[RealtimeFetcher] Service Alerts : {e}")

    # ── Méthodes privées — orchestration ─────────────────────────────────────

    def _run_trip_updates(self) -> set[str]:
        """
        Télécharge le flux Trip Updates, détecte les changements
        et sauvegarde si nécessaire.

        Retourne l'ensemble des trip_ids en retard > DELAY_THRESHOLD_SECONDS.
        Retourne un set vide si le fetch échoue ou si rien n'a changé.
        """
        try:
            feed    = self._fetch_protobuf(self.trip_updates_url)
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

        except Exception as e:
            print(f"[RealtimeFetcher] Trip Updates erreur : {e}")
            return set()

    def _run_service_alerts(self) -> set[str]:
        """
        Télécharge le flux Service Alerts, détecte les changements
        et sauvegarde si nécessaire.

        Retourne l'ensemble des trip_ids supprimés (NO_SERVICE).
        Retourne un set vide si le fetch échoue ou si rien n'a changé.
        """
        try:
            feed     = self._fetch_protobuf(self.service_alerts_url)
            nouveau  = self._parse_service_alerts(feed)
            ancien   = self._load_previous("service_alerts.json")

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

        except Exception as e:
            print(f"[RealtimeFetcher] Service Alerts erreur : {e}")
            return set()

    # ── Méthodes privées — détection de changement ───────────────────────────

    def _extract_delayed_trips(self, data: dict) -> set[str]:
        """
        Extrait les trip_ids dont le retard dépasse DELAY_THRESHOLD_SECONDS.

        On ne compare pas tout le fichier JSON — juste les trip_ids impactés.
        Pourquoi ?
            Le champ fetched_at change à chaque fetch même si les données
            sont identiques. Comparer le JSON entier déclencherait toujours
            un recalcul inutile.
        """
        delayed = set()
        for tu in data.get("trip_updates", []):
            for stu in tu.get("stop_time_updates", []):
                arr_delay = stu.get("arrival_delay_seconds") or 0
                dep_delay = stu.get("departure_delay_seconds") or 0
                if max(arr_delay, dep_delay) >= DELAY_THRESHOLD_SECONDS:
                    delayed.add(tu["trip_id"])
                    break  # un seul arrêt en retard suffit pour marquer le trip
        return delayed

    def _extract_cancelled_trips(self, data: dict) -> set[str]:
        """
        Extrait les trip_ids complètement supprimés (effect = NO_SERVICE).
        """
        return {
            trip_id
            for alert in data.get("alerts", [])
            if alert.get("effect") == "NO_SERVICE"
            for trip_id in alert.get("affected_trips", [])
        }

    def _load_previous(self, filename: str) -> dict | None:
        """
        Charge le fichier JSON du fetch précédent.
        Retourne None si le fichier n'existe pas encore
        (premier démarrage du scheduler).
        """
        path = self.output_dir / filename
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    # ── Méthodes privées — téléchargement et parsing ─────────────────────────

    def _fetch_protobuf(self, url: str):
        """
        Télécharge un flux GTFS-RT et le décode depuis le format Protobuf.
        """
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        feed = gtfs_realtime_pb2.FeedMessage()
        feed.ParseFromString(response.content)
        return feed

    def _parse_trip_updates(self, feed) -> dict:
        """
        Extrait les informations de retard depuis un FeedMessage Trip Updates.
        """
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
                "train_number":      trip.trip_headsign,
                "route_id":          trip.route_id,
                "stop_time_updates": stop_time_updates,
            })
        return {
            "fetched_at":   datetime.now().isoformat(),
            "source":       "gtfs-rt-trip-updates",
            "trip_updates": trip_updates,
        }

    def _parse_service_alerts(self, feed) -> dict:
        """
        Extrait les alertes de service depuis un FeedMessage Service Alerts.
        """
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

    def _get_translated_text(self, translated_string) -> str:
        """
        Extrait le texte français d'un TranslatedString GTFS-RT.
        Retourne le premier texte disponible si le français n'est pas trouvé.
        """
        for translation in translated_string.translation:
            if translation.language in ("fr", "fr-FR", ""):
                return translation.text
        if translated_string.translation:
            return translated_string.translation[0].text
        return ""

    def _save(self, data: dict, filename: str) -> Path:
        """
        Sauvegarde les données en JSON de façon atomique.
        Écrit dans un fichier temporaire puis renomme — évite
        qu'un lecteur lise un fichier partiellement écrit.
        """
        output_path = self.output_dir / filename
        tmp_path    = output_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp_path.replace(output_path)
        return output_path