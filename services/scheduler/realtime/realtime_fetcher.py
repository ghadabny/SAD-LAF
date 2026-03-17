import json
from datetime import datetime
from pathlib import Path

import requests
from google.transit import gtfs_realtime_pb2

from shared.config import config


class RealtimeFetcher:
    """
    Responsabilité unique : télécharger et décoder les flux GTFS-RT
    (Trip Updates et Service Alerts) puis les sauvegarder en JSON.

    Pourquoi JSON et pas Protobuf brut ?
        Le Protobuf est un format binaire optimisé pour le transport réseau
        mais difficile à lire et à requêter. On le décode une fois ici
        et on sauvegarde en JSON — ml_engine peut ensuite lire des fichiers
        JSON sans dépendre de la librairie Protobuf.

    Flux gérés :
        Trip Updates    : retards sur les trains en circulation
        Service Alerts  : suppressions et alertes sur les circulations

    Usage :
        fetcher = RealtimeFetcher()
        fetcher.fetch_trip_updates()
        fetcher.fetch_service_alerts()
        fetcher.run()  # les deux en une fois (appelé par APScheduler)
    """

    def __init__(self):
        self.trip_updates_url   = config.GTFS_RT_TRIP_UPDATES_URL
        self.service_alerts_url = config.GTFS_RT_SERVICE_ALERTS_URL
        self.output_dir         = config.GTFS_RT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Interface publique ────────────────────────────────────────────────────

    def run(self) -> None:
        """
        Télécharge et sauvegarde les deux flux GTFS-RT.
        Appelé toutes les 2 minutes par APScheduler.

        On continue même si un flux échoue — si Trip Updates plante,
        on veut quand même récupérer Service Alerts.
        """
        print(
            f"[RealtimeFetcher] Fetch GTFS-RT "
            f"— {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        self.fetch_trip_updates()
        self.fetch_service_alerts()

    def fetch_trip_updates(self) -> None:
        """
        Télécharge le flux Trip Updates (retards).

        Structure du fichier JSON produit :
        {
            "fetched_at": "2024-09-02T08:23:00",
            "source": "gtfs-rt-trip-updates",
            "trip_updates": [
                {
                    "trip_id": "TRIP_117756",
                    "train_number": "117756",
                    "route_id": "FR:Line::TER_ALSACE:",
                    "stop_time_updates": [
                        {
                            "stop_id": "87212027",
                            "stop_sequence": 0,
                            "arrival_delay_seconds": 120,
                            "departure_delay_seconds": 120,
                        },
                        ...
                    ]
                },
                ...
            ]
        }
        """
        try:
            feed = self._fetch_protobuf(self.trip_updates_url)
            data = self._parse_trip_updates(feed)
            self._save(data, "trip_updates.json")
            print(
                f"[RealtimeFetcher] ✅ Trip Updates : "
                f"{len(data['trip_updates'])} trains récupérés."
            )
        except Exception as e:
            print(f"[RealtimeFetcher] ❌ Trip Updates : {e}")

    def fetch_service_alerts(self) -> None:
        """
        Télécharge le flux Service Alerts (suppressions et alertes).

        Structure du fichier JSON produit :
        {
            "fetched_at": "2024-09-02T08:23:00",
            "source": "gtfs-rt-service-alerts",
            "alerts": [
                {
                    "alert_id": "...",
                    "cause": "STRIKE",
                    "effect": "NO_SERVICE",
                    "header": "Train supprimé",
                    "description": "Le train 117756 est supprimé...",
                    "affected_trips": ["TRIP_117756"],
                    "affected_stops": [],
                    "active_periods": [
                        {"start": "2024-09-02T06:00:00", "end": "2024-09-02T22:00:00"}
                    ]
                },
                ...
            ]
        }
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

    # ── Méthodes privées ──────────────────────────────────────────────────────

    def _fetch_protobuf(self, url: str):
        """
        Télécharge un flux GTFS-RT et le décode depuis le format Protobuf.

        Pourquoi Protobuf ?
            Le GTFS-RT est un standard Google basé sur Protocol Buffers —
            un format binaire compact. 'gtfs-realtime-bindings' est la
            librairie officielle Google pour décoder ce format en Python.

        Retourne un objet FeedMessage (structure Protobuf décodée).
        Lève une exception si le téléchargement ou le décodage échoue.
        """
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        feed = gtfs_realtime_pb2.FeedMessage()
        feed.ParseFromString(response.content)
        return feed

    def _parse_trip_updates(self, feed) -> dict:
        """
        Extrait les informations de retard depuis un FeedMessage Trip Updates.

        Pour chaque entité du feed :
            - On récupère le trip_id, route_id, trip_headsign (= numéro commercial)
            - Pour chaque arrêt : délai d'arrivée et de départ en secondes

        Le délai est en secondes dans le standard GTFS-RT.
        Ex: arrival_delay = 120 → 2 minutes de retard.
        """
        trip_updates = []

        for entity in feed.entity:
            if not entity.HasField("trip_update"):
                continue

            tu = entity.trip_update
            trip = tu.trip

            stop_time_updates = []
            for stu in tu.stop_time_update:
                stop_time_updates.append({
                    "stop_id":                   stu.stop_id,
                    "stop_sequence":             stu.stop_sequence,
                    "arrival_delay_seconds":     stu.arrival.delay  if stu.HasField("arrival")   else None,
                    "departure_delay_seconds":   stu.departure.delay if stu.HasField("departure") else None,
                })

            trip_updates.append({
                "trip_id":           trip.trip_id,
                "train_number":      trip.trip_headsign,
                "route_id":          trip.route_id,
                "stop_time_updates": stop_time_updates,
            })

        return {
            "fetched_at":    datetime.now().isoformat(),
            "source":        "gtfs-rt-trip-updates",
            "trip_updates":  trip_updates,
        }

    def _parse_service_alerts(self, feed) -> dict:
        """
        Extrait les alertes de service depuis un FeedMessage Service Alerts.

        Pour chaque entité du feed :
            - Cause (STRIKE, TECHNICAL_PROBLEM, etc.)
            - Effect (NO_SERVICE, REDUCED_SERVICE, etc.)
            - Textes header et description
            - Trips et arrêts concernés
            - Périodes d'activation

        Note SNCF : l'identifiant commun entre flux est le numéro commercial
        du train (trip_headsign), pas le trip_id technique.
        """
        alerts = []

        for entity in feed.entity:
            if not entity.HasField("alert"):
                continue

            alert = entity.alert

            # Trips et arrêts concernés
            affected_trips = []
            affected_stops = []
            for informed in alert.informed_entity:
                if informed.trip.trip_id:
                    affected_trips.append(informed.trip.trip_id)
                if informed.stop_id:
                    affected_stops.append(informed.stop_id)

            # Périodes d'activation
            active_periods = []
            for period in alert.active_period:
                active_periods.append({
                    "start": datetime.fromtimestamp(period.start).isoformat() if period.start else None,
                    "end":   datetime.fromtimestamp(period.end).isoformat()   if period.end   else None,
                })

            # Textes (GTFS-RT stocke les textes en plusieurs langues)
            header      = self._get_translated_text(alert.header_text)
            description = self._get_translated_text(alert.description_text)

            alerts.append({
                "alert_id":       entity.id,
                "cause":          gtfs_realtime_pb2.Alert.Cause.Name(alert.cause),
                "effect":         gtfs_realtime_pb2.Alert.Effect.Name(alert.effect),
                "header":         header,
                "description":    description,
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
        # Fallback : premier texte disponible quelle que soit la langue
        if translated_string.translation:
            return translated_string.translation[0].text
        return ""

    def _save(self, data: dict, filename: str) -> Path:
        """
        Sauvegarde les données décodées en JSON dans GTFS_RT_DIR.

        Pourquoi écraser le fichier précédent ?
            On veut toujours l'état le plus récent.
            L'historique n'est pas nécessaire ici — le modèle ML
            travaille sur les tronçons du jour, pas sur un historique
            de retards (ça c'est le rôle des données LAF).

        Le fichier est écrasé atomiquement : on écrit d'abord dans
        un fichier temporaire puis on renomme — évite qu'un lecteur
        lise un fichier partiellement écrit.
        """
        output_path = self.output_dir / filename
        tmp_path    = output_path.with_suffix(".tmp")

        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # Renommage atomique : sur Linux/Windows, replace() est atomique
        tmp_path.replace(output_path)
        return output_path