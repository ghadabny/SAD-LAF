import pandas as pd
from shared.config import config


class GTFSLoader:
    """
    Responsabilité unique : lire les fichiers GTFS bruts en DataFrames.

    Ne télécharge pas, ne nettoie pas, ne transforme pas.
    Chaque méthode retourne exactement ce qui est dans le fichier .txt,
    sans aucune modification.

    Usage:
        loader = GTFSLoader()
        stop_times = loader.load_stop_times()
        trips      = loader.load_trips()
    """

    def __init__(self):
        self.gtfs_dir = config.GTFS_DIR

    def _load(self, filename: str) -> pd.DataFrame:
        """
        Méthode interne : charge un fichier .txt GTFS (format CSV standard).
        Toutes les méthodes publiques passent par ici.

        Paramètres:
            filename : nom du fichier (ex: "stop_times.txt")

        Lève FileNotFoundError si le fichier est absent.
        """
        path = self.gtfs_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"[GTFSLoader] '{filename}' introuvable dans '{self.gtfs_dir}'.\n"
                f"             Lance d'abord GTFSDownloader().download()."
            )
        return pd.read_csv(path, low_memory=False)

    def load_stop_times(self) -> pd.DataFrame:
        """
        Horaires de passage à chaque arrêt pour chaque trip.

        Colonnes principales :
            trip_id          : identifiant technique du trip
            arrival_time     : heure d'arrivée (format HH:MM:SS, peut dépasser 24h)
            departure_time   : heure de départ (format HH:MM:SS, peut dépasser 24h)
            stop_id          : identifiant de l'arrêt (avec préfixe SNCF)
            stop_sequence    : numéro d'ordre de l'arrêt dans le trip
            pickup_type      : 0=normal, 1=pas de montée
            drop_off_type    : 0=normal, 1=pas de descente
        """
        return self._load("stop_times.txt")

    def load_trips(self) -> pd.DataFrame:
        """
        Un trip = une circulation d'un train sur une journée donnée.

        Colonnes principales :
            route_id         : identifiant de la ligne commerciale
            service_id       : identifiant du calendrier de circulation
            trip_id          : identifiant technique unique du trip
            trip_headsign    : numéro commercial du train (ex: 117756)
            direction_id     : sens de circulation (0 ou 1)
        """
        return self._load("trips.txt")

    def load_stops(self) -> pd.DataFrame:
        """
        Liste de tous les arrêts avec leurs informations géographiques.

        Colonnes principales :
            stop_id          : identifiant de l'arrêt (avec préfixe SNCF)
            stop_name        : nom lisible de la gare (ex: "Strasbourg")
            stop_lat         : latitude GPS
            stop_lon         : longitude GPS
            location_type    : 0=arrêt, 1=station mère
            parent_station   : station mère si location_type=0
        """
        return self._load("stops.txt")

    def load_routes(self) -> pd.DataFrame:
        """
        Lignes commerciales (TER, TGV, Transilien, etc.).

        Colonnes principales :
            route_id         : identifiant de la ligne
            agency_id        : identifiant de l'agence opératrice
            route_short_name : nom court de la ligne (ex: "P53")
            route_long_name  : nom complet (ex: "Bening - Sarreguemines")
            route_type       : type de transport
                               2 = Rail régional (TER, Intercités)
                               3 = Bus
        """
        return self._load("routes.txt")

    def load_calendar_dates(self) -> pd.DataFrame:
        """
        Jours de circulation par service_id (exceptions au calendrier).

        Colonnes :
            service_id       : identifiant du service
            date             : date au format YYYYMMDD (ex: 20260501)
            exception_type   : 1 = le service circule ce jour
                               2 = le service ne circule pas ce jour
        """
        return self._load("calendar_dates.txt")