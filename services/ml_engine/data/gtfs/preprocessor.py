import pandas as pd


class GTFSPreprocessor:
    """
    Responsabilité unique : nettoyer et construire les tronçons ferroviaires
    à partir des DataFrames bruts fournis par GTFSLoader.

    Ne sait pas d'où viennent les données (fichiers, API, etc.).
    Reçoit des DataFrames en entrée, retourne des DataFrames en sortie.

    Usage:
        preprocessor = GTFSPreprocessor()
        troncons = preprocessor.build_troncons(
            stop_times, trips, stops, routes
        )
    """

    # Préfixes SNCF à supprimer des stop_id pour obtenir le code UIC (8 chiffres)
    # Ordre important : du plus spécifique au plus générique
    STOP_ID_PREFIXES = [
        "StopPoint:OCETrain TER-",
        "StopPoint:OCETGV INOUI-",
        "StopPoint:OCETransilien-",
        "StopPoint:OCE",
    ]

    # route_type=2 = Rail régional (TER, Intercités)
    TER_ROUTE_TYPE = 2

    # ----------------------------------------------------------------
    # Méthodes de nettoyage
    # Chacune a une responsabilité unique et est testable isolément
    # ----------------------------------------------------------------

    def clean_stop_id(self, stop_id: str) -> str:
        """
        Supprime les préfixes SNCF dans un stop_id pour obtenir
        le code UIC de la gare (8 chiffres).

        Exemples :
            "StopPoint:OCETrain TER-87723197" → "87723197"
            "StopPoint:OCETGV INOUI-87723197" → "87723197"
            "87723197"                         → "87723197" (inchangé)
        """
        for prefix in self.STOP_ID_PREFIXES:
            if isinstance(stop_id, str) and stop_id.startswith(prefix):
                return stop_id[len(prefix):]
        return stop_id

    def parse_gtfs_time(self, time_series: pd.Series) -> pd.Series:
        """
        Convertit une série d'heures GTFS en minutes depuis minuit.

        Nécessaire car le GTFS peut dépasser 24:00:00 pour les trains
        de nuit (convention GTFS officielle).

        Exemples :
            "09:16:00" → 556  minutes
            "25:30:00" → 1530 minutes (train de nuit)
            valeur manquante → -1 (valeur sentinelle)
        """
        def _to_minutes(t: str) -> int:
            try:
                h, m, s = map(int, str(t).strip().split(":"))
                return h * 60 + m
            except Exception:
                return -1

        return time_series.apply(_to_minutes)

    def filter_ter(
        self,
        trips: pd.DataFrame,
        routes: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Filtre les trips pour ne conserver que les TER.

        Paramètres:
            trips  : DataFrame brut de trips.txt
            routes : DataFrame brut de routes.txt

        Retourne:
            trips filtré sur les lignes de type rail régional (route_type=2)
        """
        ter_route_ids = routes[
            routes["route_type"] == self.TER_ROUTE_TYPE
        ]["route_id"]

        filtered = trips[trips["route_id"].isin(ter_route_ids)].copy()
        print(
            f"[GTFSPreprocessor] Filtre TER : "
            f"{len(filtered):,} trips conservés sur {len(trips):,} total."
        )
        return filtered

    # ----------------------------------------------------------------
    # Construction des tronçons
    # Méthode principale — orchestre les étapes dans l'ordre
    # ----------------------------------------------------------------

    def build_troncons(
        self,
        stop_times: pd.DataFrame,
        trips: pd.DataFrame,
        stops: pd.DataFrame,
        routes: pd.DataFrame,
        ter_only: bool = True,
    ) -> pd.DataFrame:
        """
        Construit la table des tronçons origine-destination.

        Un tronçon = un train entre deux arrêts CONSÉCUTIFS.
        C'est l'unité de granularité du scoring LAF.

        Le principe repose sur un shift(-1) :
            ligne N   : stop_id_dep, dep_minutes  (arrêt courant)
            ligne N+1 : stop_id_arr, arr_minutes  (arrêt suivant)

        Paramètres:
            stop_times : DataFrame brut de stop_times.txt
            trips      : DataFrame brut de trips.txt
            stops      : DataFrame brut de stops.txt
            routes     : DataFrame brut de routes.txt
            ter_only   : True  = filtre sur TER uniquement (recommandé)
                         False = conserve tous les trains (TGV, TER, etc.)

        Retourne un DataFrame avec les colonnes :
            trip_id        : identifiant technique du trip
            train_number   : numéro commercial du train (ex: 117756)
            service_id     : identifiant du calendrier de circulation
            stop_sequence  : numéro de l'arrêt de départ dans le trip
            stop_id_dep    : code UIC gare de départ
            stop_name_dep  : nom lisible gare de départ
            dep_minutes    : heure de départ en minutes depuis minuit
            stop_id_arr    : code UIC gare d'arrivée
            stop_name_arr  : nom lisible gare d'arrivée
            arr_minutes    : heure d'arrivée en minutes depuis minuit
            duration_min   : durée du tronçon en minutes
        """
        print("[GTFSPreprocessor] Construction des tronçons en cours...")

        # Étape 1 : filtrer sur les TER si demandé
        if ter_only:
            trips = self.filter_ter(trips, routes)

        # Étape 2 : joindre trip_headsign et service_id sur stop_times
        # trip_headsign = numéro commercial du train (ex: 117756)
        trips_light = trips[["trip_id", "trip_headsign", "service_id"]].copy()
        stop_times = stop_times.merge(trips_light, on="trip_id", how="inner")

        # Étape 3 : nettoyer les stop_id (supprimer préfixes SNCF)
        stops = stops.copy()
        stops["stop_id_clean"] = stops["stop_id"].apply(self.clean_stop_id)
        stop_times["stop_id_clean"] = stop_times["stop_id"].apply(self.clean_stop_id)

        # Étape 4 : joindre les noms de gares lisibles
        # On ne garde que les arrêts (location_type=0), pas les stations mères
        stops_light = (
            stops[stops["location_type"] == 0][["stop_id_clean", "stop_name"]]
            .drop_duplicates("stop_id_clean")
        )
        stop_times = stop_times.merge(stops_light, on="stop_id_clean", how="left")

        # Étape 5 : convertir les heures en minutes depuis minuit
        stop_times["dep_minutes"] = self.parse_gtfs_time(stop_times["departure_time"])
        stop_times["arr_minutes"] = self.parse_gtfs_time(stop_times["arrival_time"])

        # Étape 6 : trier par trip et séquence d'arrêt
        # CRUCIAL : l'ordre doit être correct avant le shift
        stop_times = stop_times.sort_values(
            ["trip_id", "stop_sequence"]
        ).reset_index(drop=True)

        # Étape 7 : shift(-1) pour obtenir l'arrêt SUIVANT sur chaque ligne
        # Avant shift :                   Après shift(-1) sur colonnes _arr :
        # trip_id  stop_name              stop_id_arr   arr_minutes
        # TRAIN_A  Strasbourg      →      Sélestat      ...
        # TRAIN_A  Sélestat        →      Colmar        ...
        # TRAIN_A  Colmar          →      (TRAIN_B)     ← supprimé étape 8
        next_stop = stop_times.shift(-1)

        troncons = pd.DataFrame({
            "trip_id":       stop_times["trip_id"],
            "train_number":  stop_times["trip_headsign"],
            "service_id":    stop_times["service_id"],
            "stop_sequence": stop_times["stop_sequence"],
            "stop_id_dep":   stop_times["stop_id_clean"],
            "stop_name_dep": stop_times["stop_name"],
            "dep_minutes":   stop_times["dep_minutes"],
            "stop_id_arr":   next_stop["stop_id_clean"],
            "stop_name_arr": next_stop["stop_name"],
            "arr_minutes":   next_stop["arr_minutes"],
        })

        # Étape 8 : supprimer les fausses lignes de fin de trip
        # Quand on fait shift(-1), la dernière ligne de chaque trip
        # récupère les données du trip suivant → c'est une erreur à supprimer
        mask = stop_times["trip_id"] == next_stop["trip_id"]
        troncons = troncons[mask].reset_index(drop=True)

        # Étape 9 : calculer la durée du tronçon en minutes
        troncons["duration_min"] = troncons["arr_minutes"] - troncons["dep_minutes"]

        # Étape 10 : supprimer les tronçons avec données invalides
        # (heures manquantes = valeur sentinelle -1, durées négatives)
        before = len(troncons)
        troncons = troncons[
            (troncons["dep_minutes"] >= 0) &
            (troncons["arr_minutes"] >= 0) &
            (troncons["duration_min"] > 0)
        ].reset_index(drop=True)
        removed = before - len(troncons)
        if removed > 0:
            print(f"[GTFSPreprocessor] {removed} tronçons invalides supprimés.")

        print(
            f"[GTFSPreprocessor] {len(troncons):,} tronçons construits "
            f"({troncons['trip_id'].nunique():,} trips, "
            f"{troncons['stop_name_dep'].nunique():,} gares de départ uniques)."
        )
        return troncons