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

    # route_type=2 = Rail régional (TER, Intercités)
    TER_ROUTE_TYPE = 2

    # Préfixes de stop_id SNCF — extraits en constante de classe (principe O).
    # Si SNCF ajoute un nouveau format, on modifie UNIQUEMENT cette liste,
    # pas la logique de clean_stop_id().
    SNCF_STOP_AREA_PREFIXES: tuple[str, ...] = (
        "StopArea:OCE",
        "StopPoint:OCE",
    )

    # Colonnes finales retournées par build_troncons()
    TRONCON_COLUMNS = [
        "trip_id", "train_number", "service_id", "stop_sequence",
        "stop_id_dep", "stop_name_dep", "dep_minutes",
        "stop_id_arr", "stop_name_arr", "arr_minutes",
        "duration_min",
    ]

    # ── Méthodes publiques ────────────────────────────────────────────────────

    def clean_stop_id(self, stop_id: str) -> str:
        """
        Extrait le code UIC (8 chiffres) depuis un stop_id GTFS SNCF.

        Tous les formats SNCF suivent le même pattern :
            préfixe-XXXXXXXX  →  on prend ce qui est après le dernier tiret

        Exemples :
            "StopPoint:OCETrain TER-87723197"  → "87723197"
            "StopPoint:OCECar TER-87723197"    → "87723197"
            "StopPoint:OCELyria-87723197"      → "87723197"
            "StopArea:OCE87723197"             → "87723197"
            "87723197"                         → "87723197" (inchangé)

        Pourquoi cette approche plutôt qu'une liste de préfixes ?
            Une liste doit être mise à jour chaque fois que SNCF ajoute
            un nouveau service. Prendre ce qui suit le dernier tiret
            fonctionne pour tous les formats présents et futurs.
            C'est le O de SOLID : ouvert à l'extension, fermé à la modification.
        """
        if not isinstance(stop_id, str):
            return stop_id
        # Cas 1 : format avec tiret — on prend tout ce qui suit le dernier tiret.
        # Robuste à tout nouveau service SNCF sans modifier cette méthode.
        if "-" in stop_id:
            return stop_id.rsplit("-", 1)[-1]
        # Cas 2 : format sans tiret — on supprime le préfixe connu.
        for prefix in self.SNCF_STOP_AREA_PREFIXES:
            if stop_id.startswith(prefix):
                return stop_id[-8:]
        return stop_id

    def parse_gtfs_time(self, time_series: pd.Series) -> pd.Series:
        """
        Convertit une série d'heures GTFS en minutes depuis minuit.

        Gère les heures > 24:00:00 (trains de nuit — convention GTFS officielle).

        Exemples :
            "09:16:00" → 556  minutes
            "25:30:00" → 1530 minutes (train de nuit)
            valeur invalide → -1 (valeur sentinelle)
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
        routes: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Filtre les trips pour ne conserver que les TER (route_type=2).
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

        Paramètres:
            stop_times : DataFrame brut de stop_times.txt
            trips      : DataFrame brut de trips.txt
            stops      : DataFrame brut de stops.txt
            routes     : DataFrame brut de routes.txt
            ter_only   : True = filtre sur TER uniquement (recommandé)

        Retourne un DataFrame avec les colonnes définies dans TRONCON_COLUMNS.
        """
        print("[GTFSPreprocessor] Construction des tronçons en cours...")

        if ter_only:
            trips = self.filter_ter(trips, routes)

        stop_times = self._join_trips(stop_times, trips)
        stop_times = self._join_stop_names(stop_times, stops)
        stop_times = self._convert_times(stop_times)
        stop_times = self._sort(stop_times)
        troncons   = self._apply_shift(stop_times)
        troncons   = self._filter_invalid(troncons)

        print(
            f"[GTFSPreprocessor] {len(troncons):,} tronçons construits "
            f"({troncons['trip_id'].nunique():,} trips, "
            f"{troncons['stop_name_dep'].nunique():,} gares de départ uniques)."
        )
        return troncons[self.TRONCON_COLUMNS].reset_index(drop=True)

    # ── Méthodes privées — une étape = une méthode ────────────────────────────

    def _join_trips(
        self,
        stop_times: pd.DataFrame,
        trips: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Joint trip_headsign (numéro commercial) et service_id sur stop_times.
        On ne garde que les colonnes utiles de trips pour éviter les collisions.
        """
        trips_light = trips[["trip_id", "trip_headsign", "service_id"]].copy()
        return stop_times.merge(trips_light, on="trip_id", how="inner")

    def _join_stop_names(
        self,
        stop_times: pd.DataFrame,
        stops: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Nettoie les stop_id et joint les noms de gares lisibles.
        On ne garde que les arrêts physiques (location_type=0),
        pas les stations mères (location_type=1).
        """
        stops = stops.copy()
        stops["stop_id_clean"] = stops["stop_id"].apply(self.clean_stop_id)
        stop_times["stop_id_clean"] = stop_times["stop_id"].apply(self.clean_stop_id)

        stops_light = (
            stops[stops["location_type"] == 0][["stop_id_clean", "stop_name"]]
            .drop_duplicates("stop_id_clean")
        )
        return stop_times.merge(stops_light, on="stop_id_clean", how="left")

    def _convert_times(self, stop_times: pd.DataFrame) -> pd.DataFrame:
        """
        Convertit departure_time et arrival_time en minutes depuis minuit.
        Gère les heures > 24:00 (trains de nuit).
        """
        stop_times = stop_times.copy()
        stop_times["dep_minutes"] = self.parse_gtfs_time(stop_times["departure_time"])
        stop_times["arr_minutes"] = self.parse_gtfs_time(stop_times["arrival_time"])
        return stop_times

    def _sort(self, stop_times: pd.DataFrame) -> pd.DataFrame:
        """
        Trie par trip_id puis stop_sequence.
        CRUCIAL : le tri doit être correct avant le shift(-1).
        Si les lignes ne sont pas dans l'ordre, le shift créerait
        des tronçons entre des arrêts non consécutifs.
        """
        return stop_times.sort_values(
            ["trip_id", "stop_sequence"]
        ).reset_index(drop=True)

    def _apply_shift(self, stop_times: pd.DataFrame) -> pd.DataFrame:
        """
        Construit les tronçons via shift(-1).

        Principe : décale les colonnes d'arrêt d'une ligne vers le haut.
        Chaque ligne obtient alors les infos de l'arrêt SUIVANT.

            Avant shift :          Après shift(-1) :
            Strasbourg 08:23  →   stop_id_arr = Sélestat, arr_min = 532
            Sélestat   08:52  →   stop_id_arr = Colmar,   arr_min = 555
            Colmar     09:15  →   stop_id_arr = NaN  ← supprimé ensuite

        Le tri préalable (_sort) garantit que les trips sont regroupés.
        On filtre ensuite les fausses lignes de fin de trip
        (trip_id courant ≠ trip_id décalé).
        """
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

        # Supprimer les fausses lignes de fin de trip
        mask = stop_times["trip_id"] == next_stop["trip_id"]
        troncons = troncons[mask].reset_index(drop=True)

        troncons["duration_min"] = troncons["arr_minutes"] - troncons["dep_minutes"]
        return troncons

    def _filter_invalid(self, troncons: pd.DataFrame) -> pd.DataFrame:
        """
        Supprime les tronçons avec des données invalides :
            - dep_minutes ou arr_minutes = -1 (heure GTFS non parseable)
            - duration_min <= 0 (durée négative ou nulle)
        """
        before = len(troncons)
        troncons = troncons[
            (troncons["dep_minutes"] >= 0) &
            (troncons["arr_minutes"] >= 0) &
            (troncons["duration_min"] > 0)
        ]
        removed = before - len(troncons)
        if removed > 0:
            print(f"[GTFSPreprocessor] {removed} tronçons invalides supprimés.")
        return troncons