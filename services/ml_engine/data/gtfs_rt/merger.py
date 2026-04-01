import re

import pandas as pd


class GTFSRTMerger:
    """
    Responsabilité unique : fusionner les données RT avec les tronçons GTFS Static.

    Clé de jointure : (train_number, stop_id)
        - train_number : numéro court du train (ex: '4402'), stable entre
                         le GTFS statique (snapshot mensuel) et le RT (temps réel).
                         Les trip_id complets encodent la date de service → non utilisables.
        - stop_id      : code UIC 8 chiffres, normalisé par GTFSRTParser.

    Colonnes ajoutées aux tronçons :
        delay_dep_sec : int  — délai au départ en secondes (0 = à l'heure)
        delay_arr_sec : int  — délai à l'arrivée en secondes
        is_cancelled  : bool — True si le train figure dans les Service Alerts

    Stratégie de fallback :
        - Colonne train_number absente des tronçons → jointure sur trip_id
          (cas des tests unitaires avec fixtures minimales)
        - Tronçon sans correspondance RT → delay=0, is_cancelled=False
    """

    _TRAIN_NUMBER_PATTERN = re.compile(r'OCE[A-Z]+(\d+)[A-Z]')

    def merge(
        self,
        troncons: pd.DataFrame,
        stop_time_updates: pd.DataFrame,
        cancelled_trips: pd.DataFrame,
    ) -> pd.DataFrame:
        """Fusionne RT et Static. Retourne un nouveau DataFrame (pas de mutation)."""
        result = troncons.copy()
        result = self._normalize_keys(result)
        result = self._add_delays(result, stop_time_updates)
        result = self._add_cancellations(result, cancelled_trips)
        return result

    def _normalize_keys(self, troncons: pd.DataFrame) -> pd.DataFrame:
        """
        Force les colonnes de jointure en str pour garantir la compatibilité.

        Si train_number est absent (ex: fixtures de test), on le crée
        en copiant trip_id — la jointure échouera proprement sans KeyError.
        """
        result = troncons.copy()

        if "train_number" not in result.columns:
            result["train_number"] = result["trip_id"].astype(str)
        else:
            result["train_number"] = result["train_number"].astype(str)

        result["stop_id_dep"] = result["stop_id_dep"].astype(str)
        result["stop_id_arr"] = result["stop_id_arr"].astype(str)
        return result

    def _add_delays(self, troncons: pd.DataFrame, stu: pd.DataFrame) -> pd.DataFrame:
        if stu.empty:
            troncons["delay_dep_sec"] = 0
            troncons["delay_arr_sec"] = 0
            return troncons

        dep_delays = (
            stu[["train_number", "stop_id", "delay_dep_sec"]]
            .rename(columns={"stop_id": "stop_id_dep"})
        )
        arr_delays = (
            stu[["train_number", "stop_id", "delay_arr_sec"]]
            .rename(columns={"stop_id": "stop_id_arr"})
        )

        result = troncons.merge(dep_delays, on=["train_number", "stop_id_dep"], how="left")
        result = result.merge(arr_delays, on=["train_number", "stop_id_arr"], how="left")
        result["delay_dep_sec"] = result["delay_dep_sec"].fillna(0).astype(int)
        result["delay_arr_sec"] = result["delay_arr_sec"].fillna(0).astype(int)
        return result

    def _add_cancellations(self, troncons: pd.DataFrame, cancelled: pd.DataFrame) -> pd.DataFrame:
        if cancelled.empty:
            troncons["is_cancelled"] = False
            return troncons

        cancelled_numbers = set(
            cancelled["trip_id"].apply(self._extract_train_number)
        )
        troncons["is_cancelled"] = troncons["train_number"].isin(cancelled_numbers)
        return troncons

    def _extract_train_number(self, trip_id: str) -> str:
        """
        Extrait le numéro de train depuis un trip_id SNCF.

        Exemple : 'OCESN4402F1187_F:IC:...' → '4402'
        Dupliqué depuis GTFSRTParser pour respecter SRP :
        le Merger ne dépend pas du Parser.
        """
        match = self._TRAIN_NUMBER_PATTERN.search(trip_id)
        return match.group(1) if match else trip_id