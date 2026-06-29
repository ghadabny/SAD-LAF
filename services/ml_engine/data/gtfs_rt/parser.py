import re
from dataclasses import dataclass

import pandas as pd
from google.transit import gtfs_realtime_pb2


@dataclass(frozen=True)
class GTFSRTDataFrames:
    """Résultat du parsing : deux DataFrames prêts à être mergés."""
    stop_time_updates: pd.DataFrame
    cancelled_trips: pd.DataFrame


class GTFSRTParser:
    """
    Responsabilité unique : décoder les bytes protobuf en DataFrames pandas.

    Colonnes produites par stop_time_updates :
        trip_id       : str — identifiant complet du trip SNCF
        train_number  : str — numéro de train extrait du trip_id (clé de jointure)
        stop_id       : str — code UIC 8 chiffres (nettoyé du préfixe SNCF)
        delay_dep_sec : int — délai au départ en secondes
        delay_arr_sec : int — délai à l'arrivée en secondes

    Colonnes produites par cancelled_trips :
        trip_id      : str — identifiant complet du trip annulé

    Pourquoi train_number comme clé de jointure ?
        Les trip_id SNCF encodent la date de service en fin de chaîne.
        Le GTFS statique (snapshot mensuel) et le RT (temps réel)
        n'ont jamais la même date → jointure directe impossible.
        Le train_number (ex: '4402') est stable et présent dans les deux.
    """

    # Regex pour extraire le numéro de train depuis un trip_id SNCF
    # Exemple : 'OCESN4402F1187_F:IC:...' → '4402'
    _TRAIN_NUMBER_PATTERN = re.compile(r'OCE[A-Z]+(\d+)[A-Z]')

    # Regex pour extraire le code UIC depuis un stop_id SNCF
    # Exemple : 'StopPoint:OCEINTERCITES-87481002' → '87481002'
    _STOP_ID_PATTERN = re.compile(r'(\d{8})$')

    def parse(self, trip_updates_bytes: bytes, service_alerts_bytes: bytes) -> GTFSRTDataFrames:
        """
        Décode les deux flux protobuf.

        Paramètres:
            trip_updates_bytes  : contenu brut du flux Trip Updates
            service_alerts_bytes: contenu brut du flux Service Alerts

        Retourne un GTFSRTDataFrames.
        """
        stop_time_updates = self._parse_trip_updates(trip_updates_bytes)
        cancelled_trips = self._parse_service_alerts(service_alerts_bytes)
        return GTFSRTDataFrames(
            stop_time_updates=stop_time_updates,
            cancelled_trips=cancelled_trips,
        )

    def _parse_trip_updates(self, raw_bytes: bytes) -> pd.DataFrame:
        feed = gtfs_realtime_pb2.FeedMessage()
        feed.ParseFromString(raw_bytes)

        rows = []
        for entity in feed.entity:
            if not entity.HasField("trip_update"):
                continue
            trip_id = entity.trip_update.trip.trip_id
            train_number = self._extract_train_number(trip_id)
            for stu in entity.trip_update.stop_time_update:
                rows.append({
                    "trip_id":       trip_id,
                    "train_number":  train_number,
                    "stop_id":       self._clean_stop_id(stu.stop_id),
                    "delay_dep_sec": stu.departure.delay if stu.HasField("departure") else 0,
                    "delay_arr_sec": stu.arrival.delay if stu.HasField("arrival") else 0,
                })

        if not rows:
            return self._empty_stop_time_updates()

        return pd.DataFrame(rows).astype({
            "trip_id":       str,
            "train_number":  str,
            "stop_id":       str,
            "delay_dep_sec": int,
            "delay_arr_sec": int,
        })

    def _parse_service_alerts(self, raw_bytes: bytes) -> pd.DataFrame:
        feed = gtfs_realtime_pb2.FeedMessage()
        feed.ParseFromString(raw_bytes)

        trip_ids = []
        for entity in feed.entity:
            if not entity.HasField("alert"):
                continue
            if entity.alert.effect != gtfs_realtime_pb2.Alert.Effect.NO_SERVICE:
                    continue
            for informed in entity.alert.informed_entity:
                if informed.trip.trip_id:
                    trip_ids.append(informed.trip.trip_id)

        if not trip_ids:
            return pd.DataFrame({"trip_id": pd.Series([], dtype=str)})

        return pd.DataFrame({"trip_id": trip_ids}).drop_duplicates()

    def _extract_train_number(self, trip_id: str) -> str:
        """
        Extrait le numéro de train depuis le trip_id SNCF.

        Entrée : 'OCESN4402F1187_F:IC:FR:Line::...'
        Sortie : '4402'

        Retourne trip_id intact si le pattern ne matche pas.
        """
        match = self._TRAIN_NUMBER_PATTERN.search(trip_id)
        return match.group(1) if match else trip_id

    def _clean_stop_id(self, raw_id: str) -> str:
        """
        Normalise les stop_id du flux RT au format GTFS statique.

        Entrée : 'StopPoint:OCEINTERCITES-87481002'
        Sortie : '87481002'

        Retourne raw_id intact si le pattern ne matche pas.
        """
        match = self._STOP_ID_PATTERN.search(raw_id)
        return match.group(1) if match else raw_id

    @staticmethod
    def _empty_stop_time_updates() -> pd.DataFrame:
        return pd.DataFrame({
            "trip_id":       pd.Series([], dtype=str),
            "train_number":  pd.Series([], dtype=str),
            "stop_id":       pd.Series([], dtype=str),
            "delay_dep_sec": pd.Series([], dtype=int),
            "delay_arr_sec": pd.Series([], dtype=int),
        })