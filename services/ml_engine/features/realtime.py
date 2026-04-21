# services/ml_engine/features/realtime.py
"""
SOLID — DIP (v2) :
    GTFSRealtimeFeatureTransformer reçoit ses 4 dépendances via le constructeur
    au lieu de les instancier lui-même.

Avant :
    def __init__(self):
        self._fetcher = GTFSRTFetcher()   # new concret — impossible à mocker
        self._parser  = GTFSRTParser()    # idem
        self._merger  = GTFSRTMerger()    # idem
        self._cache   = GTFSRTCache()     # idem

Après :
    def __init__(self, fetcher=None, parser=None, merger=None, cache=None):
        self._fetcher = fetcher or GTFSRTFetcher()   # injectable
        ...

En production : GTFSRealtimeFeatureTransformer()         → defaults
En test       : GTFSRealtimeFeatureTransformer(
                    fetcher=FakeFetcher(),
                    cache=FakeCache(),
                )
→ aucun appel réseau, aucune écriture disque dans les tests.
"""
from __future__ import annotations

import pandas as pd

from services.ml_engine.data.gtfs_rt.fetcher import GTFSRTFetcher
from services.ml_engine.data.gtfs_rt.parser import GTFSRTParser
from services.ml_engine.data.gtfs_rt.merger import GTFSRTMerger
from services.ml_engine.data.gtfs_rt.cache import GTFSRTCache
from services.ml_engine.features.base import BaseFeatureTransformer


class GTFSRealtimeFeatureTransformer(BaseFeatureTransformer):
    """
    Enrichit les tronçons avec les features temps réel GTFS-RT.

    Features produites :
        delay_dep_sec   : int   — délai au départ en secondes (0 = à l'heure)
        delay_arr_sec   : int   — délai à l'arrivée en secondes
        delay_dep_min   : float — délai au départ en minutes (pour LightGBM)
        is_cancelled    : bool  — True si le trip est annulé
        is_delayed      : bool  — True si delay_dep_sec > 300 (5 min)

    Colonnes requises en entrée :
        trip_id      : str — clé de jointure avec le flux RT
        stop_id_dep  : str — arrêt d'origine du tronçon
        stop_id_arr  : str — arrêt de destination du tronçon

    Fallback :
        Si le flux RT est indisponible (réseau, API SNCF down),
        les colonnes sont remplies avec les valeurs par défaut (0 / False).
        Le pipeline ne plante pas — il dégrade gracieusement.

    SOLID — DIP :
        Les 4 dépendances (fetcher, parser, merger, cache) sont injectées.
        Aucun `new` dans le corps de la classe → testabilité maximale.

    Pourquoi fit() ne fait rien ?
        Les features RT sont calculatoires, pas statistiques.
        Même logique que TemporalFeatureTransformer.
    """

    DELAY_THRESHOLD_SECONDS: int = 300  # 5 minutes

    def __init__(
        self,
        fetcher: GTFSRTFetcher | None = None,
        parser:  GTFSRTParser  | None = None,
        merger:  GTFSRTMerger  | None = None,
        cache:   GTFSRTCache   | None = None,
    ) -> None:
        """
        Paramètres (tous optionnels) :
            fetcher : télécharge les flux RT depuis l'API SNCF
            parser  : décode le protobuf en DataFrames
            merger  : joint les tronçons statiques avec les données RT
            cache   : lit/écrit le cache Parquet local

        Usage production : GTFSRealtimeFeatureTransformer()
        Usage test       : GTFSRealtimeFeatureTransformer(fetcher=mock, cache=mock)
        """
        self._fetcher = fetcher or GTFSRTFetcher()
        self._parser  = parser  or GTFSRTParser()
        self._merger  = merger  or GTFSRTMerger()
        self._cache   = cache   or GTFSRTCache()

    def fit(self, df: pd.DataFrame) -> "GTFSRealtimeFeatureTransformer":
        """Rien à apprendre. Retourne self pour le chaînage."""
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrichit df avec les features RT.
        En cas d'échec du fetch, retourne df avec colonnes par défaut.
        """
        stop_time_updates, cancelled_trips = self._load_realtime_data()
        enriched = self._merger.merge(df, stop_time_updates, cancelled_trips)
        return self._compute_derived_features(enriched)

    # ── Méthodes privées ─────────────────────────────────────────────────────

    def _load_realtime_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Charge depuis cache si valide, sinon fetch et met en cache."""
        if self._cache.is_valid():
            return self._cache.load()
        return self._fetch_and_cache()

    def _fetch_and_cache(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch les deux flux, parse, met en cache. Fallback sur erreur."""
        try:
            feed    = self._fetcher.fetch()
            rt_data = self._parser.parse(
                feed.trip_updates_bytes,
                feed.service_alerts_bytes,
            )
            self._cache.save(rt_data.stop_time_updates, rt_data.cancelled_trips)
            return rt_data.stop_time_updates, rt_data.cancelled_trips
        except Exception as exc:
            print(
                f"[GTFSRealtimeFeatureTransformer] WARN: fetch RT échoué "
                f"({exc}). Fallback zéros."
            )
            return self._parser._empty_stop_time_updates(), self._empty_cancelled()

    def _compute_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcule les features dérivées des colonnes brutes."""
        result = df.copy()
        result["delay_dep_min"] = result["delay_dep_sec"] / 60.0
        result["is_delayed"]    = result["delay_dep_sec"] > self.DELAY_THRESHOLD_SECONDS
        return result

    @staticmethod
    def _empty_cancelled() -> pd.DataFrame:
        return pd.DataFrame({"trip_id": pd.Series([], dtype=str)})