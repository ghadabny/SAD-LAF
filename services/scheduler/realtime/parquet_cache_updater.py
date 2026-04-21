# services/scheduler/realtime/parquet_cache_updater.py
"""
SOLID — SRP : extraction de la responsabilité "mise à jour cache Parquet ML"
hors de RealtimeFetcher.

Avant ce refactoring, RealtimeFetcher cumulait 3 responsabilités :
    1. Fetch réseau des bytes protobuf
    2. Détection de changements (comparaison ancien/nouveau JSON)
    3. Mise à jour du cache Parquet pour le pipeline ML  ← extrait ici

Pourquoi cette séparation ?
    - Si le format Parquet change (colonne renommée, schéma ML évolue),
      on modifie ParquetCacheUpdater sans toucher à la logique de détection.
    - ParquetCacheUpdater est testable indépendamment avec de faux bytes protobuf.
    - RealtimeFetcher devient plus lisible : il ne sait plus rien de GTFSRTParser
      ni de GTFSRTCache — il délègue via l'interface de ParquetCacheUpdater.

SOLID — DIP :
    ParquetCacheUpdater reçoit GTFSRTParser et GTFSRTCache via son constructeur.
    En test, on injecte des doublures sans toucher au code de production.
"""
from __future__ import annotations

from services.ml_engine.data.gtfs_rt.parser import GTFSRTParser
from services.ml_engine.data.gtfs_rt.cache import GTFSRTCache


class ParquetCacheUpdater:
    """
    Responsabilité unique : mettre à jour le cache Parquet ML
    à partir de bytes protobuf GTFS-RT bruts.

    Utilisé par RealtimeFetcher après chaque fetch réseau réussi.
    Permet au pipeline GTFSRealtimeFeatureTransformer de scorer
    les tronçons avec des données temps réel sans faire d'appel réseau.

    Séquence :
        bytes protobuf → GTFSRTParser → DataFrames → GTFSRTCache → .parquet

    SOLID — DIP :
        parser et cache sont injectés, pas instanciés ici.
        En production : ParquetCacheUpdater()          (defaults)
        En test       : ParquetCacheUpdater(parser=FakeParser(), cache=FakeCache())
    """

    def __init__(
        self,
        parser: GTFSRTParser | None = None,
        cache: GTFSRTCache | None = None,
    ) -> None:
        self._parser = parser or GTFSRTParser()
        self._cache  = cache  or GTFSRTCache()

    def update(self, tu_bytes: bytes, sa_bytes: bytes) -> None:
        """
        Parse les bytes protobuf et persiste les DataFrames en Parquet.

        Non-bloquant : en cas d'erreur, log et continue.
        Le scheduler ne doit jamais planter à cause d'un cache ML.

        Paramètres :
            tu_bytes : bytes du flux Trip Updates (protobuf)
            sa_bytes : bytes du flux Service Alerts (protobuf)
        """
        try:
            rt_data = self._parser.parse(tu_bytes, sa_bytes)
            self._cache.save(rt_data.stop_time_updates, rt_data.cancelled_trips)
            print(
                f"[ParquetCacheUpdater] ✅ Cache mis à jour — "
                f"{len(rt_data.stop_time_updates):,} stop updates, "
                f"{len(rt_data.cancelled_trips):,} trips annulés."
            )
        except Exception as exc:
            print(f"[ParquetCacheUpdater] ⚠️  Erreur (non bloquant) : {exc}")