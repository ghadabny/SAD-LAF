from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from shared.config import config


class GTFSRTCache:
    """
    Responsabilité unique : persister et invalider le cache Parquet des données RT.

    Pourquoi Parquet ?
        - Lecture 10x plus rapide que CSV pour les DataFrames pandas
        - Préserve les types (int, bool) sans ambiguïté
        - Compatible avec la stack pandas existante du projet

    Le cache est invalidé après TTL_MINUTES minutes.
    En production, le pipeline est lancé 1×/jour donc le cache
    sert surtout en dev pour éviter de spammer l'API SNCF.
    """

    TTL_MINUTES: int = 10
    _TU_FILENAME = "stop_time_updates.parquet"
    _SA_FILENAME = "cancelled_trips.parquet"
    _META_FILENAME = "meta.txt"

    def __init__(self):
        self._cache_dir: Path = config.GTFS_RT_CACHE_DIR

    def is_valid(self) -> bool:
        """Retourne True si le cache existe et n'a pas expiré."""
        meta_path = self._cache_dir / self._META_FILENAME
        if not meta_path.exists():
            return False
        saved_at = datetime.fromisoformat(meta_path.read_text().strip())
        return datetime.now() - saved_at < timedelta(minutes=self.TTL_MINUTES)

    def save(self, stop_time_updates: pd.DataFrame, cancelled_trips: pd.DataFrame) -> None:
        """Persiste les deux DataFrames et horodate le cache."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        stop_time_updates.to_parquet(self._cache_dir / self._TU_FILENAME, index=False)
        cancelled_trips.to_parquet(self._cache_dir / self._SA_FILENAME, index=False)
        (self._cache_dir / self._META_FILENAME).write_text(datetime.now().isoformat())
        print(f"[GTFSRTCache] Cache sauvegardé ({len(stop_time_updates):,} stop updates).")

    def load(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Charge les DataFrames depuis le cache Parquet."""
        stu = pd.read_parquet(self._cache_dir / self._TU_FILENAME)
        cancelled = pd.read_parquet(self._cache_dir / self._SA_FILENAME)
        print(f"[GTFSRTCache] Cache chargé ({len(stu):,} stop updates).")
        return stu, cancelled