# services/ml_engine/data/cache.py
"""
Cache de checkpoints pour le préprocessing LAF/GTFS.

Problème résolu :
    Chaque entraînement retraitait ~35M lignes SC depuis zéro (~30min).
    Changer 3 hyperparamètres LightGBM impliquait une attente identique.

Solution :
    Les DataFrames intermédiaires sont sérialisés en Parquet.
    La clé de cache est basée sur l'empreinte (taille + mtime) des fichiers source.
    Si les fichiers source n'ont pas changé → on lit directement le Parquet → ~3s.

Granularité des niveaux de cache (définis dans train.py) :
    L1 - laf_unified_stats  : stats SC + CC + PV agrégées par troncon_id
         Invalidé si : n'importe quel fichier LAF change
    L2 - training_dataset   : L1 jointé avec les tronçons GTFS
         Invalidé si : fichiers LAF OU fichiers GTFS changent

Cas d'usage fréquents :
    ┌─────────────────────────────────────┬──────┬──────┐
    │ Changement                          │  L1  │  L2  │
    ├─────────────────────────────────────┼──────┼──────┤
    │ Hyperparamètres seulement           │  ✅  │  ✅  │ ~5s
    │ Nouvelle livraison LAF              │  ❌  │  ❌  │ ~30min
    │ Mise à jour GTFS seulement          │  ✅  │  ❌  │ ~5min
    │ Modification logique de features    │  ✅  │  ✅  │ ~5s *
    └─────────────────────────────────────┴──────┴──────┘
    * Les features sont calculées après le cache → toujours recalculées.
      Passer force_recompute=True si la logique d'agrégation LAF elle-même change.

Usage typique :
    cache = DataCache(config.PROCESSED_DIR)
    key   = cache.make_key("laf_unified_stats", loader.get_all_laf_paths())
    df    = cache.get(key)
    if df is None:
        df = compute_expensive_stats(...)
        cache.set(key, df)
"""
import hashlib
import pandas as pd
from pathlib import Path


class DataCache:
    """
    Gestionnaire de cache Parquet basé sur l'empreinte des fichiers source.

    Responsabilité unique : persister et restituer des DataFrames intermédiaires.
    Ne sait rien du contenu — c'est le rôle des fonctions appelantes.
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Interface publique ────────────────────────────────────────────────────

    def make_key(self, prefix: str, source_paths: list[Path]) -> str:
        """
        Génère une clé de cache lisible : "{prefix}_{fingerprint16hex}".

        Exemples :
            "laf_unified_stats_a1b2c3d4e5f6g7h8"
            "training_dataset_1234567890abcdef"

        Paramètres :
            prefix       : nom logique du checkpoint (ex: "laf_unified_stats")
            source_paths : fichiers dont dépend ce checkpoint
        """
        fp = self._fingerprint(source_paths)
        return f"{prefix}_{fp}"

    def exists(self, key: str) -> bool:
        """Retourne True si le checkpoint existe sur le disque."""
        return self._path(key).exists()

    def get(self, key: str) -> pd.DataFrame | None:
        """
        Charge un DataFrame depuis le cache.

        Retourne None si :
            - l'entrée n'existe pas (cache MISS normal)
            - le fichier Parquet est corrompu (recalcul automatique)

        Dans les deux cas l'appelant doit recalculer et appeler set().
        """
        path = self._path(key)
        if not path.exists():
            print(f"[Cache] MISS : {key}")
            return None
        try:
            df = pd.read_parquet(path)
            size_kb = path.stat().st_size // 1024
            print(
                f"[Cache] ✅ HIT  : {path.name} "
                f"({size_kb:,} Ko, {len(df):,} lignes) — préprocessing sauté."
            )
            return df
        except Exception as e:
            print(f"[Cache] ⚠️  Parquet corrompu ({path.name}) : {e} — recalcul forcé.")
            path.unlink(missing_ok=True)
            return None

    def set(self, key: str, df: pd.DataFrame) -> None:
        """
        Sauvegarde un DataFrame dans le cache.

        Format Parquet avec compression snappy :
            - Plus compact que CSV (souvent 5-10x)
            - Préserve les types pandas (int, float, datetime)
            - Lecture plus rapide que CSV pour les gros DataFrames
        """
        path = self._path(key)
        df.to_parquet(path, index=False, compression="snappy")
        size_kb = path.stat().st_size // 1024
        print(f"[Cache] 💾 MISS → sauvegardé : {path.name} ({size_kb:,} Ko)")

    def invalidate_prefix(self, prefix: str) -> int:
        """
        Supprime toutes les entrées de cache commençant par ce préfixe.

        Usage : forcer un recalcul complet d'une étape spécifique.
            cache.invalidate_prefix("laf_unified_stats")  # recalcule tout le LAF
            cache.invalidate_prefix("training_dataset")   # recalcule seulement la jointure

        Retourne le nombre de fichiers supprimés.
        """
        removed = 0
        for f in self.cache_dir.glob(f"{prefix}_*.parquet"):
            f.unlink()
            print(f"[Cache] 🗑️  Invalidé : {f.name}")
            removed += 1
        if removed == 0:
            print(f"[Cache] ℹ️  Aucun cache trouvé pour le préfixe '{prefix}'.")
        return removed

    def list_entries(self) -> list[dict]:
        """
        Retourne la liste des entrées de cache avec leurs métadonnées.
        Utile pour auditer ce qui est en cache ou libérer de l'espace.
        """
        entries = []
        for f in sorted(self.cache_dir.glob("*.parquet")):
            stat = f.stat()
            entries.append({
                "key":      f.stem,
                "size_kb":  stat.st_size // 1024,
                "created":  pd.Timestamp(stat.st_mtime, unit="s"),
            })
        return entries

    def clear_all(self) -> int:
        """Supprime tous les fichiers de cache. À utiliser avec précaution."""
        removed = 0
        for f in self.cache_dir.glob("*.parquet"):
            f.unlink()
            removed += 1
        print(f"[Cache] 🗑️  {removed} fichier(s) de cache supprimé(s).")
        return removed

    # ── Méthodes privées ──────────────────────────────────────────────────────

    def _path(self, key: str) -> Path:
        """Retourne le chemin complet du fichier Parquet pour cette clé."""
        return self.cache_dir / f"{key}.parquet"

    def _fingerprint(self, paths: list[Path]) -> str:
        """
        Calcule une empreinte MD5 (16 hex) à partir des métadonnées des fichiers.

        Pour chaque fichier : concatène "nom:taille_octets:mtime_entier".
        Les fichiers sont triés → l'ordre d'appel n'impacte pas l'empreinte.
        Un fichier absent est encodé comme "nom:absent" → une nouvelle livraison
        (fichier présent vs absent) invalide bien le cache.

        Pourquoi MD5 et pas SHA256 ?
            MD5 suffit pour identifier des changements de fichiers (pas un usage
            cryptographique). 16 caractères hex sont suffisamment uniques
            pour distinguer des checkpoints différents.
        """
        h = hashlib.md5()
        for p in sorted(paths):
            if p.exists():
                stat = p.stat()
                # int(mtime) évite les différences de précision selon l'OS
                h.update(f"{p.name}:{stat.st_size}:{int(stat.st_mtime)}".encode())
            else:
                h.update(f"{p.name}:absent".encode())
        return h.hexdigest()[:16]