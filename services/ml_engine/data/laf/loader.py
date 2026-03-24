# services/ml_engine/data/laf/loader.py
import pandas as pd
from pathlib import Path

from shared.config import config


class LAFLoader:
    """
    Responsabilité unique : lire les fichiers CSV bruts LAF depuis le disque.

    Trois sources de données :
        CC   (Contrôles Comportés)  : ~1,7M lignes, séparateur ';'
        PV   (Procès-Verbaux)       : ~0,6M lignes, séparateur ','
        SCAN (Titres scannés)       : ~35M lignes, fichiers trimestriels,
                                      même structure que CC, séparateur ';'

    Ne nettoie pas, ne transforme pas — c'est le rôle de LAFPreprocessor.
    Retourne des DataFrames bruts avec les types par défaut de pandas.

    Particularité SCAN : les fichiers sont découpés par trimestre
    (scan_YYYY_QN.csv) pour des raisons de volumétrie. load_scan()
    les concatène en un seul DataFrame. Pour les cas où la mémoire
    est limitée, utiliser load_scan_chunked() qui retourne un itérateur.

    Usage :
        loader = LAFLoader()
        cc   = loader.load_cc()
        pv   = loader.load_pv()
        scan = loader.load_scan()           # tout en mémoire (~35M lignes)
        # OU
        for chunk in loader.load_scan_chunked(chunksize=500_000):
            process(chunk)
    """

    # Colonnes à forcer en string pour éviter la perte de zéros initiaux
    # sur les codes UIC (ex: "08714007" → ne pas laisser pandas inférer int)
    _UIC_COLUMNS_CC = [
        "station_uicCode",
        "ticket_travelInformation_origin_uicCode",
        "ticket_travelInformation_destination_uicCode",
    ]
    _UIC_COLUMNS_PV = [
        "course_origin_uicCode",
        "course_destination_uicCode",
        "penalties_origin_uicCode",
        "penalties_destination_uicCode",
    ]

    def __init__(self):
        self.laf_dir = config.LAF_DIR

    # ── Interface publique ────────────────────────────────────────────────────

    def load_cc(self) -> pd.DataFrame:
        """
        Charge le fichier CC (Contrôles Comportés).

        Séparateur : point-virgule (';')
        Encoding   : utf-8 (à ajuster si les fichiers SNCF sont en latin-1)
        Volume     : ~1,7M lignes

        Retourne un DataFrame avec toutes les colonnes du fichier source.
        Lève FileNotFoundError si le fichier est absent.
        """
        path = self._get_path("cc_2022_2026.csv")
        print(f"[LAFLoader] Chargement CC depuis : {path}")
        df = pd.read_csv(
            path,
            sep=";",
            dtype={col: str for col in self._UIC_COLUMNS_CC},
            low_memory=False,
        )
        print(f"[LAFLoader] CC chargé : {len(df):,} lignes, {len(df.columns)} colonnes.")
        return df

    def load_pv(self) -> pd.DataFrame:
        """
        Charge le fichier PV (Procès-Verbaux).

        Séparateur : virgule (',')
        Volume     : ~0,6M lignes

        Retourne un DataFrame avec toutes les colonnes du fichier source.
        Lève FileNotFoundError si le fichier est absent.
        """
        path = self._get_path("pv_2022_2026.csv")
        print(f"[LAFLoader] Chargement PV depuis : {path}")
        df = pd.read_csv(
            path,
            sep=",",
            dtype={col: str for col in self._UIC_COLUMNS_PV},
            low_memory=False,
        )
        print(f"[LAFLoader] PV chargé : {len(df):,} lignes, {len(df.columns)} colonnes.")
        return df

    def load_scan(self) -> pd.DataFrame:
        """
        Charge et concatène tous les fichiers SCAN trimestriels.

        Les fichiers sont nommés scan_YYYY_QN.csv (ex: scan_2024_Q1.csv).
        Ils ont la même structure que CC et utilisent ';' comme séparateur.

        ATTENTION : ~35M lignes au total → peut nécessiter 4-6 Go de RAM.
        Préférer load_scan_chunked() si la mémoire est limitée.

        Lève ValueError si aucun fichier SCAN n'est trouvé.
        """
        fichiers = sorted(self.laf_dir.glob("scan_*.csv"))
        if not fichiers:
            raise ValueError(
                f"[LAFLoader] Aucun fichier SCAN trouvé dans '{self.laf_dir}'.\n"
                f"Fichiers attendus : scan_YYYY_QN.csv (ex: scan_2024_Q1.csv)"
            )

        print(f"[LAFLoader] {len(fichiers)} fichier(s) SCAN trouvé(s). Concaténation...")
        chunks = []
        for f in fichiers:
            df = pd.read_csv(
                f,
                sep=";",
                dtype={col: str for col in self._UIC_COLUMNS_CC},
                low_memory=False,
            )
            print(f"[LAFLoader]   - {f.name} : {len(df):,} lignes")
            chunks.append(df)

        result = pd.concat(chunks, ignore_index=True)
        print(f"[LAFLoader] SCAN total : {len(result):,} lignes.")
        return result

    def load_scan_chunked(self, chunksize: int = 500_000):
        """
        Itérateur sur les fichiers SCAN par chunks de `chunksize` lignes.

        Utilise ce mode quand load_scan() dépasse la RAM disponible.
        Chaque itération retourne un DataFrame de taille <= chunksize.

        Usage :
            for chunk in loader.load_scan_chunked(chunksize=500_000):
                df_processed = preprocessor.transform(chunk)
                features.append(df_processed)

        Lève ValueError si aucun fichier SCAN n'est trouvé.
        """
        fichiers = sorted(self.laf_dir.glob("scan_*.csv"))
        if not fichiers:
            raise ValueError(
                f"[LAFLoader] Aucun fichier SCAN trouvé dans '{self.laf_dir}'."
            )

        for f in fichiers:
            print(f"[LAFLoader] SCAN chunked : {f.name}")
            reader = pd.read_csv(
                f,
                sep=";",
                dtype={col: str for col in self._UIC_COLUMNS_CC},
                low_memory=False,
                chunksize=chunksize,
            )
            yield from reader

    # ── Méthode privée ────────────────────────────────────────────────────────

    def _get_path(self, filename: str) -> Path:
        """
        Retourne le chemin complet vers un fichier LAF.
        Lève FileNotFoundError si le fichier est absent — fail-fast.
        """
        path = self.laf_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"[LAFLoader] '{filename}' introuvable dans '{self.laf_dir}'.\n"
                f"Vérifie que le fichier a bien été déposé dans : {self.laf_dir}"
            )
        return path