# services/ml_engine/data/laf/loader.py
import pandas as pd
from pathlib import Path

from shared.config import config


class LAFLoader:
    """
    Responsabilité unique : lire les fichiers CSV bruts LAF depuis le disque.

    Trois sources de données :
        CC   (Contrôles Comportés)  : ~1,7M lignes, séparateur ';', encoding latin-1
             Nommage réel : Extract_DATA_GE_CC_*.csv
        SC   (Titres Scannés)       : ~35M lignes (fichiers trimestriels),
             même structure que CC, séparateur ',', encoding latin-1
             Nommage réel : Extract_DATA_GE_SC_*.csv
        PV   (Procès-Verbaux)       : ~0,6M lignes, séparateur ',', encoding latin-1
             Nommage réel : Extract_DATA_GE_PV_*.csv

    Pourquoi des patterns glob plutôt que des noms en dur ?
        Les fichiers sont livrés avec la date d'extraction dans leur nom :
        "Extract_DATA_GE_CC_202201-202602_20260320.csv"
        Ce nom changera à chaque nouvelle livraison. Un nom en dur casserait
        le loader dès la prochaine extraction. Le pattern *_CC_* est robuste.

    Pourquoi fail-soft (retour DataFrame vide) plutôt que fail-fast ?
        Le pipeline tourne chaque nuit. Si un seul fichier est absent,
        on veut que le reste du pipeline continue avec les sources disponibles.
        L'appelant décide si l'absence est bloquante ou non.
        Le DataFrame vide retourné a le bon schéma de colonnes → le preprocessor
        ne plantera pas sur un accès à une colonne inexistante.

    Pourquoi encoding latin-1 ?
        Les fichiers SNCF contiennent des caractères accentués (noms de gares,
        motifs de refus : "Titre périmé"). Ils sont encodés en latin-1,
        pas en UTF-8. Forcer UTF-8 lèverait un UnicodeDecodeError.

    Usage :
        loader = LAFLoader()
        cc = loader.load_cc()          # DataFrame ou vide si absent
        pv = loader.load_pv()
        sc = loader.load_sc()          # tout en mémoire (~35M lignes)
        for chunk in loader.load_sc_chunked(500_000):
            process(chunk)
    """

    # Colonnes exactes des fichiers réels (ordre extrait des headers CSV)
    CC_COLUMNS: list[str] = [
        "course_companyCode",
        "course_courseNumber",
        "course_departureDate",
        "course_operatingCarrierCode",
        "course_transportationMode",
        "verifiedTickets_missionActivity",
        "station_label",
        "station_uicCode",
        "ticket_contractTariff",
        "ticket_contractType",
        "ticket_fareCode",
        "ticket_networkId",
        "ticket_passengerCount",
        "ticket_provider",
        "ticket_travelInformation_courseNumber",
        "ticket_travelInformation_departureDateTime",
        "ticket_travelInformation_destination_label",
        "ticket_travelInformation_destination_uicCode",
        "ticket_travelInformation_origin_label",
        "ticket_travelInformation_origin_uicCode",
        "ticket_travelInformation_spaceValidity",
        "ticket_travelInformation_travelClass",
        "ticket_type",
        "ticket_validFrom",
        "ticket_validTo",
        "ticketDocument_documentType",
        "verifiedTickets_type",
        "verifiedTickets_verificationDateTime",
        "verifiedTickets_verificationStatus",
        "verifiedTickets_verificationStatusReason",
    ]

    PV_COLUMNS: list[str] = [
        "course_companyCode",
        "course_transportationMode",
        "course_courseNumber",
        "course_departureDate",
        "course_operatingCarrierCode",
        "course_destination_label",
        "course_destination_uicCode",
        "course_origin_label",
        "course_origin_uicCode",
        "penalties_issueDateTime",
        "penalties_origin_label",
        "penalties_origin_uicCode",
        "penalties_destination_label",
        "penalties_destination_uicCode",
        "penalties_travelDistance",
        "penalties_perceptionFee_amountInCents",
        "penalties_flatFee_amountInCents",
        "penalties_applicationFee_amountInCents",
        "penalties_applicationFee_currency",
        "penalties_travelClass",
        "penalties_offenceType",
        "penalties_penaltyStatus",
        "penalties_identityStatementStatus",
        "penalties_trafficCode",
        "penalties_offenceLabel",
        "penalties_penaltyCode",
        "penalties_penaltyType",
        "penalties_ticketAttached",
        "penalties_otherAttachment",
        "penalties_offender_identityDocument_documentType",
    ]

    # Colonnes UIC à forcer en str pour conserver les zéros initiaux
    # (ex: un code "08714007" serait tronqué en 8714007 si lu en int)
    _UIC_COLUMNS_CC: list[str] = [
        "station_uicCode",
        "ticket_travelInformation_origin_uicCode",
        "ticket_travelInformation_destination_uicCode",
    ]
    _UIC_COLUMNS_PV: list[str] = [
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

        Pattern : *_CC_*.csv dans config.LAF_DIR
        Séparateur détecté automatiquement (';' dans les fichiers réels CC)
        Encoding : latin-1

        Retourne un DataFrame vide avec le bon schéma si le fichier est absent.
        """
        path = self._find_file("CC")
        if path is None:
            print("[LAFLoader] CC introuvable — retour DataFrame vide.")
            return self._empty_df(self.CC_COLUMNS)

        sep = self._detect_separator(path)
        print(f"[LAFLoader] Chargement CC : {path.name} (sep='{sep}')")
        df = pd.read_csv(
            path,
            sep=sep,
            encoding="latin-1",
            dtype={col: str for col in self._UIC_COLUMNS_CC},
            low_memory=False,
        )
        print(f"[LAFLoader] CC chargé : {len(df):,} lignes, {len(df.columns)} colonnes.")
        return df

    def load_pv(self) -> pd.DataFrame:
        """
        Charge le fichier PV (Procès-Verbaux).

        Pattern : *_PV_*.csv dans config.LAF_DIR
        Séparateur détecté automatiquement (',' dans les fichiers réels PV)
        Encoding : latin-1

        Retourne un DataFrame vide avec le bon schéma si le fichier est absent.
        """
        path = self._find_file("PV")
        if path is None:
            print("[LAFLoader] PV introuvable — retour DataFrame vide.")
            return self._empty_df(self.PV_COLUMNS)

        sep = self._detect_separator(path)
        print(f"[LAFLoader] Chargement PV : {path.name} (sep='{sep}')")
        df = pd.read_csv(
            path,
            sep=sep,
            encoding="latin-1",
            dtype={col: str for col in self._UIC_COLUMNS_PV},
            low_memory=False,
        )
        print(f"[LAFLoader] PV chargé : {len(df):,} lignes, {len(df.columns)} colonnes.")
        return df

    def load_sc(self) -> pd.DataFrame:
        """
        Charge et concatène tous les fichiers SC (Titres Scannés).

        Pattern : *_SC_*.csv dans config.LAF_DIR
        Plusieurs fichiers possibles (livraisons trimestrielles).
        ATTENTION : ~35M lignes au total → peut nécessiter 4-6 Go de RAM.
        Préférer load_sc_chunked() si la mémoire est limitée.

        Retourne un DataFrame vide avec le bon schéma si aucun fichier SC trouvé.
        """
        fichiers = self._find_files("SC")
        if not fichiers:
            print("[LAFLoader] SC introuvable — retour DataFrame vide.")
            return self._empty_df(self.CC_COLUMNS)

        print(f"[LAFLoader] {len(fichiers)} fichier(s) SC trouvé(s). Concaténation...")
        chunks = []
        for f in fichiers:
            sep = self._detect_separator(f)
            df = pd.read_csv(
                f,
                sep=sep,
                encoding="latin-1",
                dtype={col: str for col in self._UIC_COLUMNS_CC},
                low_memory=False,
            )
            print(f"[LAFLoader]   - {f.name} : {len(df):,} lignes")
            chunks.append(df)

        result = pd.concat(chunks, ignore_index=True)
        print(f"[LAFLoader] SC total : {len(result):,} lignes.")
        return result

    def load_sc_chunked(self, chunksize: int = 500_000):
        """
        Itérateur sur les fichiers SC par chunks de `chunksize` lignes.

        Utilise ce mode quand load_sc() dépasse la RAM disponible.
        Chaque itération retourne un DataFrame de taille <= chunksize.

        Si aucun fichier SC n'est trouvé, l'itérateur est vide (pas d'erreur).

        Usage :
            for chunk in loader.load_sc_chunked(500_000):
                stats = preprocessor.build_troncon_stats(
                    preprocessor.clean_cc(chunk)
                )
        """
        fichiers = self._find_files("SC")
        if not fichiers:
            print("[LAFLoader] SC introuvable — itérateur vide.")
            return

        for f in fichiers:
            sep = self._detect_separator(f)
            print(f"[LAFLoader] SC chunked : {f.name} (sep='{sep}')")
            reader = pd.read_csv(
                f,
                sep=sep,
                encoding="latin-1",
                dtype={col: str for col in self._UIC_COLUMNS_CC},
                low_memory=False,
                chunksize=chunksize,
            )
            yield from reader

    # ── Méthodes privées ──────────────────────────────────────────────────────

    def _find_file(self, tag: str) -> Path | None:
        """
        Cherche un fichier unique correspondant au pattern *_{TAG}_*.csv.

        Retourne le premier fichier trouvé, ou None si absent / dossier inexistant.
        Si plusieurs fichiers correspondent, log un avertissement et prend le premier
        (comportement prévisible plutôt qu'une exception mystérieuse).
        """
        if not self.laf_dir.exists():
            return None
        pattern = f"*{tag}*.csv"
        matches = sorted(self.laf_dir.glob(pattern))
        if not matches:
            return None
        if len(matches) > 1:
            print(
                f"[LAFLoader] Avertissement : {len(matches)} fichiers trouvés "
                f"pour le pattern {pattern}. Utilisation de : {matches[0].name}"
            )
        return matches[0]

    def _find_files(self, tag: str) -> list[Path]:
        """
        Cherche tous les fichiers correspondant au pattern *_{TAG}_*.csv.

        Retourne une liste vide si le dossier est absent ou si aucun fichier
        ne correspond.
        """
        if not self.laf_dir.exists():
            return []
        pattern = f"*{tag}*.csv"
        return sorted(self.laf_dir.glob(pattern))

    def _detect_separator(self, path: Path) -> str:
        """
        Détecte le séparateur CSV en lisant la première ligne du fichier.

        Stratégie simple et fiable : compte les ';' et ',' dans le header.
        Le séparateur dominant est retenu. Par défaut ';' si égalité.

        Pourquoi ne pas utiliser csv.Sniffer ?
            Sniffer est lent et peu fiable sur les fichiers avec des valeurs
            complexes (dates, libellés avec virgules). Cette heuristique
            fonctionne parfaitement pour des headers sans ambiguïté.
        """
        try:
            with open(path, encoding="latin-1") as f:
                first_line = f.readline()
            n_semicolon = first_line.count(";")
            n_comma = first_line.count(",")
            return ";" if n_semicolon >= n_comma else ","
        except Exception:
            return ";"  # valeur par défaut safe

    def _empty_df(self, columns: list[str]) -> pd.DataFrame:
        """
        Retourne un DataFrame vide avec le schéma de colonnes spécifié.

        Utilisé par load_cc(), load_pv(), load_sc() quand le fichier est absent.
        Garantit que le preprocessor peut toujours accéder aux colonnes
        attendues sans KeyError, même sans données.
        """
        return pd.DataFrame(columns=columns)