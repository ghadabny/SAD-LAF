# services/ml_engine/data/laf/preprocessor.py
import pandas as pd


class LAFPreprocessor:
    """
    Responsabilité unique : nettoyer les données LAF brutes et construire
    les features historiques nécessaires au modèle LightGBM.

    Deux opérations principales :
        1. Nettoyage des DataFrames bruts (CC, PV)
           - Suppression des lignes avec données critiques manquantes
           - Parsing des colonnes datetime
           - Construction du troncon_id (clé d'agrégation ML)
           - Construction du gtfs_join_key (clé de jointure avec le GTFS)

        2. Agrégation par tronçon (build_troncon_stats / build_pv_stats)
           - nb_controles, nb_irregularites, taux_irregularite
           - Utilisées comme features par HistoricalFeatureTransformer

    ── Définition du troncon_id ──────────────────────────────────────────────
        Format : "{origin_uic}_{dest_uic}_{dep_hour}"
        Exemple : "87212027_87214007_8" = Strasbourg→Sélestat entre 8h et 9h

        Pourquoi inclure l'heure ?
            Le taux de fraude varie selon le créneau horaire. Un train à 8h
            (pendulaires) a un profil très différent du même tronçon à 14h.
            L'heure est présente dans les données GTFS (dep_hour) et dans les
            données LAF (ticket_travelInformation_departureDateTime).

        Note importante : ce troncon_id est l'unité ML (agrégation sur
        l'historique). La jointure avec les trips GTFS spécifiques utilise
        gtfs_join_key = "{course_courseNumber}_{course_departureDate_YYYYMMDD}".

    ── Vrais statuts d'irrégularité ─────────────────────────────────────────
        Dans les données CC réelles (Extract_DATA_GE_CC_*.csv) :
            ACCEPTED : titre valide, voyageur en règle
            REFUSED  : titre invalide → IRRÉGULARITÉ (périmé, mauvais trajet...)
            DENIED   : refus de présenter (à clarifier avec l'équipe LAF)

        Le statut REFUSED est le signal d'irrégularité dans les CC.
        Les PV sont à part : chaque ligne = une fraude constatée.

    ── Colonnes UIC dans les CC ─────────────────────────────────────────────
        station_uicCode : souvent null dans les CC (≈99% des lignes)
        ticket_travelInformation_origin_uicCode : renseigné sur le titre
        ticket_travelInformation_destination_uicCode : renseigné sur le titre

        On utilise les colonnes du titre (origin/destination_uicCode),
        pas station_uicCode, pour construire le troncon_id.

    Usage :
        preprocessor = LAFPreprocessor()
        cc_clean = preprocessor.clean_cc(loader.load_cc())
        pv_clean = preprocessor.clean_pv(loader.load_pv())
        stats    = preprocessor.build_troncon_stats(cc_clean)
        pv_stats = preprocessor.build_pv_stats(pv_clean)
    """

    # ── Colonnes datetime à parser dans CC/SC ────────────────────────────────
    _CC_DATETIME_COLS: list[str] = [
        "verifiedTickets_verificationDateTime",
        "ticket_travelInformation_departureDateTime",
    ]

    # ── Colonnes critiques pour qu'un contrôle CC soit exploitable ───────────
    # station_uicCode est EXCLU : quasi-systématiquement null dans les données réelles.
    _CC_REQUIRED: list[str] = [
        "ticket_travelInformation_origin_uicCode",
        "ticket_travelInformation_destination_uicCode",
        "ticket_travelInformation_departureDateTime",
        "verifiedTickets_verificationStatus",
    ]

    # ── Vrais statuts d'irrégularité ─────────────────────────────────────────
    # Source : inspection des données réelles CC (Extract_DATA_GE_CC_*.csv)
    # REFUSED = titre refusé → irrégularité
    STATUTS_IRREGULIERS: frozenset[str] = frozenset({"REFUSED"})

    def clean_cc(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Nettoie le DataFrame CC brut (sortie de LAFLoader.load_cc()).

        Opérations dans l'ordre :
            1. Copie défensive
            2. Suppression des lignes avec colonnes critiques manquantes
            3. Parsing des colonnes datetime (gère DD/MM/YYYY et YYYY-MM-DD)
            4. Suppression des lignes dont le parsing datetime a échoué
            5. Construction du troncon_id (agrégation ML)
            6. Construction du gtfs_join_key (jointure trips GTFS)

        Retourne un DataFrame nettoyé. Si df est vide, retourne df vide.
        """
        if df.empty:
            return df.copy()

        result = df.copy()
        avant = len(result)

        # ── Filtrer les lignes avec données critiques manquantes ──────────────
        cols_requises_presentes = [
            c for c in self._CC_REQUIRED if c in result.columns
        ]
        result = result.dropna(subset=cols_requises_presentes)
        supprimees = avant - len(result)
        if supprimees > 0:
            print(
                f"[LAFPreprocessor] {supprimees:,} lignes CC supprimées "
                f"(données critiques manquantes)."
            )

        # ── Parser les colonnes datetime ──────────────────────────────────────
        # Les deux formats de date coexistent selon les livraisons :
        #   "25/03/2022 23:09" (format français avec heure)
        #   "2022-03-25T22:09:15Z" (format ISO 8601)
        # _parse_datetime() gère les deux sans ambiguïté.
        for col in self._CC_DATETIME_COLS:
            if col in result.columns:
                result[col] = _parse_datetime(result[col])

        # Supprimer les lignes dont la datetime de départ n'a pas pu être parsée
        dep_col = "ticket_travelInformation_departureDateTime"
        if dep_col in result.columns:
            result = result.dropna(subset=[dep_col])

        if result.empty:
            print("[LAFPreprocessor] Aucune ligne CC valide après nettoyage.")
            return result.reset_index(drop=True)

        # ── Construire troncon_id ─────────────────────────────────────────────
        result["troncon_id"] = _build_troncon_id(
            origin_uic=result["ticket_travelInformation_origin_uicCode"],
            dest_uic=result["ticket_travelInformation_destination_uicCode"],
            dep_datetime=result[dep_col],
        )

        # ── Construire gtfs_join_key ──────────────────────────────────────────
        if "course_courseNumber" in result.columns and "course_departureDate" in result.columns:
            result["gtfs_join_key"] = _build_gtfs_join_key(
                course_number=result["course_courseNumber"],
                departure_date=result["course_departureDate"],
            )

        return result.reset_index(drop=True)

    def clean_pv(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Nettoie le DataFrame PV brut (sortie de LAFLoader.load_pv()).

        Particularité PV :
            Chaque ligne = une fraude constatée ayant donné lieu à un PV.
            Il n'y a pas de verifiedTickets_verificationStatus.
            L'heure de référence est penalties_issueDateTime (heure du contrôle).

        Retourne un DataFrame nettoyé avec troncon_id et gtfs_join_key ajoutés.
        """
        if df.empty:
            return df.copy()

        result = df.copy()

        # ── Parser issueDateTime ──────────────────────────────────────────────
        if "penalties_issueDateTime" in result.columns:
            result["penalties_issueDateTime"] = _parse_datetime(
                result["penalties_issueDateTime"]
            )

        # ── Filtrer les lignes invalides ──────────────────────────────────────
        cols_requises = [
            c for c in [
                "penalties_origin_uicCode",
                "penalties_destination_uicCode",
                "penalties_issueDateTime",
            ]
            if c in result.columns
        ]
        avant = len(result)
        result = result.dropna(subset=cols_requises)
        supprimees = avant - len(result)
        if supprimees > 0:
            print(
                f"[LAFPreprocessor] {supprimees:,} lignes PV supprimées "
                f"(UIC ou datetime manquants)."
            )

        if result.empty:
            return result.reset_index(drop=True)

        # ── Construire troncon_id ─────────────────────────────────────────────
        if "penalties_issueDateTime" in result.columns:
            result["troncon_id"] = _build_troncon_id(
                origin_uic=result["penalties_origin_uicCode"],
                dest_uic=result["penalties_destination_uicCode"],
                dep_datetime=result["penalties_issueDateTime"],
            )

        # ── Construire gtfs_join_key ──────────────────────────────────────────
        if "course_courseNumber" in result.columns and "course_departureDate" in result.columns:
            result["gtfs_join_key"] = _build_gtfs_join_key(
                course_number=result["course_courseNumber"],
                departure_date=result["course_departureDate"],
            )

        return result.reset_index(drop=True)

    def build_troncon_stats(self, cc_clean: pd.DataFrame) -> pd.DataFrame:
        """
        Agrège les données CC nettoyées par troncon_id.

        Produit les features historiques pour HistoricalFeatureTransformer :
            nb_controles         : nombre total de vérifications sur ce tronçon
            nb_irregularites     : nombre de REFUSED constatés
            taux_irregularite    : nb_irregularites / nb_controles ∈ [0, 1]
            derniere_date_controle : date du dernier contrôle

        Retourne une ligne par troncon_id unique. DataFrame vide si cc_clean vide.
        """
        if cc_clean.empty or "troncon_id" not in cc_clean.columns:
            print("[LAFPreprocessor] CC vide ou sans troncon_id — stats vides.")
            return pd.DataFrame(columns=[
                "troncon_id", "nb_controles", "nb_irregularites",
                "taux_irregularite", "derniere_date_controle",
            ])

        is_irreg = cc_clean["verifiedTickets_verificationStatus"].isin(
            self.STATUTS_IRREGULIERS
        )

        agg_cols = {
            "nb_controles":     ("verifiedTickets_verificationStatus", "count"),
            "nb_irregularites": ("is_irregulier", "sum"),
        }
        date_col = "verifiedTickets_verificationDateTime"
        if date_col in cc_clean.columns:
            agg_cols["derniere_date_controle"] = (date_col, "max")

        stats = (
            cc_clean
            .assign(is_irregulier=is_irreg)
            .groupby("troncon_id", as_index=False)
            .agg(**agg_cols)
        )

        stats["taux_irregularite"] = (
            stats["nb_irregularites"] / stats["nb_controles"]
        ).round(4)
        stats["nb_irregularites"] = stats["nb_irregularites"].astype(int)

        print(
            f"[LAFPreprocessor] Stats CC : "
            f"{len(stats):,} tronçons, "
            f"taux moyen = {stats['taux_irregularite'].mean():.3f}."
        )
        return stats

    def build_pv_stats(self, pv_clean: pd.DataFrame) -> pd.DataFrame:
        """
        Agrège les données PV nettoyées par troncon_id.

        Chaque ligne PV = 1 fraude constatée.
        Produit :
            nb_pv                : nombre total de PV sur ce tronçon
            nb_pv_tariff         : PV pour fraude tarifaire (sans titre)
            nb_pv_non_tariff     : PV pour fraude comportementale
            montant_moyen_pv_cents : montant moyen du PV en centimes

        Retourne DataFrame vide si pv_clean est vide.
        """
        if pv_clean.empty or "troncon_id" not in pv_clean.columns:
            print("[LAFPreprocessor] PV vide ou sans troncon_id — stats vides.")
            return pd.DataFrame(columns=[
                "troncon_id", "nb_pv", "nb_pv_tariff",
                "nb_pv_non_tariff", "montant_moyen_pv_cents",
            ])

        pv = pv_clean.copy()

        # Indicateurs booléens de type de fraude
        offence = pv.get("penalties_offenceType", pd.Series(dtype=str))
        pv["is_tariff"]     = offence.eq("TARIFF")
        pv["is_non_tariff"] = offence.eq("NON_TARIFF")

        # Montant total du PV (perception + forfait) en centimes
        perception = pd.to_numeric(
            pv.get("penalties_perceptionFee_amountInCents", 0), errors="coerce"
        ).fillna(0)
        forfait = pd.to_numeric(
            pv.get("penalties_flatFee_amountInCents", 0), errors="coerce"
        ).fillna(0)
        pv["montant_total_cents"] = perception + forfait

        stats = (
            pv
            .groupby("troncon_id", as_index=False)
            .agg(
                nb_pv=("troncon_id", "count"),
                nb_pv_tariff=("is_tariff", "sum"),
                nb_pv_non_tariff=("is_non_tariff", "sum"),
                montant_moyen_pv_cents=("montant_total_cents", "mean"),
            )
        )

        stats["nb_pv_tariff"]     = stats["nb_pv_tariff"].astype(int)
        stats["nb_pv_non_tariff"] = stats["nb_pv_non_tariff"].astype(int)
        stats["montant_moyen_pv_cents"] = stats["montant_moyen_pv_cents"].round(0)

        print(
            f"[LAFPreprocessor] Stats PV : "
            f"{len(stats):,} tronçons, "
            f"{stats['nb_pv'].sum():,} PV total."
        )
        return stats


# ── Fonctions utilitaires module-level ────────────────────────────────────────
# Définies hors de la classe car elles n'ont pas d'état et sont
# potentiellement utilisées par d'autres modules.

def _parse_datetime(series: pd.Series) -> pd.Series:
    """
    Parse une série de dates/datetimes en gérant les deux formats SNCF.

    Stratégie : détection du format par regex avant parsing.
        - Format français  DD/MM/... → dayfirst=True  forcé
        - Format ISO       YYYY-...  → format ISO standard
    Évite le problème d'ambiguïté quand les deux formats coexistent
    dans la même série (le fallback NaT ne détecte pas les mauvais parsings).
    """
    import re

    FRENCH_PATTERN = re.compile(r"^\d{2}/\d{2}/\d{4}")

    def _parse_single(val):
        if pd.isna(val):
            return pd.NaT
        s = str(val).strip()
        try:
            if FRENCH_PATTERN.match(s):
                return pd.to_datetime(s, dayfirst=True, errors="coerce")
            else:
                return pd.to_datetime(s, errors="coerce", utc=False)
        except Exception:
            return pd.NaT

    parsed = series.apply(_parse_single)

    # Supprimer le timezone si présent
    if hasattr(parsed, "dt") and parsed.dt.tz is not None:
        parsed = parsed.dt.tz_localize(None)

    return parsed


def _build_troncon_id(
    origin_uic: pd.Series,
    dest_uic: pd.Series,
    dep_datetime: pd.Series,
) -> pd.Series:
    """
    Construit le troncon_id vectorisé : "{origin}_{dest}_{dep_hour}".

    dep_datetime doit déjà être de type datetime64 (après _parse_datetime).
    Vectorisé avec .str et .dt pour performance sur millions de lignes.

    Exemple : "87212027_87214007_8" = Strasbourg→Sélestat entre 8h et 9h
    """
    hour = dep_datetime.dt.hour.astype(str)
    return origin_uic.str.strip() + "_" + dest_uic.str.strip() + "_" + hour


def _build_gtfs_join_key(
    course_number: pd.Series,
    departure_date: pd.Series,
) -> pd.Series:
    """
    Construit la clé de jointure avec les trips GTFS.

    Format : "{course_courseNumber}_{YYYYMMDD}"
    Exemple : "117756_20220325"

    Gère les deux formats de date des livraisons SNCF :
        "25/03/2022"  → "20220325"  (format français)
        "2022-03-25"  → "20220325"  (format ISO)

    Utilise _parse_datetime() pour le parsing — même logique que clean_cc().

    Pourquoi la clé de jointure est (numéro_train, date) ?
        Un même numéro de train peut avoir des trajets différents selon la date
        (variantes de desserte encodées dans des trip_id GTFS distincts).
        La clé numéro_train seul serait ambiguë.
    """
    date_normalized = _parse_datetime(
        departure_date.astype(str)
    ).dt.strftime("%Y%m%d").fillna("UNKNOWN")

    return course_number.astype(str).str.strip() + "_" + date_normalized
