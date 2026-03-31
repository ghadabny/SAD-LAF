# services/ml_engine/data/laf/preprocessor.py
import numpy as np
import pandas as pd


class LAFPreprocessor:
    """
    Responsabilité unique : nettoyer les données LAF brutes et construire
    les features historiques nécessaires au modèle LightGBM.

    Trois opérations principales :
        1. Nettoyage des DataFrames bruts (CC/SC, PV)
           - Suppression des lignes avec données critiques manquantes
           - Parsing des colonnes datetime
           - Construction du troncon_id (clé d'agrégation ML)
           - Construction du gtfs_join_key (clé de jointure avec le GTFS)

        2. Agrégation par tronçon
           - build_troncon_stats(cc_clean)  → nb_controles, nb_irregularites
           - build_pv_stats(pv_clean)       → nb_pv, types, montants

        3. Unification CC/SC + PV (build_unified_stats)
           - Fusionne les deux tables d'agrégats
           - Calcule le fraud_score unifié

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

    ── Définition du fraud_score unifié ─────────────────────────────────────
        fraud_score = (nb_irregularites + nb_pv) / (nb_controles + nb_pv)

        Intuition :
            • Numérateur   : tous les signaux de fraude connus
                              - nb_irregularites : REFUSED lors d'un scan
                              - nb_pv            : verbalisés (sans titre ou comportement)
            • Dénominateur : tous les "voyageurs observés" connus
                              - nb_controles : scans effectués (présence certifiée)
                              - nb_pv        : verbalisés sans scan préalable
                              (on ajoute nb_pv pour ne pas gonfler artificiellement
                               le taux quand il n'y a que des PV et peu de scans)

        PV-only tronçons (nb_controles = 0) :
            fraud_score = nb_pv / nb_pv = 1.0
            → Signal fort mais sans dénominateur fiable.
              Le modèle apprend à pondérer via nb_controles (feature explicite).
              À terme : lissage bayésien (prior = taux moyen du réseau).

    ── Colonnes UIC dans les CC/SC ──────────────────────────────────────────
        station_uicCode : souvent null dans les CC (≈99% des lignes)
        ticket_travelInformation_origin_uicCode : renseigné sur le titre
        ticket_travelInformation_destination_uicCode : renseigné sur le titre

        On utilise les colonnes du titre (origin/destination_uicCode),
        pas station_uicCode, pour construire le troncon_id.

    Usage :
        preprocessor = LAFPreprocessor()
        cc_clean  = preprocessor.clean_cc(loader.load_cc())
        pv_clean  = preprocessor.clean_pv(loader.load_pv())
        cc_stats  = preprocessor.build_troncon_stats(cc_clean)
        pv_stats  = preprocessor.build_pv_stats(pv_clean)
        unified   = preprocessor.build_unified_stats(cc_stats, pv_stats)
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
        Nettoie le DataFrame CC/SC brut (sortie de LAFLoader.load_cc() ou chunks SC).

        CC et SC ont exactement le même schéma de colonnes — cette méthode
        s'applique aux deux sans modification.

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

        UIC utilisés pour le troncon_id :
            penalties_origin_uicCode      → gare de montée du fraudeur
            penalties_destination_uicCode → destination du fraudeur
        Ces colonnes sont analogues aux origin/destination du titre dans CC/SC,
        ce qui garantit la cohérence du troncon_id entre les deux sources.

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
        Agrège les données CC/SC nettoyées par troncon_id.

        Produit les features historiques de base :
            nb_controles         : nombre total de vérifications sur ce tronçon
            nb_irregularites     : nombre de REFUSED constatés
            taux_irregularite    : nb_irregularites / nb_controles ∈ [0, 1]
            derniere_date_controle : date du dernier contrôle

        Note : cette méthode produit des counts bruts.
        Le taux_irregularite ici est PARTIEL (CC/SC seulement, sans PV).
        Pour le fraud_score final incluant les PV → utiliser build_unified_stats().

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
            nb_pv_tariff         : PV pour fraude tarifaire (sans titre valide)
            nb_pv_non_tariff     : PV pour fraude comportementale
            montant_moyen_pv_cents : montant moyen du PV en centimes
                                   (proxy de la gravité de la fraude)

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
        pv["is_tariff"]     = offence.str.upper().eq("TARIFF")
        pv["is_non_tariff"] = ~offence.str.upper().eq("TARIFF") & offence.notna()

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

    def build_unified_stats(
        self,
        cc_sc_stats: pd.DataFrame,
        pv_stats: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Fusionne les stats CC/SC et PV pour produire un fraud_score unifié.

        Formule :
            fraud_score = (nb_irregularites + nb_pv) / (nb_controles + nb_pv)

        Pourquoi cette formule ?
            ┌──────────────────┬──────────────────────────────────────────────┐
            │ Source           │ Signification                                │
            ├──────────────────┼──────────────────────────────────────────────┤
            │ nb_irregularites │ Fraudes détectées parmi voyageurs scannés    │
            │ nb_pv            │ Fraudes verbalisées hors scan                │
            │ nb_controles     │ Total des scans (proxy fréquentation)        │
            └──────────────────┴──────────────────────────────────────────────┘

            En ajoutant nb_pv au numérateur ET au dénominateur :
              - On ne gonfle pas artificiellement les taux des tronçons
                avec peu de scans mais beaucoup de PV.
              - Un tronçon avec 1000 scans, 0 refus et 50 PV donne :
                fraud_score = (0 + 50) / (1000 + 50) ≈ 0.048
                (cohérent : 5% de fraude détectée)

        Jointure outer :
            Tronçons avec CC/SC seulement → nb_pv = 0 (rempli)
            Tronçons avec PV seulement    → nb_controles = 0, nb_irregularites = 0
            Tronçons mixtes               → données complètes

        Features supplémentaires produites (inputs pour LightGBM) :
            pv_intensity       : nb_pv / nb_controles
                                 Mesure la pression PV relative au volume de contrôle.
                                 Élevé → le tronçon attire des fraudeurs sans titre.
            pct_pv_tariff      : part des PV tarifaires sur total PV
                                 Distingue fraude tarifaire (sans titre) vs comportementale.

        Paramètres :
            cc_sc_stats : sortie de build_troncon_stats() agrégée sur SC + CC
            pv_stats    : sortie de build_pv_stats()
                          Peut être un DataFrame vide → fraud_score = taux CC/SC seulement

        Retourne un DataFrame avec une ligne par troncon_id unique.
        """
        # ── Gestion PV vide ───────────────────────────────────────────────────
        # Si pas de données PV (fichier absent, aucun PV sur ce réseau...),
        # on crée un DataFrame PV vide avec le bon schéma pour que la logique
        # de fusion reste identique.
        if pv_stats.empty or "troncon_id" not in pv_stats.columns:
            print(
                "[LAFPreprocessor] Aucun PV disponible — "
                "fraud_score basé sur CC/SC uniquement."
            )
            pv_stats = pd.DataFrame(columns=[
                "troncon_id", "nb_pv", "nb_pv_tariff",
                "nb_pv_non_tariff", "montant_moyen_pv_cents",
            ])

        # ── Jointure outer CC/SC × PV ─────────────────────────────────────────
        # On utilise outer pour conserver les tronçons qui n'ont que des CC/SC
        # OU que des PV. Les NaN sont ensuite remplacés par 0.
        pv_cols = ["troncon_id", "nb_pv", "nb_pv_tariff",
                   "nb_pv_non_tariff", "montant_moyen_pv_cents"]
        merged = pd.merge(
            cc_sc_stats,
            pv_stats[[c for c in pv_cols if c in pv_stats.columns]],
            on="troncon_id",
            how="outer",
        )

        # ── Remplissage des NaN ───────────────────────────────────────────────
        for col in ["nb_controles", "nb_irregularites"]:
            if col not in merged.columns:
                merged[col] = 0
            merged[col] = merged[col].fillna(0).astype(int)

        for col in ["nb_pv", "nb_pv_tariff", "nb_pv_non_tariff"]:
            if col not in merged.columns:
                merged[col] = 0
            merged[col] = merged[col].fillna(0).astype(int)

        if "montant_moyen_pv_cents" not in merged.columns:
            merged["montant_moyen_pv_cents"] = 0.0
        merged["montant_moyen_pv_cents"] = merged["montant_moyen_pv_cents"].fillna(0.0)

        # ── Calcul du fraud_score unifié ──────────────────────────────────────
        denom = (merged["nb_controles"] + merged["nb_pv"]).replace(0, np.nan)
        merged["fraud_score"] = (
            (merged["nb_irregularites"] + merged["nb_pv"]) / denom
        ).fillna(0.0).round(4)

        # ── Features dérivées ─────────────────────────────────────────────────
        # pv_intensity : pression PV relative au volume de contrôle
        # → élevé = les fraudeurs sans titre sont nombreux par rapport aux scannés
        ctrl_nonzero = merged["nb_controles"].replace(0, np.nan)
        merged["pv_intensity"] = (
            merged["nb_pv"] / ctrl_nonzero
        ).fillna(0.0).round(4)

        # pct_pv_tariff : proportion de fraude tarifaire (sans titre) parmi les PV
        # → distingue la fraude intentionnelle (sans titre) de la fraude comportementale
        pv_nonzero = merged["nb_pv"].replace(0, np.nan)
        merged["pct_pv_tariff"] = (
            merged["nb_pv_tariff"] / pv_nonzero
        ).fillna(0.0).round(4)

        print(
            f"[LAFPreprocessor] Stats unifiées : "
            f"{len(merged):,} tronçons, "
            f"fraud_score moyen = {merged['fraud_score'].mean():.3f} "
            f"(CC/SC : {(merged['nb_controles'] > 0).sum():,} tronçons avec scans, "
            f"PV : {(merged['nb_pv'] > 0).sum():,} tronçons avec PV)."
        )
        return merged.reset_index(drop=True)


# ── Fonctions utilitaires module-level ────────────────────────────────────────

def _parse_datetime(series: pd.Series) -> pd.Series:
    """
    Parse une série de dates/datetimes en gérant les deux formats SNCF.

    Stratégie vectorisée (×55 plus rapide que .apply() ligne par ligne) :
        - Détection du format par regex sur toutes les valeurs non-nulles
        - Parsing séparé des deux groupes (français DD/MM vs ISO YYYY-)
        - Fusion et réalignement sur l'index d'origine

    Formats gérés :
        "25/03/2022 23:09"     → format français avec heure
        "2022-03-25T22:09:15Z" → format ISO 8601 UTC

    Note sur le timezone : les fichiers ISO avec suffixe "Z" (UTC) sont
    convertis en datetime naive via tz_convert(None), sans décalage horaire.
    """
    if series.empty:
        return series.copy()

    # 1. Isoler les valeurs non-nulles converties en string
    valid_series = series.dropna().astype(str).str.strip()
    if valid_series.empty:
        return pd.to_datetime(series, errors="coerce")

    # 2. Détection vectorisée du format
    mask_french = valid_series.str.match(r"^\d{2}/\d{2}/\d{4}")

    # 3. Parsing séparé par format
    french_parsed = pd.to_datetime(
        valid_series[mask_french],
        dayfirst=True,
        errors="coerce",
    )
    iso_parsed = pd.to_datetime(
        valid_series[~mask_french],
        errors="coerce",
        utc=False,
    )

    # 4. Fusion des deux groupes (filtre les Series vides pour éviter les warnings)
    to_concat = [s for s in (french_parsed, iso_parsed) if not s.empty]
    if to_concat:
        parsed_all = pd.concat(to_concat)
    else:
        parsed_all = pd.Series(dtype="datetime64[ns]", index=valid_series.index)

    # 5. Réalignement sur l'index d'origine (NaT pour les valeurs initialement nulles)
    result = parsed_all.reindex(series.index)

    # 6. Suppression du timezone (tz_convert sur timezone-aware, no-op sinon)
    # tz_convert(None) est correct ici car les fichiers ISO SNCF sont en UTC (suffixe Z).
    # Contrairement à tz_localize(None), cette méthode ne lève pas TypeError
    # sur les versions récentes de pandas.
    if hasattr(result, "dt") and result.dt.tz is not None:
        result = result.dt.tz_convert(None)

    return result

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

    Cohérence avec le GTFS :
        Les codes UIC ici sont les 8 chiffres présents dans les tickets.
        Les stop_id GTFS sont nettoyés de leurs préfixes SNCF par GTFSPreprocessor.
        Les deux doivent correspondre après nettoyage pour que la jointure fonctionne.
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