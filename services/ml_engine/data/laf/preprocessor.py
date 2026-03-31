# services/ml_engine/data/laf/preprocessor.py
import re

import numpy as np
import pandas as pd


class LAFPreprocessor:
    """
    Responsabilité unique : nettoyer les données LAF brutes et construire
    les features historiques nécessaires au modèle LightGBM.

    Trois opérations principales :
        1. Nettoyage des DataFrames bruts (CC/SC, PV)
        2. Agrégation par tronçon (build_troncon_stats, build_pv_stats)
        3. Unification CC/SC + PV (build_unified_stats)

    SOLID — principe I (refactoring) :
        build_unified_stats() calculait fraud_score, pv_intensity et
        pct_pv_tariff dans un bloc monolithique difficile à tester
        et à faire évoluer indépendamment.
        Ces trois calculs sont maintenant délégués à des méthodes privées :
            _compute_fraud_score()
            _compute_pv_intensity()
            _compute_pct_pv_tariff()
        build_unified_stats() orchestre uniquement la jointure et les appels.

    ── Définition du troncon_id ──────────────────────────────────────────────
        Format : "{origin_uic}_{dest_uic}_{dep_hour}"
        Exemple : "87212027_87214007_8"

    ── Définition du fraud_score unifié ─────────────────────────────────────
        fraud_score = (nb_irregularites + nb_pv) / (nb_controles + nb_pv)
    """

    _CC_DATETIME_COLS: list[str] = [
        "verifiedTickets_verificationDateTime",
        "ticket_travelInformation_departureDateTime",
    ]

    _CC_REQUIRED: list[str] = [
        "ticket_travelInformation_origin_uicCode",
        "ticket_travelInformation_destination_uicCode",
        "ticket_travelInformation_departureDateTime",
        "verifiedTickets_verificationStatus",
    ]

    STATUTS_IRREGULIERS: frozenset[str] = frozenset({"REFUSED"})

    # ── Interface publique ────────────────────────────────────────────────────

    def clean_cc(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Nettoie le DataFrame CC/SC brut.

        Opérations :
            1. Copie défensive
            2. Suppression des lignes avec colonnes critiques manquantes
            3. Parsing datetime (gère DD/MM/YYYY et YYYY-MM-DD)
            4. Suppression des lignes dont le parsing a échoué
            5. Construction du troncon_id
            6. Construction du gtfs_join_key
        """
        if df.empty:
            return df.copy()

        result = df.copy()
        avant  = len(result)

        cols_presentes = [c for c in self._CC_REQUIRED if c in result.columns]
        result = result.dropna(subset=cols_presentes)
        supprimees = avant - len(result)
        if supprimees > 0:
            print(
                f"[LAFPreprocessor] {supprimees:,} lignes CC supprimées "
                f"(données critiques manquantes)."
            )

        for col in self._CC_DATETIME_COLS:
            if col in result.columns:
                result[col] = _parse_datetime(result[col])

        dep_col = "ticket_travelInformation_departureDateTime"
        if dep_col in result.columns:
            result = result.dropna(subset=[dep_col])

        if result.empty:
            print("[LAFPreprocessor] Aucune ligne CC valide après nettoyage.")
            return result.reset_index(drop=True)

        result["troncon_id"] = _build_troncon_id(
            origin_uic=result["ticket_travelInformation_origin_uicCode"],
            dest_uic=result["ticket_travelInformation_destination_uicCode"],
            dep_datetime=result[dep_col],
        )

        if "course_courseNumber" in result.columns and "course_departureDate" in result.columns:
            result["gtfs_join_key"] = _build_gtfs_join_key(
                course_number=result["course_courseNumber"],
                departure_date=result["course_departureDate"],
            )

        return result.reset_index(drop=True)

    def clean_pv(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Nettoie le DataFrame PV brut.
        L'heure de référence est penalties_issueDateTime.
        """
        if df.empty:
            return df.copy()

        result = df.copy()

        if "penalties_issueDateTime" in result.columns:
            result["penalties_issueDateTime"] = _parse_datetime(
                result["penalties_issueDateTime"]
            )

        cols_requises = [
            c for c in [
                "penalties_origin_uicCode",
                "penalties_destination_uicCode",
                "penalties_issueDateTime",
            ]
            if c in result.columns
        ]
        avant  = len(result)
        result = result.dropna(subset=cols_requises)
        supprimees = avant - len(result)
        if supprimees > 0:
            print(
                f"[LAFPreprocessor] {supprimees:,} lignes PV supprimées "
                f"(UIC ou datetime manquants)."
            )

        if result.empty:
            return result.reset_index(drop=True)

        if "penalties_issueDateTime" in result.columns:
            result["troncon_id"] = _build_troncon_id(
                origin_uic=result["penalties_origin_uicCode"],
                dest_uic=result["penalties_destination_uicCode"],
                dep_datetime=result["penalties_issueDateTime"],
            )

        if "course_courseNumber" in result.columns and "course_departureDate" in result.columns:
            result["gtfs_join_key"] = _build_gtfs_join_key(
                course_number=result["course_courseNumber"],
                departure_date=result["course_departureDate"],
            )

        return result.reset_index(drop=True)

    def build_troncon_stats(self, cc_clean: pd.DataFrame) -> pd.DataFrame:
        """
        Agrège les données CC/SC nettoyées par troncon_id.

        Produit : nb_controles, nb_irregularites, taux_irregularite,
                  derniere_date_controle
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

        Produit : nb_pv, nb_pv_tariff, nb_pv_non_tariff, montant_moyen_pv_cents
        """
        if pv_clean.empty or "troncon_id" not in pv_clean.columns:
            print("[LAFPreprocessor] PV vide ou sans troncon_id — stats vides.")
            return pd.DataFrame(columns=[
                "troncon_id", "nb_pv", "nb_pv_tariff",
                "nb_pv_non_tariff", "montant_moyen_pv_cents",
            ])

        pv = pv_clean.copy()

        offence           = pv.get("penalties_offenceType", pd.Series(dtype=str))
        pv["is_tariff"]     = offence.str.upper().eq("TARIFF")
        pv["is_non_tariff"] = ~offence.str.upper().eq("TARIFF") & offence.notna()

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
                nb_pv=(                  "troncon_id",          "count"),
                nb_pv_tariff=(           "is_tariff",           "sum"),
                nb_pv_non_tariff=(       "is_non_tariff",       "sum"),
                montant_moyen_pv_cents=( "montant_total_cents", "mean"),
            )
        )

        stats["nb_pv_tariff"]           = stats["nb_pv_tariff"].astype(int)
        stats["nb_pv_non_tariff"]       = stats["nb_pv_non_tariff"].astype(int)
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
        Fusionne les stats CC/SC et PV et calcule les features unifiées.

        Orchestration uniquement — les calculs sont délégués aux
        méthodes privées _compute_fraud_score(), _compute_pv_intensity(),
        _compute_pct_pv_tariff().

        Jointure outer :
            Tronçons CC/SC seulement → nb_pv = 0
            Tronçons PV seulement    → nb_controles = 0
            Tronçons mixtes          → données complètes
        """
        if pv_stats.empty or "troncon_id" not in pv_stats.columns:
            print(
                "[LAFPreprocessor] Aucun PV disponible — "
                "fraud_score basé sur CC/SC uniquement."
            )
            pv_stats = pd.DataFrame(columns=[
                "troncon_id", "nb_pv", "nb_pv_tariff",
                "nb_pv_non_tariff", "montant_moyen_pv_cents",
            ])

        pv_cols = ["troncon_id", "nb_pv", "nb_pv_tariff",
                   "nb_pv_non_tariff", "montant_moyen_pv_cents"]
        merged = pd.merge(
            cc_sc_stats,
            pv_stats[[c for c in pv_cols if c in pv_stats.columns]],
            on="troncon_id",
            how="outer",
        )

        # Remplissage des NaN issus du outer join
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

        # ── Calculs délégués aux méthodes privées ──────────────────────────────
        merged["fraud_score"]   = self._compute_fraud_score(merged)
        merged["pv_intensity"]  = self._compute_pv_intensity(merged)
        merged["pct_pv_tariff"] = self._compute_pct_pv_tariff(merged)

        print(
            f"[LAFPreprocessor] Stats unifiées : "
            f"{len(merged):,} tronçons, "
            f"fraud_score moyen = {merged['fraud_score'].mean():.3f} "
            f"(CC/SC : {(merged['nb_controles'] > 0).sum():,} tronçons avec scans, "
            f"PV : {(merged['nb_pv'] > 0).sum():,} tronçons avec PV)."
        )
        return merged.reset_index(drop=True)

    # ── Méthodes privées — calculs unitaires testables ────────────────────────

    def _compute_fraud_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Calcule le fraud_score unifié CC/SC + PV.

        Formule : (nb_irregularites + nb_pv) / (nb_controles + nb_pv)

        Pourquoi cette formule ?
            nb_pv est dans le numérateur (fraude constatée) ET le dénominateur
            (chaque PV représente un voyageur contrôlé). Cela évite de compter
            deux fois un fraudeur détecté à la fois en CC et en PV.

        Retourne 0.0 si le dénominateur est nul (tronçon sans aucun contrôle).
        """
        denom = (df["nb_controles"] + df["nb_pv"]).replace(0, np.nan)
        return (
            (df["nb_irregularites"] + df["nb_pv"]) / denom
        ).fillna(0.0).round(4)

    def _compute_pv_intensity(self, df: pd.DataFrame) -> pd.Series:
        """
        Calcule l'intensité PV : nb_pv / nb_controles.

        Mesure la proportion de contrôles ayant abouti à un PV.
        Retourne 0.0 si nb_controles = 0.

        Note : pv_intensity est exclu de COLS_NON_FEATURES dans LGBMScorer
        car sa corrélation avec fraud_score > 0.95 (composant quasi-direct).
        Conservé ici pour des analyses exploratoires.
        """
        ctrl_nonzero = df["nb_controles"].replace(0, np.nan)
        return (df["nb_pv"] / ctrl_nonzero).fillna(0.0).round(4)

    def _compute_pct_pv_tariff(self, df: pd.DataFrame) -> pd.Series:
        """
        Calcule la part des PV tarifaires : nb_pv_tariff / nb_pv.

        Distingue fraude tarifaire (sans titre) vs comportementale.
        Retourne 0.0 si nb_pv = 0.

        Cette feature est utilisée dans HistoricalFeatureTransformer
        via hist_pct_pv_tariff (agrégé par paire O/D).
        """
        pv_nonzero = df["nb_pv"].replace(0, np.nan)
        return (df["nb_pv_tariff"] / pv_nonzero).fillna(0.0).round(4)


# ── Fonctions utilitaires module-level ────────────────────────────────────────

def _parse_datetime(series: pd.Series) -> pd.Series:
    """
    Parse une série de dates/datetimes en gérant les deux formats SNCF.

    Stratégie vectorisée (×55 plus rapide que .apply()) :
        - Détection du format par regex
        - Parsing séparé des deux groupes
        - Fusion et réalignement sur l'index d'origine

    Formats gérés :
        "25/03/2022 23:09"     → format français
        "2022-03-25T22:09:15Z" → format ISO 8601 UTC
    """
    if series.empty:
        return series.copy()

    valid_series = series.dropna().astype(str).str.strip()
    if valid_series.empty:
        return pd.to_datetime(series, errors="coerce")

    mask_french = valid_series.str.match(r"^\d{2}/\d{2}/\d{4}")

    french_parsed = pd.to_datetime(
        valid_series[mask_french], dayfirst=True, errors="coerce",
    )
    iso_parsed = pd.to_datetime(
        valid_series[~mask_french], errors="coerce", utc=False,
    )

    to_concat = [s for s in (french_parsed, iso_parsed) if not s.empty]
    parsed_all = (
        pd.concat(to_concat)
        if to_concat
        else pd.Series(dtype="datetime64[ns]", index=valid_series.index)
    )

    result = parsed_all.reindex(series.index)

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
    Exemple : "87212027_87214007_8"
    """
    hour = dep_datetime.dt.hour.astype(str)
    return origin_uic.str.strip() + "_" + dest_uic.str.strip() + "_" + hour


def _build_gtfs_join_key(
    course_number: pd.Series,
    departure_date: pd.Series,
) -> pd.Series:
    """
    Construit la clé de jointure GTFS : "{course_courseNumber}_{YYYYMMDD}".
    Exemple : "117756_20220325"
    """
    date_normalized = _parse_datetime(
        departure_date.astype(str)
    ).dt.strftime("%Y%m%d").fillna("UNKNOWN")
    return course_number.astype(str).str.strip() + "_" + date_normalized