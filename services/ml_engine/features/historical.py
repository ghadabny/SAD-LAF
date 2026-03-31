# services/ml_engine/features/historical.py
import math

import numpy as np
import pandas as pd

from services.ml_engine.features.base import BaseFeatureTransformer


class HistoricalFeatureTransformer(BaseFeatureTransformer):
    """
    Transformateur qui ajoute des features historiques de volume et d'intensité
    de fraude par tronçon (Origine → Destination).

    ── HISTORIQUE DES CORRECTIONS ANTI-LEAKAGE ──────────────────────────────

    v1 → hist_fraud_score_segment = mean(fraud_score) groupby(O/D)
         LEAKAGE DIRECT : copie exacte de la cible. Supprimé.

    v2 → hist_nb_pv utilisé comme feature.
         LEAKAGE DOUX : nb_pv est le numérateur ET dans le dénominateur de
         fraud_score = (nb_irregularites + nb_pv) / (nb_controles + nb_pv).
         Sur les tronçons à fort nb_pv, la corrélation avec fraud_score est
         > 0.95. Supprimé de la liste des features.

    v3 → taux_irregularite utilisé comme feature.
         LEAKAGE DIRECT : taux_irregularite = nb_irregularites / nb_controles
         ≈ fraud_score sans les PV. Exclu.

    ── CE QUE LE TRANSFORMATEUR PRODUIT (v3 — version courante) ─────────────

    Features de VOLUME (proxies d'exposition au contrôle) :
        hist_nb_controles    : nombre total de scans historiques sur ce tronçon.
                               Mesure l'ACTIVITÉ de contrôle, pas le taux.

        hist_log_controles   : log(1 + nb_controles)
                               Compresse la distribution très asymétrique des
                               volumes (quelques gares très contrôlées vs
                               beaucoup peu contrôlées).

    Feature de STRUCTURE (composition de la fraude) :
        hist_pct_pv_tariff   : proportion de PV tarifaires parmi tous les PV.
                               Distingue fraude tarifaire vs comportementale.

    ── FEATURES EXCLUES PAR CONCEPTION ─────────────────────────────────────
        hist_nb_pv           : nb_pv est dans le numérateur ET dénominateur de
                               fraud_score → corrélation > 0.95 probable.
        hist_pv_intensity    : nb_pv / nb_controles → même problème.

    ── FALLBACK POUR LES TRONÇONS INCONNUS ──────────────────────────────────
    En production, des tronçons GTFS n'auront jamais été contrôlés dans
    l'historique. Fallback = moyenne des valeurs apprises sur les tronçons
    connus, calculée après construction du lookup (cohérence garantie).

    ── INTERFACE SCIKIT-LEARN ───────────────────────────────────────────────
    Usage correct (anti-leakage) :
        transformer = HistoricalFeatureTransformer()
        df_train_feat = transformer.fit_transform(df_train)
        df_test_feat  = transformer.transform(df_test)
    """

    # Colonnes source LAF requises pour le fit()
    _REQUIRED_SOURCE_COLS: list[str] = [
        "nb_controles",
        "pct_pv_tariff",
    ]

    # Colonnes que le transform() AJOUTE au DataFrame.
    # nb_pv et pv_intensity sont intentionnellement absents.
    FEATURE_COLS: list[str] = [
        "hist_nb_controles",
        "hist_log_controles",
        "hist_pct_pv_tariff",
    ]

    def __init__(self):
        # Dictionnaire de lookup : (stop_id_dep, stop_id_arr) → valeurs
        self._stats: dict[tuple[str, str], dict[str, float]] = {}
        # Fallback (moyennes calculées sur les valeurs du lookup, pas sur df_train brut)
        self._fallback: dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> "HistoricalFeatureTransformer":
        """
        Apprend les statistiques historiques agrégées par paire O/D.

        PRÉCONDITION : df est le jeu d'entraînement UNIQUEMENT.
        Ne jamais appeler fit() ou fit_transform() sur le dataset complet.

        Paramètres :
            df : DataFrame d'entraînement avec colonnes nb_controles, pct_pv_tariff.

        Retourne self pour le chaînage fit().transform().
        """
        if "stop_id_dep" not in df.columns or "stop_id_arr" not in df.columns:
            raise ValueError(
                "[HistoricalFeatureTransformer] Colonnes manquantes : "
                "'stop_id_dep' et 'stop_id_arr' sont requises pour fit()."
            )

        cols_presentes = [c for c in self._REQUIRED_SOURCE_COLS if c in df.columns]
        if not cols_presentes:
            raise ValueError(
                "[HistoricalFeatureTransformer] Aucune colonne source LAF trouvée "
                f"(requises : {self._REQUIRED_SOURCE_COLS}). "
                "Vérifie que build_unified_stats() a bien tourné avant le pipeline."
            )

        # ── Agrégation par paire O/D ───────────────────────────────────────
        agg_dict = {}
        if "nb_controles" in df.columns:
            agg_dict["nb_controles"] = "mean"
        if "pct_pv_tariff" in df.columns:
            agg_dict["pct_pv_tariff"] = "mean"

        if not agg_dict:
            print("[HistoricalFeatureTransformer] ⚠️  Aucune colonne à agréger.")
            return self

        grouped = (
            df.groupby(["stop_id_dep", "stop_id_arr"], as_index=False)
              .agg(agg_dict)
        )

        # ── Construction du dictionnaire de lookup ─────────────────────────
        # log1p est appliqué sur la valeur AGRÉGÉE (mean par O/D),
        # pas sur chaque ligne brute — cohérence garantie avec le fallback.
        self._stats = {}
        for _, row in grouped.iterrows():
            key = (str(row["stop_id_dep"]), str(row["stop_id_arr"]))
            entry: dict[str, float] = {}
            if "nb_controles" in grouped.columns:
                nb_ctrl = float(row["nb_controles"])
                entry["hist_nb_controles"]  = nb_ctrl
                entry["hist_log_controles"] = math.log1p(nb_ctrl)
            if "pct_pv_tariff" in grouped.columns:
                entry["hist_pct_pv_tariff"] = float(row["pct_pv_tariff"])
            self._stats[key] = entry

        # ── Calcul des fallbacks DEPUIS LE LOOKUP (et non depuis df brut) ──
        # Garantit que le fallback est dans la même distribution que les valeurs
        # apprises, quelle que soit la pondération des lignes dans df_train.
        if self._stats:
            all_vals = list(self._stats.values())
            self._fallback = {
                feat: float(np.mean([v[feat] for v in all_vals if feat in v]))
                for feat in self.FEATURE_COLS
                if any(feat in v for v in all_vals)
            }
        else:
            self._fallback = {feat: 0.0 for feat in self.FEATURE_COLS}

        n_od = len(self._stats)
        print(
            f"[HistoricalFeatureTransformer] fit() : "
            f"{n_od:,} tronçons O/D appris | "
            f"fallback nb_controles={self._fallback.get('hist_nb_controles', 0):.1f} | "
            f"features produites : {self.FEATURE_COLS}"
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ajoute les features historiques au DataFrame.

        Pour les tronçons présents dans le dictionnaire → valeurs apprises.
        Pour les tronçons inconnus (nouveaux en production) → fallback.

        Ne modifie pas le DataFrame d'entrée (copie défensive).

        Paramètres :
            df : DataFrame de tronçons (colonnes requises : stop_id_dep, stop_id_arr).

        Retourne un nouveau DataFrame avec FEATURE_COLS ajoutées.
        """
        required = {"stop_id_dep", "stop_id_arr"}
        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            raise ValueError(
                f"[HistoricalFeatureTransformer] Colonnes manquantes pour transform() : "
                f"{missing}"
            )

        result = df.copy()

        # ── Mapping vectorisé via merge ────────────────────────────────────
        if self._stats:
            lookup_rows = [
                {"stop_id_dep": k[0], "stop_id_arr": k[1], **v}
                for k, v in self._stats.items()
            ]
            lookup_df = pd.DataFrame(lookup_rows)

            result["stop_id_dep"] = result["stop_id_dep"].astype(str)
            result["stop_id_arr"] = result["stop_id_arr"].astype(str)

            result = result.merge(
                lookup_df,
                on=["stop_id_dep", "stop_id_arr"],
                how="left",
            )
        else:
            for col in self.FEATURE_COLS:
                result[col] = float("nan")

        # ── Remplissage des tronçons inconnus par les fallbacks ────────────
        n_fallback = 0
        for col, val in self._fallback.items():
            if col in result.columns:
                mask = result[col].isna()
                n_fallback = max(n_fallback, int(mask.sum()))
                result.loc[mask, col] = val
            else:
                result[col] = val

        if n_fallback > 0:
            print(
                f"[HistoricalFeatureTransformer] {n_fallback:,} tronçons inconnus "
                f"→ fallback appliqué."
            )

        return result