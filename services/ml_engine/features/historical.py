# services/ml_engine/features/historical.py
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
         > 0.95. Supprimé de la liste des features (reste dans le DataFrame
         pour le calcul de fraud_score mais exclu de FEATURE_COLS).

    v3 → taux_irregularite utilisé comme feature.
         LEAKAGE DIRECT : taux_irregularite = nb_irregularites / nb_controles
         ≈ fraud_score sans les PV. Déjà exclu dans COLS_NON_FEATURES de
         LGBMScorer, confirmé ici.

    ── CE QUE LE TRANSFORMATEUR PRODUIT (v3) ────────────────────────────────

    Features de VOLUME (proxies d'exposition au contrôle) :
        hist_nb_controles    : nombre total de scans historiques sur ce tronçon
                               Mesure l'ACTIVITÉ de contrôle, pas le taux.
                               Corrélation avec fraud_score : faible à modérée
                               (un tronçon très contrôlé n'est pas forcément
                               le plus fraudé — la pression dissuade).

        hist_log_controles   : log(1 + nb_controles)
                               Compresse la distribution très asymétrique des
                               volumes (quelques gares très contrôlées vs
                               beaucoup peu contrôlées).

    Feature de STRUCTURE (composition de la fraude) :
        hist_pct_pv_tariff   : proportion de PV tarifaires parmi tous les PV
                               → distingue fraude tarifaire (voyage sans titre)
                                 vs fraude comportementale (classe/zone)
                               → peu corrélé à fraud_score car ne dépend pas
                                 du taux mais du TYPE de fraude.

    ── FEATURES EXCLUES PAR CONCEPTION ─────────────────────────────────────
        hist_nb_pv           : nb_pv est dans le numérateur ET dénominateur de
                               fraud_score → corrélation > 0.95 probable.
                               Exclu des FEATURE_COLS (gardé dans le DataFrame
                               pour la traçabilité mais pas utilisé par LightGBM).

        hist_pv_intensity    : nb_pv / nb_controles. Dépend directement de
                               nb_pv → même problème que hist_nb_pv.
                               Exclu des FEATURE_COLS.

    ── FALLBACK POUR LES TRONÇONS INCONNUS ──────────────────────────────────
    En production, des tronçons GTFS n'auront jamais été contrôlés dans
    l'historique. Fallback = moyenne globale du réseau calculée sur les
    données d'entraînement lors du fit() — jamais sur le test.

    ── INTERFACE SCIKIT-LEARN ───────────────────────────────────────────────
    fit()       : apprend les stats historiques UNIQUEMENT sur df_train
    transform() : applique le mapping à n'importe quel DataFrame (train ou prod)
    fit_transform() : hérité de BaseFeatureTransformer → fit puis transform

    Usage correct (anti-leakage) :
        transformer = HistoricalFeatureTransformer()
        df_train_feat = transformer.fit_transform(df_train)   # apprend sur train
        df_test_feat  = transformer.transform(df_test)        # applique au test
    """

    # Colonnes source LAF requises pour le fit() (produites par build_unified_stats)
    _REQUIRED_SOURCE_COLS: list[str] = [
        "nb_controles",
        "pct_pv_tariff",
    ]

    # Colonnes que le transform() AJOUTE au DataFrame.
    # UNIQUEMENT des features sans leakage (voir justification ci-dessus).
    # Note : nb_pv et pv_intensity sont intentionnellement ABSENTS.
    FEATURE_COLS: list[str] = [
        "hist_nb_controles",
        "hist_log_controles",
        "hist_pct_pv_tariff",
    ]

    def __init__(self):
        # Dictionnaire de lookup : (stop_id_dep, stop_id_arr) → valeurs
        self._stats: dict[tuple[str, str], dict[str, float]] = {}

        # Valeurs de fallback (moyennes globales calculées sur df_train uniquement)
        self._fallback: dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> "HistoricalFeatureTransformer":
        """
        Apprend les statistiques historiques agrégées par paire O/D.

        PRÉCONDITION : df est le jeu d'entraînement UNIQUEMENT.
        Ne jamais appeler fit() ou fit_transform() sur le dataset complet
        (train + test) — ce serait un leakage temporel.

        Les valeurs de fallback calculées ici servent aux tronçons jamais
        vus en production. Si elles sont calculées sur le test, elles fuient
        de l'information sur la distribution du test vers le train.

        Paramètres :
            df : DataFrame d'entraînement avec colonnes nb_controles,
                 pct_pv_tariff (et optionnellement nb_pv pour la traçabilité)

        Retourne self pour le chaînage fit().transform()
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

        import numpy as np

        # ── Calcul des fallbacks sur le train uniquement ───────────────────
        self._fallback = {
            "hist_nb_controles":  float(df["nb_controles"].mean())
                                  if "nb_controles" in df.columns else 0.0,
            "hist_log_controles": float(np.log1p(df["nb_controles"]).mean())
                                  if "nb_controles" in df.columns else 0.0,
            "hist_pct_pv_tariff": float(df["pct_pv_tariff"].mean())
                                  if "pct_pv_tariff" in df.columns else 0.0,
        }

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
        self._stats = {}
        for _, row in grouped.iterrows():
            key = (str(row["stop_id_dep"]), str(row["stop_id_arr"]))
            entry: dict[str, float] = {}
            if "nb_controles" in grouped.columns:
                entry["hist_nb_controles"]  = float(row["nb_controles"])
                entry["hist_log_controles"] = float(
                    __import__("math").log1p(row["nb_controles"])
                )
            if "pct_pv_tariff" in grouped.columns:
                entry["hist_pct_pv_tariff"] = float(row["pct_pv_tariff"])
            self._stats[key] = entry

        n_od = len(self._stats)
        print(
            f"[HistoricalFeatureTransformer] fit() : "
            f"{n_od:,} tronçons O/D appris | "
            f"fallback nb_controles={self._fallback['hist_nb_controles']:.1f} | "
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
            df : DataFrame de tronçons (Colonnes requises : stop_id_dep, stop_id_arr)

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
            # fit() non appelé ou données vides → colonnes NaN
            for col in self._fallback:
                result[col] = float("nan")

        # ── Remplissage des tronçons inconnus par les fallbacks ────────────
        for col, val in self._fallback.items():
            if col in result.columns:
                result[col] = result[col].fillna(val)
            else:
                result[col] = val

        # Log des tronçons ayant reçu le fallback
        if "hist_nb_controles" in result.columns:
            n_fallback = result["hist_nb_controles"].isna().sum()
            if n_fallback > 0:
                print(
                    f"[HistoricalFeatureTransformer] {n_fallback:,} tronçons inconnus "
                    f"→ fallback appliqué."
                )

        return result