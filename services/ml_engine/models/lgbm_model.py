# services/ml_engine/models/lgbm_model.py
import joblib
import math
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr

from services.ml_engine.models.base import BaseScorer
from shared.config import config


class LGBMScorer(BaseScorer):
    """
    Encapsule le modèle LightGBM pour la prédiction du taux de fraude par tronçon.

    ── Features utilisées ────────────────────────────────────────────────────
    Contexte temporel (TemporalFeatureTransformer) :
        dep_hour         : heure de départ (0-23)
        day_of_week      : jour de la semaine (0=lundi ... 6=dimanche)
        is_weekend       : 0/1 week-end
        is_vacances      : 0/1 vacances scolaires zone B
        is_jour_ferie    : 0/1 jour férié Alsace-Moselle
        is_peak_hour     : 0/1 heure de pointe (7h-9h / 17h-19h en semaine)

    Volume et intensité historique (HistoricalFeatureTransformer — v3) :
        hist_nb_controles   : nombre total de scans historiques sur ce tronçon O/D
        hist_log_controles  : log(1 + nb_controles) — compresse la distribution
        hist_pct_pv_tariff  : % PV tarifaires / total PV sur ce tronçon O/D
        [hist_nb_pv et hist_pv_intensity exclus : corrélation > 0.95 avec cible]

    Caractéristiques brutes du tronçon (source LAF agrégée) :
        nb_controles, nb_pv, nb_pv_tariff, nb_pv_non_tariff
        pv_intensity, pct_pv_tariff, montant_moyen_pv_cents
        stop_sequence, dep_minutes, arr_minutes, duration_min

    ── Variable cible ────────────────────────────────────────────────────────
        fraud_score = (nb_irregularites + nb_pv) / (nb_controles + nb_pv) ∈ [0, 1]

    ── Ce qui N'EST PAS une feature ─────────────────────────────────────────
        fraud_score           : c'est la cible (évite le leakage trivial)
        taux_irregularite     : ≈ fraud_score sans les PV → leakage direct
        nb_irregularites      : numérateur de la cible → leakage direct
        hist_fraud_score_segment : ancienne feature = copie exacte de la cible

    ── Anti-leakage : pourquoi les features nb_controles / pv_intensity sont OK ──
        Ces colonnes mesurent le VOLUME et l'INTENSITÉ historiques de la fraude,
        pas la valeur exacte de la cible. Elles proviennent du même pool de
        données que fraud_score mais restent des proxies distincts (un tronçon
        très contrôlé n'est pas forcément le plus fraudé — effet dissuasif).
        La corrélation Pearson de ces features avec fraud_score est < 0.95,
        vérifiée automatiquement par _check_leakage().
    """

    # ── Colonnes exclues de l'entraînement ───────────────────────────────────
    # Ces colonnes sont dans df_train mais ne doivent JAMAIS être des features.
    # Règle : toute colonne qui est algébriquement dérivée de fraud_score.
    COLS_NON_FEATURES: frozenset[str] = frozenset({
        # Cible
        "fraud_score",
        # Composants directs de la cible
        "taux_irregularite",     # = nb_irregularites / nb_controles ≈ fraud_score
        "nb_irregularites",      # numérateur de fraud_score
        # Identifiants techniques (pas prédictifs, créent du bruit)
        "trip_id",
        "train_number",
        "service_id",
        "stop_id_dep",
        "stop_id_arr",
        "stop_name_dep",
        "stop_name_arr",
        "troncon_id",
        "gtfs_join_key",
        "service_date",
        # Feature supprimée : ancienne version de HistoricalFeatureTransformer
        "hist_fraud_score_segment",
        # Autres agrégats source (déjà transformés en hist_* par le transformer)
        "derniere_date_controle",
        "dep_minute_of_day",     # redondant avec dep_hour
    })

    def __init__(self, **kwargs):
        # Hyperparamètres par défaut pour la régression d'un taux ∈ [0, 1]
        self.params = {
            "objective":    "regression",
            "metric":       "rmse",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves":   31,
            "min_child_samples": 20,   # évite les feuilles sur 1 seul tronçon
            "verbose":      -1,
        }
        self.params.update(kwargs)
        self.model: lgb.Booster | None = None
        # Mémorise les features utilisées lors du train pour vérifier en prod
        self.feature_cols: list[str] = []

    # ── Entraînement ─────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame, target_col: str = "fraud_score") -> "LGBMScorer":
        """
        Entraîne le modèle LightGBM.
        """
        if target_col not in df.columns:
            raise ValueError(
                f"[LGBMScorer] Colonne cible '{target_col}' absente du DataFrame. "
                f"Colonnes disponibles : {list(df.columns)}"
            )

        # --- CORRECTION 1 : Forcer les booléens en entiers (0/1) ---
        # Empêche les features temporelles (is_weekend, etc.) de disparaître
        for col in df.columns:
            if pd.api.types.is_bool_dtype(df[col]):
                df[col] = df[col].astype(int)

        # ── Sélection automatique des features ────────────────────────────────
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        self.feature_cols = [
            col for col in numeric_cols
            if col not in self.COLS_NON_FEATURES and col != target_col
        ]

        if not self.feature_cols:
            raise ValueError(
                "[LGBMScorer] Aucune feature numérique trouvée après exclusion. "
                "Vérifie que le pipeline de features a bien tourné (fit_transform)."
            )

        print(
            f"[LGBMScorer] Features sélectionnées ({len(self.feature_cols)}) : "
            f"{self.feature_cols}"
        )

        # ── Vérification anti-leakage ─────────────────────────────────────────
        self._check_leakage(df, target_col)

        X = df[self.feature_cols]
        y = df[target_col]

        lgb_data = lgb.Dataset(X, label=y, feature_name=self.feature_cols)

        # --- CORRECTION 2 : Gestion silencieuse de n_estimators ---
        # On extrait 'n_estimators' pour ne pas l'envoyer en double
        params_clean = self.params.copy()
        n_rounds = params_clean.pop("n_estimators", 200) # 200 par défaut si non trouvé
        params_clean.pop("num_boost_round", None)

        self.model = lgb.train(
            params_clean,
            lgb_data,
            num_boost_round=n_rounds,
        )

        # ── Métriques d'entraînement rapides ──────────────────────────────────
        y_pred_train = self.model.predict(X.values)
        import math
        rmse_train = math.sqrt(((y.values - y_pred_train) ** 2).mean())
        print(f"[LGBMScorer] RMSE train (informatif seulement) : {rmse_train:.4f}")

        return self

    # ── Prédiction ────────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Prédit le fraud_score pour de nouveaux tronçons.

        Vérifie que les features sont identiques à celles utilisées lors du train.
        Retourne une Series de scores clippés dans [0, 1].
        """
        if self.model is None:
            raise RuntimeError(
                "[LGBMScorer] Le modèle n'est pas entraîné. "
                "Appelle train() ou charge un modèle avec load()."
            )
        if not self.feature_cols:
            raise RuntimeError(
                "[LGBMScorer] feature_cols vide. Modèle non entraîné correctement."
            )

        missing = set(self.feature_cols) - set(df.columns)
        if missing:
            raise ValueError(
                f"[LGBMScorer] Colonnes manquantes pour la prédiction : {missing}"
            )

        X = df[self.feature_cols]
        raw = self.model.predict(X.values)
        return pd.Series(raw, index=df.index).clip(lower=0.0, upper=1.0)

    # ── Évaluation ────────────────────────────────────────────────────────────

    def evaluate(self, df: pd.DataFrame, target_col: str = "fraud_score") -> dict[str, float]:
        """
        Calcule les métriques d'évaluation sur un DataFrame de test.

        Métriques retournées :
            rmse         : Root Mean Square Error — pénalise les grosses erreurs
            mae          : Mean Absolute Error — interprétable en unités de fraud_score
            r2           : R² — 0=modèle nul, 1=parfait, <0=pire que la moyenne
            spearman_rho : Corrélation de rang — MÉTRIQUE CLÉ pour ce projet
                           L'objectif LAF est d'ordonner les tronçons par risque,
                           pas de prédire le taux exact. Spearman mesure si les
                           tronçons les plus risqués sont bien classés en premier.
            mape_pct     : Mean Absolute Percentage Error (%)
                           Instable quand y_true ≈ 0, interpréter avec précaution.

        Règle d'interprétation pour un modèle SAIN sur ce dataset :
            RMSE      ∈ [0.05, 0.15]   si < 0.01 → leakage probable
            R²        ∈ [0.30, 0.70]   si > 0.90 → leakage probable
            Spearman  ∈ [0.50, 0.80]   si > 0.95 → leakage probable
        """
        if target_col not in df.columns:
            raise ValueError(f"[LGBMScorer] Colonne '{target_col}' absente pour l'évaluation.")

        y_true = df[target_col].values
        y_pred = self.predict(df).values

        residuals = y_pred - y_true
        rmse = math.sqrt((residuals ** 2).mean())
        mae  = np.abs(residuals).mean()

        # R²
        ss_res = (residuals ** 2).sum()
        ss_tot = ((y_true - y_true.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        # Corrélation de Spearman (basée sur les rangs)
        from scipy.stats import spearmanr
        rho, p_val = spearmanr(y_true, y_pred)

        # MAPE (avec plancher pour éviter la division par zéro)
        mape = np.mean(np.abs(residuals) / (np.abs(y_true) + 1e-8)) * 100

        metrics = {
            "rmse":         round(float(rmse), 6),
            "mae":          round(float(mae),  6),
            "r2":           round(float(r2),   4),
            "spearman_rho": round(float(rho),  4),
            "spearman_p":   float(p_val),
            "mape_pct":     round(float(mape), 2),
        }

        # Alertes automatiques sur les indicateurs de leakage
        self._print_evaluation_report(metrics)
        return metrics

    # ── Persistance ───────────────────────────────────────────────────────────

    def save(self, filename: str = "lgbm_scorer.joblib") -> Path:
        """Sauvegarde le scorer complet (modèle + feature_cols + params)."""
        if self.model is None:
            raise RuntimeError("[LGBMScorer] Impossible de sauvegarder un modèle non entraîné.")
        save_path = config.MODELS_DIR / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, save_path)
        return save_path

    @classmethod
    def load(cls, filename: str = "lgbm_scorer.joblib") -> "LGBMScorer":
        """Charge un scorer depuis le disque."""
        load_path = config.MODELS_DIR / filename
        if not load_path.exists():
            raise FileNotFoundError(f"[LGBMScorer] Fichier introuvable : {load_path}")
        return joblib.load(load_path)

    # ── Méthodes privées ──────────────────────────────────────────────────────

    def _check_leakage(self, df: pd.DataFrame, target_col: str) -> None:
        """
        Détecte les features quasi-colinéaires avec la cible.

        Calcule la corrélation de Pearson entre chaque feature et la cible.
        Si une feature a une corrélation > 0.95, c'est un signal fort de leakage.

        Cette vérification est non-bloquante (warning, pas exception) car une
        corrélation élevée peut être légitime dans certains cas. Le juger est
        la responsabilité du data scientist qui lit le log.
        """
        y = df[target_col]
        high_corr = []
        for col in self.feature_cols:
            if col in df.columns:
                # --- AJOUT : Ignore les colonnes constantes (écart-type de 0) ---
                # Évite le RuntimeWarning de Numpy lors du calcul de la corrélation
                if df[col].nunique() <= 1:
                    continue
                # ----------------------------------------------------------------

                corr = abs(df[col].corr(y))
                if corr > 0.95:
                    high_corr.append((col, round(corr, 4)))

        if high_corr:
            print(
                f"[LGBMScorer] ⚠️  ALERTE LEAKAGE — Features avec corrélation > 0.95 "
                f"avec la cible '{target_col}' :"
            )
            for col, corr in sorted(high_corr, key=lambda x: -x[1]):
                print(f"   • {col!r:40s} : corr = {corr:.4f}  ← SUSPECT")
            print(
                "   → Vérifie que ces features ne sont pas des composants de la cible.\n"
                "   → Si leakage confirmé : ajoute la colonne à COLS_NON_FEATURES."
            )
        else:
            print("[LGBMScorer] ✅ Contrôle anti-leakage : aucune corrélation > 0.95 détectée.")

    def _print_evaluation_report(self, metrics: dict[str, float]) -> None:
        """Affiche un rapport d'évaluation structuré avec indicateurs de santé."""
        rho = metrics["spearman_rho"]
        r2  = metrics["r2"]
        rmse = metrics["rmse"]

        def status(val, lo, hi, inverse=False):
            """Retourne ✅ si la valeur est dans la plage saine, ⚠️ sinon."""
            in_range = lo <= val <= hi
            if inverse:
                in_range = not in_range
            return "✅" if in_range else "⚠️ "

        print("\n" + "─" * 55)
        print("  RAPPORT D'ÉVALUATION")
        print("─" * 55)
        print(f"  RMSE        : {rmse:.4f}  {status(rmse, 0.05, 0.15)}  (sain: 0.05–0.15)")
        print(f"  MAE         : {metrics['mae']:.4f}")
        print(f"  R²          : {r2:.4f}  {status(r2, 0.30, 0.70)}  (sain: 0.30–0.70)")
        print(f"  Spearman ρ  : {rho:.4f}  {status(rho, 0.50, 0.80)}  (sain: 0.50–0.80)")
        print(f"  MAPE        : {metrics['mape_pct']:.1f}%")
        print("─" * 55)

        # Verdict global
        leakage_signals = sum([
            rmse < 0.01,
            r2   > 0.90,
            rho  > 0.95,
        ])
        if leakage_signals >= 2:
            print(
                "  ❌ VERDICT : {}/3 indicateurs de leakage détectés.\n"
                "     Le modèle a probablement mémorisé la cible,\n"
                "     pas appris une vraie relation prédictive.".format(leakage_signals)
            )
        elif leakage_signals == 1:
            print(
                "  ⚠️  VERDICT : 1/3 indicateur suspect. Analyse feature importance."
            )
        else:
            print("  ✅ VERDICT : Pas d'indicateur de leakage. Modèle sain.")
        print("─" * 55 + "\n")

    def feature_importance_df(self) -> pd.DataFrame:
        """
        Retourne les importances des features triées par gain décroissant.

        Utile pour détecter visuellement si une feature domine anormalement
        (signe que le modèle s'est appuyé sur une quasi-identité avec la cible).
        """
        if self.model is None:
            raise RuntimeError("[LGBMScorer] Modèle non entraîné.")
        importance = self.model.feature_importance(importance_type="gain")
        return (
            pd.DataFrame({
                "feature":    self.model.feature_name(),
                "importance": importance,
            })
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )