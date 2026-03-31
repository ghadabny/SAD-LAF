# services/ml_engine/evaluate.py
"""
Diagnostic visuel du modèle SAD-LAF.

Lance ce script APRÈS un entraînement pour auditer la qualité du modèle
et détecter d'éventuels problèmes (leakage, overfitting, biais).

Usage :
    python services/ml_engine/evaluate.py

Prérequis :
    - Un modèle entraîné : data/models/lgbm_scorer.joblib
    - Un pipeline de features : data/models/feature_pipeline.joblib
    - pip install matplotlib seaborn scipy

Sorties :
    data/outputs/diagnostic_modele.png   — 5 graphiques de diagnostic
    data/outputs/feature_importance.csv  — importances des features
"""

import sys
from pathlib import Path

# Ajoute la racine du projet au sys.path
root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root))

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from shared.config import config
from services.ml_engine.features.pipeline import FeaturePipeline
from services.ml_engine.models.lgbm_model import LGBMScorer
from sklearn.model_selection import train_test_split


# ─────────────────────────────────────────────────────────────────────────────
# Chargement
# ─────────────────────────────────────────────────────────────────────────────

def load_test_data() -> pd.DataFrame:
    """
    Recharge le dataset depuis le cache L2 et reproduit exactement
    le même split train/test que train.py (random_state=42).

    Retourne uniquement le jeu de TEST pour les diagnostics.
    """
    parquet_files = sorted(config.PROCESSED_DIR.glob("training_dataset_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            "❌ Cache L2 introuvable dans data/processed/.\n"
            "Lance d'abord : python services/ml_engine/train.py"
        )

    print(f"[evaluate] Chargement depuis : {parquet_files[-1].name}")
    df_raw = pd.read_parquet(parquet_files[-1])
    print(f"[evaluate] {len(df_raw):,} tronçons chargés.")

    # ── Propagation de service_date (correctif date.today()) ─────────────
    # On recharge calendar_dates depuis le GTFS pour obtenir les vraies
    # dates de circulation, comme dans train.py.
    from services.ml_engine.data.gtfs.loader import GTFSLoader as _GTFSLoader
    from services.ml_engine.data.gtfs.preprocessor import GTFSPreprocessor as _Prep
    from services.ml_engine.train import _attach_service_dates

    try:
        _calendar = _GTFSLoader().load_calendar_dates()
        df_raw = _attach_service_dates(df_raw, _calendar)
    except Exception as e:
        from datetime import date
        print(f"[evaluate] ⚠️  Impossible de charger calendar_dates ({e}). Fallback date.today().")
        df_raw["service_date"] = date.today()

    pipeline = FeaturePipeline.load("feature_pipeline.joblib")
    df_features = pipeline.transform(df_raw)

    # Reproduit exactement le split de train.py
    _, df_test = train_test_split(df_features, test_size=0.2, random_state=42)
    print(f"[evaluate] Jeu de test : {len(df_test):,} tronçons.")
    return df_test


# ─────────────────────────────────────────────────────────────────────────────
# Graphiques
# ─────────────────────────────────────────────────────────────────────────────

def plot_diagnostics(scorer: LGBMScorer, df_test: pd.DataFrame) -> None:
    """
    Génère 5 graphiques de diagnostic et les sauvegarde dans data/outputs/.

    Graphiques :
        1. Predicted vs Actual    — leakage → points sur la diagonale parfaite
        2. Distribution des erreurs — leakage → pic ultra-étroit autour de 0
        3. Résidus vs Valeur réelle — biais systématique = problème
        4. Feature Importance (gain) — feature dominante = leakage suspect
        5. Distribution cible réelle vs prédite — divergence = biais
    """
    y_true = df_test["fraud_score"].values
    y_pred = scorer.predict(df_test).values
    residuals = y_pred - y_true
    feat_df = scorer.feature_importance_df()

    rmse = float(np.sqrt((residuals ** 2).mean()))
    rho  = float(spearmanr(y_true, y_pred)[0])

    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(
        f"Diagnostic SAD-LAF  |  N test = {len(df_test):,} tronçons  |  "
        f"RMSE = {rmse:.4f}  |  Spearman ρ = {rho:.3f}",
        fontsize=13, fontweight="bold", y=0.99,
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)

    # ── 1. Predicted vs Actual ────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(y_true, y_pred, alpha=0.25, s=8, color="steelblue", rasterized=True)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax1.plot(lims, lims, "r--", linewidth=1.5, label="Parfait")
    ax1.set_xlabel("fraud_score réel")
    ax1.set_ylabel("fraud_score prédit")
    ax1.set_title("① Predicted vs Actual\n(leakage → tous sur la diagonale)")
    ax1.legend(fontsize=8)

    # ── 2. Distribution des erreurs ───────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(residuals, bins=80, color="coral", edgecolor="white", linewidth=0.3)
    ax2.axvline(0, color="black", linewidth=1.2)
    ax2.axvline(
        residuals.mean(), color="red", linewidth=1, linestyle="--",
        label=f"Moyenne = {residuals.mean():.3f}",
    )
    ax2.set_xlabel("Erreur (prédit − réel)")
    ax2.set_ylabel("Fréquence")
    ax2.set_title("② Distribution des erreurs\n(leakage → pic < 0.01 de largeur)")
    ax2.legend(fontsize=8)

    # ── 3. Résidus vs Valeur réelle ───────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(y_true, residuals, alpha=0.25, s=8, color="mediumpurple", rasterized=True)
    ax3.axhline(0, color="black", linewidth=1.2)
    z = np.polyfit(y_true, residuals, 1)
    x_line = np.linspace(y_true.min(), y_true.max(), 100)
    ax3.plot(x_line, np.poly1d(z)(x_line), "r-", linewidth=1.5, label="Tendance")
    ax3.set_xlabel("fraud_score réel")
    ax3.set_ylabel("Résidu (prédit − réel)")
    ax3.set_title("③ Résidus vs Valeur réelle\n(tendance non plate = biais)")
    ax3.legend(fontsize=8)

    # ── 4. Feature Importance ─────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, :2])
    top_n = min(15, len(feat_df))
    feat_top = feat_df.head(top_n).iloc[::-1]

    SUSPECTS = {
        "taux_irregularite", "nb_irregularites",
        "fraud_score", "hist_fraud_score_segment",
    }
    colors = [
        "crimson" if f in SUSPECTS else "steelblue"
        for f in feat_top["feature"]
    ]

    ax4.barh(range(top_n), feat_top["importance"], color=colors)
    ax4.set_yticks(range(top_n))
    ax4.set_yticklabels(feat_top["feature"], fontsize=9)
    ax4.set_xlabel("Importance (gain)")
    ax4.set_title("④ Feature Importance (gain)\n🔴 = features suspectes (composants de la cible)")

    from matplotlib.patches import Patch
    ax4.legend(
        handles=[
            Patch(facecolor="steelblue", label="Feature normale"),
            Patch(facecolor="crimson",   label="Feature suspecte (leakage)"),
        ],
        fontsize=8, loc="lower right",
    )

    # ── 5. Distribution réelle vs prédite ─────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    bins = np.linspace(0, 1, 40)
    ax5.hist(y_true, bins=bins, alpha=0.6, label="Réel",   color="steelblue")
    ax5.hist(y_pred, bins=bins, alpha=0.6, label="Prédit", color="coral")
    ax5.set_xlabel("fraud_score")
    ax5.set_ylabel("Fréquence")
    ax5.set_title("⑤ Distribution réelle vs prédite\n(divergence = biais de prédiction)")
    ax5.legend(fontsize=8)

    # ── Sauvegarde ────────────────────────────────────────────────────────────
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.OUTPUTS_DIR / "diagnostic_modele.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[evaluate] 📊 Graphiques sauvegardés → {out_path}")
    plt.show()


def export_feature_importance(scorer: LGBMScorer) -> None:
    """Exporte les importances des features en CSV et affiche le top 10."""
    feat_df = scorer.feature_importance_df()
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.OUTPUTS_DIR / "feature_importance.csv"
    feat_df.to_csv(out, index=False, encoding="utf-8")
    print(f"[evaluate] 📋 Feature importance → {out}")
    print("\n  Top 10 features :")
    print(feat_df.head(10).to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# Point d'entrée
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * 55)
    print("🔍  SAD-LAF — Diagnostic du modèle")
    print("=" * 55 + "\n")

    try:
        print("[evaluate] Chargement du modèle...")
        scorer = LGBMScorer.load("lgbm_scorer.joblib")
        print(f"[evaluate] Features utilisées : {scorer.feature_cols}\n")

        df_test = load_test_data()

        print("\n[evaluate] Calcul des métriques...")
        scorer.evaluate(df_test, target_col="fraud_score")

        print("\n[evaluate] Génération des graphiques...")
        plot_diagnostics(scorer, df_test)

        export_feature_importance(scorer)

        print("\n✅  Diagnostic terminé.")

    except FileNotFoundError as e:
        print(f"\n❌  {e}")
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"\n❌  Erreur : {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()