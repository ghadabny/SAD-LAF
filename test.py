"""
debug_laf_columns.py
--------------------
Lance ce script AVANT de relancer train.py pour voir les vrais noms de colonnes
de tes fichiers LAF et diagnostiquer pourquoi toutes les lignes sont supprimées.

Usage :
    python debug_laf_columns.py
"""

import pandas as pd
from pathlib import Path

LAF_DIR = Path("data/raw/laf")


def inspect_file(path: Path, nrows: int = 5):
    """Lit les premières lignes d'un CSV et affiche colonnes + valeurs."""
    print(f"\n{'=' * 70}")
    print(f"Fichier : {path.name}")
    print(f"{'=' * 70}")

    # Détecte le séparateur
    with open(path, encoding="latin-1") as f:
        header = f.readline()
    sep = ";" if header.count(";") >= header.count(",") else ","
    print(f"Séparateur détecté : '{sep}'")

    df = pd.read_csv(path, sep=sep, encoding="latin-1", nrows=nrows, low_memory=False)

    print(f"\n📋 Colonnes ({len(df.columns)}) :")
    for col in df.columns:
        sample = df[col].iloc[0] if len(df) > 0 else "N/A"
        n_null = df[col].isna().sum()
        print(f"   [{n_null}/{nrows} nulls] {col!r:60s} | ex: {str(sample)[:50]!r}")

    return df


def check_required_columns(df: pd.DataFrame, required: list[str], label: str):
    """Vérifie quelles colonnes requises sont présentes / absentes."""
    print(f"\n🔍 Colonnes REQUISES par le code ({label}) :")
    for col in required:
        status = "✅ PRÉSENTE" if col in df.columns else "❌ ABSENTE"
        print(f"   {status} : {col!r}")


# ── Colonnes attendues par le preprocessor ────────────────────────────────────
CC_REQUIRED_IN_CODE = [
    "ticket_travelInformation_origin_uicCode",
    "ticket_travelInformation_destination_uicCode",
    "ticket_travelInformation_departureDateTime",
    "verifiedTickets_verificationStatus",
]

CC_DATETIME_COLS = [
    "verifiedTickets_verificationDateTime",
    "ticket_travelInformation_departureDateTime",
]


def main():
    if not LAF_DIR.exists():
        print(f"❌ Dossier LAF introuvable : {LAF_DIR}")
        return

    csv_files = sorted(LAF_DIR.glob("*.csv"))
    if not csv_files:
        print(f"❌ Aucun CSV trouvé dans {LAF_DIR}")
        return

    print(f"✅ {len(csv_files)} fichier(s) LAF trouvé(s) dans {LAF_DIR}/")

    # Inspecte le fichier CC
    cc_files = [f for f in csv_files if "CC" in f.name.upper()]
    if cc_files:
        df_cc = inspect_file(cc_files[0], nrows=3)
        check_required_columns(df_cc, CC_REQUIRED_IN_CODE, "CC")

        # Vérifie le contenu de verifiedTickets_verificationStatus
        if "verifiedTickets_verificationStatus" in df_cc.columns:
            print("\n📊 Valeurs uniques de verifiedTickets_verificationStatus (5 premières lignes) :")
            # Charge plus de lignes pour avoir un aperçu
            with open(cc_files[0], encoding="latin-1") as f:
                header = f.readline()
            sep = ";" if header.count(";") >= header.count(",") else ","
            df_sample = pd.read_csv(cc_files[0], sep=sep, encoding="latin-1",
                                    nrows=1000, low_memory=False)
            if "verifiedTickets_verificationStatus" in df_sample.columns:
                counts = df_sample["verifiedTickets_verificationStatus"].value_counts()
                print(counts.to_string())

        # Vérifie les nulls sur les colonnes critiques (sur 1000 lignes)
        with open(cc_files[0], encoding="latin-1") as f:
            header = f.readline()
        sep = ";" if header.count(";") >= header.count(",") else ","
        df_1k = pd.read_csv(cc_files[0], sep=sep, encoding="latin-1",
                             nrows=1000, low_memory=False)
        print("\n📊 Taux de null sur 1000 lignes (colonnes critiques) :")
        for col in CC_REQUIRED_IN_CODE:
            if col in df_1k.columns:
                n_null = df_1k[col].isna().sum()
                pct = n_null / len(df_1k) * 100
                print(f"   {col!r}: {n_null}/1000 nulls ({pct:.1f}%)")
    else:
        print("\n⚠️  Pas de fichier CC trouvé.")

    # Inspecte un fichier SC
    sc_files = [f for f in csv_files if "SC" in f.name.upper()]
    if sc_files:
        print(f"\n\nFichiers SC disponibles ({len(sc_files)}) :")
        for f in sc_files:
            print(f"   - {f.name}")
        df_sc = inspect_file(sc_files[0], nrows=2)
        check_required_columns(df_sc, CC_REQUIRED_IN_CODE, "SC")
    else:
        print("\n⚠️  Pas de fichier SC trouvé.")


if __name__ == "__main__":
    main()