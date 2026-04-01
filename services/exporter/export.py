# services/exporter/export.py
"""
services/exporter/export.py — Orchestration de l'export CSV vers Power Automate.

Responsabilité unique : lire les tournées générées par le solver (JSON),
les formater via TourneeFormatter, et écrire le CSV horodaté dans
data/outputs/ pour que Power Automate le détecte et injecte dans Dataverse.

Flux complet :
    POST /optimize → tournee_YYYY-MM-DD_HHMMSS.json  (solver.py ou API)
                          ↓  (export.py — ce fichier)
                    tournees_export_YYYYMMDD_HHMMSS.csv
                          ↓  (OneDrive sync automatique)
                    Power Automate (déclencheur "Quand un fichier est créé")
                          ↓
                    Dataverse → Power Apps

Principes SOLID :
    S — export.py orchestre UNIQUEMENT. Le formatage est dans formatter.py,
        la lecture JSON dans _load_tournee(), l'écriture CSV dans _write_csv().
    O — Pour ajouter un format de sortie (Excel, JSON Dataverse...), on ajoute
        une fonction _write_excel() sans modifier _run().
    D — Dépend de TourneeFormatter et TourneeSchema (abstractions), pas du CSV brut.

Usage :
    python -m services.exporter.export                    # exporte toutes les tournées du jour
    python -m services.exporter.export --date 2024-09-02  # exporte une date spécifique
    python -m services.exporter.export --all              # exporte toutes les tournées disponibles
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from services.exporter.formatter import TourneeFormatter
from shared.config import config
from shared.schemas import (
    ArcSchema,
    ArcTypeSchema,
    NodeSchema,
    OptimizeResponseSchema,
    TourneeRequestSchema,
    TourneeSchema,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration principale
# ─────────────────────────────────────────────────────────────────────────────

class TourneeExporter:
    """
    Orchestre la lecture des tournées JSON et l'écriture du CSV de sortie.

    Responsabilité unique : trouver les fichiers JSON de tournées, les charger,
    les formater et les écrire dans le dossier de sortie.

    Ne sait pas comment formater (rôle de TourneeFormatter).
    Ne sait pas comment Dataverse consomme le CSV (rôle de Power Automate).
    """

    def __init__(self):
        self.outputs_dir  = config.OUTPUTS_DIR
        self.formatter    = TourneeFormatter()

    # ── Interface publique ────────────────────────────────────────────────────

    def run(
        self,
        service_date: date | None = None,
        export_all:   bool = False,
    ) -> Path | None:
        """
        Orchestre l'export complet.

        Paramètres :
            service_date : si fourni, exporte uniquement les tournées de ce jour.
                           Si None, exporte les tournées du jour courant.
            export_all   : si True, exporte toutes les tournées disponibles
                           (ignore service_date).

        Retourne :
            Path du fichier CSV créé, ou None si aucune tournée trouvée.
        """
        logger.info("═══════════════════════════════════════")
        logger.info("🚀 SAD-LAF Exporter — Démarrage")
        logger.info("═══════════════════════════════════════")

        # ── Étape 1 : Trouver les fichiers JSON de tournées ───────────────────
        if export_all:
            json_files = self._find_all_tournee_files()
        else:
            target_date = service_date or date.today()
            json_files  = self._find_tournee_files_for_date(target_date)

        if not json_files:
            logger.warning(
                "Aucun fichier de tournée JSON trouvé dans '%s'. "
                "Lance d'abord POST /optimize pour générer des tournées.",
                self.outputs_dir,
            )
            return None

        logger.info("📁 %d fichier(s) de tournée trouvé(s).", len(json_files))

        # ── Étape 2 : Charger et valider chaque tournée ───────────────────────
        tournees = []
        for json_path in json_files:
            tournee = self._load_tournee(json_path)
            if tournee is not None:
                tournees.append(tournee)

        if not tournees:
            logger.error("Aucune tournée valide chargée. Vérifiez les fichiers JSON.")
            return None

        logger.info("✅ %d tournée(s) chargée(s) et validées.", len(tournees))

        # ── Étape 3 : Formater en DataFrame ──────────────────────────────────
        df = self.formatter.format_batch(tournees)

        if df.empty:
            logger.warning("DataFrame vide après formatage. Aucune ligne à exporter.")
            return None

        # ── Étape 4 : Écrire le CSV horodaté ─────────────────────────────────
        csv_path = self._write_csv(df)

        logger.info("═══════════════════════════════════════")
        logger.info("✅ Export terminé : %s", csv_path)
        logger.info("   Lignes : %d | Tournées : %d", len(df), len(tournees))
        logger.info("═══════════════════════════════════════")

        return csv_path

    # ── Helpers privés ────────────────────────────────────────────────────────

    def _find_tournee_files_for_date(self, service_date: date) -> list[Path]:
        """
        Trouve les fichiers JSON de tournées pour une date donnée.

        Convention de nommage attendue :
            tournee_YYYY-MM-DD_*.json
            optimize_response_YYYY-MM-DD_*.json

        Exemple :
            tournee_2024-09-02_083000.json
        """
        date_str = service_date.strftime("%Y-%m-%d")
        patterns = [
            f"tournee_{date_str}*.json",
            f"optimize_response_{date_str}*.json",
            f"*tournee*{date_str}*.json",
        ]

        found = set()
        for pattern in patterns:
            found.update(self.outputs_dir.glob(pattern))

        return sorted(found)

    def _find_all_tournee_files(self) -> list[Path]:
        """Trouve tous les fichiers JSON de tournées disponibles."""
        patterns = [
            "tournee_*.json",
            "optimize_response_*.json",
        ]
        found = set()
        for pattern in patterns:
            found.update(self.outputs_dir.glob(pattern))
        return sorted(found)

    def _load_tournee(self, json_path: Path) -> TourneeSchema | None:
        """
        Charge et valide un fichier JSON de tournée.

        Supporte deux formats :
            1. OptimizeResponseSchema (sortie directe de POST /optimize)
               → extrait le champ 'tournee'
            2. TourneeSchema directement (format simplifié)

        Retourne None si le fichier est invalide (log de l'erreur).
        """
        try:
            raw = json.loads(json_path.read_text(encoding="utf-8"))

            # Format 1 : réponse complète de l'API (contient 'tournee' + 'request')
            if "tournee" in raw and "request" in raw:
                response = OptimizeResponseSchema.model_validate(raw)
                logger.debug("Chargé (format OptimizeResponseSchema) : %s", json_path.name)
                return response.tournee

            # Format 2 : TourneeSchema directement
            if "arcs" in raw and "score_total" in raw:
                tournee = TourneeSchema.model_validate(raw)
                logger.debug("Chargé (format TourneeSchema) : %s", json_path.name)
                return tournee

            logger.warning(
                "Format JSON non reconnu dans '%s'. "
                "Attendu : OptimizeResponseSchema ou TourneeSchema.",
                json_path.name,
            )
            return None

        except json.JSONDecodeError as e:
            logger.error("JSON invalide dans '%s' : %s", json_path.name, e)
            return None
        except Exception as e:
            logger.error("Erreur lors du chargement de '%s' : %s", json_path.name, e)
            return None

    def _write_csv(self, df: pd.DataFrame) -> Path:
        """
        Écrit le DataFrame en CSV horodaté dans data/outputs/.

        Convention de nommage :
            tournees_export_YYYYMMDD_HHMMSS.csv

        Le nom horodaté garantit qu'un nouveau fichier est créé à chaque export.
        Power Automate détecte les NOUVEAUX fichiers — un fichier écrasé
        ne déclencherait pas le flux.

        Encodage : UTF-8 avec BOM (utf-8-sig) pour compatibilité Excel / Power Apps.
        Séparateur : point-virgule (convention française / SNCF).
        """
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = f"tournees_export_{timestamp}.csv"
        csv_path  = self.outputs_dir / filename

        df.to_csv(
            csv_path,
            index=False,
            sep=";",                # Séparateur français (compatible Excel FR)
            encoding="utf-8-sig",   # BOM UTF-8 → Excel ouvre sans problème d'accents
            date_format="%Y-%m-%d",
        )

        size_kb = csv_path.stat().st_size // 1024
        logger.info(
            "💾 CSV écrit : %s (%d Ko, %d lignes)",
            csv_path.name, size_kb, len(df),
        )
        return csv_path

    def _archive_processed_files(self, json_files: list[Path]) -> None:
        """
        Déplace les fichiers JSON traités vers un sous-dossier 'archives/'.
        Optionnel — appelé si --archive est passé en CLI.

        Évite de retraiter les mêmes fichiers lors du prochain export.
        """
        archive_dir = self.outputs_dir / "archives"
        archive_dir.mkdir(parents=True, exist_ok=True)

        for f in json_files:
            dest = archive_dir / f.name
            f.rename(dest)
            logger.debug("Archivé : %s → %s", f.name, dest)

        logger.info("📦 %d fichier(s) JSON archivé(s) dans '%s'.", len(json_files), archive_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Fonction utilitaire : sauvegarde d'une tournée en JSON (appelée par le solver)
# ─────────────────────────────────────────────────────────────────────────────

def save_tournee_to_json(
    response: OptimizeResponseSchema,
    output_dir: Path | None = None,
) -> Path:
    """
    Sauvegarde une OptimizeResponseSchema en fichier JSON horodaté.

    Appelée par solver.py ou l'API après chaque génération de tournée.
    L'exporter lit ensuite ces fichiers pour produire le CSV Dataverse.

    Paramètres :
        response   : réponse complète de POST /optimize
        output_dir : dossier de sortie (défaut : config.OUTPUTS_DIR)

    Retourne :
        Path du fichier JSON créé.
    """
    out_dir = Path(output_dir) if output_dir else config.OUTPUTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    service_date = response.tournee.service_date.strftime("%Y-%m-%d")
    timestamp    = datetime.now().strftime("%H%M%S_%f")
    filename     = f"tournee_{service_date}_{timestamp}.json"
    json_path    = out_dir / filename

    # Sérialisation Pydantic → JSON avec gestion des types date/datetime
    json_path.write_text(
        response.model_dump_json(indent=2),
        encoding="utf-8",
    )

    logger.info(
        "[save_tournee_to_json] Tournée sauvegardée : %s "
        "(score=%.4f, %d trains)",
        json_path.name,
        response.tournee.score_total,
        response.tournee.nb_trains,
    )
    return json_path


# ─────────────────────────────────────────────────────────────────────────────
# Point d'entrée CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SAD-LAF Exporter — Génère le CSV Dataverse depuis les tournées JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python -m services.exporter.export                     # Exporte les tournées du jour
  python -m services.exporter.export --date 2024-09-02   # Exporte un jour précis
  python -m services.exporter.export --all               # Exporte toutes les tournées
        """,
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Date de service (YYYY-MM-DD). Défaut : aujourd'hui.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="export_all",
        help="Exporte toutes les tournées disponibles (ignore --date).",
    )
    return parser.parse_args()


def _run() -> None:
    """Point d'entrée principal."""
    args = _parse_args()

    service_date = None
    if args.date and not args.export_all:
        try:
            service_date = date.fromisoformat(args.date)
        except ValueError:
            logger.error(
                "Format de date invalide : '%s'. Attendu : YYYY-MM-DD.", args.date
            )
            sys.exit(1)

    exporter = TourneeExporter()
    csv_path = exporter.run(
        service_date=service_date,
        export_all=args.export_all,
    )

    if csv_path is None:
        logger.warning("Aucun CSV produit.")
        sys.exit(0)

    print(f"\n✅ Export terminé : {csv_path}")


if __name__ == "__main__":
    _run()