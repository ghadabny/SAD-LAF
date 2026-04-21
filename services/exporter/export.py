"""
services/exporter/export.py — Export CSV des tournées.

CORRECTIONS v3.1 :
    Bug 2 — double ligne CSV :
        Cause : export() appelait formatter.format() en interne, alors que le router
                avait déjà appelé format_with_id() pour extraire l'ID. Résultat :
                deux appels à format() → deux UUIDs → deux fichiers CSV, chacun
                avec l'ID du *second* appel (DF651B6B) au lieu de celui du JSON (378F17C3).
        Fix   : Ajout de export_from_df(df, response, tournee_id) qui reçoit
                directement le DataFrame pré-formaté. export() est conservé pour
                la compatibilité (tests, export JSON).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import pandas as pd

from shared.config import config

if TYPE_CHECKING:
    from shared.schemas import OptimizeResponseV2Schema

logger = logging.getLogger(__name__)


class TourneeExporter:
    """
    Écrit la tournée formatée en CSV dans le dossier de sortie.

    Deux méthodes d'export CSV :
        export(response)                    → formate + exporte (compat. ancienne API)
        export_from_df(df, response, id)    → exporte un DataFrame déjà formaté (v2)
    """

    def __init__(self, output_dir: Optional[Path] = None):
        self._output_dir = output_dir or config.OUTPUTS_DIR

    # ── Méthode principale (endpoint v2) ──────────────────────────────────────

    def export_from_df(
        self,
        df: pd.DataFrame,
        response: "OptimizeResponseV2Schema",
        tournee_id: str,
    ) -> Path:
        """
        Exporte un DataFrame déjà formaté en CSV.

        FIX Bug 2 : cette méthode ne rappelle PAS format(), évitant la
        double génération d'UUID et la double ligne dans le CSV.

        Paramètres :
            df          : DataFrame produit par TourneeFormatter.format_with_id()
            response    : réponse pour construire le nom de fichier
            tournee_id  : ID unique déjà fixé par le router
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)

        filename = self._build_filename_from_response(response, tournee_id)
        path     = self._output_dir / filename

        df.to_csv(
            path,
            index=False,
            encoding="utf-8-sig",   # BOM pour Excel France
            sep=";",
        )

        logger.info(
            "[TourneeExporter] ✅ CSV écrit : %s (%d lignes | score=%.3f)",
            path.name, len(df), response.tournee.score_total,
        )
        return path

    # ── Méthode de compatibilité ──────────────────────────────────────────────

    def export(self, response: "OptimizeResponseV2Schema") -> Path:
        """
        Formate et exporte en CSV. Conservé pour compatibilité avec les tests.
        ATTENTION : génère un nouvel UUID → ne pas utiliser dans l'endpoint v2.
        """
        from services.exporter.formatter import TourneeFormatter

        self._output_dir.mkdir(parents=True, exist_ok=True)

        formatter  = TourneeFormatter()
        df         = formatter.format(response)
        tournee_id = df.iloc[0]["tournee_id"] if not df.empty else "NOID"
        filename   = self._build_filename_from_response(response, tournee_id)
        path       = self._output_dir / filename

        df.to_csv(
            path,
            index=False,
            encoding="utf-8-sig",
            sep=";",
        )

        logger.info("[TourneeExporter] CSV (compat) écrit : %s", path.name)
        return path

    def export_json(self, response: "OptimizeResponseV2Schema") -> Path:
        """Sauvegarde la réponse complète en JSON (archive/debugging)."""
        self._output_dir.mkdir(parents=True, exist_ok=True)

        agent_id = (response.request.agent_id or "INCONNU").replace(" ", "_")
        date_str = response.tournee.service_date.strftime("%Y%m%d")
        time_str = response.optimized_at.strftime("%H%M%S")
        filename = f"tournee_{agent_id}_{date_str}_{time_str}.json"
        path     = self._output_dir / filename

        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                response.model_dump(mode="json"),
                f,
                ensure_ascii=False,
                indent=2,
                default=str,
            )

        logger.info("[TourneeExporter] JSON archivé : %s", path.name)
        return path

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_filename_from_response(
        self,
        response: "OptimizeResponseV2Schema",
        tournee_id: str,
    ) -> str:
        """
        Construit le nom de fichier CSV.
        Utilise le tournee_id fourni — jamais de nouvel UUID ici.
        Format : tournee_{agent_id}_{YYYYMMDD}_{HHmmss}.csv
        """
        agent_id = (response.request.agent_id or "INCONNU").upper().replace(" ", "_")
        date_str = response.tournee.service_date.strftime("%Y%m%d")
        time_str = response.optimized_at.strftime("%H%M%S")
        return f"tournee_{agent_id}_{date_str}_{time_str}.csv"

    # Gardé pour compat avec les anciens tests
    def _build_filename(self, response: "OptimizeResponseV2Schema") -> str:
        return self._build_filename_from_response(response, response.tournee_id)


# ── Fonction utilitaire module-level ─────────────────────────────────────────

def save_tournee_to_json(
    response: "OptimizeResponseV2Schema",
    output_dir: Optional[Path] = None,
) -> Path:
    exporter = TourneeExporter(output_dir=output_dir)
    return exporter.export_json(response)