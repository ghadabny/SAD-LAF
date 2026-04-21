# tests/unit/test_exporter_v2.py
"""
Tests unitaires de TourneeExporter et save_tournee_to_json.

Couvre :
    TourneeExporter.export()   — création du fichier CSV
    TourneeExporter.export_json() — création du fichier JSON
    _build_filename()          — nommage déterministe des fichiers
    Encodage CSV UTF-8 BOM     — compatibilité Excel France
    Séparateur point-virgule   — standard French locale
"""
from __future__ import annotations

import csv
import json
from datetime import date, datetime
from pathlib import Path

import pytest

from services.exporter.export import TourneeExporter, save_tournee_to_json
from shared.schemas import (
    ArcSchema,
    ArcTypeSchema,
    NodeSchema,
    OptimizeResponseV2Schema,
    TourneeRequestV2Schema,
    TourneeSchema,
)

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

SERVICE_DATE = date(2026, 4, 14)
NOW          = datetime(2026, 4, 14, 14, 30, 22)


def _make_node(stop_id, name, t):
    return NodeSchema(stop_id=stop_id, stop_name=name,
                      time_minutes=t, service_date=SERVICE_DATE)


def _make_train_arc(dep_id, dep_name, dep_t, arr_id, arr_name, arr_t, score=0.75):
    return ArcSchema(
        source=_make_node(dep_id, dep_name, dep_t),
        destination=_make_node(arr_id, arr_name, arr_t),
        arc_type=ArcTypeSchema.TRAIN,
        duration_min=arr_t - dep_t,
        trip_id="TRIP_TEST",
        train_number="117756",
        fraud_score=score,
    )


@pytest.fixture
def response(tmp_path) -> OptimizeResponseV2Schema:
    arc = _make_train_arc("87212027", "Strasbourg", 480, "87214007", "Sélestat", 509)
    tournee = TourneeSchema(
        arcs=[arc],
        score_total=0.75,
        duree_totale_minutes=29,
        nb_trains=1,
        gare_depart=_make_node("87212027", "Strasbourg", 480),
        gare_arrivee=_make_node("87214007", "Sélestat", 509),
        service_date=SERVICE_DATE,
        generated_at=NOW,
    )
    req = TourneeRequestV2Schema(
        gare_depart_id="87212027",
        heure_ps_min=270,
        heure_fs_min=840,
        service_date=SERVICE_DATE,
        agent_id="LAF_042",
    )
    return OptimizeResponseV2Schema(
        tournee_id="TRN_TEST_20260414_00000001",
        tournee=tournee,
        request=req,
        score_perte_pct=0.0,
        optimized_at=NOW,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tests export() — fichier CSV
# ─────────────────────────────────────────────────────────────────────────────

class TestTourneeExporter:

    def test_export_cree_fichier(self, response, tmp_path):
        exporter = TourneeExporter(output_dir=tmp_path)
        path     = exporter.export(response)
        assert path.exists()

    def test_export_retourne_path(self, response, tmp_path):
        exporter = TourneeExporter(output_dir=tmp_path)
        path     = exporter.export(response)
        assert isinstance(path, Path)

    def test_export_nom_fichier_format(self, response, tmp_path):
        """Format : tournee_{AGENT}_{YYYYMMDD}_{HHmmss}.csv"""
        exporter = TourneeExporter(output_dir=tmp_path)
        path     = exporter.export(response)
        assert path.name.startswith("tournee_LAF_042_20260414_")
        assert path.suffix == ".csv"

    def test_export_separateur_point_virgule(self, response, tmp_path):
        """Le fichier CSV doit utiliser ';' comme séparateur (Excel France)."""
        exporter = TourneeExporter(output_dir=tmp_path)
        path     = exporter.export(response)
        content  = path.read_text(encoding="utf-8-sig")
        # La première ligne (header) doit contenir des ';'
        first_line = content.split("\n")[0]
        assert ";" in first_line

    def test_export_encodage_utf8_bom(self, response, tmp_path):
        """Le fichier doit commencer par le BOM UTF-8 (EF BB BF)."""
        exporter = TourneeExporter(output_dir=tmp_path)
        path     = exporter.export(response)
        raw_bytes = path.read_bytes()
        assert raw_bytes[:3] == b"\xef\xbb\xbf", "BOM UTF-8 attendu en début de fichier"

    def test_export_contient_numero_train(self, response, tmp_path):
        exporter = TourneeExporter(output_dir=tmp_path)
        path     = exporter.export(response)
        content  = path.read_text(encoding="utf-8-sig")
        assert "117756" in content

    def test_export_contient_gare_dep(self, response, tmp_path):
        exporter = TourneeExporter(output_dir=tmp_path)
        path     = exporter.export(response)
        content  = path.read_text(encoding="utf-8-sig")
        assert "Strasbourg" in content

    def test_export_cree_dossier_si_inexistant(self, response, tmp_path):
        output_dir = tmp_path / "sous_dossier" / "exports"
        exporter   = TourneeExporter(output_dir=output_dir)
        path       = exporter.export(response)
        assert path.exists()

    def test_export_autant_de_lignes_que_trains(self, response, tmp_path):
        """Une tournée à 1 train → CSV avec 1 ligne de données + 1 header."""
        exporter = TourneeExporter(output_dir=tmp_path)
        path     = exporter.export(response)
        with open(path, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f, delimiter=";")
            rows   = list(reader)
        assert len(rows) == 1   # 1 train = 1 ligne

    def test_export_json(self, response, tmp_path):
        exporter = TourneeExporter(output_dir=tmp_path)
        path     = exporter.export_json(response)
        assert path.exists()
        assert path.suffix == ".json"

    def test_export_json_contenu_valide(self, response, tmp_path):
        exporter = TourneeExporter(output_dir=tmp_path)
        path     = exporter.export_json(response)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert "tournee" in data
        assert "request" in data

    def test_export_json_nom_fichier_format(self, response, tmp_path):
        exporter = TourneeExporter(output_dir=tmp_path)
        path     = exporter.export_json(response)
        assert path.name.startswith("tournee_LAF_042_20260414_")
        assert path.suffix == ".json"


# ─────────────────────────────────────────────────────────────────────────────
# Tests save_tournee_to_json() (fonction utilitaire)
# ─────────────────────────────────────────────────────────────────────────────

class TestSaveTourneeToJson:

    def test_cree_fichier_json(self, response, tmp_path):
        path = save_tournee_to_json(response, output_dir=tmp_path)
        assert path.exists()
        assert path.suffix == ".json"

    def test_json_est_valide(self, response, tmp_path):
        path = save_tournee_to_json(response, output_dir=tmp_path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict)


# ─────────────────────────────────────────────────────────────────────────────
# Tests _build_filename()
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildFilename:

    def test_agent_id_dans_nom(self, response, tmp_path):
        exporter  = TourneeExporter(output_dir=tmp_path)
        filename  = exporter._build_filename(response)
        assert "LAF_042" in filename

    def test_date_dans_nom(self, response, tmp_path):
        exporter = TourneeExporter(output_dir=tmp_path)
        filename = exporter._build_filename(response)
        assert "20260414" in filename

    def test_heure_dans_nom(self, response, tmp_path):
        exporter = TourneeExporter(output_dir=tmp_path)
        filename = exporter._build_filename(response)
        assert "143022" in filename   # NOW = 14:30:22

    def test_sans_agent_id_utilise_inconnu(self, tmp_path):
        arc = _make_train_arc("87212027", "Strasbourg", 480, "87214007", "Sélestat", 509)
        tournee = TourneeSchema(
            arcs=[arc], score_total=0.75, duree_totale_minutes=29, nb_trains=1,
            gare_depart=_make_node("87212027", "Strasbourg", 480),
            gare_arrivee=_make_node("87214007", "Sélestat", 509),
            service_date=SERVICE_DATE, generated_at=NOW,
        )
        req = TourneeRequestV2Schema(
            gare_depart_id="87212027", heure_ps_min=270, heure_fs_min=840,
            service_date=SERVICE_DATE, agent_id=None,
        )
        response_no_agent = OptimizeResponseV2Schema(
            tournee_id="TRN_INCONNU_20260414_00000001",
            tournee=tournee, request=req, optimized_at=NOW,
        )
        exporter = TourneeExporter(output_dir=tmp_path)
        filename = exporter._build_filename(response_no_agent)
        assert "INCONNU" in filename