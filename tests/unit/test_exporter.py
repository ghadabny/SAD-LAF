# tests/unit/test_exporter.py
"""
Tests unitaires du module exporter.

Couvre :
    TourneeFormatter  (formatter.py) — mise en forme des tournées
    TourneeExporter   (export.py)    — orchestration lecture/écriture
    save_tournee_to_json             — sauvegarde JSON d'une tournée
"""
from __future__ import annotations

import json
import math
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

from services.exporter.formatter import (
    TourneeFormatter,
    _minutes_to_hhmm,
    _score_to_priorite,
    SEUIL_HAUTE,
    SEUIL_MOYENNE,
)
from services.exporter.export import TourneeExporter, save_tournee_to_json
from shared.schemas import (
    ArcSchema,
    ArcTypeSchema,
    NodeSchema,
    OptimizeResponseSchema,
    TourneeRequestSchema,
    TourneeSchema,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

SERVICE_DATE = date(2024, 9, 2)


def _make_node(stop_id: str, stop_name: str, time_minutes: int) -> NodeSchema:
    return NodeSchema(
        stop_id=stop_id,
        stop_name=stop_name,
        time_minutes=time_minutes,
        service_date=SERVICE_DATE,
    )


def _make_arc_train(
    dep_id: str, dep_name: str, dep_min: int,
    arr_id: str, arr_name: str, arr_min: int,
    train_number: str = "117756",
    fraud_score: float = 0.75,
) -> ArcSchema:
    return ArcSchema(
        source=_make_node(dep_id, dep_name, dep_min),
        destination=_make_node(arr_id, arr_name, arr_min),
        arc_type=ArcTypeSchema.TRAIN,
        duration_min=arr_min - dep_min,
        trip_id="TRIP_001",
        train_number=train_number,
        fraud_score=fraud_score,
    )


def _make_arc_correspondance(
    dep_id: str, dep_name: str, dep_min: int,
    arr_id: str, arr_name: str, arr_min: int,
) -> ArcSchema:
    return ArcSchema(
        source=_make_node(dep_id, dep_name, dep_min),
        destination=_make_node(arr_id, arr_name, arr_min),
        arc_type=ArcTypeSchema.CORRESPONDANCE,
        duration_min=arr_min - dep_min,
        fraud_score=0.0,
    )


@pytest.fixture
def arc_train_stras_selestat() -> ArcSchema:
    """Arc TRAIN Strasbourg → Sélestat, score 0.75 (HAUTE)."""
    return _make_arc_train(
        "87212027", "Strasbourg", 503,
        "87214007", "Sélestat", 532,
        fraud_score=0.75,
    )


@pytest.fixture
def arc_train_selestat_colmar() -> ArcSchema:
    """Arc TRAIN Sélestat → Colmar, score 0.40 (MOYENNE)."""
    return _make_arc_train(
        "87214007", "Sélestat", 545,
        "87214080", "Colmar", 568,
        train_number="117758",
        fraud_score=0.40,
    )


@pytest.fixture
def arc_correspondance() -> ArcSchema:
    """Arc CORRESPONDANCE Sélestat, score 0.0."""
    return _make_arc_correspondance(
        "87214007", "Sélestat", 532,
        "87214007", "Sélestat", 545,
    )


@pytest.fixture
def tournee_simple(
    arc_train_stras_selestat,
    arc_correspondance,
    arc_train_selestat_colmar,
) -> TourneeSchema:
    """Tournée avec 2 trains + 1 correspondance."""
    arcs = [arc_train_stras_selestat, arc_correspondance, arc_train_selestat_colmar]
    return TourneeSchema(
        arcs=arcs,
        score_total=0.75 + 0.40,
        duree_totale_minutes=65,
        nb_trains=2,
        gare_depart=_make_node("87212027", "Strasbourg", 503),
        gare_arrivee=_make_node("87214080", "Colmar", 568),
        service_date=SERVICE_DATE,
        generated_at=datetime(2024, 9, 2, 10, 0, 0),
    )


@pytest.fixture
def optimize_response(tournee_simple) -> OptimizeResponseSchema:
    """OptimizeResponseSchema complète."""
    return OptimizeResponseSchema(
        tournee=tournee_simple,
        request=TourneeRequestSchema(
            gare_depart_id="87212027",
            heure_depart_min=480,
            duree_max_minutes=240,
            service_date=SERVICE_DATE,
        ),
        optimized_at=datetime(2024, 9, 2, 10, 0, 0),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tests utilitaires
# ─────────────────────────────────────────────────────────────────────────────

class TestMinutesToHHMM:

    def test_heure_normale(self):
        assert _minutes_to_hhmm(503) == "08:23"

    def test_minuit(self):
        assert _minutes_to_hhmm(0) == "00:00"

    def test_23h59(self):
        assert _minutes_to_hhmm(1439) == "23:59"

    def test_heure_ronde(self):
        assert _minutes_to_hhmm(480) == "08:00"

    def test_midi(self):
        assert _minutes_to_hhmm(720) == "12:00"


class TestScoreToPriorite:

    def test_score_haute(self):
        assert _score_to_priorite(0.8) == "HAUTE"
        assert _score_to_priorite(SEUIL_HAUTE) == "HAUTE"

    def test_score_moyenne(self):
        assert _score_to_priorite(0.5) == "MOYENNE"
        assert _score_to_priorite(SEUIL_MOYENNE) == "MOYENNE"

    def test_score_basse(self):
        assert _score_to_priorite(0.1) == "BASSE"
        assert _score_to_priorite(0.0) == "BASSE"

    def test_seuil_juste_en_dessous_haute(self):
        assert _score_to_priorite(SEUIL_HAUTE - 0.01) == "MOYENNE"

    def test_seuil_juste_en_dessous_moyenne(self):
        assert _score_to_priorite(SEUIL_MOYENNE - 0.01) == "BASSE"


# ─────────────────────────────────────────────────────────────────────────────
# Tests TourneeFormatter
# ─────────────────────────────────────────────────────────────────────────────

class TestTourneeFormatter:

    def setup_method(self):
        self.formatter = TourneeFormatter()

    def test_format_retourne_dataframe(self, tournee_simple):
        result = self.formatter.format(tournee_simple)
        assert isinstance(result, pd.DataFrame)

    def test_format_une_ligne_par_arc_train(self, tournee_simple):
        """Tournée avec 2 arcs TRAIN → 2 lignes dans le CSV."""
        result = self.formatter.format(tournee_simple)
        assert len(result) == 2

    def test_format_exclut_correspondances(self, tournee_simple):
        """Les arcs CORRESPONDANCE ne doivent pas apparaître dans le CSV."""
        result = self.formatter.format(tournee_simple)
        # 3 arcs au total, 1 correspondance → 2 lignes
        assert len(result) == tournee_simple.nb_trains

    def test_format_colonnes_presentes(self, tournee_simple):
        """Toutes les colonnes Dataverse doivent être présentes."""
        result = self.formatter.format(tournee_simple)
        colonnes_requises = [
            "tournee_id", "service_date", "generated_at",
            "gare_depart_id", "gare_depart_nom",
            "gare_arrivee_id", "gare_arrivee_nom",
            "duree_totale_minutes", "score_total", "nb_trains",
            "arc_ordre", "train_numero", "trip_id",
            "gare_dep_id", "gare_dep_nom", "heure_dep",
            "gare_arr_id", "gare_arr_nom", "heure_arr",
            "duree_troncon_minutes", "fraud_score", "priorite",
        ]
        for col in colonnes_requises:
            assert col in result.columns, f"Colonne manquante : {col}"

    def test_format_service_date_correcte(self, tournee_simple):
        result = self.formatter.format(tournee_simple)
        assert (result["service_date"] == "2024-09-02").all()

    def test_format_gare_depart_correcte(self, tournee_simple):
        result = self.formatter.format(tournee_simple)
        assert (result["gare_depart_id"] == "87212027").all()
        assert (result["gare_depart_nom"] == "Strasbourg").all()

    def test_format_arc_ordre_croissant(self, tournee_simple):
        result = self.formatter.format(tournee_simple)
        assert list(result["arc_ordre"]) == [1, 2]

    def test_format_heure_dep_format_hhmm(self, tournee_simple):
        result = self.formatter.format(tournee_simple)
        # Premier arc : Strasbourg 503 min = 08:23
        assert result.iloc[0]["heure_dep"] == "08:23"

    def test_format_fraud_score_correct(self, tournee_simple):
        result = self.formatter.format(tournee_simple)
        assert result.iloc[0]["fraud_score"] == pytest.approx(0.75)
        assert result.iloc[1]["fraud_score"] == pytest.approx(0.40)

    def test_format_priorite_haute(self, tournee_simple):
        result = self.formatter.format(tournee_simple)
        assert result.iloc[0]["priorite"] == "HAUTE"   # 0.75 ≥ SEUIL_HAUTE

    def test_format_priorite_moyenne(self, tournee_simple):
        result = self.formatter.format(tournee_simple)
        assert result.iloc[1]["priorite"] == "MOYENNE"  # 0.40 ≥ SEUIL_MOYENNE

    def test_format_score_total_correct(self, tournee_simple):
        result = self.formatter.format(tournee_simple)
        for val in result["score_total"]:
            assert float(val) == pytest.approx(1.15)

    def test_format_tournee_id_est_uuid(self, tournee_simple):
        """tournee_id doit être un UUID valide."""
        import uuid
        result = self.formatter.format(tournee_simple)
        tournee_ids = result["tournee_id"].unique()
        assert len(tournee_ids) == 1
        # Lève ValueError si pas un UUID valide
        uuid.UUID(tournee_ids[0])

    def test_format_meme_tournee_id_toutes_lignes(self, tournee_simple):
        """Toutes les lignes d'une même tournée doivent avoir le même tournee_id."""
        result = self.formatter.format(tournee_simple)
        assert result["tournee_id"].nunique() == 1

    def test_format_tournee_sans_arc_train_retourne_vide(self):
        """Une tournée avec uniquement des correspondances → DataFrame vide."""
        arc_corr = _make_arc_correspondance(
            "87212027", "Strasbourg", 503,
            "87214007", "Sélestat", 520,
        )
        tournee_vide = TourneeSchema(
            arcs=[arc_corr],
            score_total=0.0,
            duree_totale_minutes=17,
            nb_trains=1,
            gare_depart=_make_node("87212027", "Strasbourg", 503),
            gare_arrivee=_make_node("87214007", "Sélestat", 520),
            service_date=SERVICE_DATE,
            generated_at=datetime(2024, 9, 2, 10, 0, 0),
        )
        result = self.formatter.format(tournee_vide)
        assert result.empty

    def test_format_input_non_modifie(self, tournee_simple):
        """format() ne doit pas modifier la tournée d'entrée."""
        nb_arcs_avant = len(tournee_simple.arcs)
        self.formatter.format(tournee_simple)
        assert len(tournee_simple.arcs) == nb_arcs_avant

    def test_format_batch_plusieurs_tournees(self, tournee_simple):
        """format_batch() doit concaténer les DataFrames de plusieurs tournées."""
        result = self.formatter.format_batch([tournee_simple, tournee_simple])
        assert len(result) == 4  # 2 arcs TRAIN × 2 tournées

    def test_format_batch_liste_vide(self):
        result = self.formatter.format_batch([])
        assert result.empty

    def test_format_deux_tournees_ids_differents(self, tournee_simple):
        """Deux tournées doivent avoir des tournee_id distincts."""
        result = self.formatter.format_batch([tournee_simple, tournee_simple])
        assert result["tournee_id"].nunique() == 2


# ─────────────────────────────────────────────────────────────────────────────
# Tests save_tournee_to_json
# ─────────────────────────────────────────────────────────────────────────────

class TestSaveTourneeToJson:

    def test_cree_fichier_json(self, optimize_response, tmp_path):
        path = save_tournee_to_json(optimize_response, output_dir=tmp_path)
        assert path.exists()
        assert path.suffix == ".json"

    def test_nom_fichier_contient_date(self, optimize_response, tmp_path):
        path = save_tournee_to_json(optimize_response, output_dir=tmp_path)
        assert "2024-09-02" in path.name

    def test_json_valide_et_parseable(self, optimize_response, tmp_path):
        path = save_tournee_to_json(optimize_response, output_dir=tmp_path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        assert "tournee" in raw
        assert "request" in raw

    def test_json_rechargeable_comme_optimize_response(self, optimize_response, tmp_path):
        path = save_tournee_to_json(optimize_response, output_dir=tmp_path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        reloaded = OptimizeResponseSchema.model_validate(raw)
        assert reloaded.tournee.score_total == pytest.approx(optimize_response.tournee.score_total)
        assert reloaded.tournee.nb_trains == optimize_response.tournee.nb_trains


# ─────────────────────────────────────────────────────────────────────────────
# Tests TourneeExporter (intégration légère)
# ─────────────────────────────────────────────────────────────────────────────

class TestTourneeExporter:

    @pytest.fixture
    def exporter_tmp(self, tmp_path, monkeypatch):
        """Exporter avec config pointant vers un dossier temporaire."""
        monkeypatch.setattr(
            "services.exporter.export.config.OUTPUTS_DIR", tmp_path
        )
        exporter = TourneeExporter()
        exporter.outputs_dir = tmp_path
        return exporter, tmp_path

    def test_run_sans_fichier_retourne_none(self, exporter_tmp):
        exporter, tmp_path = exporter_tmp
        result = exporter.run(service_date=date(2024, 9, 2))
        assert result is None

    def test_run_avec_fichier_json_cree_csv(
        self, exporter_tmp, optimize_response
    ):
        exporter, tmp_path = exporter_tmp

        # Sauvegarder une tournée JSON dans le dossier temporaire
        save_tournee_to_json(optimize_response, output_dir=tmp_path)

        # Lancer l'export
        csv_path = exporter.run(service_date=date(2024, 9, 2))

        assert csv_path is not None
        assert csv_path.exists()
        assert csv_path.suffix == ".csv"

    def test_csv_contient_bonnes_colonnes(self, exporter_tmp, optimize_response):
        exporter, tmp_path = exporter_tmp
        save_tournee_to_json(optimize_response, output_dir=tmp_path)

        csv_path = exporter.run(service_date=date(2024, 9, 2))
        df = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig")

        assert "tournee_id" in df.columns
        assert "fraud_score" in df.columns
        assert "priorite" in df.columns

    def test_csv_encodage_utf8_bom(self, exporter_tmp, optimize_response):
        """Le CSV doit être encodé UTF-8 avec BOM pour compatibilité Excel."""
        exporter, tmp_path = exporter_tmp
        save_tournee_to_json(optimize_response, output_dir=tmp_path)

        csv_path = exporter.run(service_date=date(2024, 9, 2))
        raw_bytes = csv_path.read_bytes()
        # UTF-8 BOM = EF BB BF
        assert raw_bytes[:3] == b"\xef\xbb\xbf"

    def test_csv_separateur_point_virgule(self, exporter_tmp, optimize_response):
        """Le séparateur doit être ; (convention française / Power Automate)."""
        exporter, tmp_path = exporter_tmp
        save_tunnee = save_tournee_to_json(optimize_response, output_dir=tmp_path)

        csv_path = exporter.run(service_date=date(2024, 9, 2))
        first_line = csv_path.read_text(encoding="utf-8-sig").split("\n")[0]
        assert ";" in first_line

    def test_csv_nom_horodate(self, exporter_tmp, optimize_response):
        """Le nom du CSV doit contenir un horodatage (fichier unique à chaque export)."""
        exporter, tmp_path = exporter_tmp
        save_tournee_to_json(optimize_response, output_dir=tmp_path)

        csv_path = exporter.run(service_date=date(2024, 9, 2))
        assert "tournees_export_" in csv_path.name

    def test_run_export_all(self, exporter_tmp, optimize_response):
        """--all doit exporter toutes les tournées disponibles."""
        exporter, tmp_path = exporter_tmp

        # Sauvegarder deux tournées
        save_tournee_to_json(optimize_response, output_dir=tmp_path)
        save_tournee_to_json(optimize_response, output_dir=tmp_path)

        csv_path = exporter.run(export_all=True)
        assert csv_path is not None

        df = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig")
        # 2 tournées × 2 arcs TRAIN = 4 lignes minimum
        assert len(df) >= 4

    def test_json_invalide_ignore_sans_crash(self, exporter_tmp):
        """Un fichier JSON invalide doit être ignoré sans crasher l'export."""
        exporter, tmp_path = exporter_tmp

        # Créer un fichier JSON invalide
        bad_file = tmp_path / "tournee_2024-09-02_invalid.json"
        bad_file.write_text("{invalid json", encoding="utf-8")

        # L'exporter doit retourner None (aucune tournée valide), pas crasher
        result = exporter.run(service_date=date(2024, 9, 2))
        assert result is None