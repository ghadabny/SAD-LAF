# tests/unit/test_listener.py
"""
Tests unitaires du listener event-driven SAD-LAF.

Couverture :
    test_process_request_valid          — fichier valide → tournée générée → processed/
    test_process_request_invalid_json   — JSON invalide → fichier dans errors/
    test_process_request_schema_error   — champs manquants → fichier dans errors/
    test_process_request_atomic         — deux fichiers simultanés → séquentiel sans doublon
    test_process_decision_validate      — VALIDATE → store.validate_tournee() appelé
    test_process_decision_refuse        — REFUSE avec motif → store.refuse_tournee() appelé
    test_process_decision_cancel        — CANCEL → store.cancel_tournee() appelé
    test_process_decision_invalid_action — action inconnue → fichier dans errors/

Stratégie de mock :
    - Le solver_run et les méthodes du store sont mockés pour isoler le listener.
    - Les vrais dossiers temporaires sont créés avec tmp_path (pytest).
    - FileListener reçoit sa config via monkeypatch sur shared.config.config.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from services.listener.schemas import DecisionFileSchema, TourneeRequestFileSchema


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def dossiers_tmp(tmp_path: Path):
    """
    Crée la structure de dossiers listener dans un répertoire temporaire.

    Retourne un dict avec les chemins utiles pour les tests.
    """
    requests_dir   = tmp_path / "requests"
    decisions_dir  = tmp_path / "decisions"
    for d in (
        requests_dir / "processed",
        requests_dir / "errors",
        decisions_dir / "processed",
        decisions_dir / "errors",
    ):
        d.mkdir(parents=True)
    return {
        "requests":            requests_dir,
        "requests_processed":  requests_dir / "processed",
        "requests_errors":     requests_dir / "errors",
        "decisions":           decisions_dir,
        "decisions_processed": decisions_dir / "processed",
        "decisions_errors":    decisions_dir / "errors",
    }


@pytest.fixture()
def mock_config(monkeypatch, dossiers_tmp):
    """
    Remplace les chemins de config par les dossiers temporaires.
    Remplace aussi les intervalles pour éviter toute dépendance à l'env.
    """
    cfg = MagicMock()
    cfg.REQUESTS_DIR  = dossiers_tmp["requests"]
    cfg.DECISIONS_DIR = dossiers_tmp["decisions"]
    cfg.REQUEST_LISTENER_INTERVAL_SECONDS  = 30
    cfg.DECISION_LISTENER_INTERVAL_SECONDS = 15
    monkeypatch.setattr("services.listener.listener.config", cfg)
    return cfg


@pytest.fixture()
def listener(mock_config):
    """Instancie FileListener avec la config mockée."""
    from services.listener.listener import FileListener
    return FileListener()


def _ecrire_json(dossier: Path, nom: str, contenu: dict) -> Path:
    """Utilitaire : crée un fichier JSON dans le dossier donné."""
    chemin = dossier / nom
    chemin.write_text(json.dumps(contenu, default=str), encoding="utf-8")
    return chemin


REQUETE_VALIDE = {
    "agent_id":             "AGENT_001",
    "gare_depart_id":       "87214056",
    "heure_ps_min":         480,
    "heure_fs_min":         780,
    "service_date":         "2026-05-14",
    "mode":                 "aller_retour",
    "show_scores":          False,
    "max_agents_per_train": 1,
}


# ─────────────────────────────────────────────────────────────────────────────
# Tests — schémas Pydantic
# ─────────────────────────────────────────────────────────────────────────────

class TestTourneeRequestFileSchema:
    """Validation des schémas de fichiers entrants (indépendant du filesystem)."""

    def test_schema_valide(self):
        """Un payload complet et correct est accepté sans erreur."""
        schema = TourneeRequestFileSchema.model_validate(REQUETE_VALIDE)
        assert schema.agent_id == "AGENT_001"
        assert schema.gare_depart_id == "87214056"
        assert schema.heure_ps_min == 480

    def test_gare_depart_invalide(self):
        """Un code UIC non numérique ou de longueur incorrecte est rejeté."""
        from pydantic import ValidationError
        payload = {**REQUETE_VALIDE, "gare_depart_id": "ABC123"}
        with pytest.raises(ValidationError, match="gare_depart_id invalide"):
            TourneeRequestFileSchema.model_validate(payload)

    def test_fs_inferieur_ps(self):
        """heure_fs_min <= heure_ps_min doit lever une ValidationError."""
        from pydantic import ValidationError
        payload = {**REQUETE_VALIDE, "heure_fs_min": 480, "heure_ps_min": 480}
        with pytest.raises(ValidationError):
            TourneeRequestFileSchema.model_validate(payload)

    def test_mode_invalide(self):
        """Un mode non reconnu doit être rejeté par Pydantic."""
        from pydantic import ValidationError
        payload = {**REQUETE_VALIDE, "mode": "aller_simple"}
        with pytest.raises(ValidationError):
            TourneeRequestFileSchema.model_validate(payload)


class TestDecisionFileSchema:
    """Validation des schémas de décision."""

    @pytest.mark.parametrize("action", ["VALIDATE", "validate", "Validate"])
    def test_action_normalisee(self, action: str):
        """L'action est normalisée en majuscules quelle que soit la casse."""
        schema = DecisionFileSchema.model_validate({
            "action":     action,
            "tournee_id": "TRN_AGENT_001_20260514_3A7F",
            "acteur_id":  "MANAGER_007",
        })
        assert schema.action == "VALIDATE"

    @pytest.mark.parametrize("action", ["REFUSE", "CANCEL"])
    def test_actions_valides(self, action: str):
        """REFUSE et CANCEL sont acceptés."""
        schema = DecisionFileSchema.model_validate({
            "action":     action,
            "tournee_id": "TRN_X",
            "acteur_id":  "MGR",
        })
        assert schema.action == action

    def test_action_inconnue(self):
        """Une action non reconnue doit être rejetée."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="Action inconnue"):
            DecisionFileSchema.model_validate({
                "action":     "APPROVE",
                "tournee_id": "TRN_X",
                "acteur_id":  "MGR",
            })

    def test_tournee_id_vide(self):
        """Un tournee_id vide doit être rejeté."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            DecisionFileSchema.model_validate({
                "action":     "VALIDATE",
                "tournee_id": "   ",
                "acteur_id":  "MGR",
            })


# ─────────────────────────────────────────────────────────────────────────────
# Tests — traitement des requêtes
# ─────────────────────────────────────────────────────────────────────────────

class TestProcessPendingRequests:
    """Tests du traitement des fichiers dans data/requests/."""

    def test_process_request_valid(self, listener, dossiers_tmp):
        """
        Fichier valide → generer() appelé → fichier déplacé vers processed/.
        """
        chemin = _ecrire_json(
            dossiers_tmp["requests"],
            "req_AGENT_001_20260514T083000.json",
            REQUETE_VALIDE,
        )

        # 🔧 FIX: Mock 'generer' on the injected TourneeGenerationService
        with patch.object(listener._generation_service, "generer") as mock_gen:
            mock_gen.return_value = "TRN_MOCK"  # Return a mock ID
            listener.process_pending_requests()

        # Le fichier source a disparu de requests/
        assert not chemin.exists()
        # Le fichier est dans processed/
        dest = dossiers_tmp["requests_processed"] / chemin.name
        assert dest.exists()
        # _generer_tournee a été appelé une fois
        mock_gen.assert_called_once()

    def test_process_request_invalid_json(self, listener, dossiers_tmp):
        """
        Fichier avec JSON invalide → fichier déplacé vers errors/ + rapport .error.json.
        """
        chemin = dossiers_tmp["requests"] / "req_invalide.json"
        chemin.write_text("{ pas du json valide !!! }", encoding="utf-8")

        listener.process_pending_requests()

        # Le fichier original est dans errors/
        assert (dossiers_tmp["requests_errors"] / "req_invalide.json").exists()
        # Un rapport d'erreur est présent
        rapport = dossiers_tmp["requests_errors"] / "req_invalide.error.json"
        assert rapport.exists()
        contenu = json.loads(rapport.read_text())
        assert "erreur" in contenu
        assert "timestamp" in contenu

    def test_process_request_schema_error(self, listener, dossiers_tmp):
        """
        Fichier avec champs manquants (gare_depart_id absent) → errors/.
        """
        payload = {k: v for k, v in REQUETE_VALIDE.items() if k != "gare_depart_id"}
        chemin = _ecrire_json(
            dossiers_tmp["requests"], "req_incomplet.json", payload
        )

        listener.process_pending_requests()

        assert not chemin.exists()
        assert (dossiers_tmp["requests_errors"] / "req_incomplet.json").exists()
        assert (dossiers_tmp["requests_errors"] / "req_incomplet.error.json").exists()

    def test_process_request_atomic(self, listener, dossiers_tmp):
        """
        Deux fichiers simultanés → traitement séquentiel, aucun doublon.

        Simulation : on enregistre l'ordre de traitement et on vérifie
        que chaque fichier est traité exactement une fois.
        """
        fichiers_traites = []

        def mock_generer(request):
            fichiers_traites.append(request.agent_id)
            return f"TRN_{request.agent_id}"

        _ecrire_json(dossiers_tmp["requests"], "req_001.json", {
            **REQUETE_VALIDE, "agent_id": "AGENT_001",
        })
        _ecrire_json(dossiers_tmp["requests"], "req_002.json", {
            **REQUETE_VALIDE, "agent_id": "AGENT_002",
        })

        # 🔧 FIX: Mock 'generer' on the injected TourneeGenerationService
        with patch.object(listener._generation_service, "generer", side_effect=mock_generer):
            listener.process_pending_requests()

        # Chaque agent traité exactement une fois
        assert len(fichiers_traites) == 2
        assert "AGENT_001" in fichiers_traites
        assert "AGENT_002" in fichiers_traites
        # Les deux fichiers sont dans processed/
        assert len(list(dossiers_tmp["requests_processed"].iterdir())) == 2
        # Aucun fichier restant dans requests/
        assert not any(
            f.suffix == ".json"
            for f in dossiers_tmp["requests"].iterdir()
            if f.is_file()
        )

    def test_dossier_vide_ne_plante_pas(self, listener, dossiers_tmp):
        """process_pending_requests() sur dossier vide ne lève aucune exception."""
        listener.process_pending_requests()  # doit passer silencieusement


# ─────────────────────────────────────────────────────────────────────────────
# Tests — traitement des décisions
# ─────────────────────────────────────────────────────────────────────────────

class TestProcessPendingDecisions:
    """Tests du traitement des fichiers dans data/decisions/."""

    def _decision(self, action: str, motif: str | None = None) -> dict:
        return {
            "action":     action,
            "tournee_id": "TRN_AGENT_001_20260514_3A7F",
            "acteur_id":  "MANAGER_007",
            "motif":      motif,
        }

    def test_process_decision_validate(self, listener, dossiers_tmp):
        """
        Décision VALIDATE → store.validate_tournee() appelé avec les bons arguments.
        """
        _ecrire_json(
            dossiers_tmp["decisions"],
            "dec_VALIDATE_TRN001.json",
            self._decision("VALIDATE"),
        )

        mock_store = MagicMock()
        mock_store.validate_tournee.return_value = {}

        with patch("services.listener.listener.get_booking_store", return_value=mock_store):
            listener.process_pending_decisions()

        mock_store.validate_tournee.assert_called_once_with(
            tournee_id           = "TRN_AGENT_001_20260514_3A7F",
            validated_by         = "MANAGER_007",
            max_agents_per_train = 1,
        )
        assert (dossiers_tmp["decisions_processed"] / "dec_VALIDATE_TRN001.json").exists()

    def test_process_decision_refuse(self, listener, dossiers_tmp):
        """
        Décision REFUSE avec motif → store.refuse_tournee() appelé avec le motif.
        """
        _ecrire_json(
            dossiers_tmp["decisions"],
            "dec_REFUSE_TRN001.json",
            self._decision("REFUSE", motif="Conflit avec autre tournée"),
        )

        mock_store = MagicMock()
        mock_store.refuse_tournee.return_value = {}

        with patch("services.listener.listener.get_booking_store", return_value=mock_store):
            listener.process_pending_decisions()

        mock_store.refuse_tournee.assert_called_once_with(
            tournee_id = "TRN_AGENT_001_20260514_3A7F",
            refused_by = "MANAGER_007",
            motif      = "Conflit avec autre tournée",
        )

    def test_process_decision_cancel(self, listener, dossiers_tmp):
        """
        Décision CANCEL → store.cancel_tournee() appelé.
        """
        _ecrire_json(
            dossiers_tmp["decisions"],
            "dec_CANCEL_TRN001.json",
            self._decision("CANCEL"),
        )

        mock_store = MagicMock()
        mock_store.cancel_tournee.return_value = {}

        with patch("services.listener.listener.get_booking_store", return_value=mock_store):
            listener.process_pending_decisions()

        mock_store.cancel_tournee.assert_called_once_with(
            tournee_id   = "TRN_AGENT_001_20260514_3A7F",
            cancelled_by = "MANAGER_007",
        )

    def test_process_decision_invalid_action(self, listener, dossiers_tmp):
        """
        Action inconnue dans le JSON → validation Pydantic échoue → errors/.
        """
        _ecrire_json(
            dossiers_tmp["decisions"],
            "dec_APPROVE_TRN001.json",
            {"action": "APPROVE", "tournee_id": "TRN_X", "acteur_id": "MGR"},
        )

        listener.process_pending_decisions()

        assert (dossiers_tmp["decisions_errors"] / "dec_APPROVE_TRN001.json").exists()
        rapport = dossiers_tmp["decisions_errors"] / "dec_APPROVE_TRN001.error.json"
        assert rapport.exists()
        contenu = json.loads(rapport.read_text())
        assert "APPROVE" in contenu["erreur"] or "Action" in contenu["erreur"]

    def test_process_decision_tournee_introuvable(self, listener, dossiers_tmp):
        """
        Store lève KeyError (tournée introuvable) → fichier dans errors/.
        """
        _ecrire_json(
            dossiers_tmp["decisions"],
            "dec_VALIDATE_INTROUVABLE.json",
            self._decision("VALIDATE"),
        )

        mock_store = MagicMock()
        mock_store.validate_tournee.side_effect = KeyError("Tournée inconnue")

        with patch("services.listener.listener.get_booking_store", return_value=mock_store):
            listener.process_pending_decisions()

        assert (
            dossiers_tmp["decisions_errors"] / "dec_VALIDATE_INTROUVABLE.json"
        ).exists()
