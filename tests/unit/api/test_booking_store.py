# tests/unit/api/test_booking_store_v3.py
"""
Tests du cycle de vie complet des tournées dans TripBookingStore v3.

Couvre :
    register_tournee()   — enregistrement EN_ATTENTE (pas de booking)
    validate_tournee()   — booking effectif + passage VALIDEE
    refuse_tournee()     — refus sans booking
    cancel_tournee()     — annulation + libération des trains
    is_conflict()        — uniquement sur trains VALIDÉS
    get_tournee()        — consultation par ID
    get_tournees_en_attente() — dashboard N+1
    Persistance v2.0     — format JSON avec bookings + tournees
"""
from __future__ import annotations

import json
from datetime import date, timedelta, datetime
from pathlib import Path

import pytest

from services.api.booking_store import (
    STATUT_ANNULEE,
    STATUT_EN_ATTENTE,
    STATUT_REFUSEE,
    STATUT_VALIDEE,
    TripBookingStore,
)

TODAY = date.today()


@pytest.fixture
def store(tmp_path) -> TripBookingStore:
    return TripBookingStore(persist_path=tmp_path / "bookings.json")


# ─────────────────────────────────────────────────────────────────────────────
# register_tournee
# ─────────────────────────────────────────────────────────────────────────────

class TestRegisterTournee:

    def test_register_cree_record_en_attente(self, store):
        store.register_tournee("TRN_001", "LAF_042", TODAY, ["T1", "T2"])
        record = store.get_tournee("TRN_001")
        assert record is not None
        assert record["statut"]    == STATUT_EN_ATTENTE
        assert record["agent_id"]  == "LAF_042"
        assert record["trip_ids"]  == ["T1", "T2"]

    def test_register_ne_book_pas_les_trains(self, store):
        """Les trains NE SONT PAS réservés à la génération."""
        store.register_tournee("TRN_001", "LAF_042", TODAY, ["T1", "T2"])
        booked = store.get_all_for_date(TODAY)
        assert "T1" not in booked
        assert "T2" not in booked

    def test_register_pas_de_conflit_sur_meme_trains_en_attente(self, store):
        """Deux tournées en attente peuvent avoir les mêmes trains — pas de conflit."""
        store.register_tournee("TRN_001", "LAF_042", TODAY, ["T1"])
        store.register_tournee("TRN_002", "LAF_043", TODAY, ["T1"])
        # Les deux sont EN_ATTENTE, aucun conflit car booking pas effectif
        conflicts = store.is_conflict(["T1"], TODAY)
        assert conflicts == []

    def test_register_plusieurs_tournees_meme_agent(self, store):
        store.register_tournee("TRN_001", "LAF_042", TODAY, ["T1"])
        store.register_tournee("TRN_002", "LAF_042", TODAY, ["T2"])
        records = store.get_tournees_by_agent("LAF_042", TODAY)
        assert len(records) == 2


# ─────────────────────────────────────────────────────────────────────────────
# validate_tournee
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateTournee:

    def test_validate_change_statut_a_validee(self, store):
        store.register_tournee("TRN_001", "LAF_042", TODAY, ["T1", "T2"])
        record = store.validate_tournee("TRN_001", "MANAGER_007")
        assert record["statut"]       == STATUT_VALIDEE
        assert record["validated_by"] == "MANAGER_007"
        assert record["validated_at"] is not None

    def test_validate_effectue_le_booking(self, store):
        """Le booking devient effectif à la validation, pas avant."""
        store.register_tournee("TRN_001", "LAF_042", TODAY, ["T1", "T2"])
        store.validate_tournee("TRN_001", "MANAGER_007")
        booked = store.get_all_for_date(TODAY)
        assert booked.get("T1") == "LAF_042"
        assert booked.get("T2") == "LAF_042"

    def test_validate_puis_conflit_sur_meme_train(self, store):
        """Après validation de TRN_001, TRN_002 ne peut plus valider T1."""
        store.register_tournee("TRN_001", "LAF_042", TODAY, ["T1"])
        store.register_tournee("TRN_002", "LAF_043", TODAY, ["T1"])
        store.validate_tournee("TRN_001", "MANAGER_007")

        with pytest.raises(ValueError, match="Conflit"):
            store.validate_tournee("TRN_002", "MANAGER_007")

    def test_validate_tournee_inconnue_leve_keyerror(self, store):
        with pytest.raises(KeyError, match="introuvable"):
            store.validate_tournee("TRN_INEXISTANT", "MANAGER_007")

    def test_validate_tournee_deja_validee_leve_valueerror(self, store):
        store.register_tournee("TRN_001", "LAF_042", TODAY, ["T1"])
        store.validate_tournee("TRN_001", "MANAGER_007")
        with pytest.raises(ValueError, match="statut"):
            store.validate_tournee("TRN_001", "MANAGER_007")

    def test_validate_meme_agent_pas_de_conflit_sur_propres_trains(self, store):
        """Un agent peut valider une deuxième tournée avec des trains différents."""
        store.register_tournee("TRN_001", "LAF_042", TODAY, ["T1"])
        store.register_tournee("TRN_002", "LAF_042", TODAY, ["T2"])
        store.validate_tournee("TRN_001", "MANAGER_007")
        record = store.validate_tournee("TRN_002", "MANAGER_007")
        assert record["statut"] == STATUT_VALIDEE

    def test_validate_max_agents_4_autorise_meme_train(self, store):
        """Avec max_agents_per_train=4, deux agents peuvent valider le même train."""
        store.register_tournee("TRN_001", "LAF_042", TODAY, ["T1"])
        store.register_tournee("TRN_002", "LAF_043", TODAY, ["T1"])
        store.validate_tournee("TRN_001", "MANAGER_007")
        # max=4 : pas de conflit → validation autorisée
        record = store.validate_tournee("TRN_002", "MANAGER_007", max_agents_per_train=4)
        assert record["statut"] == STATUT_VALIDEE


# ─────────────────────────────────────────────────────────────────────────────
# refuse_tournee
# ─────────────────────────────────────────────────────────────────────────────

class TestRefuseTournee:

    def test_refuse_change_statut_a_refusee(self, store):
        store.register_tournee("TRN_001", "LAF_042", TODAY, ["T1"])
        record = store.refuse_tournee("TRN_001", "MANAGER_007", motif="Chevauchement")
        assert record["statut"]      == STATUT_REFUSEE
        assert record["refused_by"]  == "MANAGER_007"
        assert record["motif_refus"] == "Chevauchement"

    def test_refuse_ne_book_pas_les_trains(self, store):
        store.register_tournee("TRN_001", "LAF_042", TODAY, ["T1"])
        store.refuse_tournee("TRN_001", "MANAGER_007")
        assert store.get_all_for_date(TODAY) == {}

    def test_refuse_libere_les_trains_pour_autres_agents(self, store):
        """Après refus, un autre agent peut récupérer les mêmes trains."""
        store.register_tournee("TRN_001", "LAF_042", TODAY, ["T1"])
        store.refuse_tournee("TRN_001", "MANAGER_007")
        store.register_tournee("TRN_002", "LAF_043", TODAY, ["T1"])
        record = store.validate_tournee("TRN_002", "MANAGER_007")
        assert record["statut"] == STATUT_VALIDEE

    def test_refuse_tournee_inconnue_leve_keyerror(self, store):
        with pytest.raises(KeyError):
            store.refuse_tournee("TRN_INEXISTANT", "MANAGER_007")

    def test_refuse_tournee_deja_validee_leve_valueerror(self, store):
        store.register_tournee("TRN_001", "LAF_042", TODAY, ["T1"])
        store.validate_tournee("TRN_001", "MANAGER_007")
        with pytest.raises(ValueError, match="statut"):
            store.refuse_tournee("TRN_001", "MANAGER_007")


# ─────────────────────────────────────────────────────────────────────────────
# cancel_tournee
# ─────────────────────────────────────────────────────────────────────────────

class TestCancelTournee:

    def test_cancel_change_statut_a_annulee(self, store):
        store.register_tournee("TRN_001", "LAF_042", TODAY, ["T1"])
        store.validate_tournee("TRN_001", "MANAGER_007")
        record = store.cancel_tournee("TRN_001", "MANAGER_007")
        assert record["statut"]       == STATUT_ANNULEE
        assert record["cancelled_by"] == "MANAGER_007"

    def test_cancel_libere_les_trains(self, store):
        store.register_tournee("TRN_001", "LAF_042", TODAY, ["T1", "T2"])
        store.validate_tournee("TRN_001", "MANAGER_007")
        store.cancel_tournee("TRN_001", "MANAGER_007")
        booked = store.get_all_for_date(TODAY)
        assert "T1" not in booked
        assert "T2" not in booked

    def test_cancel_permet_revalidation_par_autre_agent(self, store):
        store.register_tournee("TRN_001", "LAF_042", TODAY, ["T1"])
        store.validate_tournee("TRN_001", "MANAGER_007")
        store.cancel_tournee("TRN_001", "MANAGER_007")
        # Maintenant LAF_043 peut valider une tournée avec T1
        store.register_tournee("TRN_002", "LAF_043", TODAY, ["T1"])
        record = store.validate_tournee("TRN_002", "MANAGER_007")
        assert record["statut"] == STATUT_VALIDEE

    def test_cancel_tournee_en_attente_leve_valueerror(self, store):
        store.register_tournee("TRN_001", "LAF_042", TODAY, ["T1"])
        with pytest.raises(ValueError, match="statut"):
            store.cancel_tournee("TRN_001", "MANAGER_007")


# ─────────────────────────────────────────────────────────────────────────────
# get_tournees_en_attente (dashboard N+1)
# ─────────────────────────────────────────────────────────────────────────────

class TestGetTourneesEnAttente:

    def test_retourne_toutes_les_tournees_en_attente(self, store):
        store.register_tournee("TRN_001", "LAF_042", TODAY, ["T1"])
        store.register_tournee("TRN_002", "LAF_043", TODAY, ["T2"])
        records = store.get_tournees_en_attente()
        assert len(records) == 2

    def test_n_inclut_pas_les_tournees_validees(self, store):
        store.register_tournee("TRN_001", "LAF_042", TODAY, ["T1"])
        store.register_tournee("TRN_002", "LAF_043", TODAY, ["T2"])
        store.validate_tournee("TRN_001", "MANAGER_007")
        records = store.get_tournees_en_attente()
        assert len(records) == 1
        assert records[0]["tournee_id"] == "TRN_002"

    def test_filtre_par_date(self, store):
        tomorrow = TODAY + timedelta(days=1)
        store.register_tournee("TRN_001", "LAF_042", TODAY,    ["T1"])
        store.register_tournee("TRN_002", "LAF_042", tomorrow, ["T2"])
        records_today    = store.get_tournees_en_attente(TODAY)
        records_tomorrow = store.get_tournees_en_attente(tomorrow)
        assert len(records_today)    == 1
        assert len(records_tomorrow) == 1

    def test_retourne_liste_vide_si_aucune_en_attente(self, store):
        assert store.get_tournees_en_attente() == []


# ─────────────────────────────────────────────────────────────────────────────
# Persistance v2.0
# ─────────────────────────────────────────────────────────────────────────────

class TestPersistanceV2:

    def test_persistance_sauvegarde_tournees(self, tmp_path):
        path  = tmp_path / "bookings.json"
        store = TripBookingStore(persist_path=path)
        store.register_tournee("TRN_001", "LAF_042", TODAY, ["T1"])
        store.validate_tournee("TRN_001", "MANAGER_007")

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        assert data["version"]  == "2.0"
        assert "tournees"       in data
        assert "TRN_001"        in data["tournees"]
        assert data["tournees"]["TRN_001"]["statut"] == STATUT_VALIDEE

    def test_rechargement_restaure_tournees_et_bookings(self, tmp_path):
        path   = tmp_path / "bookings.json"
        store1 = TripBookingStore(persist_path=path)
        store1.register_tournee("TRN_001", "LAF_042", TODAY, ["T1"])
        store1.validate_tournee("TRN_001", "MANAGER_007")

        # Recharger depuis le disque
        store2 = TripBookingStore(persist_path=path)
        assert store2.get_tournee("TRN_001")["statut"] == STATUT_VALIDEE
        assert store2.get_all_for_date(TODAY).get("T1") == "LAF_042"

    def test_migration_depuis_format_v1(self, tmp_path):
        """Un fichier v1 (sans 'tournees') doit charger sans erreur."""
        path = tmp_path / "bookings.json"
        with open(path, "w") as f:
            json.dump({
                "version":    "1.0",
                "updated_at": "2026-01-01T00:00:00",
                "bookings":   {TODAY.isoformat(): {"T1": "LAF_042"}},
            }, f)

        store = TripBookingStore(persist_path=path)
        booked = store.get_all_for_date(TODAY)
        assert booked.get("T1") == "LAF_042"
        assert store.get_tournees_en_attente() == []  # pas de tournees en v1

# ─────────────────────────────────────────────────────────────────────────────
# clean_old_dates
# ─────────────────────────────────────────────────────────────────────────────
class TestCleanOldDates:
    def test_clean_old_dates_preserve_en_attente(self, store):
        """_clean_old_dates ne doit PAS supprimer une tournée EN_ATTENTE expirée."""
        past_date = date.today() - timedelta(days=30)  # bien au-delà du TTL de 14j
        tournee_id = "TRN_TEST_PASSEE"
        store._tournees[tournee_id] = {
            "tournee_id":   tournee_id,
            "agent_id":     "LAF_042",
            "service_date": past_date.isoformat(),
            "trip_ids":     ["TRAIN_X"],
            "statut":       STATUT_EN_ATTENTE,
            "created_at":   datetime.now().isoformat(),
            "validated_by": None, "validated_at": None,
            "refused_by":   None, "refused_at":   None,
        }
        store._clean_old_dates()
        assert tournee_id in store._tournees, (
            "Une tournée EN_ATTENTE ne doit jamais être purgée par le TTL"
        )

    def test_clean_old_dates_purge_validee_expiree(self, store):
        """_clean_old_dates DOIT supprimer une tournée VALIDEE expirée."""
        past_date = date.today() - timedelta(days=30)
        tournee_id = "TRN_TEST_VALIDEE_PASSEE"
        store._tournees[tournee_id] = {
            "tournee_id":   tournee_id,
            "agent_id":     "LAF_042",
            "service_date": past_date.isoformat(),
            "trip_ids":     ["TRAIN_Y"],
            "statut":       STATUT_VALIDEE,
            "created_at":   datetime.now().isoformat(),
            "validated_by": "MGR_001", "validated_at": datetime.now().isoformat(),
            "refused_by":   None, "refused_at":   None,
        }
        store._clean_old_dates()
        assert tournee_id not in store._tournees, (
            "Une tournée VALIDEE expirée doit être purgée par le TTL"
        )