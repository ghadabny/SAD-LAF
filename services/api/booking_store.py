"""
services/api/booking_store.py — Store thread-safe des tournées réservées.

CORRECTIONS v3.1 :
    Bug 3 — tournées introuvables / pending vide après validation :
        Cause 1 : Le singleton get_booking_store() utilisait une variable globale
                  initialisée à None, avec un double-checked locking. En mode
                  uvicorn --reload, le module est réimporté → variable remise à None
                  → nouveau store vide sans charger le JSON persisté.
        Cause 2 : Deux requêtes successives pouvaient voir deux instances distinctes
                  du store si le module était rechargé entre-temps.
        Fix     : Le singleton est maintenant initialisé au niveau module (import time)
                  via _BOOKING_STORE = TripBookingStore(). Cette instance est créée
                  une seule fois par processus Python. La persistance JSON garantit
                  que les données survivent aux redémarrages complets.

        Cause 3 : Le message d'erreur "introuvable" apparaissait car le store chargé
                  au démarrage ne trouvait pas le fichier bookings.json (chemin incorrect
                  ou permissions). Ajout de logs explicites pour diagnostiquer.
"""
from __future__ import annotations

import json
import threading
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

from shared.config import config

# ── Statuts ───────────────────────────────────────────────────────────────────
STATUT_EN_ATTENTE = "EN_ATTENTE_VALIDATION"
STATUT_VALIDEE    = "VALIDEE"
STATUT_REFUSEE    = "REFUSEE"
STATUT_ANNULEE    = "ANNULEE"


class TripBookingStore:
    """
    Store thread-safe des tournées et réservations de trains.

    Structure interne :
        _store    : date_iso → {trip_id: agent_id}  (trains VALIDÉS)
        _tournees : tournee_id → record              (toutes les tournées)

    Persistance :
        Toute modification est immédiatement écrite dans bookings.json
        via écriture atomique (write .tmp → rename).
        Au démarrage, le fichier est rechargé automatiquement.
    """

    TTL_DAYS: int = 14

    def __init__(self, persist_path: Optional[Path] = None):
        self._lock     = threading.Lock()
        self._path     = persist_path or (config.OUTPUTS_DIR / "bookings.json")
        self._store:    dict[str, dict[str, str]] = {}
        self._tournees: dict[str, dict]           = {}
        self._load()

    # ── Anti-doublon ──────────────────────────────────────────────────────────

    def is_conflict(
            self,
            trip_ids: list[str],
            service_date: date,
            exclude_agent_id: Optional[str] = None,
            max_agents_per_train: int = 1,
    ) -> list[tuple[str, str]]:
        """
        Retourne la liste des (trip_id, agent_id) en conflit.

        Règles :
            max_agents_per_train <= 0  → ValueError (valeur invalide)
            max_agents_per_train >= 4  → mode "pas de contrainte", retourne []
            sinon                      → vérifie les trains déjà bookés
        """
        if max_agents_per_train <= 0:
            raise ValueError(
                f"[BookingStore] max_agents_per_train doit être >= 1, "
                f"reçu : {max_agents_per_train}"
            )
        if max_agents_per_train >= 4:
            return []  # mode sans contrainte (ex: formation, test)

        key = service_date.isoformat()
        with self._lock:
            booked = self._store.get(key, {})
            return [
                (tid, booked[tid])
                for tid in trip_ids
                if tid in booked
                   and (exclude_agent_id is None or booked[tid] != exclude_agent_id)
            ]

    # ── Cycle de vie ──────────────────────────────────────────────────────────

    def register_tournee(
        self,
        tournee_id: str,
        agent_id: str,
        service_date: date,
        trip_ids: list[str],
        auto_validate: bool = True,
    ) -> None:
        """
        Enregistre une nouvelle tournée.

        auto_validate=True (défaut) : la tournée est immédiatement VALIDEE
        et les trains sont réservés. C'est le comportement normal lors d'une
        génération initiale (décision métier : pas de validation manager requise).

        auto_validate=False : la tournée passe EN_ATTENTE_VALIDATION.
        Utilisé uniquement pour les demandes de modification d'une tournée
        existante qui nécessitent une approbation N+1.
        """
        statut = STATUT_VALIDEE if auto_validate else STATUT_EN_ATTENTE
        now    = datetime.now().isoformat()
        record = {
            "tournee_id":   tournee_id,
            "agent_id":     agent_id,
            "service_date": service_date.isoformat(),
            "trip_ids":     trip_ids,
            "statut":       statut,
            "created_at":   now,
            "validated_by": agent_id if auto_validate else None,
            "validated_at": now       if auto_validate else None,
            "refused_by":   None,
            "refused_at":   None,
        }
        with self._lock:
            self._tournees[tournee_id] = record
            if auto_validate:
                day_key   = service_date.isoformat()
                day_store = self._store.setdefault(day_key, {})
                for tid in trip_ids:
                    day_store[tid] = agent_id
            self._persist()
        label = "✅ validée automatiquement" if auto_validate else "📋 en attente"
        print(
            f"[BookingStore] {label} : {tournee_id} "
            f"(agent={agent_id}, {len(trip_ids)} trains)"
        )

    def validate_tournee(
        self,
        tournee_id: str,
        validated_by: str,
        max_agents_per_train: int = 1,
    ) -> dict:
        with self._lock:
            if tournee_id not in self._tournees:
                raise KeyError(
                    f"Tournée '{tournee_id}' introuvable. "
                    "Vérifiez l'identifiant ou régénérez une tournée."
                )
            record = self._tournees[tournee_id]

            if record["statut"] != STATUT_EN_ATTENTE:
                raise ValueError(
                    f"Impossible de valider : la tournée '{tournee_id}' "
                    f"est en statut '{record['statut']}' (attendu: {STATUT_EN_ATTENTE})."
                )

            service_date = date.fromisoformat(record["service_date"])
            trip_ids     = record["trip_ids"]
            agent_id     = record["agent_id"]
            day_key      = service_date.isoformat()
            booked       = self._store.get(day_key, {})

            conflicts: list[tuple[str, str]] = []
            if max_agents_per_train < 4:
                conflicts = [
                    (tid, booked[tid])
                    for tid in trip_ids
                    if tid in booked and booked[tid] != agent_id
                ]

            if conflicts:
                trains  = [c[0] for c in conflicts]
                agents  = list({c[1] for c in conflicts})
                preview = trains[:3]
                suffix  = f" et {len(trains)-3} autres" if len(trains) > 3 else ""
                raise ValueError(
                    f"Conflit de validation : {len(trains)} train(s) de la tournée "
                    f"'{tournee_id}' sont déjà validés par {', '.join(agents)}. "
                    f"Trains : {', '.join(preview)}{suffix}."
                )

            day_store = self._store.setdefault(day_key, {})
            for tid in trip_ids:
                day_store[tid] = agent_id

            record["statut"]       = STATUT_VALIDEE
            record["validated_by"] = validated_by
            record["validated_at"] = datetime.now().isoformat()

            self._clean_old_dates()
            self._persist()

        print(f"[BookingStore] ✅ Tournée validée : {tournee_id} par {validated_by}")
        return dict(record)

    def refuse_tournee(
        self,
        tournee_id: str,
        refused_by: str,
        motif: Optional[str] = None,
    ) -> dict:
        with self._lock:
            if tournee_id not in self._tournees:
                raise KeyError(f"Tournée '{tournee_id}' introuvable.")
            record = self._tournees[tournee_id]
            if record["statut"] != STATUT_EN_ATTENTE:
                raise ValueError(
                    f"Impossible de refuser : statut '{record['statut']}' "
                    f"(attendu: {STATUT_EN_ATTENTE})."
                )
            record["statut"]     = STATUT_REFUSEE
            record["refused_by"] = refused_by
            record["refused_at"] = datetime.now().isoformat()
            if motif:
                record["motif_refus"] = motif
            self._persist()

        print(f"[BookingStore] ❌ Tournée refusée : {tournee_id} par {refused_by}")
        return dict(record)

    def select_tournee(
            self,
            tournee_id: str,
            agent_id: str,
            max_agents_per_train: int = 1,
    ) -> dict:
        """
        L'agent confirme sa tournée : EN_ATTENTE → VALIDEE, trains réservés.

        Alias sémantique de validate_tournee() pour l'action 'Sélectionner'
        côté agent (vs. 'Valider' côté manager N+1).
        """
        return self.validate_tournee(
            tournee_id=tournee_id,
            validated_by=agent_id,
            max_agents_per_train=max_agents_per_train,
        )

    def get_trip_ids_for_tournee(self, tournee_id: str) -> list[str]:
        """Retourne les trip_ids d'une tournée (pour exclusion lors d'une régénération)."""
        with self._lock:
            record = self._tournees.get(tournee_id)
            return list(record["trip_ids"]) if record else []

    def cancel_tournee(self, tournee_id: str, cancelled_by: str) -> dict:
        with self._lock:
            if tournee_id not in self._tournees:
                raise KeyError(f"Tournée '{tournee_id}' introuvable.")
            record = self._tournees[tournee_id]
            if record["statut"] != STATUT_VALIDEE:
                raise ValueError(
                    f"Impossible d'annuler : statut '{record['statut']}' "
                    f"(attendu: {STATUT_VALIDEE})."
                )
            day_key = record["service_date"]
            booked  = self._store.get(day_key, {})
            for tid in record["trip_ids"]:
                booked.pop(tid, None)

            record["statut"]       = STATUT_ANNULEE
            record["cancelled_by"] = cancelled_by
            record["cancelled_at"] = datetime.now().isoformat()
            self._persist()

        print(f"[BookingStore] 🔄 Tournée annulée : {tournee_id} par {cancelled_by}")
        return dict(record)

    # ── Consultation ──────────────────────────────────────────────────────────

    def get_tournee(self, tournee_id: str) -> Optional[dict]:
        with self._lock:
            r = self._tournees.get(tournee_id)
            return dict(r) if r else None

    def get_tournees_en_attente(
        self, service_date: Optional[date] = None
    ) -> list[dict]:
        with self._lock:
            records = [
                dict(r) for r in self._tournees.values()
                if r["statut"] == STATUT_EN_ATTENTE
                and (service_date is None or r["service_date"] == service_date.isoformat())
            ]
        return sorted(records, key=lambda r: r["created_at"])

    def get_tournees_by_agent(
        self, agent_id: str, service_date: Optional[date] = None
    ) -> list[dict]:
        with self._lock:
            records = [
                dict(r) for r in self._tournees.values()
                if r["agent_id"] == agent_id
                and (service_date is None or r["service_date"] == service_date.isoformat())
            ]
        return sorted(records, key=lambda r: r["created_at"], reverse=True)

    def get_all_for_date(self, service_date: date) -> dict[str, str]:
        with self._lock:
            return dict(self._store.get(service_date.isoformat(), {}))

    def stats(self) -> dict:
        with self._lock:
            return {
                "nb_trains_valides":      sum(len(v) for v in self._store.values()),
                "nb_tournees_total":      len(self._tournees),
                "nb_tournees_en_attente": sum(
                    1 for r in self._tournees.values()
                    if r["statut"] == STATUT_EN_ATTENTE
                ),
                "persist_path": str(self._path),
            }

    # ── Compatibilité v1 ──────────────────────────────────────────────────────

    def book(self, trip_ids: list[str], service_date: date, agent_id: str) -> None:
        key = service_date.isoformat()
        with self._lock:
            booked = self._store.setdefault(key, {})
            for tid in trip_ids:
                booked[tid] = agent_id
            self._clean_old_dates()
            self._persist()

    def release(
        self, trip_ids: list[str], service_date: date,
        agent_id: Optional[str] = None
    ) -> int:
        key = service_date.isoformat()
        removed = 0
        with self._lock:
            booked = self._store.get(key, {})
            for tid in list(trip_ids):
                if tid in booked and (agent_id is None or booked[tid] == agent_id):
                    del booked[tid]
                    removed += 1
            self._persist()
        return removed

    def clear_date(self, service_date: date) -> int:
        key = service_date.isoformat()
        with self._lock:
            removed = len(self._store.pop(key, {}))
            self._persist()
        return removed

    # ── Persistance ───────────────────────────────────────────────────────────

    def _persist(self) -> None:
        """Écriture atomique : .tmp → rename, évite la corruption en cas d'interruption."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({
                    "version":    "2.0",
                    "updated_at": datetime.now().isoformat(),
                    "bookings":   self._store,
                    "tournees":   self._tournees,
                }, f, ensure_ascii=False, indent=2)
            tmp.replace(self._path)
        except Exception as exc:
            print(f"[BookingStore] ⚠️  Persistance impossible : {exc}")

    def _load(self) -> None:
        """
        Charge le store depuis bookings.json au démarrage.

        FIX Bug 3 : logs explicites pour diagnostiquer les problèmes de chemin.
        """
        try:
            if self._path.exists():
                with open(self._path, encoding="utf-8") as f:
                    data = json.load(f)
                self._store    = data.get("bookings", {})
                self._tournees = data.get("tournees", {})
                self._clean_old_dates()
                nb_trains   = sum(len(v) for v in self._store.values())
                nb_tournees = len(self._tournees)
                nb_attente  = sum(
                    1 for r in self._tournees.values()
                    if r.get("statut") == STATUT_EN_ATTENTE
                )
                print(
                    f"[BookingStore] ✅ Chargé depuis {self._path} — "
                    f"{nb_trains} trains validés, "
                    f"{nb_tournees} tournées ({nb_attente} en attente)."
                )
            else:
                print(
                    f"[BookingStore] ℹ️  Aucun fichier de persistance trouvé "
                    f"({self._path}) — store vide."
                )
        except Exception as exc:
            print(
                f"[BookingStore] ⚠️  Chargement impossible ({exc}) — store vide. "
                f"Chemin : {self._path}"
            )
            self._store    = {}
            self._tournees = {}

    def _clean_old_dates(self) -> None:
        cutoff = (datetime.now().date() - timedelta(days=self.TTL_DAYS)).isoformat()
        stale_dates = [d for d in self._store if d < cutoff]

        # Ne purger QUE les tournées terminales (VALIDEE, REFUSEE, ANNULEE)
        # Les EN_ATTENTE sont toujours actives métier et ne doivent jamais être purgées silencieusement
        STATUTS_PURGABLES = {STATUT_VALIDEE, STATUT_REFUSEE, STATUT_ANNULEE}
        stale_ids = [
            tid for tid, r in self._tournees.items()
            if r.get("service_date", "") < cutoff
               and r.get("statut") in STATUTS_PURGABLES
        ]

        for d in stale_dates: del self._store[d]
        for tid in stale_ids:   del self._tournees[tid]

        if stale_dates or stale_ids:
            print(
                f"[BookingStore] 🗑️  TTL nettoyage : "
                f"{len(stale_dates)} jours, {len(stale_ids)} tournées supprimés."
            )

        # Alerter si des tournées EN_ATTENTE sont expirées (anomalie opérationnelle)
        orphaned = [
            tid for tid, r in self._tournees.items()
            if r.get("service_date", "") < cutoff
               and r.get("statut") == STATUT_EN_ATTENTE
        ]
        if orphaned:
            print(
                f"[BookingStore] ⚠️  {len(orphaned)} tournée(s) EN_ATTENTE expirée(s) "
                f"(service_date < {cutoff}) — non purgées, action manuelle requise : "
                f"{orphaned[:5]}"
            )


# ── Singleton initialisé au niveau module (import time) ───────────────────────
#
# FIX Bug 3 : initialisation au niveau module garantit qu'il y a
# exactement UNE instance par processus Python. En mode uvicorn --reload,
# le processus redémarre complètement → _load() relit bookings.json.
# Pas de risque de double-init car Python met les modules en cache dans sys.modules.
#
_BOOKING_STORE: TripBookingStore = TripBookingStore()


def get_booking_store() -> TripBookingStore:
    """Retourne le singleton du store — créé une seule fois par processus."""
    return _BOOKING_STORE