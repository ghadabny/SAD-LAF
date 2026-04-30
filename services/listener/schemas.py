# services/listener/schemas.py
"""
Schémas Pydantic pour la validation des fichiers JSON entrants du listener.

Deux familles de fichiers :
    - TourneeRequestFileSchema  : fichiers déposés dans data/requests/
                                  par Power Automate pour déclencher une génération.
    - DecisionFileSchema        : fichiers déposés dans data/decisions/
                                  par Power Automate pour valider/refuser/annuler.

Ces schémas sont volontairement distincts de ceux de shared/schemas.py :
    - Ils valident les données *brutes* telles qu'elles arrivent du filesystem.
    - Ils sont convertis en TourneeRequestV2Schema / ValidateTourneeRequest
      par le listener avant d'appeler la logique métier.
"""
from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ─────────────────────────────────────────────────────────────────────────────
# Demande de génération de tournée
# ─────────────────────────────────────────────────────────────────────────────

class TourneeRequestFileSchema(BaseModel):
    """
    Schéma du fichier JSON déposé dans data/requests/ par Power Automate.

    Chaque champ correspond exactement aux colonnes du calendrier opérationnel
    (fichier calendrier_tournee_IA_LAF) et aux paramètres de TourneeRequestV2Schema.

    Exemple de fichier : req_AGENT_001_20260514T083000.json
    """
    agent_id:             str  = Field(description="Identifiant de l'agent (ex: AGENT_001)")
    gare_depart_id:       str  = Field(description="Code UIC 8 chiffres de la gare de départ")
    heure_ps_min:         int  = Field(ge=0, le=1439, description="Prise de service en minutes depuis minuit")
    heure_fs_min:         int  = Field(ge=0, le=1439, description="Fin de service en minutes depuis minuit")
    service_date:         date = Field(description="Date du service (YYYY-MM-DD)")
    mode:                 Literal["aller_retour", "decouche"] = Field(
        default="aller_retour",
        description="aller_retour = retour gare départ ; decouche = pas de contrainte retour",
    )
    show_scores:          bool = Field(default=False, description="Inclure fraud_score dans le CSV")
    max_agents_per_train: int  = Field(
        default=1,
        ge=0,
        description="Nombre max d'agents autorisés par train (1=normal, 4=heure de pointe)",
    )
    gare_arrivee_id: Optional[str] = Field(
        default=None,
        description="Code UIC 8 chiffres de la gare d'arrivée (None = gare de départ en aller_retour)",
    )

    @field_validator("gare_depart_id")
    @classmethod
    def valider_gare_depart(cls, v: str) -> str:
        """Valide que l'identifiant de gare est un code UIC à 8 chiffres."""
        if not v.isdigit() or len(v) != 8:
            raise ValueError(
                f"gare_depart_id invalide : '{v}'. "
                "Attendu : 8 chiffres (code UIC SNCF, ex: 87212027)."
            )
        return v

    @field_validator("gare_arrivee_id", mode="before")
    @classmethod
    def valider_gare_arrivee(cls, v: Optional[str]) -> Optional[str]:
        """Valide optionnellement l'identifiant de gare d'arrivée."""
        if v is not None and (not v.isdigit() or len(v) != 8):
            raise ValueError(
                f"gare_arrivee_id invalide : '{v}'. "
                "Attendu : 8 chiffres ou null."
            )
        return v

    @field_validator("heure_fs_min")
    @classmethod
    def valider_plage_horaire(cls, v: int, info) -> int:
        """Valide que la fin de service est postérieure à la prise de service."""
        # info.data contient les champs déjà validés
        ps = info.data.get("heure_ps_min")
        if ps is not None and v <= ps:
            raise ValueError(
                f"heure_fs_min ({v}) doit être strictement supérieure "
                f"à heure_ps_min ({ps})."
            )
        return v


# ─────────────────────────────────────────────────────────────────────────────
# Décision manager (validate / refuse / cancel)
# ─────────────────────────────────────────────────────────────────────────────

# Actions reconnues par le listener
_ACTIONS_VALIDES = {"VALIDATE", "REFUSE", "CANCEL"}


class DecisionFileSchema(BaseModel):
    """
    Schéma du fichier JSON déposé dans data/decisions/ par Power Automate.

    Représente une décision d'un manager (N+1) sur une tournée existante.
    Le champ `action` est normalisé en majuscules et validé contre la liste
    des actions reconnues.

    Exemple de fichier : dec_VALIDATE_TRN_AGENT_001_20260514_3A7F_20260513T141500.json
    """
    action:     str            = Field(description="Action : VALIDATE, REFUSE ou CANCEL")
    tournee_id: str            = Field(description="Identifiant de la tournée (format TRN_…)")
    acteur_id:  str            = Field(description="Identifiant du manager effectuant la décision")
    motif:      Optional[str]  = Field(
        default=None,
        description="Motif textuel (obligatoire pour REFUSE, optionnel pour CANCEL)",
    )

    @field_validator("action", mode="before")
    @classmethod
    def normaliser_et_valider_action(cls, v: str) -> str:
        """
        Normalise l'action en majuscules et vérifie qu'elle est reconnue.

        Exemples valides : "VALIDATE", "validate", "Refuse"
        Exemples invalides : "APPROVE", "DELETE", ""
        """
        normalise = str(v).strip().upper()
        if normalise not in _ACTIONS_VALIDES:
            raise ValueError(
                f"Action inconnue : '{v}'. "
                f"Valeurs acceptées : {', '.join(sorted(_ACTIONS_VALIDES))}."
            )
        return normalise

    @field_validator("tournee_id")
    @classmethod
    def valider_tournee_id(cls, v: str) -> str:
        """Vérifie que l'identifiant de tournée n'est pas vide."""
        if not v or not v.strip():
            raise ValueError("tournee_id ne peut pas être vide.")
        return v.strip()

    @field_validator("acteur_id")
    @classmethod
    def valider_acteur_id(cls, v: str) -> str:
        """Vérifie que l'identifiant de l'acteur n'est pas vide."""
        if not v or not v.strip():
            raise ValueError("acteur_id ne peut pas être vide.")
        return v.strip()
