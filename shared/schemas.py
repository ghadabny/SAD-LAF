"""
shared/schemas.py — Schémas Pydantic partagés entre tous les services.

Nouveautés v3 (cycle de vie validation) :
    ValidateTourneeRequest  : body de POST /optimize/v2/validate
    RefuseTourneeRequest    : body de POST /optimize/v2/refuse
    CancelTourneeRequest    : body de POST /optimize/v2/cancel
    TourneeRecordSchema     : représentation d'une tournée avec son statut
    ValidateTourneeResponse : réponse après validation/refus/annulation
"""

from datetime import date, datetime
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from shared.constants import MISSION_START_OFFSET_MINUTES


# ─────────────────────────────────────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────────────────────────────────────

class ArcTypeSchema(str, Enum):
    TRAIN          = "train"
    CORRESPONDANCE = "correspondance"


class TourneeStatutSchema(str, Enum):
    """Statuts possibles d'une tournée dans son cycle de vie."""
    EN_ATTENTE_VALIDATION = "EN_ATTENTE_VALIDATION"
    VALIDEE               = "VALIDEE"
    REFUSEE               = "REFUSEE"
    ANNULEE               = "ANNULEE"


# ─────────────────────────────────────────────────────────────────────────────
# GTFS — Tronçons
# ─────────────────────────────────────────────────────────────────────────────

class TronconSchema(BaseModel):
    trip_id:       str
    train_number:  str
    service_id:    str
    stop_sequence: int

    stop_id_dep:   str
    stop_name_dep: str
    dep_minutes:   int = Field(ge=0)

    stop_id_arr:   str
    stop_name_arr: str
    arr_minutes:   int = Field(ge=0)

    duration_min:  int = Field(gt=0)

    @field_validator("stop_id_dep", "stop_id_arr")
    @classmethod
    def validate_stop_id(cls, v: str) -> str:
        if not v.isdigit() or len(v) != 8:
            raise ValueError(f"stop_id invalide : '{v}'. Attendu : 8 chiffres (code UIC).")
        return v


class TronconScoreSchema(TronconSchema):
    fraud_score: float = Field(default=0.0, ge=0.0, le=1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Graphe temps-étendu
# ─────────────────────────────────────────────────────────────────────────────

class NodeSchema(BaseModel):
    stop_id:      str
    stop_name:    str
    time_minutes: int  = Field(ge=0)
    service_date: date

    @property
    def label(self) -> str:
        h = self.time_minutes // 60
        m = self.time_minutes % 60
        return f"{self.stop_name} {h:02d}:{m:02d}"

    model_config = {"frozen": True}


class ArcSchema(BaseModel):
    source:       NodeSchema
    destination:  NodeSchema
    arc_type:     ArcTypeSchema
    duration_min: int  = Field(gt=0)
    trip_id:      Optional[str]   = None
    train_number: Optional[str]   = None
    fraud_score:  float = Field(default=0.0, ge=0.0, le=1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Optimisation v1 (conservé pour compatibilité)
# ─────────────────────────────────────────────────────────────────────────────

class TourneeRequestSchema(BaseModel):
    gare_depart_id:    str = Field(description="Code UIC 8 chiffres")
    heure_depart_min:  int = Field(ge=0, le=1439)
    duree_max_minutes: int = Field(default=360, gt=0, le=720)
    service_date:      date

    @field_validator("gare_depart_id")
    @classmethod
    def validate_gare_depart(cls, v: str) -> str:
        if not v.isdigit() or len(v) != 8:
            raise ValueError(f"gare_depart_id invalide : '{v}'.")
        return v


class TourneeSchema(BaseModel):
    arcs:                  list[ArcSchema]
    score_total:           float = Field(ge=0.0)
    duree_totale_minutes:  int   = Field(gt=0)
    nb_trains:             int   = Field(ge=1)
    gare_depart:           NodeSchema
    gare_arrivee:          NodeSchema
    service_date:          date
    generated_at:          datetime = Field(default_factory=datetime.now)

    @property
    def arcs_train(self) -> list[ArcSchema]:
        return [a for a in self.arcs if a.arc_type == ArcTypeSchema.TRAIN]

    @property
    def trains_visites(self) -> list[str]:
        return [a.train_number for a in self.arcs_train if a.train_number is not None]


# ─────────────────────────────────────────────────────────────────────────────
# Optimisation v2 — Requête enrichie
# ─────────────────────────────────────────────────────────────────────────────

class TourneeRequestV2Schema(BaseModel):
    """
    Requête de génération de tournée v2 (PowerApps).

    PS/FS correspondent directement aux colonnes Heure_Origine / Heure_Fin
    du fichier calendrier. Le système dérive heure_depart_min = PS + 10 min.
    """
    gare_depart_id:  str  = Field(description="Code UIC 8 chiffres de la gare de départ")
    heure_ps_min:    int  = Field(ge=0, le=1439, description="Prise de service (minutes depuis minuit)")
    heure_fs_min:    int  = Field(ge=0, le=1439, description="Fin de service (minutes depuis minuit)")
    service_date:    date

    gare_arrivee_id: Optional[str] = Field(
        default=None,
        description=(
            "Code UIC gare d'arrivée. "
            "None = retour gare de départ (aller_retour) ou pas de contrainte (decouche)."
        ),
    )
    agent_id:        Optional[str] = Field(
        default=None,
        description="Identifiant de l'agent — utilisé pour l'anti-doublon et le nommage CSV.",
    )
    mode:            Literal["aller_retour", "decouche"] = Field(
        default="aller_retour",
        description="aller_retour = retour gare départ. decouche = pas de contrainte retour J1.",
    )
    show_scores:     bool = Field(
        default=False,
        description="Si True, inclure fraud_score dans le CSV (usage interne uniquement).",
    )
    max_agents_per_train: int = Field(
        default=1,
        ge=0,
        description=(
            "Nombre max d'agents autorisés sur le même train. "
            "1 = JS normale. 4 = heure de pointe / opération civil. 0 = désactivé."
        ),
    )
    pause_debut_min: Optional[int] = Field(
        default=None, ge=0, le=1439,
        description="Début de la pause agent (minutes depuis minuit). None = pas de pause.",
    )
    pause_fin_min: Optional[int] = Field(
        default=None, ge=0, le=1439,
        description="Fin de la pause agent (minutes depuis minuit). None = pas de pause.",
    )

    @field_validator("gare_depart_id")
    @classmethod
    def validate_gare_depart(cls, v: str) -> str:
        if not v.isdigit() or len(v) != 8:
            raise ValueError(f"gare_depart_id invalide : '{v}'.")
        return v

    @field_validator("gare_arrivee_id")
    @classmethod
    def validate_gare_arrivee(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and (not v.isdigit() or len(v) != 8):
            raise ValueError(f"gare_arrivee_id invalide : '{v}'.")
        return v

    @model_validator(mode="after")
    def validate_ps_fs(self) -> "TourneeRequestV2Schema":
        if self.heure_fs_min <= self.heure_ps_min:
            raise ValueError(
                f"heure_fs_min ({self.heure_fs_min}) doit être > heure_ps_min ({self.heure_ps_min})."
            )
        if self.duree_max_minutes < 30:
            raise ValueError(
                f"La fenêtre PS/FS est trop courte : {self.duree_max_minutes} min "
                f"(minimum 30 min requis pour une tournée valide)."
            )
        if (self.pause_debut_min is None) != (self.pause_fin_min is None):
            raise ValueError(
                "pause_debut_min et pause_fin_min doivent être fournis ensemble."
            )
        if self.pause_debut_min is not None:
            if self.pause_fin_min <= self.pause_debut_min:
                raise ValueError(
                    f"pause_fin_min ({self.pause_fin_min}) doit être > "
                    f"pause_debut_min ({self.pause_debut_min})."
                )
            if self.pause_debut_min < self.heure_depart_min or self.pause_fin_min > self.heure_fs_min:
                raise ValueError(
                    f"La pause [{self.pause_debut_min}, {self.pause_fin_min}] doit être "
                    f"dans la journée de service [{self.heure_depart_min}, {self.heure_fs_min}]."
                )
        return self

    @property
    def heure_depart_min(self) -> int:
        """Heure de début de mission = PS + 10 min (temps de positionnement)."""
        return self.heure_ps_min + MISSION_START_OFFSET_MINUTES

    @property
    def duree_max_minutes(self) -> int:
        """Budget temps = FS − heure_depart_min."""
        return self.heure_fs_min - self.heure_depart_min

    @property
    def gare_arrivee_effective(self) -> Optional[str]:
        """
        Gare d'arrivée réelle selon la logique métier :
            - gare_arrivee_id explicite → respectée strictement
            - mode aller_retour         → retour gare de départ
            - mode decouche             → pas de contrainte (None)
        """
        if self.gare_arrivee_id is not None:
            return self.gare_arrivee_id
        if self.mode == "aller_retour":
            return self.gare_depart_id
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Optimisation v2 — Réponse génération
# ─────────────────────────────────────────────────────────────────────────────

class OptimizeResponseV2Schema(BaseModel):
    """
    Réponse de POST /optimize/v2.

    Changement v3 : la tournée est en EN_ATTENTE_VALIDATION.
    Le booking N'EST PAS encore effectué — il le sera à la validation N+1.
    Le champ tournee_id permet au manager de retrouver et valider la tournée.
    """
    tournee:           TourneeSchema
    request:           TourneeRequestV2Schema
    tournee_id:        str   = Field(description="Identifiant unique de la tournée — à transmettre au N+1 pour validation")
    statut:            str   = Field(default="EN_ATTENTE_VALIDATION", description="Statut initial de la tournée")
    score_perte_pct:   float = Field(default=0.0, ge=0.0)
    trains_en_conflit: list[str]  = Field(default_factory=list)
    csv_path:          Optional[str] = None
    warning_messages:  list[str]  = Field(default_factory=list)
    optimized_at:      datetime   = Field(default_factory=datetime.now)

# ─────────────────────────────────────────────────────────────────────────────
# Selection et Regénération de Tournées
# ─────────────────────────────────────────────────────────────────────────────

class SelectTourneeRequest(BaseModel):
    """Body de POST /optimize/v2/select — l'agent confirme sa tournée."""
    tournee_id:           str = Field(description="ID de la tournée à sélectionner")
    agent_id:             str = Field(description="Identifiant de l'agent qui sélectionne")
    max_agents_per_train: int = Field(
        default=1, ge=1,
        description="Nombre max d'agents autorisés sur le même train.",
    )


class RegenerateTourneeRequest(TourneeRequestV2Schema):
    """
    Body de POST /optimize/v2/regenerate.

    L'agent rejette sa tournée courante et en demande une nouvelle
    avec les mêmes paramètres. La tournée rejetée redevient disponible
    pour d'autres agents.

    Hérite de TourneeRequestV2Schema pour réutiliser tous les paramètres
    de génération (gare, PS/FS, date, mode…).
    """
    tournee_id_rejetee: str           = Field(description="ID de la tournée à rejeter")
    motif:              Optional[str] = Field(
        default=None,
        description="Motif du rejet (optionnel, usage log).",
    )

# ─────────────────────────────────────────────────────────────────────────────
# Validation — Requêtes et Réponses (POST /optimize/v2/validate, /refuse, /cancel)
# ─────────────────────────────────────────────────────────────────────────────

class ValidateTourneeRequest(BaseModel):
    """
    Body de POST /optimize/v2/validate.
    Appelé par le N+1 depuis PowerApps pour valider une tournée.
    C'est à cet appel que le booking des trains devient effectif.
    """
    tournee_id:           str = Field(description="ID de la tournée à valider (fourni dans la réponse de génération)")
    validated_by:         str = Field(description="Identifiant du manager validant (N+1 ou N+2)")
    max_agents_per_train: int = Field(
        default=1,
        ge=0,
        description="1 = normal, 4 = heure de pointe/civil, 0 = désactivé",
    )


class RefuseTourneeRequest(BaseModel):
    """Body de POST /optimize/v2/refuse. Aucun booking effectué."""
    tournee_id: str             = Field(description="ID de la tournée à refuser")
    refused_by: str             = Field(description="Identifiant du manager refusant")
    motif:      Optional[str]   = Field(default=None, description="Motif du refus (optionnel)")


class CancelTourneeRequest(BaseModel):
    """Body de POST /optimize/v2/cancel. Libère les trains d'une tournée validée."""
    tournee_id:   str = Field(description="ID de la tournée validée à annuler")
    cancelled_by: str = Field(description="Identifiant de l'auteur de l'annulation")


class TourneeRecordSchema(BaseModel):
    """
    Représentation complète d'une tournée avec son statut et son historique.
    Retourné par GET /optimize/v2/tournee/{tournee_id} et les endpoints de validation.
    """
    tournee_id:   str
    agent_id:     str
    service_date: str
    trip_ids:     list[str]
    statut:       str
    created_at:   str
    validated_by: Optional[str] = None
    validated_at: Optional[str] = None
    refused_by:   Optional[str] = None
    refused_at:   Optional[str] = None
    motif_refus:  Optional[str] = None
    cancelled_by: Optional[str] = None
    cancelled_at: Optional[str] = None


class ValidateTourneeResponse(BaseModel):
    """Réponse des endpoints de validation/refus/annulation."""
    tournee_id: str
    statut:     str
    message:    str
    record:     TourneeRecordSchema


# ─────────────────────────────────────────────────────────────────────────────
# GTFS-RT
# ─────────────────────────────────────────────────────────────────────────────

class StopTimeUpdateSchema(BaseModel):
    stop_id:                   str
    stop_sequence:             int
    arrival_delay_seconds:     Optional[int] = None
    departure_delay_seconds:   Optional[int] = None


class TripUpdateSchema(BaseModel):
    trip_id:           str
    train_number:      Optional[str] = None
    route_id:          Optional[str] = None
    stop_time_updates: list[StopTimeUpdateSchema] = []


class ServiceAlertSchema(BaseModel):
    alert_id:       str
    cause:          Optional[str] = None
    effect:         Optional[str] = None
    affected_trips: list[str]     = []
    affected_stops: list[str]     = []

    @property                                          # ← ajouter ces 3 lignes
    def is_suppression(self) -> bool:
        return self.effect == "NO_SERVICE"
# ─────────────────────────────────────────────────────────────────────────────
# API — Schemas génériques
# ─────────────────────────────────────────────────────────────────────────────

class HealthSchema(BaseModel):
    status:  str = "ok"
    version: str = "0.3.0"


class ErrorSchema(BaseModel):
    error:  str
    detail: Optional[str] = None
    code:   int


class PredictResponseSchema(BaseModel):
    troncons:     list[TronconScoreSchema]
    service_date: date
    scored_at:    datetime = Field(default_factory=datetime.now)
    nb_troncons:  int

    model_config = {"from_attributes": True}


class OptimizeResponseSchema(BaseModel):
    """Réponse v1 — conservée pour compatibilité."""
    tournee:      TourneeSchema
    request:      TourneeRequestSchema
    optimized_at: datetime = Field(default_factory=datetime.now)


# ─────────────────────────────────────────────────────────────────────────────
# ML API — Scoring batch
# ─────────────────────────────────────────────────────────────────────────────

class TronconInput(BaseModel):
    trip_id:       str
    train_number:  str
    service_id:    str
    stop_sequence: int = Field(ge=0)

    stop_id_dep:   str  = Field(description="Code UIC 8 chiffres")
    stop_name_dep: str  = Field(default="")
    dep_minutes:   int  = Field(ge=0)

    stop_id_arr:   str
    stop_name_arr: str  = Field(default="")
    arr_minutes:   int  = Field(ge=0)

    duration_min:  int  = Field(gt=0)
    service_date:  date


class PredictRequest(BaseModel):
    troncons: list[TronconInput] = Field(min_length=1)


class PredictScoreItem(BaseModel):
    trip_id:       str
    stop_sequence: int   = Field(ge=0)
    stop_id_dep:   str
    dep_minutes:   int   = Field(ge=0)
    fraud_score:   float = Field(default=0.0, ge=0.0, le=1.0)


class PredictResponse(BaseModel):
    scores:      list[PredictScoreItem]
    scored_at:   datetime = Field(default_factory=datetime.now)
    nb_troncons: int      = Field(ge=0)