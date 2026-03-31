"""
shared/schemas.py — Schémas Pydantic partagés entre tous les services.

Convention de nommage : suffixe "Schema" pour distinguer des classes métier.

Structure des schémas par domaine :
    GTFS         → TronconSchema, TronconScoreSchema
    Graphe       → NodeSchema, ArcSchema, ArcTypeSchema
    Optimisation → TourneeRequestSchema, TourneeSchema
    ML API       → TronconInput, PredictRequest, PredictScoreItem, PredictResponse
    GTFS-RT      → StopTimeUpdateSchema, TripUpdateSchema, ServiceAlertSchema
    API          → HealthSchema, ErrorSchema, PredictResponseSchema, OptimizeResponseSchema

Pourquoi Pydantic ?
    Validation automatique des types à l'instanciation.
    FastAPI l'utilise nativement comme types de paramètres et réponses.
"""

from datetime import date, datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ─────────────────────────────────────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────────────────────────────────────

class ArcTypeSchema(str, Enum):
    """
    Type d'arc dans le graphe temps-étendu.
    Hérite de str pour la sérialisation JSON automatique :
        ArcTypeSchema.TRAIN → "train" dans le JSON de l'API
    Sans héritage str, Pydantic sérialiserait "ArcTypeSchema.TRAIN".
    """
    TRAIN          = "train"
    CORRESPONDANCE = "correspondance"


# ─────────────────────────────────────────────────────────────────────────────
# GTFS — Tronçons
# ─────────────────────────────────────────────────────────────────────────────

class TronconSchema(BaseModel):
    """
    Représente un tronçon ferroviaire entre deux arrêts consécutifs.
    Sortie de GTFSPreprocessor.build_troncons().

    C'est l'unité de granularité du scoring ML :
    le modèle prédit un fraud_score par tronçon.
    """
    trip_id:       str
    train_number:  str
    service_id:    str
    stop_sequence: int

    stop_id_dep:   str
    stop_name_dep: str
    dep_minutes:   int = Field(ge=0, description="Minutes depuis minuit")

    stop_id_arr:   str
    stop_name_arr: str
    arr_minutes:   int = Field(ge=0, description="Minutes depuis minuit")

    duration_min:  int = Field(gt=0, description="Durée en minutes, strictement positive")

    @field_validator("stop_id_dep", "stop_id_arr")
    @classmethod
    def validate_stop_id(cls, v: str) -> str:
        """Un code UIC valide = 8 chiffres."""
        if not v.isdigit() or len(v) != 8:
            raise ValueError(
                f"stop_id invalide : '{v}'. "
                f"Attendu : 8 chiffres (code UIC). "
                f"Vérifie que GTFSPreprocessor.clean_stop_id() a bien été appelé."
            )
        return v


class TronconScoreSchema(TronconSchema):
    """
    Tronçon enrichi avec le score de fraude prédit par LightGBM.
    Hérite de TronconSchema — ajoute uniquement fraud_score.

    fraud_score : probabilité de fraude sur ce tronçon (0.0 → 1.0)
        0.0 = aucun risque détecté
        1.0 = risque maximum
    """
    fraud_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Score de fraude prédit par LightGBM (0.0 à 1.0)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Graphe temps-étendu — Nœuds et Arcs
# ─────────────────────────────────────────────────────────────────────────────

class NodeSchema(BaseModel):
    """
    Représente un nœud du graphe temps-étendu.
    Un nœud = une position dans l'espace ET le temps.
    """
    stop_id:      str
    stop_name:    str
    time_minutes: int  = Field(ge=0, description="Minutes depuis minuit")
    service_date: date

    @property
    def label(self) -> str:
        """Représentation lisible : 'Strasbourg 08:23'"""
        h = self.time_minutes // 60
        m = self.time_minutes % 60
        return f"{self.stop_name} {h:02d}:{m:02d}"

    model_config = {"frozen": True}


class ArcSchema(BaseModel):
    """
    Représente un arc orienté entre deux nœuds du graphe.
    Un arc = une action possible pour un agent LAF.
    """
    source:       NodeSchema
    destination:  NodeSchema
    arc_type:     ArcTypeSchema
    duration_min: int  = Field(gt=0)
    trip_id:      Optional[str]   = None   # None pour les arcs CORRESPONDANCE
    train_number: Optional[str]   = None   # None pour les arcs CORRESPONDANCE
    fraud_score:  float = Field(default=0.0, ge=0.0, le=1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Optimisation — Requête et Réponse
# ─────────────────────────────────────────────────────────────────────────────

class TourneeRequestSchema(BaseModel):
    """
    Paramètres d'une requête d'optimisation de tournée.

    L'optimiseur ne sait rien des agents — il reçoit des contraintes
    opérationnelles et génère la meilleure tournée possible.
    L'affectation agent ↔ tournée est faite en dehors du système.
    """
    gare_depart_id:    str = Field(description="Code UIC 8 chiffres de la gare de départ")
    heure_depart_min:  int = Field(ge=0, le=1439, description="Heure de départ en minutes depuis minuit")
    duree_max_minutes: int = Field(default=360, gt=0, le=720,
                                   description="Durée max de la tournée en minutes (max 12h)")
    service_date:      date

    @field_validator("gare_depart_id")
    @classmethod
    def validate_gare_depart(cls, v: str) -> str:
        if not v.isdigit() or len(v) != 8:
            raise ValueError(f"gare_depart_id invalide : '{v}'. Attendu : code UIC 8 chiffres.")
        return v


class TourneeSchema(BaseModel):
    """
    Résultat d'une optimisation de tournée.

    Une tournée = une séquence d'arcs TRAIN et CORRESPONDANCE
    optimisée pour maximiser le score de fraude total
    sous contrainte de durée.
    """
    arcs:                  list[ArcSchema]
    score_total:           float = Field(ge=0.0, description="Somme des scores sur arcs TRAIN")
    duree_totale_minutes:  int   = Field(gt=0)
    nb_trains:             int   = Field(ge=1, description="Nombre de trains contrôlés")
    gare_depart:           NodeSchema
    gare_arrivee:          NodeSchema
    service_date:          date
    generated_at:          datetime = Field(default_factory=datetime.now)

    @property
    def arcs_train(self) -> list[ArcSchema]:
        """Retourne uniquement les arcs TRAIN (sans les correspondances)."""
        return [a for a in self.arcs if a.arc_type == ArcTypeSchema.TRAIN]

    @property
    def trains_visites(self) -> list[str]:
        """Liste des numéros de trains visités dans l'ordre."""
        return [
            a.train_number for a in self.arcs_train
            if a.train_number is not None
        ]


# ─────────────────────────────────────────────────────────────────────────────
# GTFS-RT — Temps réel
# ─────────────────────────────────────────────────────────────────────────────

class StopTimeUpdateSchema(BaseModel):
    """
    Mise à jour d'horaire pour un arrêt donné (issu du flux Trip Updates).

    delay_seconds > 0 → retard
    delay_seconds < 0 → avance (rare)
    delay_seconds = None → pas d'info disponible
    """
    stop_id:                   str
    stop_sequence:             int
    arrival_delay_seconds:     Optional[int] = None
    departure_delay_seconds:   Optional[int] = None


class TripUpdateSchema(BaseModel):
    """
    Mise à jour temps réel pour un trip complet.
    Issu du fichier trip_updates.json produit par RealtimeFetcher.
    """
    trip_id:           str
    train_number:      Optional[str] = None
    route_id:          Optional[str] = None
    stop_time_updates: list[StopTimeUpdateSchema] = Field(default_factory=list)


class ActivePeriodSchema(BaseModel):
    """Période d'activation d'une alerte de service."""
    start: Optional[datetime] = None
    end:   Optional[datetime] = None


class ServiceAlertSchema(BaseModel):
    """
    Alerte de service (suppression, perturbation).
    Issu du fichier service_alerts.json produit par RealtimeFetcher.
    """
    alert_id:        str
    cause:           str
    effect:          str
    header:          str = ""
    description:     str = ""
    affected_trips:  list[str] = Field(default_factory=list)
    affected_stops:  list[str] = Field(default_factory=list)
    active_periods:  list[ActivePeriodSchema] = Field(default_factory=list)

    @property
    def is_suppression(self) -> bool:
        """True si le train est complètement supprimé."""
        return self.effect == "NO_SERVICE"


class RealtimeFeedSchema(BaseModel):
    """Contenu complet d'un fichier JSON produit par RealtimeFetcher."""
    fetched_at: datetime
    source:     str


class TripUpdateFeedSchema(RealtimeFeedSchema):
    """Feed Trip Updates complet."""
    trip_updates: list[TripUpdateSchema] = Field(default_factory=list)


class ServiceAlertFeedSchema(RealtimeFeedSchema):
    """Feed Service Alerts complet."""
    alerts: list[ServiceAlertSchema] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# API — Réponses standardisées (endpoints GTFS)
# ─────────────────────────────────────────────────────────────────────────────

class HealthSchema(BaseModel):
    """Réponse du endpoint GET /health."""
    status:  str = "ok"
    version: str = "0.1.0"


class ErrorSchema(BaseModel):
    """Format standard des erreurs retournées par l'API."""
    error:   str
    detail:  Optional[str] = None
    code:    int


class PredictResponseSchema(BaseModel):
    """
    Réponse du endpoint GET /predict.
    Retourne les tronçons GTFS enrichis avec leurs scores de fraude.
    """
    troncons:     list[TronconScoreSchema]
    service_date: date
    scored_at:    datetime = Field(default_factory=datetime.now)
    nb_troncons:  int

    model_config = {"from_attributes": True}


class OptimizeResponseSchema(BaseModel):
    """
    Réponse du endpoint POST /optimize.
    Retourne la tournée optimisée.
    """
    tournee:      TourneeSchema
    request:      TourneeRequestSchema
    optimized_at: datetime = Field(default_factory=datetime.now)


# ─────────────────────────────────────────────────────────────────────────────
# ML API — Scoring batch (Phase 1 : câblage post-entraînement)
# ─────────────────────────────────────────────────────────────────────────────

class TronconInput(BaseModel):
    """
    Données brutes d'un tronçon envoyées au modèle ML pour prédiction.

    Utilisé par l'OR Engine pour appeler POST /predict/batch avant l'optimisation.
    Contient les colonnes minimales requises par le FeaturePipeline.

    Colonnes requises par TemporalFeatureTransformer :
        dep_minutes  → calcul de dep_hour, is_peak_hour, day_of_week…
        service_date → calcul de is_weekend, is_vacances, is_jour_ferie…

    Colonnes requises par HistoricalFeatureTransformer :
        stop_id_dep, stop_id_arr → lookup du taux de fraude historique par O/D

    Propagation de service_date depuis le GTFS :
        La date n'est pas dans les tronçons bruts GTFS (elle vient de
        calendar_dates via service_id). Elle est injectée dans le solver
        depuis le contexte de la requête d'optimisation — voir solver.py,
        fonction _build_predict_request().
    """
    trip_id:       str
    train_number:  str
    service_id:    str
    stop_sequence: int = Field(ge=0, description="Position dans le trip (0-indexed)")

    stop_id_dep:   str  = Field(description="Code UIC 8 chiffres de la gare de départ")
    stop_name_dep: str  = Field(default="", description="Nom lisible (ex: Strasbourg)")
    dep_minutes:   int  = Field(ge=0, description="Heure de départ en minutes depuis minuit")

    stop_id_arr:   str  = Field(description="Code UIC 8 chiffres de la gare d'arrivée")
    stop_name_arr: str  = Field(default="", description="Nom lisible (ex: Sélestat)")
    arr_minutes:   int  = Field(ge=0, description="Heure d'arrivée en minutes depuis minuit")

    duration_min:  int  = Field(gt=0, description="Durée du tronçon en minutes")

    service_date:  date = Field(
        description=(
            "Date réelle de circulation du train. "
            "Requise par TemporalFeatureTransformer pour calculer "
            "is_weekend, is_vacances, is_jour_ferie, etc. "
            "Propagée depuis le contexte GTFS (calendar_dates × service_id)."
        )
    )


class PredictRequest(BaseModel):
    """
    Requête de scoring ML en batch.
    Corps de POST /predict/batch — envoyé par l'OR Engine via httpx.

    Exemple :
        {
            "troncons": [
                {
                    "trip_id": "TRIP_001",
                    "train_number": "117756",
                    "service_id": "000001",
                    "stop_sequence": 0,
                    "stop_id_dep": "87212027",
                    "dep_minutes": 503,
                    "stop_id_arr": "87214007",
                    "arr_minutes": 532,
                    "duration_min": 29,
                    "service_date": "2024-09-02"
                }
            ]
        }
    """
    troncons: list[TronconInput] = Field(
        min_length=1,
        description="Liste de tronçons à scorer — doit contenir au moins 1 élément.",
    )


class PredictScoreItem(BaseModel):
    """
    Score de fraude prédit par LightGBM pour un tronçon identifié.

    Les champs trip_id + stop_id_dep + dep_minutes permettent à l'OR Engine
    de faire la correspondance avec l'arc correct dans le graphe temps-étendu :

        arc.trip_id           == item.trip_id
        arc.source.stop_id    == item.stop_id_dep
        arc.source.time_minutes == item.dep_minutes

    Voir inject_scores() dans solver.py pour l'utilisation de cette clé.
    """
    trip_id:       str
    stop_sequence: int  = Field(ge=0)
    stop_id_dep:   str  = Field(description="Code UIC 8 chiffres — clé de jointure avec Arc.source")
    dep_minutes:   int  = Field(ge=0, description="Heure de départ — clé de jointure avec Arc.source.time_minutes")
    fraud_score:   float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Score de fraude prédit par LightGBM ∈ [0.0, 1.0]",
    )


class PredictResponse(BaseModel):
    """
    Réponse de l'endpoint POST /predict/batch.

    Retourne exactement autant de scores qu'il y avait de tronçons en entrée
    (même ordre garanti par l'implémentation de predict_batch).
    """
    scores:      list[PredictScoreItem]
    scored_at:   datetime = Field(default_factory=datetime.now)
    nb_troncons: int = Field(ge=0, description="Nombre de tronçons scorés")