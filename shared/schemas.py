from datetime import date, datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


"""""
Pourquoi Pydantic ?
   Pydantic valide automatiquement les types à l'instanciation.
   Si on crée TronconSchema(dep_minutes="pas_un_int"), Pydantic lève
   une ValidationError claire au lieu d'un TypeError cryptique plus tard.

   FastAPI utilise Pydantic nativement — ces schémas servent directement
   comme types des paramètres et réponses des endpoints.

   Convention : on suffixe tous les schémas avec "Schema" pour les
   distinguer des classes métier (Node, Arc, etc.).

"""

# ENUMS

class ArcTypeSchema(str, Enum):
    """
    Type d'arc dans le graphe temps-étendu.
    Hérite de str pour la sérialisation JSON automatique :
        ArcTypeSchema.TRAIN → "train" dans le JSON de l'API
    Sans héritage str, Pydantic sérialiserait "ArcTypeSchema.TRAIN".
    """
    TRAIN          = "train"
    CORRESPONDANCE = "correspondance"


# GTFS — Tronçons

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


# Graphe temps-étendu — Nœuds et Arcs

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


# Optimisation — Requête et Réponse

class TourneeRequestSchema(BaseModel):
    """
    Paramètres d'une requête d'optimisation de tournée.

    L'optimiseur ne sait rien des agents — il reçoit des contraintes
    opérationnelles et génère la meilleure tournée possible.
    L'affectation agent ↔ tournée est faite en dehors du système.

    Paramètres :
        gare_depart_id    : code UIC de la gare de départ (8 chiffres)
        heure_depart_min  : heure de départ en minutes depuis minuit
                            Ex: 08:30 → 510
        duree_max_minutes : durée maximale de la tournée en minutes
                            Par défaut : MAX_MISSION_DURATION_HOURS × 60
        service_date      : date de la tournée
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

    Attributs :
        arcs          : séquence ordonnée des arcs de la tournée
        score_total   : somme des fraud_score sur les arcs TRAIN uniquement
                        (les arcs CORRESPONDANCE ne sont pas scorés)
        duree_totale_minutes : durée réelle de la tournée
        nb_trains     : nombre de trains contrôlés
        gare_depart   : nœud de départ
        gare_arrivee  : nœud d'arrivée (dernier arrêt de la tournée)
        service_date  : date de la tournée
        generated_at  : horodatage de génération
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


# GTFS-RT — Temps réel

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

    cause  : raison de la perturbation (STRIKE, TECHNICAL_PROBLEM, etc.)
    effect : impact sur le service (NO_SERVICE, REDUCED_SERVICE, etc.)
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
    """
    Contenu complet d'un fichier JSON produit par RealtimeFetcher.
    Encapsule les métadonnées + les données du flux.
    """
    fetched_at: datetime
    source:     str


class TripUpdateFeedSchema(RealtimeFeedSchema):
    """Feed Trip Updates complet."""
    trip_updates: list[TripUpdateSchema] = Field(default_factory=list)


class ServiceAlertFeedSchema(RealtimeFeedSchema):
    """Feed Service Alerts complet."""
    alerts: list[ServiceAlertSchema] = Field(default_factory=list)


# API — Réponses standardisées

class HealthSchema(BaseModel):
    """Réponse du endpoint GET /health."""
    status:  str = "ok"
    version: str = "0.1.0"


class ErrorSchema(BaseModel):
    """
    Format standard des erreurs retournées par l'API.
    Permet au client (Power Automate, front) de parser les erreurs
    de façon cohérente.
    """
    error:   str
    detail:  Optional[str] = None
    code:    int


class PredictResponseSchema(BaseModel):
    """
    Réponse du endpoint POST /predict.
    Retourne les tronçons enrichis avec leurs scores de fraude.
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


# LAF — À définir quand les données arrivent

# TODO: LAFControleSchema
#   À définir quand les données LAF seront reçues.
#   Colonnes attendues (à confirmer) :
#       - trip_id / train_number
#       - date_controle
#       - stop_id_dep / stop_id_arr (ou tronçon_id ?)
#       - nb_voyageurs_controles
#       - nb_irregularites
#       - taux_fraude
#       - agent_id (anonymisé ?)
#
# TODO: HistoricalFeatureSchema
#   Features calculées depuis LAFControleSchema.
#   Dépend du contenu exact des données LAF.