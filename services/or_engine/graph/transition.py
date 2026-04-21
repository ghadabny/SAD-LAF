from dataclasses import dataclass, field
from enum import Enum
from datetime import date


class ArcType(Enum):
    """
    Type d'arc dans le graphe temps-étendu.

    Pourquoi une Enum et pas des chaînes de caractères ?
        - Typage strict : ArcType.TRAIN est vérifiable par Python,
          "train" est une chaîne libre qui peut contenir une faute de frappe
        - Autocomplétion dans l'IDE
        - Comparaison sûre : arc.type == ArcType.TRAIN (pas de risque de casse)
    """
    TRAIN = "train"               # déplacement physique dans un train
    CORRESPONDANCE = "correspondance"  # attente entre deux trains en gare


# ─────────────────────────────────────────────────────────────────────────────
# FIX B8 — frozen=True + eq=False sur Node
#
# AVANT : @dataclass sans frozen=True
#   → Node était MUTABLE mais possédait un __hash__ manuel.
#   → Un Node pouvait être modifié après avoir été inséré dans un dict ou un set,
#     cassant silencieusement la cohérence du graphe (le hash ne correspondait
#     plus à la clé réelle dans la table de hachage).
#
# APRÈS : @dataclass(frozen=True, eq=False)
#   frozen=True  → Python lève AttributeError à tout setattr après __post_init__.
#                  Le Node est garanti immuable pour toute la durée de vie du graphe.
#   eq=False     → On désactive la génération automatique de __eq__ et __hash__
#                  par le décorateur. Nos méthodes manuelles prennent le relais.
#                  CRITIQUE : le __hash__ auto (frozen=True, eq=True) inclut
#                  stop_name — ce que l'on veut ÉVITER car deux nœuds identiques
#                  peuvent avoir des variantes de nom SNCF légèrement différentes.
#                  Nos méthodes n'utilisent que (stop_id, time_minutes, service_date).
#
# Compatibilité avec __post_init__ :
#   frozen=True bloque uniquement les setattr après la construction.
#   La validation dans __post_init__ (if self.time_minutes < 0) est une LECTURE,
#   pas une écriture → aucun problème.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, eq=False)
class Node:
    """
    Un nœud du graphe temps-étendu.

    Un nœud = une position dans l'espace ET le temps.
    Ce n'est pas juste une gare — c'est "être à cette gare à cette heure".

    Attributs :
        stop_id      : code UIC de la gare (8 chiffres, ex: '87212027')
        stop_name    : nom lisible (ex: 'Strasbourg')
        time_minutes : heure en minutes depuis minuit (ex: 503 pour 08:23)
        service_date : date de circulation

    Pourquoi frozen=True, eq=False ?
        frozen=True  → immuabilité garantie. Un Node inséré dans un dict ou un set
                       ne peut plus être modifié. Sans frozen, on risque de corrompre
                       silencieusement la table de hachage du graphe.
        eq=False     → on conserve nos __hash__ et __eq__ personnalisés qui excluent
                       stop_name du hash. Deux nœuds identiques (même stop_id, même
                       heure, même date) sont "égaux" même si leur nom diffère
                       légèrement (variantes de nommage SNCF : "Strasbourg" vs
                       "Strasbourg-Ville").
    """
    stop_id: str
    stop_name: str
    time_minutes: int
    service_date: date

    def __post_init__(self):
        """
        Validation fail-fast à la construction.

        Avec frozen=True, __post_init__ ne peut PAS écrire dans self.
        C'est correct ici car on ne fait que lire time_minutes pour valider.
        Si une validation nécessitait de corriger une valeur, il faudrait utiliser
        object.__setattr__(self, "champ", valeur_corrigee).
        """
        if self.time_minutes < 0:
            raise ValueError(
                f"[Node] time_minutes ne peut pas être négatif : {self.time_minutes}"
            )

    # ── Propriétés calculées ──────────────────────────────────────────────────

    @property
    def hour(self) -> int:
        """Heure de départ (0-23). Pratique pour les filtres."""
        return self.time_minutes // 60

    @property
    def label(self) -> str:
        """Représentation lisible pour le débogage et les logs."""
        h = self.time_minutes // 60
        m = self.time_minutes % 60
        return f"{self.stop_name} {h:02d}:{m:02d}"

    # ── Hash & Égalité personnalisés (excluent stop_name) ─────────────────────

    def __hash__(self):
        """
        Hash sur (stop_id, time_minutes, service_date).

        stop_name est exclu intentionnellement :
            Deux nœuds avec le même stop_id, la même heure et la même date
            représentent le MÊME point dans le graphe, même si le nom diffère
            légèrement selon la source (GTFS static vs GTFS-RT vs historique LAF).
            Les inclure casserait les lookups dans les dicts/sets du graphe.
        """
        return hash((self.stop_id, self.time_minutes, self.service_date))

    def __eq__(self, other):
        """
        Égalité sur (stop_id, time_minutes, service_date).

        Cohérent avec __hash__ : deux objets égaux doivent avoir le même hash.
        stop_name exclu pour les mêmes raisons que dans __hash__.
        """
        if not isinstance(other, Node):
            return False
        return (
            self.stop_id == other.stop_id
            and self.time_minutes == other.time_minutes
            and self.service_date == other.service_date
        )


@dataclass
class Arc:
    """
    Un arc orienté entre deux nœuds du graphe temps-étendu.

    Un arc = une action possible pour un agent LAF.

    Attributs :
        source       : nœud de départ
        destination  : nœud d'arrivée
        arc_type     : TRAIN ou CORRESPONDANCE
        duration_min : durée de l'arc en minutes
        trip_id      : identifiant technique du trip GTFS (None pour correspondance)
        train_number : numéro commercial du train (None pour correspondance)
        fraud_score  : score de fraude du tronçon (0.0 par défaut, mis à jour
                       par le modèle ML après entraînement)

    Pourquoi Arc n'est PAS frozen ?
        fraud_score est initialisé à 0.0 à la construction du graphe, puis injecté
        par le pipeline ML via inject_scores(). Cette mutation est intentionnelle
        et documentée. Rendre Arc frozen casserait l'injection de scores.

    Pourquoi fraud_score est initialisé à 0.0 ?
        À la construction du graphe, on ne connaît pas encore le score ML.
        Le score est injecté plus tard par le pipeline :
            1. build_graph()   → construit la structure
            2. LGBMScorer      → calcule les scores
            3. inject_scores() → met à jour fraud_score sur chaque arc TRAIN
        Cette séparation respecte le S de SOLID : builder.py ne sait pas
        comment fonctionne le modèle ML.
    """
    source: Node
    destination: Node
    arc_type: ArcType
    duration_min: int
    trip_id: str | None = None
    train_number: str | None = None
    fraud_score: float = field(default=0.0)

    def __post_init__(self):
        if self.duration_min <= 0:
            raise ValueError(
                f"[Arc] duration_min doit être > 0 : {self.duration_min}\n"
                f"  source      : {self.source.label}\n"
                f"  destination : {self.destination.label}"
            )

    @property
    def is_train(self) -> bool:
        return self.arc_type == ArcType.TRAIN

    @property
    def is_correspondance(self) -> bool:
        return self.arc_type == ArcType.CORRESPONDANCE

    def __repr__(self) -> str:
        return (
            f"Arc({self.arc_type.value} | "
            f"{self.source.label} → {self.destination.label} | "
            f"{self.duration_min}min | score={self.fraud_score:.2f})"
        )