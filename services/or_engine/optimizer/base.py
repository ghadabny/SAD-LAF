# services/or_engine/optimizer/base.py
from abc import ABC, abstractmethod

from services.or_engine.graph.transition import Arc, Node


class BaseOptimizer(ABC):
    """
    Contrat abstrait pour tous les optimiseurs de tournées LAF.

    Garantit que solver.run() peut échanger OrienteeringOptimizer
    contre toute autre implémentation (heuristique, métaheuristique...)
    sans modifier une ligne du solver.

    SOLID — principe L :
        Toute implémentation concrète doit retourner un dict avec
        exactement les clés : arcs, score_total, duree_minutes, nb_trains.
        Si une implémentation ne peut pas trouver de solution,
        elle lève ValueError avec un message explicatif — jamais
        elle ne retourne None ou un dict incomplet.

    SOLID — principe D :
        solver.run() dépend de BaseOptimizer (abstraction),
        jamais de OrienteeringOptimizer directement.
    """

    @abstractmethod
    def solve(
        self,
        graph: dict[Node, list[Arc]],
        gare_depart_id: str,
        heure_depart_min: int,
        duree_max_minutes: int,
    ) -> dict:
        """
        Génère la tournée optimale sous contrainte de budget temps.

        Paramètres :
            graph             : graphe temps-étendu (Node → list[Arc])
                                Les Arc.fraud_score doivent déjà être injectés.
            gare_depart_id    : code UIC 8 chiffres de la gare de départ
            heure_depart_min  : heure de départ en minutes depuis minuit
            duree_max_minutes : budget temps total en minutes

        Retourne un dict avec exactement :
            {
                "arcs":          list[Arc],   — séquence TRAIN + CORRESPONDANCE
                "score_total":   float,        — somme des fraud_score des arcs TRAIN
                "duree_minutes": int,           — durée totale en minutes
                "nb_trains":     int,           — nombre d'arcs TRAIN dans la tournée
            }

        Lève :
            ValueError  : aucun train disponible depuis la gare demandée,
                          budget insuffisant, ou graphe vide.
        """
        ...