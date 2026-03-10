from collections import defaultdict
from datetime import date

import pandas as pd

from services.or_engine.graph.transition import Arc, ArcType, Node
from shared.constants import MIN_TRANSFER_MINUTES, MIN_BOARD_DURATION_MINUTES


class TimeExpandedGraphBuilder:
    """
    Construit le graphe temps-étendu à partir des tronçons GTFS.

    Un graphe temps-étendu modélise le réseau ferroviaire en ajoutant
    la dimension temporelle : chaque nœud est (gare, heure) et non
    pas juste une gare.

    Responsabilité unique : construire la structure du graphe.
    Ne sait pas comment les scores sont calculés (rôle du modèle ML).
    Ne sait pas comment le graphe est optimisé (rôle de l'optimiseur).

    Usage :
        builder = TimeExpandedGraphBuilder()
        graph = builder.build(troncons, service_date=date(2024, 9, 2))
        # graph : dict[Node, list[Arc]]

    Structure retournée :
        {
            Node(Strasbourg, 08:23): [
                Arc(TRAIN, Strasbourg 08:23 → Sélestat 08:52, train=117756),
            ],
            Node(Sélestat, 08:52): [
                Arc(TRAIN,          Sélestat 08:52 → Colmar 09:15, ...),
                Arc(CORRESPONDANCE, Sélestat 08:52 → Sélestat 09:05, ...),
            ],
            ...
        }
    """

    def __init__(self):
        # Durée minimale d'un tronçon pour qu'un contrôle soit possible
        # (valeur métier LAF depuis constants.py)
        self.min_board_duration = MIN_BOARD_DURATION_MINUTES
        # Délai minimal pour qu'une correspondance soit réalisable
        self.min_transfer = MIN_TRANSFER_MINUTES

    def build(
        self,
        troncons: pd.DataFrame,
        service_date: date,
    ) -> dict[Node, list[Arc]]:
        """
        Construit le graphe temps-étendu pour une date de service donnée.

        Paramètres :
            troncons     : DataFrame produit par GTFSPreprocessor.build_troncons()
                           Colonnes requises : trip_id, train_number,
                           stop_id_dep, stop_name_dep, dep_minutes,
                           stop_id_arr, stop_name_arr, arr_minutes,
                           duration_min
            service_date : date de circulation (ex: date(2024, 9, 2))

        Retourne :
            dict[Node, list[Arc]] — liste d'adjacence du graphe
        """
        self._validate(troncons)

        # defaultdict(list) : si une clé n'existe pas encore, crée
        # automatiquement une liste vide. Évite les KeyError.
        graph: dict[Node, list[Arc]] = defaultdict(list)

        # ── Étape 1 : arcs TRAIN ─────────────────────────────────────────────
        # Un arc TRAIN par tronçon GTFS valide
        graph = self._add_train_arcs(graph, troncons, service_date)

        # ── Étape 2 : arcs CORRESPONDANCE ────────────────────────────────────
        # Pour chaque gare, on regarde quels trains arrivent et repartent
        # et on crée des arcs de correspondance si le délai est suffisant
        graph = self._add_correspondance_arcs(graph, troncons, service_date)

        n_nodes = len(graph)
        n_arcs = sum(len(arcs) for arcs in graph.values())
        print(
            f"[GraphBuilder] Graphe construit : "
            f"{n_nodes} nœuds, {n_arcs} arcs "
            f"({service_date})"
        )
        return dict(graph)

    # ── Méthodes privées ──────────────────────────────────────────────────────

    def _add_train_arcs(
        self,
        graph: dict,
        troncons: pd.DataFrame,
        service_date: date,
    ) -> dict:
        """
        Crée un arc TRAIN pour chaque tronçon GTFS.

        Filtre les tronçons trop courts : si un train s'arrête moins de
        MIN_BOARD_DURATION_MINUTES, un agent LAF ne peut pas physiquement
        monter, contrôler et descendre — ce tronçon n'est pas contrôlable.
        """
        troncons_valides = troncons[
            troncons["duration_min"] >= self.min_board_duration
        ]

        skipped = len(troncons) - len(troncons_valides)
        if skipped > 0:
            print(
                f"[GraphBuilder] {skipped} tronçons ignorés "
                f"(durée < {self.min_board_duration} min)"
            )

        for _, row in troncons_valides.iterrows():
            node_dep = Node(
                stop_id=row["stop_id_dep"],
                stop_name=row["stop_name_dep"],
                time_minutes=int(row["dep_minutes"]),
                service_date=service_date,
            )
            node_arr = Node(
                stop_id=row["stop_id_arr"],
                stop_name=row["stop_name_arr"],
                time_minutes=int(row["arr_minutes"]),
                service_date=service_date,
            )
            arc = Arc(
                source=node_dep,
                destination=node_arr,
                arc_type=ArcType.TRAIN,
                duration_min=int(row["duration_min"]),
                trip_id=str(row["trip_id"]),
                train_number=str(row["train_number"]),
                fraud_score=0.0,  # sera mis à jour par LGBMScorer
            )
            graph[node_dep].append(arc)

        return graph

    def _add_correspondance_arcs(
        self,
        graph: dict,
        troncons: pd.DataFrame,
        service_date: date,
    ) -> dict:
        """
        Crée les arcs de correspondance entre trains dans la même gare.

        Logique :
            Pour chaque gare, on collecte :
                - les heures d'ARRIVÉE  (train_A arrive à 08:52)
                - les heures de DÉPART  (train_B part  à 09:05)

            Pour chaque paire (arrivée, départ) dans la même gare :
                si départ - arrivée >= MIN_TRANSFER_MINUTES
                → on crée un arc CORRESPONDANCE

        Exemple :
            Sélestat : train_A arrive 08:52, train_B part 09:05
            09:05 - 08:52 = 13 min >= 5 min → correspondance possible ✅

            Sélestat : train_A arrive 08:52, train_C part 08:55
            08:55 - 08:52 = 3 min < 5 min → trop court, ignoré ❌
        """
        # Collecte toutes les arrivées par gare
        # {stop_id: [(arr_minutes, stop_name), ...]}
        arrivees: dict[str, list[tuple[int, str]]] = defaultdict(list)
        for _, row in troncons.iterrows():
            arrivees[row["stop_id_arr"]].append(
                (int(row["arr_minutes"]), row["stop_name_arr"])
            )

        # Collecte tous les départs par gare
        # {stop_id: [(dep_minutes, stop_name), ...]}
        departs: dict[str, list[tuple[int, str]]] = defaultdict(list)
        for _, row in troncons.iterrows():
            departs[row["stop_id_dep"]].append(
                (int(row["dep_minutes"]), row["stop_name_dep"])
            )

        n_correspondances = 0

        # Pour chaque gare qui a des arrivées ET des départs
        for stop_id in set(arrivees.keys()) & set(departs.keys()):
            for arr_min, stop_name in arrivees[stop_id]:
                for dep_min, _ in departs[stop_id]:

                    delai = dep_min - arr_min

                    # Le délai doit être suffisant pour la correspondance
                    # mais pas absurde (> 2h = peu probable pour une tournée)
                    if self.min_transfer <= delai <= 120:

                        node_arrivee = Node(
                            stop_id=stop_id,
                            stop_name=stop_name,
                            time_minutes=arr_min,
                            service_date=service_date,
                        )
                        node_depart = Node(
                            stop_id=stop_id,
                            stop_name=stop_name,
                            time_minutes=dep_min,
                            service_date=service_date,
                        )
                        arc = Arc(
                            source=node_arrivee,
                            destination=node_depart,
                            arc_type=ArcType.CORRESPONDANCE,
                            duration_min=delai,
                        )
                        graph[node_arrivee].append(arc)
                        n_correspondances += 1

        print(
            f"[GraphBuilder] {n_correspondances} arcs de correspondance ajoutés"
        )
        return graph

    def _validate(self, troncons: pd.DataFrame) -> None:
        """Vérifie que les colonnes requises sont présentes. Fail-fast."""
        required = {
            "trip_id", "train_number",
            "stop_id_dep", "stop_name_dep", "dep_minutes",
            "stop_id_arr", "stop_name_arr", "arr_minutes",
            "duration_min",
        }
        missing = required - set(troncons.columns)
        if missing:
            raise ValueError(
                f"[GraphBuilder] Colonnes manquantes dans troncons : {missing}"
            )

    def get_nodes_at_stop(
        self,
        graph: dict[Node, list[Arc]],
        stop_id: str,
    ) -> list[Node]:
        """
        Retourne tous les nœuds d'une gare donnée dans le graphe.
        Utile pour l'optimiseur : 'quels trains puis-je prendre
        depuis Strasbourg ?'
        """
        return [node for node in graph if node.stop_id == stop_id]

    def get_reachable_arcs(
        self,
        graph: dict[Node, list[Arc]],
        node: Node,
    ) -> list[Arc]:
        """
        Retourne les arcs accessibles depuis un nœud donné.
        Retourne une liste vide si le nœud n'est pas dans le graphe
        (pas d'exception — comportement safe pour l'optimiseur).
        """
        return graph.get(node, [])