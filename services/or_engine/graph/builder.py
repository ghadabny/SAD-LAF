# services/or_engine/graph/builder.py
import bisect
from collections import defaultdict
from datetime import date

import pandas as pd

from services.or_engine.graph.transition import Arc, ArcType, Node
from shared.constants import MIN_TRANSFER_MINUTES, MIN_BOARD_DURATION_MINUTES


class TimeExpandedGraphBuilder:
    """
    Construit le graphe temps-étendu à partir des tronçons GTFS.

    Responsabilité unique : construire la structure du graphe.
    Ne sait pas comment les scores sont calculés (rôle du modèle ML).
    Ne sait pas comment le graphe est optimisé (rôle de l'optimiseur).

    SOLID — principe D (refactoring) :
        _add_correspondance_arcs() utilisait une double boucle O(|arr| × |dep|)
        par gare. Sur les grandes gares (ex: Strasbourg avec ~300 arrivées et
        ~300 départs), cela génère 90 000 paires à tester — dont 99% sont
        invalides (délai hors fenêtre).

        Nouvelle approche — tri + recherche par intervalle :
            1. Trier les départs par minute croissante (O(n log n))
            2. Pour chaque arrivée, utiliser bisect_left pour trouver le premier
               départ dans la fenêtre [arr + MIN_TRANSFER, arr + 120]
            3. Itérer uniquement sur les départs valides

        Complexité : O(n log n) au lieu de O(n²).
        Sur Strasbourg : 300 arrivées × log(300) ≈ 2 400 opérations
        vs 300 × 300 = 90 000 opérations avant.
    """

    def __init__(self):
        self.min_board_duration = MIN_BOARD_DURATION_MINUTES
        self.min_transfer       = MIN_TRANSFER_MINUTES

    def build(
        self,
        troncons: pd.DataFrame,
        service_date: date,
    ) -> dict[Node, list[Arc]]:
        """
        Construit le graphe temps-étendu pour une date de service donnée.

        Paramètres :
            troncons     : DataFrame produit par GTFSPreprocessor.build_troncons()
            service_date : date de circulation (ex: date(2024, 9, 2))

        Retourne :
            dict[Node, list[Arc]] — liste d'adjacence du graphe
        """
        self._validate(troncons)
        graph: dict[Node, list[Arc]] = defaultdict(list)
        graph = self._add_train_arcs(graph, troncons, service_date)
        graph = self._add_correspondance_arcs(graph, troncons, service_date)

        n_nodes = len(graph)
        n_arcs  = sum(len(arcs) for arcs in graph.values())
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
        """Crée un arc TRAIN pour chaque tronçon GTFS de durée suffisante."""
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
                fraud_score=0.0,
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

        Algorithme O(n log n) — tri + bisect :
            Pour chaque gare :
                1. Trier les départs par dep_minutes (O(k log k) où k = nb départs)
                2. Extraire les minutes de départ dans une liste triée
                3. Pour chaque arrivée, bisect_left trouve le premier départ
                   valide en O(log k) au lieu d'itérer sur tous les départs

        La fenêtre de correspondance est [arr + MIN_TRANSFER, arr + 120 min].
        Les paires hors fenêtre ne sont jamais visitées.
        """
        # Collecte arrivées par gare : {stop_id: [(arr_min, stop_name), ...]}
        arrivees: dict[str, list[tuple[int, str]]] = defaultdict(list)
        for _, row in troncons.iterrows():
            arrivees[row["stop_id_arr"]].append(
                (int(row["arr_minutes"]), row["stop_name_arr"])
            )

        # Collecte départs par gare : {stop_id: [(dep_min, stop_name), ...]}
        # Triés par dep_minutes pour le bisect
        departs: dict[str, list[tuple[int, str]]] = defaultdict(list)
        for _, row in troncons.iterrows():
            departs[row["stop_id_dep"]].append(
                (int(row["dep_minutes"]), row["stop_name_dep"])
            )

        # Tri des départs par minute (requis pour bisect)
        for stop_id in departs:
            departs[stop_id].sort(key=lambda x: x[0])

        n_correspondances = 0

        for stop_id in set(arrivees.keys()) & set(departs.keys()):
            deps        = departs[stop_id]        # liste triée de (dep_min, stop_name)
            dep_minutes = [d[0] for d in deps]    # liste triée des minutes seules

            for arr_min, stop_name in arrivees[stop_id]:
                # Borne basse : premier départ >= arr_min + MIN_TRANSFER
                lo = bisect.bisect_left(dep_minutes, arr_min + self.min_transfer)
                # Borne haute : dernier départ <= arr_min + 120
                hi = bisect.bisect_right(dep_minutes, arr_min + 120)

                # On n'itère que sur les départs dans la fenêtre [lo, hi)
                for dep_min, _ in deps[lo:hi]:
                    delai = dep_min - arr_min

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

        print(f"[GraphBuilder] {n_correspondances} arcs de correspondance ajoutés")
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
        """Retourne tous les nœuds d'une gare donnée dans le graphe."""
        return [node for node in graph if node.stop_id == stop_id]

    def get_reachable_arcs(
        self,
        graph: dict[Node, list[Arc]],
        node: Node,
    ) -> list[Arc]:
        """Retourne les arcs accessibles depuis un nœud. Liste vide si absent."""
        return graph.get(node, [])