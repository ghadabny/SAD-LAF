from dataclasses import dataclass, field


@dataclass
class RealtimeUpdate:
    """
    Résultat d'un fetch GTFS-RT.

    Retourné par RealtimeFetcher.run() pour indiquer au scheduler
    si un recalcul est nécessaire et quels trains sont impactés.

    Attributs :
        has_changed         : True si au moins un changement détecté
                              par rapport au fetch précédent
        trips_delayed       : trip_ids avec un retard > seuil
        trips_cancelled     : trip_ids complètement supprimés
        trips_impacted      : union des deux (tous les trips à recalculer)

    Pourquoi un dataclass ?
        Un dataclass génère automatiquement __init__, __repr__ et __eq__.
        C'est parfait pour un objet de transport de données sans logique —
        on veut juste transporter l'information du fetcher vers le scheduler.

    Exemple :
        update = RealtimeUpdate(
            has_changed=True,
            trips_delayed={"TRIP_117756"},
            trips_cancelled={"TRIP_117758"},
        )
        if update.has_changed:
            recalcule(update.trips_impacted)
    """
    has_changed:     bool       = False
    trips_delayed:   set[str]   = field(default_factory=set)
    trips_cancelled: set[str]   = field(default_factory=set)

    @property
    def trips_impacted(self) -> set[str]:
        """
        Union des trains en retard et des trains supprimés.
        Ce sont tous les trips pour lesquels il faut recalculer la tournée.
        """
        return self.trips_delayed | self.trips_cancelled

    @property
    def nb_impacted(self) -> int:
        """Nombre total de trains impactés — utile pour les logs."""
        return len(self.trips_impacted)