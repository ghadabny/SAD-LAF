from datetime import datetime

from services.ml_engine.data.gtfs.downloader import GTFSDownloader


class DailyGTFSFetcher:
    """
    Responsabilité unique : déclencher le téléchargement quotidien
    du fichier GTFS Static.

    Ne sait pas comment télécharger — c'est le rôle de GTFSDownloader.
    Ne sait pas quand être appelé — c'est le rôle de main.py (APScheduler).

    Séparation des responsabilités :
        DailyGTFSFetcher  → décide QUOI faire (forcer le re-téléchargement)
        GTFSDownloader    → sait COMMENT télécharger
        APScheduler       → sait QUAND appeler
    """

    def __init__(self):
        self.downloader = GTFSDownloader()

    def run(self) -> None:
        """
        Télécharge le fichier GTFS Static en forçant le re-téléchargement.

        force=True car on est appelé quotidiennement — on veut toujours
        la version la plus récente (SNCF met à jour les horaires J-1 à 17h).
        """
        print(
            f"[DailyGTFSFetcher] Démarrage téléchargement quotidien "
            f"— {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        try:
            self.downloader.download(force=True)
            print("[DailyGTFSFetcher] Téléchargement quotidien terminé.")
        except Exception as e:
            # On log l'erreur mais on ne la propage pas —
            # le scheduler doit continuer à tourner même si un téléchargement échoue
            print(f"[DailyGTFSFetcher] Erreur : {e}")