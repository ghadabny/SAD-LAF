import io
import zipfile

import requests

from shared.config import config


class GTFSDownloader:
    """
    Responsabilité unique : télécharger et extraire le fichier ZIP GTFS
    depuis l'URL OpenData SNCF.

    Ne sait rien du contenu des fichiers — c'est le rôle de GTFSLoader.

    Usage:
        downloader = GTFSDownloader()
        downloader.download()              # télécharge si absent
        downloader.download(force=True)    # force le re-téléchargement
    """

    def __init__(self):
        self.gtfs_dir = config.GTFS_DIR
        self.url = config.GTFS_URL

    def download(self, force: bool = False) -> None:
        """
        Télécharge et extrait le ZIP GTFS dans le dossier configuré.

        Paramètres:
            force=False : ne re-télécharge pas si les fichiers existent déjà.
                          Utile pour éviter un téléchargement inutile en dev.
            force=True  : force le re-téléchargement.
                          À utiliser pour la mise à jour quotidienne en prod.
        """
        if self.gtfs_dir.exists() and not force:
            print(
                f"[GTFSDownloader] Données déjà présentes dans '{self.gtfs_dir}'.\n"
                f"                 Utilise force=True pour forcer la mise à jour."
            )
            return

        self.gtfs_dir.mkdir(parents=True, exist_ok=True)

        print(f"[GTFSDownloader] Téléchargement depuis : {self.url}")
        response = requests.get(self.url, stream=True, timeout=120)

        # Lève une erreur explicite si le serveur répond 404 ou 500
        response.raise_for_status()

        print("[GTFSDownloader] Extraction du ZIP en cours...")
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall(self.gtfs_dir)

        fichiers = list(self.gtfs_dir.glob("*.txt"))
        print(f"[GTFSDownloader] OK — {len(fichiers)} fichiers extraits dans '{self.gtfs_dir}'.")
        for f in fichiers:
            print(f"                 - {f.name}")