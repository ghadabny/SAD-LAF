import requests
from dataclasses import dataclass

from shared.config import config


@dataclass(frozen=True)
class GTFSRTFeed:
    """Contient les bytes bruts d'un flux GTFS-RT."""
    trip_updates_bytes: bytes
    service_alerts_bytes: bytes


class GTFSRTFetcher:
    """
    Responsabilité unique : télécharger les flux GTFS Realtime (protobuf).

    Ne parse pas — c'est le rôle de GTFSRTParser.
    Ne connaît pas la sémantique métier — c'est le rôle de GTFSRTMerger.

    Usage:
        fetcher = GTFSRTFetcher()
        feed = fetcher.fetch()
    """

    TIMEOUT_SECONDS: int = 30

    def __init__(self):
        self._tu_url: str = config.GTFS_RT_TRIP_UPDATES_URL
        self._sa_url: str = config.GTFS_RT_SERVICE_ALERTS_URL

    def fetch(self) -> GTFSRTFeed:
        """
        Télécharge les deux flux RT et retourne un GTFSRTFeed.

        Lève requests.HTTPError si l'un des endpoints répond en erreur.
        """
        print("[GTFSRTFetcher] Téléchargement Trip Updates...")
        tu_bytes = self._fetch_url(self._tu_url)

        print("[GTFSRTFetcher] Téléchargement Service Alerts...")
        sa_bytes = self._fetch_url(self._sa_url)

        print(f"[GTFSRTFetcher] OK — TU: {len(tu_bytes):,} bytes, SA: {len(sa_bytes):,} bytes")
        return GTFSRTFeed(trip_updates_bytes=tu_bytes, service_alerts_bytes=sa_bytes)

    def _fetch_url(self, url: str) -> bytes:
        proxies = (
            {"http": config.HTTP_PROXY, "https": config.HTTPS_PROXY}
            if config.HTTP_PROXY
            else None
        )
        response = requests.get(url, timeout=self.TIMEOUT_SECONDS, proxies=proxies)
        response.raise_for_status()
        return response.content