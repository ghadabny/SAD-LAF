# bootstrap_gtfs.py — lance ce script d'abord
from services.ml_engine.data.gtfs.downloader import GTFSDownloader
GTFSDownloader().download(force=True)