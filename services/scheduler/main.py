import time
import signal
import sys
from datetime import datetime

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from services.scheduler.daily.daily_fetcher import DailyGTFSFetcher
from services.scheduler.realtime.realtime_fetcher import RealtimeFetcher


def build_scheduler() -> BackgroundScheduler:
    """
    Configure et retourne le scheduler APScheduler.

    Deux jobs :
        1. GTFS Static  → tous les jours à 03:00 (après la mise à jour SNCF à 17h J-1)
        2. GTFS-RT      → toutes les 2 minutes (fréquence de mise à jour SNCF)

    Pourquoi BackgroundScheduler ?
        Il tourne dans un thread séparé — le thread principal reste libre
        pour écouter les signaux d'arrêt (SIGTERM, SIGINT).
        Alternatif : BlockingScheduler qui occupe le thread principal,
        moins flexible pour la gestion propre de l'arrêt.

    Pourquoi CronTrigger pour le Static ?
        CronTrigger permet de spécifier une heure précise (03:00).
        IntervalTrigger déclencherait à partir du démarrage du container,
        ce qui ne garantit pas l'heure d'exécution.

    Pourquoi IntervalTrigger pour le Realtime ?
        On veut une fréquence fixe (toutes les 2 min) quel que soit
        l'heure de démarrage. CronTrigger serait trop verbeux pour ça.
    """
    scheduler = BackgroundScheduler(
        job_defaults={
            # Si un job tourne encore au prochain déclenchement, on attend
            # qu'il se termine (pas de parallélisme involontaire)
            "coalesce":        True,
            "max_instances":   1,
            "misfire_grace_time": 60,  # tolérance de 60s si le scheduler était arrêté
        }
    )

    daily_fetcher   = DailyGTFSFetcher()
    realtime_fetcher = RealtimeFetcher()

    # ── Job 1 : GTFS Static quotidien ────────────────────────────────────────
    scheduler.add_job(
        func=daily_fetcher.run,
        trigger=CronTrigger(hour=3, minute=0),
        id="gtfs_static_daily",
        name="Téléchargement GTFS Static quotidien",
        replace_existing=True,
    )

    # ── Job 2 : GTFS-RT toutes les 2 minutes ────────────────────────────────
    scheduler.add_job(
        func=realtime_fetcher.run,
        trigger=IntervalTrigger(minutes=2),
        id="gtfs_rt_realtime",
        name="Fetch GTFS-RT Trip Updates + Service Alerts",
        replace_existing=True,
    )

    return scheduler


def main() -> None:
    """
    Point d'entrée du service scheduler.

    Démarre le scheduler et attend les signaux d'arrêt (SIGTERM, SIGINT).
    SIGTERM est envoyé par Docker lors d'un 'docker stop'.
    SIGINT  est envoyé par Ctrl+C en développement.
    """
    print(
        f"[Scheduler] Démarrage — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    scheduler = build_scheduler()
    scheduler.start()

    # Affiche les jobs configurés au démarrage
    print("[Scheduler] Jobs configurés :")
    for job in scheduler.get_jobs():
        print(f"  - {job.name} | prochain run : {job.next_run_time}")

    # ── Gestion propre de l'arrêt ─────────────────────────────────────────────
    # On enregistre un handler pour SIGTERM et SIGINT
    # pour arrêter le scheduler proprement avant de quitter

    def handle_shutdown(signum, frame):
        print(f"\n[Scheduler] Signal {signum} reçu — arrêt en cours...")
        scheduler.shutdown(wait=True)  # attend la fin du job en cours
        print("[Scheduler] Arrêt propre. Au revoir.")
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT,  handle_shutdown)

    # Boucle principale : dort indéfiniment, les jobs tournent en arrière-plan
    print("[Scheduler] En attente... (Ctrl+C pour arrêter)")
    while True:
        time.sleep(60)


if __name__ == "__main__":
    main()