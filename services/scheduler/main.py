import json
import time
import signal
import sys
from datetime import datetime
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from services.scheduler.daily.daily_fetcher import DailyGTFSFetcher
from services.scheduler.realtime.realtime_fetcher import RealtimeFetcher
from services.scheduler.realtime.realtime_update import RealtimeUpdate
from shared.config import config


_NOTIFICATIONS_DIR = Path(config.DATA_DIR) / "notifications"


def _run_realtime_with_notifications() -> None:
    """
    Cycle GTFS-RT enrichi : fetch → détection → notification agents impactés.

    Après chaque fetch, vérifie si des tournées VALIDEES contiennent des trains
    annulés ou très en retard, et écrit un fichier JSON dans data/notifications/
    pour chaque agent concerné.

    Les fichiers de notification sont nommés :
        RT_UPDATE_{AGENT_ID}_{YYYYMMDD_HHMMSS}.json
    Ils peuvent être lus par PowerAutomate ou tout autre système de notification.
    """
    fetcher = RealtimeFetcher()
    update  = fetcher.run()

    if not update.has_changed or not update.trips_impacted:
        return

    try:
        _notify_affected_agents(update)
    except Exception as exc:
        print(f"[Scheduler] ⚠️  Notification RT échouée : {exc}")


def _notify_affected_agents(update: RealtimeUpdate) -> None:
    """
    Identifie les tournées VALIDEES impactées par un changement RT
    et écrit un fichier de notification par agent concerné.
    """
    from services.api.booking_store import STATUT_VALIDEE, get_booking_store

    store    = get_booking_store()
    affected = []

    with store._lock:
        for tournee_id, record in store._tournees.items():
            if record.get("statut") != STATUT_VALIDEE:
                continue
            impacted_in_tournee = update.trips_impacted & set(record.get("trip_ids", []))
            if impacted_in_tournee:
                affected.append({
                    "tournee_id":     tournee_id,
                    "agent_id":       record.get("agent_id", "?"),
                    "service_date":   record.get("service_date", ""),
                    "impacted_trips": list(impacted_in_tournee),
                })

    if not affected:
        return

    _NOTIFICATIONS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    for entry in affected:
        agent_id  = entry["agent_id"].upper()
        notif_path = _NOTIFICATIONS_DIR / f"RT_UPDATE_{agent_id}_{ts}.json"
        payload = {
            "type":            "GTFS_RT_UPDATE",
            "agent_id":        entry["agent_id"],
            "tournee_id":      entry["tournee_id"],
            "service_date":    entry["service_date"],
            "impacted_trips":  entry["impacted_trips"],
            "trips_delayed":   list(update.trips_delayed & set(entry["impacted_trips"])),
            "trips_cancelled": list(update.trips_cancelled & set(entry["impacted_trips"])),
            "action_required": "REGENERATE_TOURNEE",
            "message": (
                f"⚠️ Des trains de votre tournée {entry['tournee_id']} ont été "
                f"modifiés en temps réel. Veuillez régénérer une nouvelle tournée."
            ),
            "timestamp": datetime.now().isoformat(),
        }
        notif_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        print(
            f"[Scheduler] 🔔 Notification RT → agent {entry['agent_id']} "
            f"(tournée {entry['tournee_id']}, {len(entry['impacted_trips'])} trains impactés)"
        )


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

    daily_fetcher = DailyGTFSFetcher()

    # ── Job 1 : GTFS Static quotidien ────────────────────────────────────────
    scheduler.add_job(
        func=daily_fetcher.run,
        trigger=CronTrigger(hour=3, minute=0),
        id="gtfs_static_daily",
        name="Téléchargement GTFS Static quotidien",
        replace_existing=True,
    )

    # ── Job 2 : GTFS-RT toutes les 2 minutes + notification agents ───────────
    scheduler.add_job(
        func=_run_realtime_with_notifications,
        trigger=IntervalTrigger(minutes=2),
        id="gtfs_rt_realtime",
        name="Fetch GTFS-RT + Notifications agents impactés",
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