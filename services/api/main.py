# services/api/main.py
"""
Point d'entrée de l'API FastAPI SAD-LAF.

Modifications v3.3 — Listener event-driven :
    Le listener FileListener est intégré dans un scheduler APScheduler
    démarré via le lifespan de FastAPI. Deux jobs sont ajoutés :
        - request_listener  : toutes les 30s → traite data/requests/
        - decision_listener : toutes les 15s → traite data/decisions/

    Pourquoi dans l'API et non dans services/scheduler/ ?
        Le listener a besoin du TripBookingStore (in-memory, même processus)
        et du solver_run. Il doit tourner dans le même processus que l'API.
        Le service scheduler/ gère uniquement les flux GTFS (autre conteneur).
"""
import logging
from contextlib import asynccontextmanager

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from services.api.routers import health, predict, optimize
from services.api.routers import optimize_v2
from services.listener import FileListener
from shared.config import config

# ── Configuration logging ─────────────────────────────────────────────────────
# Uvicorn ne configure que ses propres loggers.
# Ce bloc active les loggers Python de tous les modules du projet
# pour que les messages [Listener], [TourneeExporter], etc. s'affichent.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
# Réduire le bruit des bibliothèques tierces
logging.getLogger("apscheduler").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.INFO)

logger = logging.getLogger(__name__)


# ── Lifespan — démarrage et arrêt propres du scheduler ───────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestionnaire de durée de vie de l'application FastAPI.

    Au démarrage :
        - Instancie FileListener
        - Démarre BackgroundScheduler avec les deux jobs listener
    À l'arrêt (SIGTERM Docker / Ctrl+C dev) :
        - Arrête le scheduler proprement (attend la fin du job en cours)
    """
    listener  = FileListener()
    scheduler = BackgroundScheduler(
        job_defaults={
            "coalesce":           True,
            "max_instances":      1,
            "misfire_grace_time": 30,
        }
    )

    scheduler.add_job(
        func    = listener.process_pending_requests,
        trigger = IntervalTrigger(seconds=config.REQUEST_LISTENER_INTERVAL_SECONDS),
        id      = "request_listener",
        name    = "Listener — demandes de génération de tournée",
        replace_existing=True,
    )

    scheduler.add_job(
        func    = listener.process_pending_decisions,
        trigger = IntervalTrigger(seconds=config.DECISION_LISTENER_INTERVAL_SECONDS),
        id      = "decision_listener",
        name    = "Listener — décisions manager (validate/refuse/cancel)",
        replace_existing=True,
    )

    scheduler.start()
    logger.info(
        "[API] ✅ Scheduler listener démarré — "
        "request_listener toutes les %ds | decision_listener toutes les %ds",
        config.REQUEST_LISTENER_INTERVAL_SECONDS,
        config.DECISION_LISTENER_INTERVAL_SECONDS,
    )
    logger.info(
        "[API] Dossiers surveillés — requests: %s | decisions: %s",
        config.REQUESTS_DIR,
        config.DECISIONS_DIR,
    )

    yield  # l'application tourne ici

    scheduler.shutdown(wait=True)
    logger.info("[API] Scheduler listener arrêté proprement.")


# ── Application FastAPI ───────────────────────────────────────────────────────

app = FastAPI(
    title       = "SAD-LAF API",
    description = (
        "Système d'Aide à la Décision — Lutte Anti-Fraude. "
        "Génère des scores de fraude par tronçon ferroviaire "
        "et optimise les tournées d'inspection des agents LAF."
    ),
    version     = "0.2.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

app.include_router(health.router)
app.include_router(predict.router)
app.include_router(optimize.router)
app.include_router(optimize_v2.router)