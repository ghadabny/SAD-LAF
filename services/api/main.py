from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from services.api.routers import health, predict, optimize
from services.api.routers import optimize_v2

app = FastAPI(
    title="SAD-LAF API",
    description=(
        "Système d'Aide à la Décision — Lutte Anti-Fraude. "
        "Génère des scores de fraude par tronçon ferroviaire "
        "et optimise les tournées d'inspection des agents LAF."
    ),
    version="0.2.0",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
# Permet à Power Apps / Power Automate d'appeler l'API depuis un navigateur.
# En production, restreindre allow_origins aux domaines SNCF uniquement.
""""
    allow_origins=[
    "https://apps.powerapps.com",
    "https://*.powerautomate.com",
    # ajouter les domaines SNCF en prod
    ]
    """
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # à restreindre en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(health.router)
app.include_router(predict.router)
app.include_router(optimize.router)          # v1 — conservé pour compatibilité
app.include_router(optimize_v2.router)       # v2 — PowerApps