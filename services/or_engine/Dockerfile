# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — service or_engine
#
# Service batch : lancé après ml_engine.
# Consomme les scores produits par ml_engine et génère les tournées
# optimisées via le solveur MILP PuLP/CBC.
#
# Comme ml_engine, ce container ne tourne pas en continu —
# il s'exécute, produit les tournées et s'arrête.
#
# TODO : solver.py sera implémenté quand orienteering.py sera codé
#        (en attente du solveur MILP PuLP).
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

WORKDIR /app

# ── Dépendances système ───────────────────────────────────────────────────────
# PuLP/CBC nécessite coinor-cbc pour résoudre les problèmes MILP.
# C'est le solveur open source inclus avec PuLP.
RUN apt-get update && apt-get install -y --no-install-recommends \
    coinor-cbc \
    && rm -rf /var/lib/apt/lists/*

# ── Dépendances Python ────────────────────────────────────────────────────────
COPY services/or_engine/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Code source ───────────────────────────────────────────────────────────────
COPY services/or_engine/  services/or_engine/
COPY services/ml_engine/  services/ml_engine/
COPY shared/              shared/

# ── Lancement ─────────────────────────────────────────────────────────────────
# solver.py : point d'entrée du moteur d'optimisation
# Lit les scores depuis data/outputs/, génère les tournées
CMD ["python", "-m", "services.or_engine.solver"]
