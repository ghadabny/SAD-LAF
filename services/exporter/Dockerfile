# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — service exporter
#
# Service batch : lancé après ml_engine.
# Lit les scores produits par ml_engine dans data/outputs/,
# les formate et les exporte vers Dataverse (ou fichier JSON local en POC).
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

WORKDIR /app

# ── Dépendances ───────────────────────────────────────────────────────────────
COPY services/exporter/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Code source ───────────────────────────────────────────────────────────────
# L'exporter a besoin de :
#   - services/exporter/  → export.py, formatter.py
#   - shared/             → config (chemins), schemas (format JSON)
COPY services/exporter/  services/exporter/
COPY shared/             shared/

# ── Lancement ─────────────────────────────────────────────────────────────────
# TODO : sera implémenté avec export.py
CMD ["python", "-m", "services.exporter.export"]
