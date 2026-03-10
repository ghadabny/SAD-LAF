import sys
from pathlib import Path

# Ajoute la racine du projet au sys.path
# Permet d'importer services.ml_engine, shared, etc.
root = Path(__file__).parent
sys.path.insert(0, str(root))