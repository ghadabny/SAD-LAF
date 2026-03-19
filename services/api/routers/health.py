from fastapi import APIRouter

from shared.schemas import HealthSchema

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthSchema,
    summary="Vérification de l'état de l'API",
)
def health() -> HealthSchema:
    """
    Retourne le statut de l'API.
    Utilisé par Docker et les outils de monitoring pour vérifier
    que le service est opérationnel.
    """
    return HealthSchema(status="ok", version="0.1.0")