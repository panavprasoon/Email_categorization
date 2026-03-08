"""
Model Management Routes

Provides endpoints for model information and management.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from api.dependencies import get_db, verify_api_key, get_email_categorizer, reload_model
from api.models import ModelInfoResponse
from api.services.model_service import ModelService
from inference.categorizer import EmailCategorizer

router = APIRouter(prefix="/models", tags=["Models"])


@router.get("/info", response_model=ModelInfoResponse)
async def get_model_info(
    db: Session = Depends(get_db),
    categorizer: EmailCategorizer = Depends(get_email_categorizer),
    api_key: str = Depends(verify_api_key)
):
    """
    Get information about the current model.
    
    **Authentication Required**: X-API-Key header
    
    Returns:
        Model metadata including version, algorithm, and performance metrics
    """
    service = ModelService(db, categorizer)
    
    return service.get_model_info()


@router.get("/categories")
async def get_categories(
    categorizer: EmailCategorizer = Depends(get_email_categorizer),
    api_key: str = Depends(verify_api_key)
):
    """
    Get list of available categories.
    
    **Authentication Required**: X-API-Key header
    
    Returns:
        List of category names
    """
    service = ModelService(None, categorizer)
    
    return {"categories": service.get_categories()}


@router.post("/reload")
async def reload_current_model(
    api_key: str = Depends(verify_api_key)
):
    """
    Reload the model (useful after retraining).
    
    **Authentication Required**: X-API-Key header
    
    Returns:
        Success message
    """
    reload_model()
    
    return {"message": "Model reloaded successfully"}
