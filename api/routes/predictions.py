"""
Prediction History Routes

Provides endpoints for retrieving prediction history and statistics.
"""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import Optional
from datetime import datetime, timedelta

from api.dependencies import get_db, verify_api_key, get_pagination_params
from api.models import (
    PredictionListResponse,
    PredictionHistoryResponse,
    StatisticsResponse
)
from api.services.prediction_service import PredictionService
from api.exceptions import ResourceNotFoundError

router = APIRouter(prefix="/predictions", tags=["Predictions"])


@router.get("/", response_model=PredictionListResponse)
async def list_predictions(
    category: Optional[str] = Query(None, description="Filter by category"),
    min_confidence: Optional[float] = Query(None, ge=0, le=1, description="Minimum confidence"),
    start_date: Optional[datetime] = Query(None, description="Start date filter"),
    end_date: Optional[datetime] = Query(None, description="End date filter"),
    pagination: dict = Depends(get_pagination_params),
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """
    Get paginated list of predictions with optional filters.
    
    **Query Parameters**:
    - category: Filter by predicted category
    - min_confidence: Only return predictions above this confidence
    - start_date: Filter predictions after this date
    - end_date: Filter predictions before this date
    - page: Page number (default: 1)
    - page_size: Items per page (default: 20, max: 100)
    
    **Authentication Required**: X-API-Key header
    
    Returns:
        Paginated list of predictions
    """
    service = PredictionService(db)
    
    filters = {
        "category": category,
        "min_confidence": min_confidence,
        "start_date": start_date,
        "end_date": end_date
    }
    
    result = service.get_predictions(
        skip=pagination["skip"],
        limit=pagination["limit"],
        filters=filters
    )
    
    # Calculate total pages
    total_pages = (result["total_count"] + pagination["page_size"] - 1) // pagination["page_size"]
    
    return PredictionListResponse(
        predictions=result["predictions"],
        total_count=result["total_count"],
        page=pagination["page"],
        page_size=pagination["page_size"],
        total_pages=total_pages
    )


@router.get("/{prediction_id}", response_model=PredictionHistoryResponse)
async def get_prediction(
    prediction_id: int,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """
    Get details of a specific prediction by ID.
    
    **Authentication Required**: X-API-Key header
    
    Returns:
        Prediction details
        
    Raises:
        404: If prediction not found
    """
    service = PredictionService(db)
    
    prediction = service.get_prediction_by_id(prediction_id)
    
    if not prediction:
        raise ResourceNotFoundError(resource="Prediction")
    
    return prediction


@router.get("/statistics/overview", response_model=StatisticsResponse)
async def get_statistics(
    days: int = Query(7, ge=1, le=365, description="Number of days to analyze"),
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """
    Get prediction statistics for the specified time period.
    
    **Query Parameters**:
    - days: Number of days to include in statistics (default: 7, max: 365)
    
    **Authentication Required**: X-API-Key header
    
    Returns:
        Statistics including counts, accuracy, and trends
    """
    service = PredictionService(db)
    
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    
    stats = service.get_statistics(start_date, end_date)
    
    return stats
