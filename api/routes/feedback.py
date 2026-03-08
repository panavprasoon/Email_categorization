"""
Feedback Routes

Provides endpoints for submitting and retrieving feedback on predictions.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from api.dependencies import get_db, verify_api_key
from api.models import FeedbackRequest, FeedbackResponse
from api.services.feedback_service import FeedbackService

router = APIRouter(prefix="/feedback", tags=["Feedback"])


@router.post("/", response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackRequest,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """
    Submit feedback for a prediction.
    
    Allows users to provide feedback on prediction accuracy.
    This feedback can be used for model retraining and improvement.
    
    **Authentication Required**: X-API-Key header
    
    Returns:
        Feedback confirmation
    """
    service = FeedbackService(db)
    
    return service.submit_feedback(request)


@router.get("/{prediction_id}")
async def get_feedback(
    prediction_id: int,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """
    Get feedback for a specific prediction.
    
    **Authentication Required**: X-API-Key header
    
    Returns:
        Feedback details or None if no feedback exists
    """
    service = FeedbackService(db)
    
    feedback = service.get_feedback_for_prediction(prediction_id)
    
    if not feedback:
        return {"message": "No feedback found for this prediction"}
    
    return feedback
