"""
Feedback Service

Business logic for collecting and managing user feedback on predictions.
"""

from sqlalchemy.orm import Session
from typing import Optional
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from database.models import Feedback, Prediction
from api.models import FeedbackRequest, FeedbackResponse
from api.exceptions import ResourceNotFoundError


class FeedbackService:
    """Service for feedback operations"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def submit_feedback(self, request: FeedbackRequest) -> FeedbackResponse:
        """
        Submit feedback for a prediction.
        
        Args:
            request: FeedbackRequest
            
        Returns:
            FeedbackResponse
            
        Raises:
            ResourceNotFoundError: If prediction not found
        """
        # Verify prediction exists
        prediction = self.db.query(Prediction).filter(
            Prediction.id == request.prediction_id
        ).first()
        
        if not prediction:
            raise ResourceNotFoundError(resource="Prediction")
        
        # Create feedback record
        feedback = Feedback(
            prediction_id=request.prediction_id,
            feedback_type=request.feedback_type.value,
            correct_category=request.correct_category.value if request.correct_category else None,
            comments=request.comments,
            created_at=datetime.utcnow()
        )
        
        self.db.add(feedback)
        self.db.commit()
        self.db.refresh(feedback)
        
        return FeedbackResponse(
            feedback_id=feedback.id,
            prediction_id=feedback.prediction_id,
            feedback_type=feedback.feedback_type,
            correct_category=feedback.correct_category,
            timestamp=feedback.created_at
        )
    
    def get_feedback_for_prediction(self, prediction_id: int) -> Optional[Feedback]:
        """
        Get feedback for a specific prediction.
        
        Args:
            prediction_id: Prediction ID
            
        Returns:
            Feedback record or None
        """
        return self.db.query(Feedback).filter(
            Feedback.prediction_id == prediction_id
        ).first()
