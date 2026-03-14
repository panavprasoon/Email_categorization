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
from api.monitoring import feedback_counter


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

        if request.feedback_type.value == "correct":
            return FeedbackResponse(
                feedback_id=None,
                prediction_id=request.prediction_id,
                feedback_type=request.feedback_type.value,
                correct_category=prediction.predicted_label,
                timestamp=datetime.utcnow(),
                message="Prediction marked correct; no correction record created"
            )

        existing_feedback = self.db.query(Feedback).filter(
            Feedback.prediction_id == request.prediction_id
        ).first()

        if existing_feedback:
            return FeedbackResponse(
                feedback_id=existing_feedback.id,
                prediction_id=existing_feedback.prediction_id,
                feedback_type=existing_feedback.feedback_source,
                correct_category=existing_feedback.corrected_label,
                timestamp=existing_feedback.created_at,
                message="Feedback already exists for this prediction"
            )
        
        # Create correction record using current schema
        feedback = Feedback(
            prediction_id=request.prediction_id,
            corrected_label=request.correct_category.value if request.correct_category else prediction.predicted_label,
            feedback_source=request.feedback_type.value,
            created_at=datetime.utcnow()
        )
        
        self.db.add(feedback)
        self.db.commit()
        self.db.refresh(feedback)

        # Track metric
        feedback_counter.labels(feedback_type=request.feedback_type.value).inc()

        return FeedbackResponse(
            feedback_id=feedback.id,
            prediction_id=feedback.prediction_id,
            feedback_type=feedback.feedback_source,
            correct_category=feedback.corrected_label,
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
