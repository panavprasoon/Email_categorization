"""
Prediction Service

Business logic for retrieving and analyzing predictions.
"""

from sqlalchemy.orm import Session
from typing import Optional, Dict, List, Any
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from database.models import Prediction, Feedback, Email
from api.models import PredictionHistoryResponse, StatisticsResponse


class PredictionService:
    """Service for prediction history operations"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_predictions(
        self,
        skip: int = 0,
        limit: int = 20,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get paginated predictions with optional filters.
        
        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return
            filters: Optional filters (category, min_confidence, date range)
            
        Returns:
            Dictionary with predictions and total count
        """
        query = self.db.query(Prediction).join(Email)
        
        # Apply filters
        if filters:
            if filters.get("category"):
                query = query.filter(Prediction.predicted_category == filters["category"])
            
            if filters.get("min_confidence"):
                query = query.filter(Prediction.confidence_score >= filters["min_confidence"])
            
            if filters.get("start_date"):
                query = query.filter(Prediction.created_at >= filters["start_date"])
            
            if filters.get("end_date"):
                query = query.filter(Prediction.created_at <= filters["end_date"])
        
        # Get total count
        total_count = query.count()
        
        # Get paginated results
        predictions = query.order_by(Prediction.created_at.desc()).offset(skip).limit(limit).all()
        
        # Convert to response models
        prediction_responses = []
        for pred in predictions:
            # Check if feedback exists
            feedback_exists = self.db.query(Feedback).filter(
                Feedback.prediction_id == pred.id
            ).first() is not None
            
            prediction_responses.append(
                PredictionHistoryResponse(
                    prediction_id=pred.id,
                    email_sender=pred.email.sender,
                    email_subject=pred.email.subject,
                    predicted_category=pred.predicted_category,
                    confidence=pred.confidence_score,
                    probabilities=pred.probabilities,
                    timestamp=pred.created_at,
                    feedback_provided=feedback_exists
                )
            )
        
        return {
            "predictions": prediction_responses,
            "total_count": total_count
        }
    
    def get_prediction_by_id(self, prediction_id: int) -> Optional[PredictionHistoryResponse]:
        """
        Get a specific prediction by ID.
        
        Args:
            prediction_id: Prediction ID
            
        Returns:
            PredictionHistoryResponse or None if not found
        """
        pred = self.db.query(Prediction).filter(
            Prediction.id == prediction_id
        ).first()
        
        if not pred:
            return None
        
        # Check if feedback exists
        feedback_exists = self.db.query(Feedback).filter(
            Feedback.prediction_id == pred.id
        ).first() is not None
        
        return PredictionHistoryResponse(
            prediction_id=pred.id,
            email_sender=pred.email.sender,
            email_subject=pred.email.subject,
            predicted_category=pred.predicted_category,
            confidence=pred.confidence_score,
            probabilities=pred.probabilities,
            timestamp=pred.created_at,
            feedback_provided=feedback_exists
        )
    
    def get_statistics(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> StatisticsResponse:
        """
        Get statistics for predictions in date range.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            StatisticsResponse
        """
        # Get all predictions in date range
        predictions = self.db.query(Prediction).filter(
            Prediction.created_at >= start_date,
            Prediction.created_at <= end_date
        ).all()
        
        total_predictions = len(predictions)
        
        # Count by category
        category_counts = {}
        total_confidence = 0
        
        for pred in predictions:
            category = pred.predicted_category
            category_counts[category] = category_counts.get(category, 0) + 1
            total_confidence += pred.confidence_score
        
        # Calculate average confidence
        avg_confidence = total_confidence / total_predictions if total_predictions > 0 else 0
        
        # Get feedback statistics
        feedbacks = self.db.query(Feedback).join(Prediction).filter(
            Prediction.created_at >= start_date,
            Prediction.created_at <= end_date
        ).all()
        
        feedback_count = len(feedbacks)
        
        # Calculate accuracy from feedback
        correct_count = sum(1 for f in feedbacks if f.feedback_type == "correct")
        accuracy = correct_count / feedback_count if feedback_count > 0 else None
        
        return StatisticsResponse(
            total_predictions=total_predictions,
            predictions_by_category=category_counts,
            average_confidence=round(avg_confidence, 4),
            feedback_count=feedback_count,
            accuracy_from_feedback=round(accuracy, 4) if accuracy else None,
            date_range={
                "start": start_date,
                "end": end_date
            }
        )
