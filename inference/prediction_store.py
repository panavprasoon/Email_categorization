# This module saves predictions to the database.

"""
Prediction storage and database integration.

This module provides functionality to save predictions to the database,
linking them with emails and tracking metadata.
"""

import os
import sys
from datetime import datetime
from typing import Dict, Optional, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from database.connection import DatabaseConnection
from database.models import Prediction, InferenceMetadata
from database.repository import PredictionRepository


class PredictionStore:
    """
    Store predictions in database.
    
    Example:
        >>> store = PredictionStore()
        >>> store.save_prediction(
        ...     email_id=12345,
        ...     category='incident',
        ...     confidence=0.945,
        ...     model_id=1
        ... )
    """
    
    def __init__(self):
        """Initialize prediction store."""
        self.db = DatabaseConnection()
        self.repo = PredictionRepository()
    
    def save_prediction(
        self,
        email_id: int,
        category: str,
        confidence: float,
        model_id: int,
        all_probabilities: Optional[Dict[str, float]] = None,
        inference_time_ms: Optional[float] = None
    ) -> int:
        """
        Save prediction to database.
        
        Args:
            email_id: ID of email in emails table
            category: Predicted category
            confidence: Prediction confidence (0.0-1.0)
            model_id: ID of model used
            all_probabilities: Dictionary of all category probabilities (optional)
            inference_time_ms: Inference time in milliseconds (optional)
            
        Returns:
            Prediction ID from database
        
        Example:
            >>> prediction_id = store.save_prediction(
            ...     email_id=100,
            ...     category='meeting',
            ...     confidence=0.89,
            ...     model_id=1
            ... )
            >>> print(f"Saved as prediction ID: {prediction_id}")
        """
        # Create prediction record
        with self.db.get_session() as session:
            prediction = self.repo.create(
                session=session,
                email_id=email_id,
                model_version_id=model_id,
                predicted_label=category,
                confidence=confidence,
                prediction_probabilities=all_probabilities or {},
                processing_time_ms=inference_time_ms or 0.0
            )
            prediction_id = prediction.id

        # Optionally save inference metadata
        if inference_time_ms is not None or all_probabilities is not None:
            self._save_metadata(
                inference_time_ms=inference_time_ms,
                all_probabilities=all_probabilities
            )

        return prediction_id
    
    def _save_metadata(
        self,
        inference_time_ms: Optional[float] = None,
        all_probabilities: Optional[Dict[str, float]] = None
    ) -> None:
        """Save inference metadata.

        This stores a lightweight record of inference activity for monitoring.
        The current schema is designed for daily aggregates, so we insert a record
        per prediction for now (one row per inference) and rely on downstream
        aggregation when needed.
        """
        from datetime import date

        with self.db.get_session() as session:
            metadata = InferenceMetadata(
                date=date.today(),
                total_predictions=1,
                category_distribution=all_probabilities or {},
                avg_confidence=None,
                low_confidence_count=None,
                created_at=datetime.now()
            )
            session.add(metadata)
            session.commit()
    
    def get_prediction(self, prediction_id: int) -> Optional[Prediction]:
        """Get prediction by ID.

        Args:
            prediction_id: Prediction ID

        Returns:
            Prediction object or None
        """
        with self.db.get_session() as session:
            prediction = self.repo.get_by_id(session, prediction_id)
            if prediction is None:
                return None

            # Force load all core attributes before closing session so the
            # returned object can be accessed outside the session.
            # This avoids DetachedInstanceError when consumers access properties.
            _ = prediction.predicted_label
            _ = prediction.confidence
            _ = prediction.email_id
            _ = prediction.model_version_id
            session.expunge(prediction)

            return prediction
    
    def get_predictions_for_email(self, email_id: int) -> list:
        """Get all predictions for an email."""
        with self.db.get_session() as session:
            return self.repo.get_by_email(session, email_id)
    
    def get_recent_predictions(self, limit: int = 100) -> list:
        """
        Get recent predictions.
        
        Args:
            limit: Maximum number to return
            
        Returns:
            List of Prediction objects
        """
        with self.db.get_session() as session:
            from sqlalchemy import text
            result = session.execute(
                text("""
                    SELECT * FROM predictions 
                    ORDER BY predicted_at DESC 
                    LIMIT :limit
                """),
                {"limit": limit}
            ).fetchall()
            
            return result