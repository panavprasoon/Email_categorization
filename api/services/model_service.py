"""
Model Service

Business logic for model information and management.
"""

from sqlalchemy.orm import Session
from typing import Optional, List
from datetime import datetime
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from inference.categorizer import EmailCategorizer
from database.models import Prediction
from api.models import ModelInfoResponse


class ModelService:
    """Service for model management operations"""
    
    def __init__(self, db: Session, categorizer: EmailCategorizer):
        self.db = db
        self.categorizer = categorizer
    
    def get_model_info(self) -> ModelInfoResponse:
        """
        Get information about the current model.
        
        Returns:
            ModelInfoResponse
        """
        # Get total prediction count
        total_predictions = self.db.query(Prediction).count() if self.db else 0
        
        # Get model metadata if available
        model_metadata = getattr(self.categorizer, 'metadata', {})
        
        # Get categories
        categories = []
        if hasattr(self.categorizer, 'model') and self.categorizer.model:
            if hasattr(self.categorizer.model, 'classes_'):
                categories = list(self.categorizer.model.classes_)
        
        return ModelInfoResponse(
            model_name="Email Categorizer",
            version=model_metadata.get("version", "1.0.0"),
            algorithm=model_metadata.get("algorithm", "Random Forest"),
            accuracy=model_metadata.get("accuracy"),
            training_date=model_metadata.get("training_date"),
            total_predictions=total_predictions,
            status="active",
            categories=categories if categories else ["Work", "Personal", "Spam", "Promotions", "Social"]
        )
    
    def get_categories(self) -> List[str]:
        """
        Get list of available categories.
        
        Returns:
            List of category names
        """
        if hasattr(self.categorizer, 'model') and self.categorizer.model:
            if hasattr(self.categorizer.model, 'classes_'):
                return list(self.categorizer.model.classes_)
        return ["Work", "Personal", "Spam", "Promotions", "Social"]
