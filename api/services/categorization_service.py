"""
Categorization Service

Business logic for email categorization operations.
"""

from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from datetime import datetime
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from inference.categorizer import EmailCategorizer
from database.models import Email, Prediction
from api.models import EmailCategorizationRequest, EmailCategorizationResponse


class CategorizationService:
    """Service for email categorization operations"""
    
    def __init__(self, db: Session, categorizer: EmailCategorizer):
        self.db = db
        self.categorizer = categorizer
    
    def categorize_single_email(
        self,
        sender: str,
        subject: str,
        body: str,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> EmailCategorizationResponse:
        """
        Categorize a single email and store the prediction.
        
        Args:
            sender: Email sender
            subject: Email subject
            body: Email body
            timestamp: Email timestamp
            metadata: Additional metadata
            
        Returns:
            EmailCategorizationResponse
        """
        start_time = time.time()
        
        # Create email record
        email = Email(
            sender=sender,
            recipient="",  # Not provided in request
            subject=subject,
            body=body,
            received_at=timestamp or datetime.utcnow()
        )
        self.db.add(email)
        self.db.commit()
        self.db.refresh(email)
        
        # Predict category
        prediction = self.categorizer.categorize(subject=subject, body=body)
        
        # Store prediction
        pred_record = Prediction(
            email_id=email.id,
            predicted_category=prediction["category"],
            confidence_score=prediction["confidence"],
            probabilities=prediction["probabilities"],
            model_version="1.0.0",
            created_at=datetime.utcnow()
        )
        self.db.add(pred_record)
        self.db.commit()
        self.db.refresh(pred_record)
        
        processing_time = (time.time() - start_time) * 1000
        
        return EmailCategorizationResponse(
            prediction_id=pred_record.id,
            category=prediction["category"],
            confidence=prediction["confidence"],
            probabilities=prediction["probabilities"],
            processing_time_ms=processing_time,
            timestamp=timestamp or datetime.utcnow()
        )
    
    def categorize_batch_emails(
        self,
        emails: List[EmailCategorizationRequest]
    ) -> List[EmailCategorizationResponse]:
        """
        Categorize multiple emails in batch.
        
        Args:
            emails: List of email requests
            
        Returns:
            List of EmailCategorizationResponse
        """
        results = []
        
        for email in emails:
            try:
                result = self.categorize_single_email(
                    sender=email.sender,
                    subject=email.subject,
                    body=email.body,
                    timestamp=email.timestamp,
                    metadata=email.metadata
                )
                results.append(result)
            except Exception as e:
                # Log error but continue processing
                print(f"Error processing email: {str(e)}")
                continue
        
        return results