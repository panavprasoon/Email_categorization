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
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from inference.categorizer import EmailCategorizer
from database.models import Email, Prediction, ModelVersion
from database.repository import InferenceMetadataRepository, ModelVersionRepository
from api.models import EmailCategorizationRequest, EmailCategorizationResponse
from api.monitoring import prediction_counter, model_confidence, batch_prediction_counter

logger = logging.getLogger(__name__)


class CategorizationService:
    """Service for email categorization operations"""
    
    def __init__(self, db: Session, categorizer: EmailCategorizer):
        self.db = db
        self.categorizer = categorizer
        self.model_repo = ModelVersionRepository()
        self.metadata_repo = InferenceMetadataRepository()

    @staticmethod
    def _build_email_text(sender: str, subject: str, body: str) -> str:
        """Store email content in a structured text block compatible with current schema."""
        return f"Sender: {sender}\nSubject: {subject}\n\n{body}"

    def _get_model_version(self) -> ModelVersion:
        """Get the active model version, or fall back to the most recent registered model."""
        model_version = self.model_repo.get_active(self.db)
        if model_version is None:
            model_version = self.db.query(ModelVersion).order_by(ModelVersion.created_at.desc()).first()

        if model_version is None:
            raise ValueError("No model version is registered in the database")

        return model_version
    
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

        try:
            # Create email record using current schema
            email = Email(
                email_text=self._build_email_text(sender=sender, subject=subject, body=body),
                created_at=timestamp or datetime.utcnow()
            )
            self.db.add(email)
            self.db.flush()

            # Predict category
            prediction = self.categorizer.categorize(subject=subject, body=body)
            processing_time = (time.time() - start_time) * 1000
            model_version = self._get_model_version()

            # Store prediction using the actual schema
            pred_record = Prediction(
                email_id=email.id,
                model_version_id=model_version.id,
                predicted_label=prediction["category"],
                confidence=prediction["confidence"],
                prediction_probabilities=prediction["probabilities"],
                processing_time_ms=processing_time,
                created_at=timestamp or datetime.utcnow()
            )
            self.db.add(pred_record)

            # Update daily BI aggregates
            self.metadata_repo.upsert_prediction_aggregate(
                session=self.db,
                prediction_date=(timestamp or datetime.utcnow()).date(),
                predicted_label=prediction["category"],
                confidence=prediction["confidence"]
            )

            self.db.commit()
            self.db.refresh(pred_record)

            # Track Prometheus metrics
            prediction_counter.labels(category=prediction["category"]).inc()
            model_confidence.observe(prediction["confidence"])

            return EmailCategorizationResponse(
                prediction_id=pred_record.id,
                category=prediction["category"],
                confidence=prediction["confidence"],
                probabilities=prediction["probabilities"],
                processing_time_ms=processing_time,
                timestamp=pred_record.created_at
            )
        except Exception:
            self.db.rollback()
            raise
    
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
                logger.error("Error processing email in batch: %s", e)
                continue
            finally:
                batch_prediction_counter.inc()
        
        return results