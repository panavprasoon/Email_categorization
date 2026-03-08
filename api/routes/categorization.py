"""
Email Categorization Routes

Provides endpoints for categorizing emails (single and batch).
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
import time

from api.dependencies import get_db, verify_api_key, get_email_categorizer
from api.models import (
    EmailCategorizationRequest,
    EmailCategorizationResponse,
    BatchEmailCategorizationRequest,
    BatchEmailCategorizationResponse
)
from api.services.categorization_service import CategorizationService
from inference.categorizer import EmailCategorizer
from api.exceptions import BatchSizeError, ProcessingError

router = APIRouter(prefix="/categorize", tags=["Categorization"])


@router.post("/", response_model=EmailCategorizationResponse)
async def categorize_email(
    request: EmailCategorizationRequest,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key),
    categorizer: EmailCategorizer = Depends(get_email_categorizer)
):
    """
    Categorize a single email.
    
    This endpoint accepts an email (sender, subject, body) and returns
    the predicted category with confidence scores.
    
    **Authentication Required**: X-API-Key header
    
    **Rate Limit**: 100 requests per minute
    
    Returns:
        EmailCategorizationResponse with prediction results
    """
    service = CategorizationService(db, categorizer)
    
    try:
        start_time = time.time()
        
        result = service.categorize_single_email(
            sender=request.sender,
            subject=request.subject,
            body=request.body,
            timestamp=request.timestamp,
            metadata=request.metadata
        )
        
        processing_time = (time.time() - start_time) * 1000
        result.processing_time_ms = processing_time
        
        return result
        
    except Exception as e:
        raise ProcessingError(detail=f"Failed to categorize email: {str(e)}")


@router.post("/batch", response_model=BatchEmailCategorizationResponse)
async def categorize_batch(
    request: BatchEmailCategorizationRequest,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key),
    categorizer: EmailCategorizer = Depends(get_email_categorizer)
):
    """
    Categorize multiple emails in a batch.
    
    This endpoint accepts up to 100 emails and processes them efficiently
    in batch mode.
    
    **Authentication Required**: X-API-Key header
    
    **Rate Limit**: 100 requests per minute
    
    **Batch Size Limit**: 100 emails per request
    
    Returns:
        BatchEmailCategorizationResponse with all predictions
    """
    service = CategorizationService(db, categorizer)
    
    # Validate batch size
    if len(request.emails) > 100:
        raise BatchSizeError(max_size=100)
    
    try:
        start_time = time.time()
        
        results = service.categorize_batch_emails(request.emails)
        
        total_processing_time = (time.time() - start_time) * 1000
        
        # Count successful and failed
        successful = len(results)
        failed = len(request.emails) - successful
        
        return BatchEmailCategorizationResponse(
            predictions=results,
            total_count=len(request.emails),
            successful_count=successful,
            failed_count=failed,
            total_processing_time_ms=total_processing_time
        )
        
    except Exception as e:
        raise ProcessingError(detail=f"Failed to process batch: {str(e)}")


@router.post("/preview")
async def preview_categorization(
    request: EmailCategorizationRequest,
    categorizer: EmailCategorizer = Depends(get_email_categorizer)
):
    """
    Preview email categorization without saving to database.
    
    Useful for testing or UI preview before submitting.
    Does not require authentication.
    
    Returns:
        Category prediction with probabilities (not saved)
    """
    try:
        # Combine email text
        email_text = f"Subject: {request.subject}\n\n{request.body}"
        
        # Predict
        prediction = categorizer.categorize(
            subject=request.subject,
            body=request.body
        )
        
        return {
            "category": prediction["category"],
            "confidence": prediction["confidence"],
            "probabilities": prediction["probabilities"]
        }
        
    except Exception as e:
        raise ProcessingError(detail=f"Preview failed: {str(e)}")
