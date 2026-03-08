"""
Pydantic Models

Defines all request and response models for API validation and documentation.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


# ============================================================================
# ENUMS
# ============================================================================

class EmailCategory(str, Enum):
    """Email category options"""
    SPAM = "Spam"
    PROMOTIONS = "Promotions"
    WORK = "Work"
    PERSONAL = "Personal"
    SOCIAL = "Social"
    URGENT = "Urgent"
    NEWSLETTER = "Newsletter"


class FeedbackType(str, Enum):
    """Feedback type options"""
    CORRECT = "correct"
    INCORRECT = "incorrect"
    PARTIALLY_CORRECT = "partially_correct"


class ModelStatus(str, Enum):
    """Model status options"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    TRAINING = "training"
    DEPRECATED = "deprecated"


# ============================================================================
# REQUEST MODELS
# ============================================================================

class EmailCategorizationRequest(BaseModel):
    """Request model for single email categorization"""
    
    sender: str = Field(
        ...,
        description="Email sender address",
        examples=["john.doe@example.com"]
    )
    subject: str = Field(
        ...,
        description="Email subject line",
        max_length=500,
        examples=["Meeting tomorrow at 3 PM"]
    )
    body: str = Field(
        ...,
        description="Email body content",
        max_length=10000,
        examples=["Hi team, just a reminder about our meeting tomorrow..."]
    )
    timestamp: Optional[datetime] = Field(
        default=None,
        description="Email timestamp (defaults to current time)"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata"
    )
    
    @validator('body')
    def validate_body(cls, v):
        if not v or len(v.strip()) < 5:
            raise ValueError('Email body must be at least 5 characters')
        return v
    
    @validator('subject')
    def validate_subject(cls, v):
        if not v or len(v.strip()) < 1:
            raise ValueError('Subject cannot be empty')
        return v


class BatchEmailCategorizationRequest(BaseModel):
    """Request model for batch email categorization"""
    
    emails: List[EmailCategorizationRequest] = Field(
        ...,
        description="List of emails to categorize",
        max_length=100
    )
    
    @validator('emails')
    def validate_batch_size(cls, v):
        if len(v) < 1:
            raise ValueError('Batch must contain at least 1 email')
        if len(v) > 100:
            raise ValueError('Batch size cannot exceed 100 emails')
        return v


class FeedbackRequest(BaseModel):
    """Request model for prediction feedback"""
    
    prediction_id: int = Field(
        ...,
        description="ID of the prediction to provide feedback for",
        gt=0
    )
    feedback_type: FeedbackType = Field(
        ...,
        description="Type of feedback"
    )
    correct_category: Optional[EmailCategory] = Field(
        default=None,
        description="Correct category if prediction was wrong"
    )
    comments: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Additional comments"
    )
    
    @validator('correct_category')
    def validate_correct_category(cls, v, values):
        if values.get('feedback_type') == FeedbackType.INCORRECT and not v:
            raise ValueError('correct_category required when feedback_type is incorrect')
        return v


# ============================================================================
# RESPONSE MODELS
# ============================================================================

class EmailCategorizationResponse(BaseModel):
    """Response model for email categorization"""
    
    prediction_id: int = Field(..., description="Unique prediction ID")
    category: str = Field(..., description="Predicted category")
    confidence: float = Field(..., description="Prediction confidence (0-1)", ge=0, le=1)
    probabilities: Dict[str, float] = Field(..., description="Probabilities for all categories")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    timestamp: datetime = Field(..., description="Prediction timestamp")


class BatchEmailCategorizationResponse(BaseModel):
    """Response model for batch email categorization"""
    
    predictions: List[EmailCategorizationResponse] = Field(
        ...,
        description="List of predictions"
    )
    total_count: int = Field(..., description="Total number of emails processed")
    successful_count: int = Field(..., description="Number of successful predictions")
    failed_count: int = Field(..., description="Number of failed predictions")
    total_processing_time_ms: float = Field(..., description="Total processing time")


class PredictionHistoryResponse(BaseModel):
    """Response model for prediction history"""
    
    prediction_id: int
    email_sender: str
    email_subject: str
    predicted_category: str
    confidence: float
    probabilities: Dict[str, float]
    timestamp: datetime
    feedback_provided: bool
    
    class Config:
        from_attributes = True


class PredictionListResponse(BaseModel):
    """Response model for paginated prediction list"""
    
    predictions: List[PredictionHistoryResponse]
    total_count: int
    page: int
    page_size: int
    total_pages: int


class FeedbackResponse(BaseModel):
    """Response model for feedback submission"""
    
    feedback_id: int
    prediction_id: int
    feedback_type: str
    correct_category: Optional[str]
    timestamp: datetime
    message: str = "Feedback recorded successfully"


class ModelInfoResponse(BaseModel):
    """Response model for model information"""
    
    model_name: str
    version: str
    algorithm: str
    accuracy: Optional[float]
    training_date: Optional[datetime]
    total_predictions: int
    status: str
    categories: List[str]


class HealthCheckResponse(BaseModel):
    """Response model for health check"""
    
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Current timestamp")
    version: str = Field(..., description="API version")
    database: str = Field(..., description="Database connection status")
    model: str = Field(..., description="Model loading status")


class ErrorResponse(BaseModel):
    """Response model for errors"""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(..., description="Error timestamp")


class StatisticsResponse(BaseModel):
    """Response model for statistics"""
    
    total_predictions: int
    predictions_by_category: Dict[str, int]
    average_confidence: float
    feedback_count: int
    accuracy_from_feedback: Optional[float]
    date_range: Dict[str, datetime]
