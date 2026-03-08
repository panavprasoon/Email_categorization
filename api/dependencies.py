"""
FastAPI Dependencies

Provides dependency injection for database sessions, authentication,
and other shared resources.
"""

from fastapi import Depends, Header
from sqlalchemy.orm import Session
from typing import Optional
import sys
import os

# Add parent directory to path to import from Steps 1-4
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.connection import DatabaseConnection
from api.config import get_settings, Settings
from api.exceptions import AuthenticationError, ModelNotFoundError
from inference.categorizer import EmailCategorizer


# ============================================================================
# DATABASE DEPENDENCY
# ============================================================================

def get_db() -> Session:
    """Dependency to get database session.

    Automatically closes session after request.
    """
    db_conn = DatabaseConnection()
    # get_session() returns a context manager that yields a Session
    with db_conn.get_session() as db:
        yield db


# ============================================================================
# AUTHENTICATION DEPENDENCY
# ============================================================================

def verify_api_key(
    x_api_key: str = Header(..., alias="X-API-Key"),
    settings: Settings = Depends(get_settings)
) -> str:
    """
    Dependency to verify API key from request header.
    
    Args:
        x_api_key: API key from X-API-Key header
        settings: Application settings
        
    Returns:
        The validated API key
        
    Raises:
        AuthenticationError: If API key is invalid
    """
    if x_api_key not in settings.VALID_API_KEYS:
        raise AuthenticationError(detail="Invalid API key")
    
    return x_api_key


def optional_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    settings: Settings = Depends(get_settings)
) -> Optional[str]:
    """
    Optional API key dependency for public endpoints.
    
    Args:
        x_api_key: Optional API key from header
        settings: Application settings
        
    Returns:
        The API key if valid, None otherwise
    """
    if x_api_key and x_api_key in settings.VALID_API_KEYS:
        return x_api_key
    return None


# ============================================================================
# MODEL DEPENDENCY
# ============================================================================

# Global model instance (lazy loaded)
_email_categorizer: Optional[EmailCategorizer] = None


def get_email_categorizer(
    settings: Settings = Depends(get_settings)
) -> EmailCategorizer:
    """
    Dependency to get EmailCategorizer instance.
    Lazy loads the model on first request.
    
    Args:
        settings: Application settings
        
    Returns:
        EmailCategorizer instance
        
    Raises:
        ModelNotFoundError: If model cannot be loaded
    """
    global _email_categorizer
    
    if _email_categorizer is None:
        try:
            # Use settings for model paths
            model_path = settings.MODEL_PATH
            vectorizer_path = settings.VECTORIZER_PATH
           
            # If paths are relative, make them absolute from project root
            if not os.path.isabs(model_path):
                model_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    model_path
                )
            if not os.path.isabs(vectorizer_path):
                vectorizer_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    vectorizer_path
                )
           
            _email_categorizer = EmailCategorizer(
                model_path=model_path,
                vectorizer_path=vectorizer_path
            )
        except Exception as e:
            raise ModelNotFoundError(detail=f"Failed to load model: {str(e)}")
   
    return _email_categorizer


def reload_model():
    """
    Reload the email categorizer model.
    Useful after model retraining.
    """
    global _email_categorizer
    _email_categorizer = None


# ============================================================================
# PAGINATION DEPENDENCY
# ============================================================================

def get_pagination_params(
    page: int = 1,
    page_size: int = 20,
    settings: Settings = Depends(get_settings)
) -> dict:
    """
    Dependency for pagination parameters.
    
    Args:
        page: Page number (1-indexed)
        page_size: Number of items per page
        settings: Application settings
        
    Returns:
        Dictionary with skip and limit values
    """
    # Validate parameters
    page = max(1, page)
    page_size = min(page_size, settings.MAX_PAGE_SIZE)
    page_size = max(1, page_size)
    
    # Calculate skip and limit
    skip = (page - 1) * page_size
    
    return {
        "skip": skip,
        "limit": page_size,
        "page": page,
        "page_size": page_size
    }
