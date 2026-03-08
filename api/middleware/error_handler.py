"""
Error Handling Middleware

Catches and formats all exceptions into consistent error responses.
"""

from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Callable
import logging
from datetime import datetime

from api.exceptions import APIException

logger = logging.getLogger(__name__)


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """
    Middleware to handle all exceptions and return formatted error responses.
    """
    
    async def dispatch(self, request: Request, call_next: Callable):
        """
        Catch exceptions and return formatted error responses.
        
        Args:
            request: Incoming request
            call_next: Next middleware/handler
            
        Returns:
            Response or error response
        """
        try:
            response = await call_next(request)
            return response
            
        except APIException as e:
            # Our custom exceptions
            logger.warning(f"API Exception: {e.detail}")
            return JSONResponse(
                status_code=e.status_code,
                content={
                    "error": e.__class__.__name__,
                    "message": e.detail,
                    "timestamp": datetime.utcnow().isoformat()
                },
                headers=e.headers
            )
            
        except ValueError as e:
            # Validation errors
            logger.warning(f"Validation error: {str(e)}")
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content={
                    "error": "ValidationError",
                    "message": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            # Unexpected errors
            logger.error(f"Unexpected error: {str(e)}", exc_info=True)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": "InternalServerError",
                    "message": "An unexpected error occurred",
                    "detail": str(e) if logger.level == logging.DEBUG else None,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
