"""
Logging Middleware

Logs all incoming requests and outgoing responses.
"""

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
import time
import logging
from typing import Callable

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to log all requests and responses.
    """
    
    async def dispatch(self, request: Request, call_next: Callable):
        """
        Log request details and response status.
        
        Args:
            request: Incoming request
            call_next: Next middleware/handler
            
        Returns:
            Response from next handler
        """
        # Generate request ID
        request_id = f"{time.time()}-{id(request)}"
        
        # Start timer
        start_time = time.time()
        
        # Log request
        logger.info(
            f"REQUEST [{request_id}] {request.method} {request.url.path} "
            f"from {request.client.host if request.client else 'unknown'}"
        )
        
        # Add request ID to state
        request.state.request_id = request_id
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate processing time
            process_time = (time.time() - start_time) * 1000  # ms
            
            # Add processing time header
            response.headers["X-Process-Time"] = f"{process_time:.2f}ms"
            response.headers["X-Request-ID"] = request_id
            
            # Log response
            logger.info(
                f"RESPONSE [{request_id}] {response.status_code} "
                f"in {process_time:.2f}ms"
            )
            
            return response
            
        except Exception as e:
            # Calculate processing time
            process_time = (time.time() - start_time) * 1000  # ms
            
            # Log error
            logger.error(
                f"ERROR [{request_id}] {str(e)} "
                f"after {process_time:.2f}ms",
                exc_info=True
            )
            
            raise
