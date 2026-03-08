"""
Authentication Middleware

Handles API key authentication and token validation.
"""

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import time
from typing import Callable
import logging

from api.config import settings

logger = logging.getLogger(__name__)


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """
    Middleware to handle authentication for protected endpoints.
    """
    
    # Endpoints that don't require authentication
    PUBLIC_ENDPOINTS = [
        "/",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/health",
        "/health/ready",
        "/health/live",
    ]
    
    async def dispatch(self, request: Request, call_next: Callable):
        """
        Process each request for authentication.
        
        Args:
            request: Incoming request
            call_next: Next middleware/handler
            
        Returns:
            Response from next handler or error response
        """
        # Check if endpoint is public
        if any(request.url.path.startswith(endpoint) for endpoint in self.PUBLIC_ENDPOINTS):
            return await call_next(request)
        
        # Get API key from header
        api_key = request.headers.get("X-API-Key")
        
        if not api_key:
            logger.warning(f"Request to {request.url.path} missing API key")
            return JSONResponse(
                status_code=401,
                content={
                    "error": "AuthenticationError",
                    "message": "Missing API key",
                    "detail": "API key required in X-API-Key header"
                }
            )
        
        # Validate API key
        if api_key not in settings.VALID_API_KEYS:
            logger.warning(f"Invalid API key attempted: {api_key[:8]}...")
            return JSONResponse(
                status_code=401,
                content={
                    "error": "AuthenticationError",
                    "message": "Invalid API key",
                    "detail": "The provided API key is not valid"
                }
            )
        
        # Add API key to request state for later use
        request.state.api_key = api_key
        
        # Process request
        response = await call_next(request)
        return response


class APIKeyRateLimiter:
    """
    Simple in-memory rate limiter for API keys.
    In production, use Redis for distributed rate limiting.
    """
    
    def __init__(self, requests_per_minute: int = 100):
        self.requests_per_minute = requests_per_minute
        self.requests = {}  # {api_key: [(timestamp, timestamp, ...)]}
    
    def is_allowed(self, api_key: str) -> bool:
        """
        Check if request is allowed based on rate limit.
        
        Args:
            api_key: API key to check
            
        Returns:
            True if allowed, False if rate limited
        """
        current_time = time.time()
        cutoff_time = current_time - 60  # 1 minute ago
        
        # Initialize if first request from this key
        if api_key not in self.requests:
            self.requests[api_key] = []
        
        # Remove old requests outside the time window
        self.requests[api_key] = [
            ts for ts in self.requests[api_key]
            if ts > cutoff_time
        ]
        
        # Check if under limit
        if len(self.requests[api_key]) >= self.requests_per_minute:
            return False
        
        # Add current request
        self.requests[api_key].append(current_time)
        return True


# Global rate limiter instance
rate_limiter = APIKeyRateLimiter(requests_per_minute=settings.RATE_LIMIT_REQUESTS)
