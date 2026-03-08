"""
Custom Exception Classes

Defines custom exceptions for the API with appropriate status codes
and error messages.
"""

from fastapi import HTTPException, status


class APIException(HTTPException):
    """Base exception for all API errors"""
    def __init__(
        self,
        status_code: int,
        detail: str,
        headers: dict = None
    ):
        super().__init__(status_code=status_code, detail=detail, headers=headers)


class AuthenticationError(APIException):
    """Raised when authentication fails"""
    def __init__(self, detail: str = "Invalid or missing API key"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "ApiKey"}
        )


class AuthorizationError(APIException):
    """Raised when user lacks permission"""
    def __init__(self, detail: str = "Insufficient permissions"):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail
        )


class ResourceNotFoundError(APIException):
    """Raised when requested resource is not found"""
    def __init__(self, resource: str = "Resource"):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{resource} not found"
        )


class ValidationError(APIException):
    """Raised when request validation fails"""
    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=detail
        )


class RateLimitError(APIException):
    """Raised when rate limit is exceeded"""
    def __init__(self, detail: str = "Rate limit exceeded"):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=detail,
            headers={"Retry-After": "60"}
        )


class ModelNotFoundError(APIException):
    """Raised when ML model is not found or not loaded"""
    def __init__(self, detail: str = "Model not found or not loaded"):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail
        )


class ProcessingError(APIException):
    """Raised when email processing fails"""
    def __init__(self, detail: str = "Error processing email"):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail
        )


class DatabaseError(APIException):
    """Raised when database operation fails"""
    def __init__(self, detail: str = "Database operation failed"):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail
        )


class BatchSizeError(APIException):
    """Raised when batch size exceeds limit"""
    def __init__(self, max_size: int = 100):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch size exceeds maximum allowed size of {max_size}"
        )
