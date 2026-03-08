
"""
FastAPI Main Application

This is the entry point for the Email Categorization API.
Initializes FastAPI app, configures middleware, and includes all routes.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.config import settings
from api.middleware.logging import RequestLoggingMiddleware
from api.middleware.error_handler import ErrorHandlerMiddleware
from api.routes import health, categorization, predictions, feedback, models

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format=settings.LOG_FORMAT,
    handlers=[
        logging.FileHandler(settings.LOG_FILE) if settings.LOG_FILE else logging.StreamHandler(),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=settings.APP_DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
if settings.CORS_ENABLED:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
        allow_methods=settings.CORS_ALLOW_METHODS,
        allow_headers=settings.CORS_ALLOW_HEADERS,
    )

# Add custom middleware
app.add_middleware(ErrorHandlerMiddleware)
app.add_middleware(RequestLoggingMiddleware)

# Include routers
app.include_router(health.router)
app.include_router(categorization.router)
app.include_router(predictions.router)
app.include_router(feedback.router)
app.include_router(models.router)

# Root endpoint
@app.get("/")
async def root():
    """
    Root endpoint - API information
    """
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "description": settings.APP_DESCRIPTION,
        "docs": "/docs",
        "health": "/health/"
    }


# Startup event
@app.on_event("startup")
async def startup_event():
    """
    Execute on application startup
    """
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Debug mode: {settings.DEBUG}")
    logger.info("API server is ready to accept requests")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """
    Execute on application shutdown
    """
    logger.info("Shutting down API server")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        workers=settings.WORKERS if not settings.RELOAD else 1
    )
