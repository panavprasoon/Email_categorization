"""
API Configuration Module

Manages all configuration settings for the FastAPI application including
database connections, authentication, rate limiting, and CORS settings.
"""

from pydantic import ConfigDict, field_validator
from pydantic_settings import BaseSettings
from typing import Any, List, Optional
from functools import lru_cache
import os
import json
import re
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore",
    )
    """
    Application settings loaded from environment variables.
    Uses Pydantic for validation and type checking.
    """
    
    # Application Settings
    APP_NAME: str = "Email Categorization API"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = "Production-ready API for email categorization using ML"
    DEBUG: bool = False
    ENVIRONMENT: str = "production"  # development, staging, production
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = False
    WORKERS: int = 4
    
    # Database Settings (from Step 1)
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    DB_POOL_TIMEOUT: int = 30
    
    # Authentication Settings
    API_KEY_HEADER_NAME: str = "X-API-Key"
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Valid API Keys (in production, store in database)
    VALID_API_KEYS: List[str] = [
        os.getenv("API_KEY_1", "dev-api-key-12345"),
        os.getenv("API_KEY_2", "test-api-key-67890"),
    ]
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_PERIOD: int = 60  # seconds
    
    # CORS Settings
    CORS_ENABLED: bool = True
    CORS_ORIGINS: Any = [
        "http://localhost:3000",
        "http://localhost:8000",
        "https://yourdomain.com",
    ]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["*"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]

    # Support comma-separated environment var (e.g., in .env) and JSON list values
    @field_validator("CORS_ORIGINS", mode="before")
    def _parse_cors_origins(cls, v):
        """Allow CORS_ORIGINS to be set via JSON array or delimited string."""
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return []
            # JSON array string (recommended)
            try:
                return json.loads(v)
            except Exception:
                # Fallback to delimiter-separated string (comma or semicolon)
                parts = [p.strip() for p in re.split(r"[;,]", v) if p.strip()]
                return parts
        return v
    
    # Logging Settings
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE: Optional[str] = "logs/api.log"
    
    # Model Settings
    MODEL_PATH: str = "artifacts/best_model.pkl"
    VECTORIZER_PATH: str = "artifacts/tfidf_vectorizer.pkl"
    MAX_EMAIL_LENGTH: int = 10000
    
    # Batch Processing Settings
    MAX_BATCH_SIZE: int = 100
    BATCH_TIMEOUT: int = 300  # seconds
    
    # Cache Settings
    CACHE_ENABLED: bool = True
    CACHE_TTL: int = 300  # seconds
    
    # Pagination Settings
    DEFAULT_PAGE_SIZE: int = 20
    MAX_PAGE_SIZE: int = 100
    


@lru_cache()
def get_settings() -> Settings:
    """
    Cached settings instance.
    Using lru_cache ensures settings are loaded only once.
    """
    return Settings()


# Global settings instance
settings = get_settings()
