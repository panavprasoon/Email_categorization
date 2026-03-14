"""
Configuration management for Email Categorization System (Neon Cloud)
Loads environment variables and provides type-safe configuration
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

class Config:
    """Application configuration for Neon cloud database"""
    
    # Neon Database settings
    DB_HOST = os.getenv('DB_HOST', '')
    DB_PORT = int(os.getenv('DB_PORT', 5432))
    DB_NAME = os.getenv('DB_NAME', 'neondb')
    DB_USER = os.getenv('DB_USER', '')
    DB_PASSWORD = os.getenv('DB_PASSWORD', '')
    DB_SSLMODE = os.getenv('DB_SSLMODE', 'require')
    DATABASE_URL = os.getenv('DATABASE_URL', '')

    # Connection pool settings
    DB_POOL_SIZE = int(os.getenv('DB_POOL_SIZE', 5))
    DB_MAX_OVERFLOW = int(os.getenv('DB_MAX_OVERFLOW', 10))
    DB_POOL_TIMEOUT = int(os.getenv('DB_POOL_TIMEOUT', 30))
    
    # Environment
    ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
    
    @classmethod
    def get_database_url(cls) -> str:
        """
        Construct PostgreSQL connection URL for Neon
        Includes SSL mode requirement
        """
        if cls.DATABASE_URL:
            return cls.DATABASE_URL

        if not all([cls.DB_HOST, cls.DB_USER, cls.DB_PASSWORD, cls.DB_NAME]):
            return 'sqlite:///./email_system_test.db'

        return (
            f"postgresql://{cls.DB_USER}:{cls.DB_PASSWORD}"
            f"@{cls.DB_HOST}:{cls.DB_PORT}/{cls.DB_NAME}"
            f"?sslmode={cls.DB_SSLMODE}"
        )
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        if cls.DATABASE_URL:
            return

        required = ['DB_HOST', 'DB_PASSWORD', 'DB_NAME', 'DB_USER']
        missing = [key for key in required if not getattr(cls, key)]
        if missing:
            raise ValueError(
                f"Missing required configuration: {', '.join(missing)}\n"
                f"Please check your .env file and ensure all Neon connection details are set."
            )
        
        # Validate host looks like Neon hostname
        if cls.DB_HOST and not ('neon.tech' in cls.DB_HOST or 'localhost' in cls.DB_HOST):
            print(f"Warning: DB_HOST '{cls.DB_HOST}' doesn't look like a Neon hostname")
    
    @classmethod
    def get_connection_info(cls):
        """Display connection info (without password)"""
        return {
            'host': cls.DB_HOST,
            'port': cls.DB_PORT,
            'database': cls.DB_NAME,
            'user': cls.DB_USER,
            'ssl_mode': cls.DB_SSLMODE,
            'environment': cls.ENVIRONMENT,
            'database_url_configured': bool(cls.DATABASE_URL)
        }

