"""
Database connection management for Neon cloud PostgreSQL
Provides session management and connection lifecycle
"""
from contextlib import contextmanager
from typing import Generator
import logging

from sqlalchemy import create_engine, event,text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError

from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create database engine with connection pooling for Neon
engine = create_engine(
    Config.get_database_url(),
    poolclass=QueuePool,
    pool_size=Config.DB_POOL_SIZE,
    max_overflow=Config.DB_MAX_OVERFLOW,
    pool_timeout=Config.DB_POOL_TIMEOUT,
    pool_pre_ping=True,  # Verify connections before using (important for cloud)
    echo=Config.ENVIRONMENT == 'development',  # Log SQL in development
    connect_args={
        'sslmode': Config.DB_SSLMODE,  # Required for Neon
        'connect_timeout': 10,  # 10 second timeout for cloud connections
    }
)

# Create session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Event listeners for connection pool monitoring
@event.listens_for(engine, "connect")
def receive_connect(dbapi_conn, connection_record):
    """Log new database connections"""
    logger.debug("New Neon database connection established")

@event.listens_for(engine, "close")
def receive_close(dbapi_conn, connection_record):
    """Log closed connections"""
    logger.debug("Neon database connection closed")

def init_database():
    """
    Initialize database connection to Neon
    Test connectivity and log connection info
    """
    try:
        logger.info("Connecting to Neon cloud database...")
        logger.info(f"Connection info: {Config.get_connection_info()}")
        
        with engine.connect() as conn:
            # Test query
            result = conn.execute(text("SELECT version();"))
            version = result.fetchone()[0]
            logger.info(f"✓ Connected successfully to Neon!")
            logger.info(f"PostgreSQL version: {version}")
            
            # Check tables exist
            result = conn.execute(
                text("SELECT COUNT(*) FROM information_schema.tables "
                "WHERE table_schema = 'public';")
            )
            table_count = result.fetchone()[0]
            logger.info(f"✓ Found {table_count} tables in database")
            
            logger.info(f"Connection pool: size={Config.DB_POOL_SIZE}, max_overflow={Config.DB_MAX_OVERFLOW}")
            return True
            
    except SQLAlchemyError as e:
        logger.error(f"❌ Failed to connect to Neon database!")
        logger.error(f"Error: {e}")
        logger.error("Please check:")
        logger.error("  1. Your .env file has correct Neon connection details")
        logger.error("  2. Your Neon project is not suspended (check dashboard)")
        logger.error("  3. Your internet connection is working")
        return False

@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Provide a transactional scope for database operations
    
    Usage:
        with get_db_session() as session:
            session.query(Email).all()
    
    Yields:
        Session: SQLAlchemy database session
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database session error: {e}")
        raise
    finally:
        session.close()

def get_session() -> Session:
    """
    Get a new database session (for dependency injection)
    Caller is responsible for closing the session
    
    Returns:
        Session: SQLAlchemy database session
    """
    return SessionLocal()

def close_database():
    """
    Dispose of the connection pool
    Call this on application shutdown
    """
    logger.info("Closing Neon database connection pool")
    engine.dispose()


class DatabaseConnection:
    """Simple wrapper to provide a consistent interface for registry and other components.

    This class exists for compatibility with older code that expected a
    DatabaseConnection object with a `get_session()` context manager.
    """

    def __init__(self):
        # No internal state required; connection pooling is managed by SQLAlchemy.
        pass

    def get_session(self):
        """Return a context manager yielding a SQLAlchemy Session."""
        return get_db_session()

    def test_connection(self) -> bool:
        """Validate database connectivity."""
        return init_database()
