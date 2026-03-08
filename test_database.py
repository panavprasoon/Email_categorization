"""
Test script to verify Neon database setup
Run with: python test_database.py
"""
from datetime import datetime
import logging

from database import init_database, get_db_session
from database.models import Email, ModelVersion, Prediction, Feedback
from database.repository import (
    EmailRepository,
    ModelVersionRepository,
    PredictionRepository,
    FeedbackRepository
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_connection():
    """Test 1: Neon database connectivity"""
    logger.info("=" * 60)
    logger.info("TEST 1: Neon Database Connection")
    logger.info("=" * 60)
    success = init_database()
    assert success, "Neon database connection failed"
    logger.info("✓ Neon database connection successful\n")

def test_email_operations():
    """Test 2: Email CRUD operations"""
    logger.info("=" * 60)
    logger.info("TEST 2: Email Operations")
    logger.info("=" * 60)
    
    with get_db_session() as session:
        # Create email
        email = EmailRepository.create(
            session,
            email_text="This is a test email about a work meeting tomorrow."
        )
        logger.info(f"✓ Created email with ID: {email.id}")
        
        # Retrieve email
        retrieved = EmailRepository.get_by_id(session, email.id)
        assert retrieved is not None
        assert retrieved.email_text == email.email_text
        logger.info(f"✓ Retrieved email: {retrieved}\n")

def test_model_version_operations():
    """Test 3: Model version operations"""
    logger.info("=" * 60)
    logger.info("TEST 3: Model Version Operations")
    logger.info("=" * 60)
    
    with get_db_session() as session:
        # Ensure a test model version exists; update if present to avoid foreign key issues
        existing = session.query(ModelVersion).filter(ModelVersion.version == "v1.0.0-test").first()
        if existing:
            # update metrics in-place so the test can run repeatedly
            existing.accuracy = 0.85
            existing.precision_score = 0.83
            existing.recall_score = 0.87
            existing.f1_score = 0.85
            existing.model_path = "/models/v1.0.0.pkl"
            existing.vectorizer_path = "/models/v1.0.0_vectorizer.pkl"
            existing.training_samples = 1000
            existing.training_metrics = {"confusion_matrix": [[50, 10], [5, 35]]}
            model = existing
        else:
            model = ModelVersionRepository.create(
                session,
                version="v1.0.0-test",
                accuracy=0.85,
                precision_score=0.83,
                recall_score=0.87,
                f1_score=0.85,
                model_path="/models/v1.0.0.pkl",
                vectorizer_path="/models/v1.0.0_vectorizer.pkl",
                training_samples=1000,
                training_metrics={"confusion_matrix": [[50, 10], [5, 35]]}
            )
        logger.info(f"✓ Created/updated model version: {model.version}")
        
        # Activate model
        success = ModelVersionRepository.activate(session, "v1.0.0-test")
        assert success
        logger.info("✓ Activated model version")
        
        # Get active model
        active = ModelVersionRepository.get_active(session)
        assert active is not None
        assert active.version == "v1.0.0-test"
        assert active.is_active == True
        logger.info(f"✓ Retrieved active model: {active.version}\n")
        
        return model

def test_prediction_operations():
    """Test 4: Prediction operations"""
    logger.info("=" * 60)
    logger.info("TEST 4: Prediction Operations")
    logger.info("=" * 60)
    
    with get_db_session() as session:
        # Get test email and model
        email = EmailRepository.get_recent(session, limit=1)[0]
        model = ModelVersionRepository.get_active(session)
        
        # Create prediction
        prediction = PredictionRepository.create(
            session,
            email_id=email.id,
            model_version_id=model.id,
            predicted_label="Work",
            confidence=0.87,
            prediction_probabilities={"Work": 0.87, "Personal": 0.10, "Spam": 0.03},
            processing_time_ms=45.2
        )
        logger.info(f"✓ Created prediction: {prediction}")
        
        # Retrieve prediction
        retrieved = PredictionRepository.get_by_id(session, prediction.id)
        assert retrieved is not None
        logger.info(f"✓ Retrieved prediction with confidence: {retrieved.confidence}\n")
        
        return prediction

def test_feedback_operations():
    """Test 5: Feedback operations"""
    logger.info("=" * 60)
    logger.info("TEST 5: Feedback Operations")
    logger.info("=" * 60)
    
    with get_db_session() as session:
        # Get test prediction
        prediction = PredictionRepository.get_recent(session, limit=1)[0]
        
        # Create feedback
        feedback = FeedbackRepository.create(
            session,
            prediction_id=prediction.id,
            corrected_label="Personal",
            user_id="test_user",
            feedback_source="manual"
        )
        logger.info(f"✓ Created feedback: {feedback}")
        
        # Calculate correction rate
        correction_rate = PredictionRepository.get_correction_rate(session)
        logger.info(f"✓ Correction rate: {correction_rate:.2f}%\n")

def test_relationships():
    """Test 6: ORM relationships"""
    logger.info("=" * 60)
    logger.info("TEST 6: ORM Relationships")
    logger.info("=" * 60)
    
    with get_db_session() as session:
        # Get email with predictions
        email = EmailRepository.get_recent(session, limit=1)[0]
        logger.info(f"Email ID {email.id} has {len(email.predictions)} prediction(s)")
        
        # Get prediction with email and model
        prediction = PredictionRepository.get_recent(session, limit=1)[0]
        logger.info(f"Prediction {prediction.id}:")
        logger.info(f"  - Email text: {prediction.email.email_text[:50]}...")
        logger.info(f"  - Model version: {prediction.model_version.version}")
        if prediction.feedback:
            logger.info(f"  - Corrected to: {prediction.feedback.corrected_label}")
        logger.info("✓ Relationships working correctly\n")

def cleanup():
    """Test 7: Cleanup test data"""
    logger.info("=" * 60)
    logger.info("TEST 7: Cleanup")
    logger.info("=" * 60)
    
    with get_db_session() as session:
        # Delete test model version
        session.query(Feedback).delete()
        
        # 2. Delete predictions (references model_versions and emails)
        session.query(Prediction).delete()
        
        # 3. Now we can delete model version
        session.query(ModelVersion).filter(ModelVersion.version == "v1.0.0-test").delete()
        
        # 4. Delete test email
        session.query(Email).delete()
        
        logger.info("✓ Cleaned up test data\n")

def main():
    """Run all tests"""
    try:
        logger.info("\n" + "=" * 60)
        logger.info("STARTING NEON DATABASE TESTS")
        logger.info("=" * 60 + "\n")
        
        test_connection()
        test_email_operations()
        test_model_version_operations()
        test_prediction_operations()
        test_feedback_operations()
        test_relationships()
        cleanup()
        
        logger.info("=" * 60)
        logger.info("ALL TESTS PASSED ✓")
        logger.info("Your Neon database is ready to use!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"TEST FAILED: {e}", exc_info=True)
        logger.error("\nPlease check:")
        logger.error("  1. Your .env file has correct Neon credentials")
        logger.error("  2. Schema was created successfully in Neon")
        logger.error("  3. Your internet connection is working")
        raise

if __name__ == "__main__":
    main()
