"""
API Integration Tests

Comprehensive tests for all API endpoints using pytest.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import app
from api.dependencies import get_db
from database.models import Base

# Test database URL
TEST_DATABASE_URL = "sqlite:///./test.db"

# Create test engine
engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    """Override database dependency for testing"""
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


# Override dependency
app.dependency_overrides[get_db] = override_get_db

# Create test client
client = TestClient(app)

# Test API key
TEST_API_KEY = "dev-api-key-12345"


@pytest.fixture(scope="module")
def setup_database():
    """Setup test database"""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


def test_health_check(setup_database):
    """Test health check endpoint"""
    response = client.get("/health/")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "version" in data


def test_liveness_probe(setup_database):
    """Test liveness probe"""
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json()["status"] == "alive"


def test_categorize_single_email(setup_database):
    """Test single email categorization"""
    email_data = {
        "sender": "boss@company.com",
        "subject": "Urgent: Project deadline",
        "body": "We need to complete the project by Friday. Please prioritize this."
    }
    
    response = client.post(
        "/categorize/",
        json=email_data,
        headers={"X-API-Key": TEST_API_KEY}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "prediction_id" in data
    assert "category" in data
    assert "confidence" in data
    assert "probabilities" in data
    assert 0 <= data["confidence"] <= 1


def test_categorize_without_auth(setup_database):
    """Test categorization without authentication"""
    email_data = {
        "sender": "test@example.com",
        "subject": "Test",
        "body": "Test email"
    }
    
    response = client.post("/categorize/", json=email_data)
    assert response.status_code == 401


def test_categorize_invalid_api_key(setup_database):
    """Test categorization with invalid API key"""
    email_data = {
        "sender": "test@example.com",
        "subject": "Test",
        "body": "Test email"
    }
    
    response = client.post(
        "/categorize/",
        json=email_data,
        headers={"X-API-Key": "invalid-key"}
    )
    assert response.status_code == 401


def test_categorize_invalid_email(setup_database):
    """Test categorization with invalid email data"""
    email_data = {
        "sender": "test@example.com",
        "subject": "",  # Empty subject
        "body": "Test"
    }
    
    response = client.post(
        "/categorize/",
        json=email_data,
        headers={"X-API-Key": TEST_API_KEY}
    )
    assert response.status_code == 422  # Validation error


def test_batch_categorization(setup_database):
    """Test batch email categorization"""
    batch_data = {
        "emails": [
            {
                "sender": "newsletter@company.com",
                "subject": "Weekly Newsletter",
                "body": "Check out this week's top articles and updates."
            },
            {
                "sender": "friend@gmail.com",
                "subject": "Dinner tonight?",
                "body": "Hey! Want to grab dinner tonight? Let me know!"
            }
        ]
    }
    
    response = client.post(
        "/categorize/batch",
        json=batch_data,
        headers={"X-API-Key": TEST_API_KEY}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert "total_count" in data
    assert len(data["predictions"]) == 2
    assert data["total_count"] == 2


def test_preview_categorization(setup_database):
    """Test preview categorization (no auth required)"""
    email_data = {
        "sender": "test@example.com",
        "subject": "Test Email",
        "body": "This is a test email for preview."
    }
    
    response = client.post("/categorize/preview", json=email_data)
    assert response.status_code == 200
    data = response.json()
    assert "category" in data
    assert "confidence" in data
    assert "probabilities" in data


def test_list_predictions(setup_database):
    """Test listing predictions"""
    # First create a prediction
    email_data = {
        "sender": "test@example.com",
        "subject": "Test",
        "body": "Test email for listing"
    }
    client.post(
        "/categorize/",
        json=email_data,
        headers={"X-API-Key": TEST_API_KEY}
    )
    
    # Now list predictions
    response = client.get(
        "/predictions/",
        headers={"X-API-Key": TEST_API_KEY}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert "total_count" in data
    assert data["total_count"] > 0


def test_get_statistics(setup_database):
    """Test getting statistics"""
    response = client.get(
        "/predictions/statistics/overview?days=7",
        headers={"X-API-Key": TEST_API_KEY}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "total_predictions" in data
    assert "predictions_by_category" in data
    assert "average_confidence" in data


def test_submit_feedback(setup_database):
    """Test submitting feedback"""
    # First create a prediction
    email_data = {
        "sender": "test@example.com",
        "subject": "Test",
        "body": "Test email for feedback"
    }
    response = client.post(
        "/categorize/",
        json=email_data,
        headers={"X-API-Key": TEST_API_KEY}
    )
    prediction_id = response.json()["prediction_id"]
    
    # Submit feedback
    feedback_data = {
        "prediction_id": prediction_id,
        "feedback_type": "correct",
        "comments": "Good prediction!"
    }
    
    response = client.post(
        "/feedback/",
        json=feedback_data,
        headers={"X-API-Key": TEST_API_KEY}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "feedback_id" in data
    assert data["prediction_id"] == prediction_id


def test_model_info(setup_database):
    """Test getting model information"""
    response = client.get(
        "/models/info",
        headers={"X-API-Key": TEST_API_KEY}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "version" in data
    assert "status" in data


def test_pagination(setup_database):
    """Test pagination parameters"""
    response = client.get(
        "/predictions/?page=1&page_size=10",
        headers={"X-API-Key": TEST_API_KEY}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["page"] == 1
    assert data["page_size"] == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
