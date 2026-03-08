"""
Authentication Tests

Tests for API key authentication and security.
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import app

client = TestClient(app)

VALID_API_KEY = "dev-api-key-12345"
INVALID_API_KEY = "invalid-key-99999"


def test_missing_api_key():
    """Test request without API key"""
    response = client.get("/predictions/")
    assert response.status_code == 401
    assert "API key" in response.json()["message"].lower()


def test_invalid_api_key():
    """Test request with invalid API key"""
    response = client.get(
        "/predictions/",
        headers={"X-API-Key": INVALID_API_KEY}
    )
    assert response.status_code == 401


def test_valid_api_key():
    """Test request with valid API key"""
    response = client.get(
        "/predictions/",
        headers={"X-API-Key": VALID_API_KEY}
    )
    assert response.status_code == 200


def test_public_endpoint_no_auth():
    """Test that public endpoints don't require auth"""
    # Health check should be public
    response = client.get("/health/")
    assert response.status_code == 200
    
    # Docs should be public
    response = client.get("/docs")
    assert response.status_code == 200


def test_case_sensitive_api_key():
    """Test that API key is case-sensitive"""
    response = client.get(
        "/predictions/",
        headers={"X-API-Key": VALID_API_KEY.upper()}
    )
    assert response.status_code == 401


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
