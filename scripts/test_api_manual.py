"""
Manual API Testing Script

Interactive script to test API endpoints manually.
"""

import requests
import json
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:8000"
API_KEY = "dev-api-key-12345"

headers = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}


def print_response(response, title="Response"):
    """Pretty print API response"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"Status Code: {response.status_code}")
    print(f"Headers: {dict(response.headers)}")
    print(f"\nBody:")
    try:
        print(json.dumps(response.json(), indent=2))
    except:
        print(response.text)
    print(f"{'='*60}\n")


def test_health():
    """Test health check endpoint"""
    print("\n🏥 Testing Health Check...")
    response = requests.get(f"{BASE_URL}/health/")
    print_response(response, "Health Check")


def test_categorize_single():
    """Test single email categorization"""
    print("\n📧 Testing Single Email Categorization...")
    
    email = {
        "sender": "boss@company.com",
        "subject": "Urgent: Q4 Report Due Tomorrow",
        "body": "Hi team, I need everyone to submit their Q4 reports by tomorrow EOD. This is critical for the board meeting next week. Please ensure all data is accurate and complete."
    }
    
    response = requests.post(
        f"{BASE_URL}/categorize/",
        headers=headers,
        json=email
    )
    print_response(response, "Single Email Categorization")
    
    return response.json().get("prediction_id") if response.status_code == 200 else None


def test_batch_categorize():
    """Test batch email categorization"""
    print("\n📨 Testing Batch Email Categorization...")
    
    batch = {
        "emails": [
            {
                "sender": "newsletter@techcrunch.com",
                "subject": "This Week in Tech",
                "body": "Check out the top technology stories from this week, including AI breakthroughs and startup news."
            },
            {
                "sender": "friend@gmail.com",
                "subject": "Coffee tomorrow?",
                "body": "Hey! Want to grab coffee tomorrow morning? I'm free after 10am."
            },
            {
                "sender": "noreply@promotions.com",
                "subject": "50% OFF Everything! Limited Time Only!",
                "body": "Don't miss out on our biggest sale of the year! Use code SAVE50 at checkout."
            }
        ]
    }
    
    response = requests.post(
        f"{BASE_URL}/categorize/batch",
        headers=headers,
        json=batch
    )
    print_response(response, "Batch Categorization")


def test_preview():
    """Test preview categorization"""
    print("\n👀 Testing Preview Categorization (No Auth)...")
    
    email = {
        "sender": "spam@suspicious.com",
        "subject": "You've won $1,000,000!",
        "body": "Congratulations! Click here to claim your prize now!"
    }
    
    # Note: Preview doesn't require API key
    response = requests.post(
        f"{BASE_URL}/categorize/preview",
        json=email
    )
    print_response(response, "Preview Categorization")


def test_list_predictions():
    """Test listing predictions"""
    print("\n📋 Testing List Predictions...")
    
    params = {
        "page": 1,
        "page_size": 5
    }
    
    response = requests.get(
        f"{BASE_URL}/predictions/",
        headers=headers,
        params=params
    )
    print_response(response, "List Predictions")


def test_statistics():
    """Test getting statistics"""
    print("\n📊 Testing Statistics...")
    
    params = {"days": 7}
    
    response = requests.get(
        f"{BASE_URL}/predictions/statistics/overview",
        headers=headers,
        params=params
    )
    print_response(response, "Statistics")


def test_feedback(prediction_id):
    """Test submitting feedback"""
    if not prediction_id:
        print("\n⚠️ Skipping feedback test (no prediction ID)")
        return
    
    print(f"\n💬 Testing Feedback for Prediction {prediction_id}...")
    
    feedback = {
        "prediction_id": prediction_id,
        "feedback_type": "correct",
        "comments": "Great prediction!"
    }
    
    response = requests.post(
        f"{BASE_URL}/feedback/",
        headers=headers,
        json=feedback
    )
    print_response(response, "Submit Feedback")


def test_model_info():
    """Test getting model information"""
    print("\n🤖 Testing Model Info...")
    
    response = requests.get(
        f"{BASE_URL}/models/info",
        headers=headers
    )
    print_response(response, "Model Info")


def test_invalid_auth():
    """Test invalid authentication"""
    print("\n🔒 Testing Invalid Authentication...")
    
    bad_headers = {"X-API-Key": "invalid-key"}
    
    response = requests.get(
        f"{BASE_URL}/predictions/",
        headers=bad_headers
    )
    print_response(response, "Invalid Auth (Expected 401)")


def run_all_tests():
    """Run all manual tests"""
    print("\n" + "="*60)
    print("EMAIL CATEGORIZATION API - MANUAL TESTING")
    print("="*60)
    print(f"Base URL: {BASE_URL}")
    print(f"API Key: {API_KEY}")
    print(f"Time: {datetime.now()}")
    
    try:
        # Run tests
        test_health()
        prediction_id = test_categorize_single()
        test_batch_categorize()
        test_preview()
        test_list_predictions()
        test_statistics()
        test_feedback(prediction_id)
        test_model_info()
        test_invalid_auth()
        
        print("\n✅ All tests completed!")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to API server")
        print("Make sure the API is running: uvicorn api.main:app --reload")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")


if __name__ == "__main__":
    run_all_tests()
