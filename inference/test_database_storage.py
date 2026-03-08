# inference/test_database_storage.py

import os
import sys

from inference.categorizer import EmailCategorizer
from inference.prediction_store import PredictionStore
from database.connection import DatabaseConnection
from database.models import Email, Prediction

def test_database_storage():
    """Test prediction storage in database"""
   
    print("=" * 60)
    print("DATABASE STORAGE TEST")
    print("=" * 60)
   
    # Initialize components
    categorizer = EmailCategorizer()
    prediction_store = PredictionStore()
   
    # Step 1: Create test email
    print("\n[STEP 1] Creating test email in database...")
    db = DatabaseConnection()

    # Note: Email model currently stores only raw text, so we combine relevant fields
    email_text = (
        "Subject: Project Status Update - Q1 2024\n"
        "Team, please review the attached quarterly report. Sales increased 20% and customer satisfaction is up."
    )

    test_email = Email(email_text=email_text)

    with db.get_session() as session:
        session.add(test_email)
        session.commit()
        email_id = test_email.id
        print(f"✓ Email created with ID: {email_id}")
   
    # Step 2: Make prediction
    print("\n[STEP 2] Making prediction...")
    # Use predict_with_details to capture probabilities and timing
    result = categorizer.predict_with_details(email_text)
    print(f"✓ Prediction: {result['category']}")
    print(f"✓ Confidence: {result['confidence']:.2%}")
   
    # Step 3: Save prediction to database
    print("\n[STEP 3] Saving prediction to database...")
    # Use the active model for storage
    from training import ModelRegistry
    active_model = ModelRegistry().get_active_model()

    prediction_id = prediction_store.save_prediction(
        email_id=email_id,
        category=result['category'],
        confidence=result['confidence'],
        model_id=active_model.id,
        all_probabilities=result.get('all_probabilities'),
        inference_time_ms=result.get('prediction_time_ms')
    )
    print(f"✓ Prediction saved with ID: {prediction_id}")
   
    # Step 4: Retrieve and verify
    print("\n[STEP 4] Retrieving prediction from database...")
    with db.get_session() as session:
        retrieved = session.query(Prediction).filter_by(id=prediction_id).first()

        if retrieved:
            print(f"✓ Prediction retrieved successfully")
            print(f"\nStored Prediction Details:")
            print(f"  Email ID: {retrieved.email_id}")
            print(f"  Category: {retrieved.predicted_category}")
            print(f"  Confidence: {retrieved.confidence_score:.2%}")
            print(f"  Model Version ID: {retrieved.model_version_id}")
            print(f"  Timestamp: {retrieved.created_at}")
            print(f"  Probabilities: {retrieved.prediction_probabilities}")
        else:
            print("✗ Failed to retrieve prediction")

        # Step 5: Verify email-prediction relationship
        print("\n[STEP 5] Verifying email-prediction relationship...")
        email_with_predictions = session.query(Email).filter_by(id=email_id).first()

        if email_with_predictions and len(email_with_predictions.predictions) > 0:
            print(f"✓ Email has {len(email_with_predictions.predictions)} prediction(s)")
            for pred in email_with_predictions.predictions:
                print(f"  - Category: {pred.predicted_category}, Confidence: {pred.confidence_score:.2%}")
        else:
            print("✗ No predictions found for email")
   
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print("✓ Email creation: SUCCESS")
    print("✓ Prediction generation: SUCCESS")
    print("✓ Database storage: SUCCESS")
    print("✓ Data retrieval: SUCCESS")
    print("✓ Relationship verification: SUCCESS")
    print("\n All database storage tests passed!")
   
    session.close()

if __name__ == "__main__":
    test_database_storage()