from inference import EmailCategorizer
from inference.prediction_store import PredictionStore
from database.repository import EmailRepository
from database.connection import DatabaseConnection

# Create an email record in the database
email_text = 'Server down need help'
with DatabaseConnection().get_session() as session:
    email = EmailRepository.create(session, email_text=email_text)
    session.commit()
    email_id = email.id

# Predict category
categorizer = EmailCategorizer()
result = categorizer.predict(email_text)

# Save prediction
store = PredictionStore()
prediction_id = store.save_prediction(
    email_id=email_id,
    category=result['category'],
    confidence=result['confidence'],
    model_id=result['model_id'],
    all_probabilities=result.get('all_probabilities'),
    inference_time_ms=result.get('prediction_time_ms')
)

print(f'Saved prediction ID: {prediction_id}')
