
"""
Email Categorizer - Loads and uses the trained model
"""

import joblib
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class EmailCategorizer:
    """Categorizes emails using trained ML model"""
   
    def __init__(self, model_path=None, vectorizer_path=None, artifacts_dir=None):
        """
        Initialize EmailCategorizer
       
        Args:
            model_path: Direct path to model file
            vectorizer_path: Direct path to vectorizer file
            artifacts_dir: Directory containing both files (alternative to individual paths)
        """
        if artifacts_dir:
            # Using artifacts directory
            self.model_path = os.path.join(artifacts_dir, 'best_model.pkl')
            self.vectorizer_path = os.path.join(artifacts_dir, 'tfidf_vectorizer.pkl')
        elif model_path and vectorizer_path:
            # Using individual paths
            self.model_path = model_path
            self.vectorizer_path = vectorizer_path
        else:
            # Auto-resolve from active model registry first, then env/default.
            resolved_model_path = None
            resolved_vectorizer_path = None

            try:
                from training.registry import ModelRegistry

                active_model = ModelRegistry().get_active_model()
                if active_model is not None:
                    resolved_model_path = getattr(active_model, 'model_path', None)
                    resolved_vectorizer_path = getattr(active_model, 'vectorizer_path', None)
            except Exception:
                # Fall through to env/default path resolution
                pass

            if not resolved_model_path or not resolved_vectorizer_path:
                resolved_model_path = os.getenv('MODEL_PATH', 'artifacts/best_model.pkl')
                resolved_vectorizer_path = os.getenv('VECTORIZER_PATH', 'artifacts/tfidf_vectorizer.pkl')

            self.model_path = resolved_model_path
            self.vectorizer_path = resolved_vectorizer_path
       
        self.model = None
        self.vectorizer = None
        self.metadata = {}
        self.load_model()
   
    def load_model(self):
        """Load model and vectorizer from disk"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        if not os.path.exists(self.vectorizer_path):
            raise FileNotFoundError(f"Vectorizer file not found: {self.vectorizer_path}")
       
        self.model = joblib.load(self.model_path)
        self.vectorizer = joblib.load(self.vectorizer_path)
       
        # Load metadata if available
        if hasattr(self.model, 'metadata'):
            self.metadata = self.model.metadata
       
        print(f"✓ Model loaded from {self.model_path}")
        print(f"✓ Vectorizer loaded from {self.vectorizer_path}")
   
    def categorize(self, subject, body):
        """
        Categorize an email
       
        Args:
            subject: Email subject line
            body: Email body text
           
        Returns:
            dict with category, confidence, and probabilities
        """
        # Combine subject and body
        text = f"{subject} {body}"
       
        # Vectorize
        text_vectorized = self.vectorizer.transform([text])
       
        # Predict
        category = self.model.predict(text_vectorized)[0]
        probabilities = self.model.predict_proba(text_vectorized)[0]
       
        # Get confidence (probability of predicted class)
        category_index = list(self.model.classes_).index(category)
        confidence = probabilities[category_index]
       
        # Create probability dictionary
        prob_dict = {}
        for i, cat in enumerate(self.model.classes_):
            prob_dict[cat] = float(probabilities[i])
       
        return {
            'category': category,
            'confidence': float(confidence),
            'probabilities': prob_dict
        }
   
    def predict(self, text):
        """Alternative method name for compatibility"""
        return self.categorize("", text)


# Test function
if __name__ == "__main__":
    # Test the categorizer with direct paths
    categorizer = EmailCategorizer(
        model_path='../artifacts/best_model.pkl',
        vectorizer_path='../artifacts/tfidf_vectorizer.pkl'
    )
   
    test_emails = [
        ("Meeting Tomorrow", "Team meeting at 3 PM in conference room"),
        ("Dinner Plans", "Want to grab dinner with friends tonight?"),
        ("WINNER!!!", "You have won $1,000,000! Click here now!"),
        ("50% OFF Sale", "Limited time offer on all products"),
    ]
   
    print("\nTesting Email Categorizer:")
    print("=" * 60)
   
    for subject, body in test_emails:
        result = categorizer.categorize(subject, body)
        print(f"\nSubject: {subject}")
        print(f"Category: {result['category']}")
        print(f"Confidence: {result['confidence']:.2%}")
