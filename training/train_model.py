"""
Simple Training Script - Creates Model Files for Step 5
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os
from datetime import datetime

# Create artifacts directory
os.makedirs('artifacts', exist_ok=True)

print("Creating sample training data...")

# Sample training data (you can expand this)
emails = [
    # Work emails
    ("Meeting tomorrow at 3 PM", "Work"),
    ("Please review the quarterly report", "Work"),
    ("Team standup in 10 minutes", "Work"),
    ("Project deadline Friday", "Work"),
    ("Budget approval needed", "Work"),
   
    # Personal emails
    ("Dinner with family tonight", "Personal"),
    ("Birthday party this weekend", "Personal"),
    ("Gym membership renewal", "Personal"),
    ("Doctor appointment reminder", "Personal"),
    ("Coffee with friends", "Personal"),
   
    # Spam emails
    ("You won $1,000,000", "Spam"),
    ("Click here to claim prize", "Spam"),
    ("Enlarge your muscles now", "Spam"),
    ("Hot singles in your area", "Spam"),
    ("Free iPhone click now", "Spam"),
   
    # Promotions
    ("50% OFF Sale Today Only", "Promotions"),
    ("Special discount for members", "Promotions"),
    ("Flash sale ends tonight", "Promotions"),
    ("Buy one get one free", "Promotions"),
    ("Limited time offer", "Promotions"),
   
    # Newsletter
    ("Weekly tech news roundup", "Newsletter"),
    ("Monthly industry updates", "Newsletter"),
    ("Subscribe to our newsletter", "Newsletter"),
    ("This week in technology", "Newsletter"),
    ("Daily digest: Top stories", "Newsletter"),
]

# Create DataFrame
df = pd.DataFrame(emails, columns=['text', 'category'])
print(f"Created {len(df)} sample emails")
print(f"Categories: {df['category'].unique()}")

# Split data
X = df['text']
y = df['category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining TF-IDF Vectorizer...")
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

print("Training Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_vectorized, y_train)

# Evaluate
accuracy = model.score(X_test_vectorized, y_test)
print(f"\nModel Accuracy: {accuracy:.2%}")

# Save model
print("\nSaving model files...")
joblib.dump(model, 'artifacts/best_model.pkl')
print("✓ Saved: artifacts/best_model.pkl")

# Save vectorizer
joblib.dump(vectorizer, 'artifacts/tfidf_vectorizer.pkl')
print("✓ Saved: artifacts/tfidf_vectorizer.pkl")

# Add metadata
model.metadata = {
    'version': '1.0.0',
    'algorithm': 'Random Forest',
    'accuracy': accuracy,
    'training_date': datetime.now(),
    'categories': list(model.classes_)
}

print("\n✅ Training complete!")
print(f"Model can classify: {list(model.classes_)}")