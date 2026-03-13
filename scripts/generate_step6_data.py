"""Generate synthetic Step 6 analytics data.

This script exercises the FastAPI app through TestClient so that:
- predictions are created
- feedback corrections are created
- audit logs are created
- inference metadata is updated

Usage:
    python scripts/generate_step6_data.py --emails 120 --target-feedback 25
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

from api.config import settings
from api.main import app
from database.connection import DatabaseConnection
from database.models import Prediction, Feedback, AuditLog, InferenceMetadata


CATEGORIES: Dict[str, List[Tuple[str, str, str]]] = {
    "Work": [
        (
            "manager@company.com",
            "Quarterly KPI review",
            "Please review the quarterly KPI dashboard and join the leadership meeting tomorrow at 10 AM.",
        ),
        (
            "hr@company.com",
            "Project deadline update",
            "The project timeline has shifted. Please send the revised status report before end of day.",
        ),
        (
            "ops@company.com",
            "Client escalation",
            "A client issue needs immediate investigation. Share the root cause analysis by afternoon.",
        ),
    ],
    "Personal": [
        (
            "friend@gmail.com",
            "Dinner this weekend?",
            "Hey, are you free for dinner on Saturday evening? Let me know what time works for you.",
        ),
        (
            "cousin@yahoo.com",
            "Family trip plans",
            "We are finalizing the family trip itinerary. Please confirm your travel dates.",
        ),
        (
            "neighbor@mail.com",
            "Need a small favor",
            "Could you collect my package tomorrow if it arrives before I get home?",
        ),
    ],
    "Spam": [
        (
            "winner@claim-now.biz",
            "You won $1,000,000",
            "Congratulations! Click here right now to claim your cash prize before it expires.",
        ),
        (
            "crypto@fastprofit.co",
            "Guaranteed returns today",
            "Double your money instantly with this secret trading trick. Limited offer for selected users only.",
        ),
        (
            "lottery@jackpot.xyz",
            "Urgent prize notification",
            "Your account has been selected for a grand prize. Submit bank details immediately.",
        ),
    ],
    "Promotions": [
        (
            "deals@shopnow.com",
            "50% off this week",
            "Huge sale on all electronics this week only. Use code SAVE50 at checkout.",
        ),
        (
            "offers@brandstore.com",
            "Flash sale ends tonight",
            "Exclusive limited-time promotion on premium products. Shop before midnight.",
        ),
        (
            "rewards@marketplace.com",
            "Special member discount",
            "As a valued member, you get an extra discount on your next purchase.",
        ),
    ],
    "Newsletter": [
        (
            "newsletter@techdaily.com",
            "Weekly product roundup",
            "Here is your weekly digest covering product launches, AI trends, and startup funding news.",
        ),
        (
            "updates@insights.org",
            "Monthly industry digest",
            "This month\'s newsletter covers market analysis, policy updates, and expert commentary.",
        ),
        (
            "briefing@mediahub.net",
            "Daily briefing",
            "Top stories and curated reads for today, including tech, business, and culture highlights.",
        ),
    ],
}


API_PATHS = [
    "/health/",
    "/predictions/?page=1&page_size=5",
    "/predictions/statistics/overview?days=30",
    "/models/info",
    "/models/categories",
]


def build_email_samples(email_count: int) -> List[Tuple[str, str, str, str]]:
    """Build a list of synthetic emails with expected categories."""
    category_names = list(CATEGORIES.keys())
    samples: List[Tuple[str, str, str, str]] = []

    for index in range(email_count):
        category = category_names[index % len(category_names)]
        sender, subject, body = random.choice(CATEGORIES[category])
        samples.append((category, sender, subject, body))

    random.shuffle(samples)
    return samples


def alternate_category(predicted: str) -> str:
    """Pick a different category for demo feedback when needed."""
    for category in CATEGORIES.keys():
        if category != predicted:
            return category
    return predicted


def count_rows() -> Dict[str, int]:
    """Return current BI-relevant row counts."""
    db = DatabaseConnection()
    with db.get_session() as session:
        return {
            "predictions": session.query(Prediction).count(),
            "feedback": session.query(Feedback).count(),
            "audit_logs": session.query(AuditLog).count(),
            "inference_metadata": session.query(InferenceMetadata).count(),
        }


def warm_up_api(client: TestClient, headers: Dict[str, str]) -> None:
    """Generate non-prediction API traffic for operations metrics."""
    for path in API_PATHS:
        client.get(path, headers=headers)

    # Intentionally create a couple of auth failures for error metrics.
    client.get("/predictions/", headers={"X-API-Key": "invalid-key"})
    client.get("/models/info", headers={"X-API-Key": "invalid-key"})


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Step 6 analytics data")
    parser.add_argument("--emails", type=int, default=120, help="Number of categorization requests to create")
    parser.add_argument("--target-feedback", type=int, default=25, help="Target number of correction records to create")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible demo data")
    args = parser.parse_args()

    random.seed(args.seed)
    client = TestClient(app)
    headers = {
        "X-API-Key": settings.VALID_API_KEYS[0],
        "Content-Type": "application/json",
    }

    before = count_rows()
    warm_up_api(client, headers)

    created_predictions = 0
    created_feedback = 0
    predicted_counter: Counter[str] = Counter()
    corrected_counter: Counter[str] = Counter()

    samples = build_email_samples(args.emails)

    for expected_category, sender, subject, body in samples:
        response = client.post(
            "/categorize/",
            headers=headers,
            json={
                "sender": sender,
                "subject": subject,
                "body": body,
            },
        )

        if response.status_code != 200:
            print(f"[WARN] categorize failed: {response.status_code} -> {response.text}")
            continue

        created_predictions += 1
        payload = response.json()
        prediction_id = payload["prediction_id"]
        predicted = payload["category"]
        predicted_counter[predicted] += 1

        should_correct = created_feedback < args.target_feedback and (
            predicted != expected_category or random.random() < 0.25
        )
        if should_correct:
            corrected_label = expected_category if expected_category != predicted else alternate_category(predicted)
            feedback_response = client.post(
                "/feedback/",
                headers=headers,
                json={
                    "prediction_id": prediction_id,
                    "feedback_type": "incorrect",
                    "correct_category": corrected_label,
                },
            )
            if feedback_response.status_code == 200:
                created_feedback += 1
                corrected_counter[corrected_label] += 1
            else:
                print(f"[WARN] feedback failed: {feedback_response.status_code} -> {feedback_response.text}")

        if created_predictions % 20 == 0:
            client.get("/predictions/?page=1&page_size=10", headers=headers)
            client.get("/health/", headers=headers)

    after = count_rows()

    print("\n" + "=" * 72)
    print("STEP 6 DATA GENERATION COMPLETE")
    print("=" * 72)
    print(f"Predictions created this run : {created_predictions}")
    print(f"Feedback created this run    : {created_feedback}")
    print(f"Audit logs added this run    : {after['audit_logs'] - before['audit_logs']}")
    print(f"Inference metadata rows delta: {after['inference_metadata'] - before['inference_metadata']}")
    print("-" * 72)
    print("Row counts before -> after")
    for key in ["predictions", "feedback", "audit_logs", "inference_metadata"]:
        print(f"{key:18}: {before[key]} -> {after[key]}")
    print("-" * 72)
    print("Predicted category mix:")
    for label, count in predicted_counter.most_common():
        print(f"  {label:12} {count}")
    print("-" * 72)
    print("Correction label mix:")
    for label, count in corrected_counter.most_common():
        print(f"  {label:12} {count}")
    print("=" * 72)


if __name__ == "__main__":
    main()
