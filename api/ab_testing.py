"""
A/B Testing Framework

Manages traffic splitting between a champion (current production) and
a challenger (candidate) model version.  The manager is initialised
from environment variables and is a lightweight in-process singleton —
no external store is required for simple 2-arm tests.

Environment variables:
    AB_CHAMPION_VERSION   — version string of the champion model
    AB_CHALLENGER_VERSION — version string of the challenger model
    AB_TRAFFIC_SPLIT      — fraction (0–1) of traffic routed to challenger
                            (default 0.20)
"""

import logging
import os
import pickle
import random
from datetime import datetime, timedelta
from typing import Optional, Tuple

from sqlalchemy.orm import Session

from database.connection import DatabaseConnection
from database.models import Feedback, ModelVersion, Prediction

logger = logging.getLogger(__name__)


class ABTestingManager:
    """
    Simple two-arm A/B testing controller.

    Args:
        champion_version: Version string of the production model.
        challenger_version: Version string of the candidate model.
        traffic_split: Fraction of requests routed to the challenger (0–1).
    """

    def __init__(
        self,
        champion_version: str,
        challenger_version: str,
        traffic_split: float = 0.20,
    ):
        if not 0 <= traffic_split <= 1:
            raise ValueError("traffic_split must be between 0 and 1")

        self.champion_version = champion_version
        self.challenger_version = challenger_version
        self.traffic_split = traffic_split

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def select_model_version(self) -> str:
        """Return the version string to use for the current request."""
        return (
            self.challenger_version
            if random.random() < self.traffic_split
            else self.champion_version
        )

    def load_model_version(self, version: str, db: Session) -> Tuple:
        """
        Load model + vectorizer for a specific version from disk.

        Returns:
            (model, vectorizer, model_version_id)
        """
        mv = (
            db.query(ModelVersion)
            .filter(ModelVersion.version == version)
            .first()
        )
        if mv is None:
            raise ValueError(f"Model version '{version}' not found in database")

        if not mv.model_path or not os.path.exists(mv.model_path):
            raise FileNotFoundError(f"Model artifact missing: {mv.model_path}")
        if not mv.vectorizer_path or not os.path.exists(mv.vectorizer_path):
            raise FileNotFoundError(f"Vectorizer artifact missing: {mv.vectorizer_path}")

        with open(mv.model_path, "rb") as fh:
            model = pickle.load(fh)
        with open(mv.vectorizer_path, "rb") as fh:
            vectorizer = pickle.load(fh)

        return model, vectorizer, mv.id

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def compare_models(self, days: int = 7) -> dict:
        """
        Compare champion vs challenger on recent predictions.

        Returns per-arm summary: count, avg_confidence, correction_rate.
        """
        conn = DatabaseConnection()
        with conn.get_session() as db:
            cutoff = datetime.utcnow() - timedelta(days=days)

            def arm_stats(version: str) -> dict:
                mv = (
                    db.query(ModelVersion)
                    .filter(ModelVersion.version == version)
                    .first()
                )
                if mv is None:
                    return {"error": f"Version {version} not found"}

                preds = (
                    db.query(Prediction)
                    .filter(
                        Prediction.model_version_id == mv.id,
                        Prediction.created_at >= cutoff,
                    )
                    .all()
                )
                if not preds:
                    return {"version": version, "count": 0}

                total = len(preds)
                avg_conf = sum(p.confidence for p in preds) / total
                corrections = sum(
                    1
                    for p in preds
                    if p.feedback and p.feedback.feedback_source == "correction"
                )
                return {
                    "version": version,
                    "count": total,
                    "avg_confidence": round(avg_conf, 4),
                    "corrections": corrections,
                    "correction_rate": round(corrections / total, 4),
                }

            return {
                "champion": arm_stats(self.champion_version),
                "challenger": arm_stats(self.challenger_version),
                "traffic_split": self.traffic_split,
                "window_days": days,
            }

    # ------------------------------------------------------------------
    # Promotion
    # ------------------------------------------------------------------

    def promote_challenger(self) -> dict:
        """
        Make the challenger the active production model.

        Deactivates the current champion and activates the challenger.
        """
        conn = DatabaseConnection()
        with conn.get_session() as db:
            # Deactivate champion
            db.query(ModelVersion).filter(
                ModelVersion.version == self.champion_version
            ).update({"is_active": False})

            # Activate challenger
            updated = (
                db.query(ModelVersion)
                .filter(ModelVersion.version == self.challenger_version)
                .first()
            )
            if updated is None:
                raise ValueError(
                    f"Challenger version '{self.challenger_version}' not found"
                )
            updated.is_active = True
            db.commit()

        logger.info(
            "A/B promotion: %s -> %s",
            self.champion_version,
            self.challenger_version,
        )

        return {
            "promoted": self.challenger_version,
            "demoted": self.champion_version,
        }


# ---------------------------------------------------------------------------
# Factory — build from environment variables
# ---------------------------------------------------------------------------

def build_ab_manager_from_env() -> Optional[ABTestingManager]:
    """
    Return an ABTestingManager if AB_CHAMPION_VERSION and
    AB_CHALLENGER_VERSION env vars are both set, else None.
    """
    champion = os.getenv("AB_CHAMPION_VERSION")
    challenger = os.getenv("AB_CHALLENGER_VERSION")
    if not champion or not challenger:
        return None

    split = float(os.getenv("AB_TRAFFIC_SPLIT", "0.20"))
    logger.info(
        "A/B testing enabled: champion=%s  challenger=%s  split=%.0f%%",
        champion,
        challenger,
        split * 100,
    )
    return ABTestingManager(champion, challenger, split)
