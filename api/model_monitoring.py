"""
Model Drift Detection Module

Checks for statistically significant shifts in prediction distribution
(chi-square test) and confidence scores (KS test) relative to a baseline
window. Designed to be called by the scheduled batch processor.
"""

import logging
from collections import Counter
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
from scipy import stats
from sqlalchemy.orm import Session

from database.connection import DatabaseConnection
from database.models import Prediction

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default baseline anchor — first date predictions were stored in production.
# Override via DRIFT_BASELINE_START env var if needed.
# ---------------------------------------------------------------------------
_BASELINE_DEFAULT = datetime(2026, 1, 1)


class DriftDetector:
    """
    Detects distribution drift using statistical tests.

    Args:
        window_days: Width (days) of the comparison window.
        threshold: p-value below which drift is flagged.
        baseline_start: Start of the baseline period (defaults to Jan 2026).
    """

    def __init__(
        self,
        window_days: int = 7,
        threshold: float = 0.05,
        baseline_start: Optional[datetime] = None,
    ):
        self.window_days = window_days
        self.threshold = threshold
        self.baseline_start = baseline_start or _BASELINE_DEFAULT
        self.baseline_end = self.baseline_start + timedelta(days=window_days)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_db(self) -> Session:
        conn = DatabaseConnection()
        return conn.get_session().__enter__()

    def _baseline_query(self, db: Session):
        return (
            db.query(Prediction)
            .filter(
                Prediction.created_at >= self.baseline_start,
                Prediction.created_at < self.baseline_end,
            )
            .all()
        )

    def _recent_query(self, db: Session):
        cutoff = datetime.utcnow() - timedelta(days=self.window_days)
        return (
            db.query(Prediction)
            .filter(Prediction.created_at >= cutoff)
            .all()
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_prediction_drift(self) -> dict:
        """
        Chi-square test on category distribution.

        Returns a dict with 'status', 'p_value', and 'recommendation'.
        """
        db = DatabaseConnection().get_session().__enter__()
        try:
            baseline = self._baseline_query(db)
            recent = self._recent_query(db)

            if len(baseline) < 30 or len(recent) < 30:
                return {
                    "status": "insufficient_data",
                    "baseline_count": len(baseline),
                    "recent_count": len(recent),
                    "recommendation": "Collect more predictions before drift analysis",
                }

            b_dist = Counter(p.predicted_label for p in baseline)
            r_dist = Counter(p.predicted_label for p in recent)
            all_cats = sorted(set(b_dist) | set(r_dist))

            b_counts = [b_dist.get(c, 0) for c in all_cats]
            r_counts = [r_dist.get(c, 0) for c in all_cats]

            # Avoid zero expected-frequency warnings
            b_counts_safe = [max(c, 1e-6) for c in b_counts]
            _, p_value = stats.chisquare(r_counts, f_exp=b_counts_safe)

            drift_detected = p_value < self.threshold
            result = {
                "status": "drift_detected" if drift_detected else "no_drift",
                "p_value": float(p_value),
                "threshold": self.threshold,
                "baseline_count": len(baseline),
                "recent_count": len(recent),
                "recommendation": (
                    "Consider retraining model"
                    if drift_detected
                    else "Category distribution is stable"
                ),
            }

            if drift_detected:
                logger.warning("Prediction category drift detected: %s", result)

            return result
        finally:
            db.close()

    def check_confidence_drift(self) -> dict:
        """
        Kolmogorov-Smirnov test on confidence score distribution.

        Returns a dict with 'status', 'ks_statistic', 'p_value', and means.
        """
        db = DatabaseConnection().get_session().__enter__()
        try:
            baseline = [p.confidence for p in self._baseline_query(db)]
            recent = [p.confidence for p in self._recent_query(db)]

            if len(baseline) < 30 or len(recent) < 30:
                return {
                    "status": "insufficient_data",
                    "recommendation": "Collect more predictions before drift analysis",
                }

            ks_stat, p_value = stats.ks_2samp(baseline, recent)
            drift_detected = p_value < self.threshold

            result = {
                "status": "drift_detected" if drift_detected else "no_drift",
                "ks_statistic": float(ks_stat),
                "p_value": float(p_value),
                "baseline_mean_confidence": float(np.mean(baseline)),
                "recent_mean_confidence": float(np.mean(recent)),
                "recommendation": (
                    "Investigate confidence drop — model may be degrading"
                    if drift_detected
                    else "Confidence distribution is stable"
                ),
            }

            if drift_detected:
                logger.warning("Confidence drift detected: %s", result)

            return result
        finally:
            db.close()

    def run_full_check(self) -> dict:
        """Run both drift checks and return combined result."""
        return {
            "checked_at": datetime.utcnow().isoformat(),
            "window_days": self.window_days,
            "prediction_drift": self.check_prediction_drift(),
            "confidence_drift": self.check_confidence_drift(),
        }
