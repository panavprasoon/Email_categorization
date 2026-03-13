import os
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv

load_dotenv()


class APIClient:
    def __init__(self) -> None:
        self.base_url = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
        self.api_key = os.getenv("API_KEY", os.getenv("API_KEY_1", "dev-api-key-12345"))
        self.timeout = int(os.getenv("API_TIMEOUT_SECONDS", "30"))

    def _headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "X-API-Key": self.api_key,
        }

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None, auth: bool = True) -> Dict[str, Any]:
        try:
            response = requests.get(
                f"{self.base_url}{path}",
                params=params,
                headers=self._headers() if auth else None,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as error:
            return {"error": str(error)}

    def _post(self, path: str, payload: Dict[str, Any], auth: bool = True) -> Dict[str, Any]:
        try:
            response = requests.post(
                f"{self.base_url}{path}",
                json=payload,
                headers=self._headers() if auth else None,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as error:
            return {"error": str(error)}

    def health(self) -> Dict[str, Any]:
        return self._get("/health/", auth=False)

    def categorize(self, sender: str, subject: str, body: str) -> Dict[str, Any]:
        payload = {
            "sender": sender,
            "subject": subject,
            "body": body,
        }
        return self._post("/categorize/", payload)

    def submit_feedback(
        self,
        prediction_id: int,
        feedback_type: str,
        correct_category: Optional[str] = None,
        comments: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "prediction_id": prediction_id,
            "feedback_type": feedback_type,
        }
        if correct_category:
            payload["correct_category"] = correct_category
        if comments:
            payload["comments"] = comments
        return self._post("/feedback/", payload)

    def stats(self, days: int = 7) -> Dict[str, Any]:
        return self._get("/predictions/statistics/overview", params={"days": days})

    def predictions(self, days: int = 7, page: int = 1, page_size: int = 100) -> Dict[str, Any]:
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        params = {
            "page": page,
            "page_size": page_size,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        }
        return self._get("/predictions/", params=params)

    def model_info(self) -> Dict[str, Any]:
        return self._get("/models/info")

    def categories(self) -> Dict[str, Any]:
        return self._get("/models/categories")

    def reload_model(self) -> Dict[str, Any]:
        return self._post("/models/reload", {})


api_client = APIClient()
