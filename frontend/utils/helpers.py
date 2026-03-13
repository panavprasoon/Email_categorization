from datetime import datetime
from typing import Any, Dict

import pandas as pd
import plotly.express as px


def format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def safe_timestamp(value: str) -> str:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return value


def category_pie(category_counts: Dict[str, int]):
    if not category_counts:
        return None
    frame = pd.DataFrame({"category": list(category_counts.keys()), "count": list(category_counts.values())})
    figure = px.pie(frame, values="count", names="category", title="Category Distribution")
    figure.update_traces(textposition="inside", textinfo="percent+label")
    return figure


def predictions_timeline(predictions: list[dict[str, Any]]):
    if not predictions:
        return None
    frame = pd.DataFrame(predictions)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp")
    if frame.empty:
        return None
    frame["date"] = frame["timestamp"].dt.date
    grouped = frame.groupby("date", as_index=False).size().rename(columns={"size": "count"})
    return px.line(grouped, x="date", y="count", markers=True, title="Predictions Over Time")
