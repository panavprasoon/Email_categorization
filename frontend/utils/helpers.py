from datetime import datetime
from typing import Any, Dict

import pandas as pd
import plotly.express as px
import streamlit as st


def format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def get_confidence_color(confidence: float) -> str:
    if confidence >= 0.9:
        return "green"
    if confidence >= 0.7:
        return "orange"
    return "red"


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


def display_prediction_result(result: Dict[str, Any]):
    category = result.get("predicted_category", result.get("category", "Unknown"))
    prediction_id = result.get("prediction_id", "N/A")
    confidence = float(result.get("confidence", 0.0))

    st.success("Prediction complete")
    st.write(f"**Prediction ID:** {prediction_id}")
    st.write(f"**Category:** {category}")
    st.progress(max(0.0, min(1.0, confidence)))

    color = get_confidence_color(confidence)
    st.markdown(f"Confidence indicator: :{color}[{format_pct(confidence)}]")


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
