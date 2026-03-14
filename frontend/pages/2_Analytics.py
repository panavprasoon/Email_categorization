import streamlit as st
import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.api_client import api_client
from utils.helpers import category_pie, format_pct, predictions_timeline

st.set_page_config(page_title="Analytics", page_icon="📊", layout="wide")
st.title("📊 Analytics")

top_left, top_mid, top_right = st.columns([2, 1, 1])

with top_left:
    days = st.slider("Time window (days)", min_value=1, max_value=90, value=7)
with top_mid:
    st.metric("Selected Period", f"{days} days")
with top_right:
    if st.button("Refresh Data"):
        st.rerun()

stats = api_client.stats(days=days)
if "error" in stats:
    st.error(f"Failed to load stats: {stats['error']}")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Predictions", stats.get("total_predictions", 0))
with col2:
    st.metric("Avg Confidence", format_pct(stats.get("average_confidence", 0.0)))
with col3:
    st.metric("Feedback Count", stats.get("feedback_count", 0))
with col4:
    accuracy = stats.get("accuracy_from_feedback")
    st.metric("Accuracy from Feedback", "N/A" if accuracy is None else format_pct(accuracy))

st.markdown("---")
left, right = st.columns(2)

with left:
    fig = category_pie(stats.get("predictions_by_category", {}))
    if fig is None:
        st.info("No category data in selected range.")
    else:
        st.plotly_chart(fig, use_container_width=True)

with right:
    prediction_page = api_client.predictions(days=days, page=1, page_size=200)
    rows = prediction_page.get("predictions", []) if isinstance(prediction_page, dict) else []
    timeline = predictions_timeline(rows)
    if timeline is None:
        st.info("No timeline data in selected range.")
    else:
        st.plotly_chart(timeline, use_container_width=True)

st.markdown("---")
st.subheader("Recent Predictions")
if rows:
    table_rows = [
        {
            "prediction_id": item.get("prediction_id"),
            "sender": item.get("email_sender"),
            "subject": item.get("email_subject"),
            "category": item.get("predicted_category"),
            "confidence": item.get("confidence"),
            "timestamp": item.get("timestamp"),
            "feedback_provided": item.get("feedback_provided"),
        }
        for item in rows[:25]
    ]
    st.dataframe(table_rows, use_container_width=True, hide_index=True)
else:
    st.info("No recent predictions to display.")

st.markdown("---")
if st.button("Download Analytics Report (CSV)"):
    report_rows = [
        {"metric": "total_predictions", "value": stats.get("total_predictions", 0)},
        {"metric": "average_confidence", "value": stats.get("average_confidence", 0.0)},
        {"metric": "feedback_count", "value": stats.get("feedback_count", 0)},
        {"metric": "accuracy_from_feedback", "value": stats.get("accuracy_from_feedback")},
    ]
    report_df = pd.DataFrame(report_rows)
    st.download_button(
        label="Download CSV",
        data=report_df.to_csv(index=False),
        file_name=f"analytics_report_{pd.Timestamp.utcnow().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )
