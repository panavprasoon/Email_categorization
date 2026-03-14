import streamlit as st
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.api_client import api_client
from utils.helpers import display_prediction_result, format_pct

st.set_page_config(page_title="Email Classifier", page_icon="📨", layout="wide")
st.title("📨 Email Classifier")

st.markdown(
    """
Paste or upload an email, then classify it and optionally submit feedback.
"""
)

input_method = st.radio("Input Method", ["Type/Paste", "Upload .txt"], horizontal=True)

uploaded_body = ""
if input_method == "Upload .txt":
    uploaded_file = st.file_uploader("Upload email text file", type=["txt"])
    if uploaded_file is not None:
        uploaded_body = uploaded_file.read().decode("utf-8")

col1, col2 = st.columns(2)
with col1:
    sender = st.text_input("Sender", value="user@example.com")
with col2:
    subject = st.text_input("Subject", placeholder="Enter email subject")

body = st.text_area(
    "Body",
    value=uploaded_body,
    height=240,
    placeholder="Paste the email body here",
)

email_id = st.text_input("Email ID (optional)", placeholder="email_12345")

if st.button("Classify Email", type="primary"):
    if not subject.strip() or not body.strip():
        st.error("Subject and body are required.")
    else:
        result = api_client.categorize(sender=sender.strip(), subject=subject.strip(), body=body.strip())
        if "error" in result:
            st.error(f"Classification failed: {result['error']}")
        else:
            st.session_state["last_prediction"] = result
            st.session_state["last_email_id"] = email_id

if "last_prediction" in st.session_state:
    result = st.session_state["last_prediction"]
    display_prediction_result(result)
    st.write(f"**Confidence:** {format_pct(result.get('confidence', 0.0))}")
    st.write(f"**Processing Time (ms):** {result.get('processing_time_ms', 0):.2f}")

    probabilities = result.get("probabilities", {})
    if probabilities:
        st.markdown("#### Category Probabilities")
        st.dataframe(
            [{"category": key, "probability": value} for key, value in probabilities.items()],
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("---")
    st.markdown("### Submit Feedback")

    categories_payload = api_client.categories()
    categories = categories_payload.get("categories", ["Work", "Personal", "Spam", "Promotions", "Social"])

    feedback_type = st.selectbox(
        "Feedback Type",
        ["correct", "incorrect", "partially_correct"],
        index=0,
    )

    correct_category = None
    if feedback_type in {"incorrect", "partially_correct"}:
        correct_category = st.selectbox("Correct Category", categories)

    comments = st.text_input("Comments (optional)")

    if st.button("Submit Feedback"):
        feedback_result = api_client.submit_feedback(
            prediction_id=int(result["prediction_id"]),
            feedback_type=feedback_type,
            correct_category=correct_category,
            comments=comments or None,
        )
        if "error" in feedback_result:
            st.error(f"Feedback failed: {feedback_result['error']}")
        else:
            st.success(feedback_result.get("message", "Feedback submitted"))

st.markdown("---")
if st.checkbox("Show Recent Activity"):
    stats = api_client.stats(days=7)
    if "error" in stats:
        st.warning("Unable to fetch recent activity.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total Predictions", stats.get("total_predictions", 0))
        with c2:
            st.metric("Feedback Count", stats.get("feedback_count", 0))
        with c3:
            st.metric("Avg Confidence", format_pct(stats.get("average_confidence", 0.0)))
