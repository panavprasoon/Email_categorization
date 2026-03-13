import streamlit as st
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.api_client import api_client
from utils.helpers import format_pct

st.set_page_config(page_title="Email Classifier", page_icon="📨", layout="wide")
st.title("📨 Email Classifier")

col1, col2 = st.columns(2)
with col1:
    sender = st.text_input("Sender", value="user@example.com")
with col2:
    subject = st.text_input("Subject", placeholder="Enter email subject")

body = st.text_area("Body", height=240, placeholder="Paste the email body here")

if st.button("Classify Email", type="primary"):
    if not subject.strip() or not body.strip():
        st.error("Subject and body are required.")
    else:
        result = api_client.categorize(sender=sender.strip(), subject=subject.strip(), body=body.strip())
        if "error" in result:
            st.error(f"Classification failed: {result['error']}")
        else:
            st.success("Prediction complete")
            st.write(f"**Prediction ID:** {result.get('prediction_id')}")
            st.write(f"**Category:** {result.get('category')}")
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
