import streamlit as st

from utils.api_client import api_client

st.set_page_config(
    page_title="Email Categorization System",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📧 Email Categorization System")
st.caption("Unified frontend: classify emails, track analytics, and manage model operations.")

health = api_client.health()

col1, col2, col3 = st.columns(3)
if "error" in health:
    with col1:
        st.metric("API", "Unreachable")
    with col2:
        st.metric("Database", "Unknown")
    with col3:
        st.metric("Model", "Unknown")
    st.error(f"Health check failed: {health['error']}")
else:
    with col1:
        st.metric("API Status", health.get("status", "unknown").upper())
    with col2:
        st.metric("Database", health.get("database", "unknown"))
    with col3:
        st.metric("Model", health.get("model", "unknown"))
    st.success("Backend reachable. Use the sidebar pages to continue.")

st.markdown("---")
st.markdown(
    """
### Pages
- `1_Email_Classifier`: submit emails and feedback
- `2_Analytics`: live metrics from API + prediction timeline
- `3_Admin_Panel`: model info and reload action

### Required manual UI steps
1. Ensure API is running locally or deployed.
2. Set `API_BASE_URL` and `API_KEY` in `.env`.
3. Run frontend: `streamlit run frontend/app.py`.
"""
)
