import streamlit as st

from utils.api_client import api_client

st.set_page_config(
    page_title="Email Categorization System",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📧 Email Categorization System")
st.caption("Classify emails, review analytics, and manage model operations from one place.")

st.markdown(
    """
### Welcome
This system categorizes incoming emails using your deployed ML model.

**Features**
- Email classification with confidence scores
- Feedback submission for continuous improvement
- Analytics dashboard for activity and quality tracking
- Admin panel for model visibility and operations
"""
)

@st.cache_data(ttl=30)
def get_health() -> dict:
    return api_client.health()


health = get_health()

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
st.markdown("### Navigation Guide")

left, middle, right = st.columns(3)

with left:
    st.markdown(
        """
#### 📩 Email Classifier
- Enter sender, subject, and body
- Run prediction instantly
- Review confidence and probabilities
- Submit feedback
"""
    )

with middle:
    st.markdown(
        """
#### 📊 Analytics
- Track total predictions
- Monitor average confidence
- View category distribution
- Inspect recent predictions
"""
    )

with right:
    st.markdown(
        """
#### 🛠️ Admin Panel
- View model metadata
- Check live health status
- Reload active model
- Review runtime configuration
"""
    )

st.markdown("---")
st.markdown(
    """
<div style='text-align: center; color: #666; padding: 0.5rem;'>
    Email Categorization System | Streamlit + FastAPI
</div>
""",
    unsafe_allow_html=True,
)
