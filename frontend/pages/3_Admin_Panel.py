import streamlit as st
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.api_client import api_client
from utils.helpers import format_pct

st.set_page_config(page_title="Admin Panel", page_icon="🛠️", layout="wide")
st.title("🛠️ Admin Panel")

model_info = api_client.model_info()
if "error" in model_info:
    st.error(f"Failed to load model info: {model_info['error']}")
else:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model Name", model_info.get("model_name", "N/A"))
    with col2:
        st.metric("Version", model_info.get("version", "N/A"))
    with col3:
        st.metric("Status", model_info.get("status", "N/A"))

    st.write(f"**Algorithm:** {model_info.get('algorithm', 'N/A')}")
    accuracy = model_info.get("accuracy")
    st.write(f"**Accuracy:** {'N/A' if accuracy is None else format_pct(accuracy)}")
    st.write(f"**Total Predictions:** {model_info.get('total_predictions', 0)}")

    categories = model_info.get("categories", [])
    if categories:
        st.write("**Supported Categories:**")
        st.write(", ".join(categories))

st.markdown("---")
st.subheader("Operations")

if st.button("Reload Active Model"):
    response = api_client.reload_model()
    if "error" in response:
        st.error(f"Reload failed: {response['error']}")
    else:
        st.success(response.get("message", "Model reloaded"))

health = api_client.health()
st.markdown("---")
st.subheader("Health")
if "error" in health:
    st.error(health["error"])
else:
    st.json(health)
