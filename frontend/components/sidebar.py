import streamlit as st


def render_sidebar() -> None:
    st.sidebar.title("Navigation")
    st.sidebar.markdown("Use the built-in Streamlit page menu to switch pages.")
    st.sidebar.markdown("---")
    st.sidebar.caption("Email Categorization System")
