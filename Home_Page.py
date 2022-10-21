import streamlit as st

st.set_page_config(
    page_title="Home page"
)

st.sidebar.success("Select a model above")
st.session_state["X"] = None
st.session_state["y"] = None
st.session_state["data"] = None
