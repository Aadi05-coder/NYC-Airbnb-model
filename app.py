import streamlit as st

st.set_page_config(
    page_title="NYC Airbnb Price Intelligence",
    page_icon="🗽",
    layout="wide",
    initial_sidebar_state="expanded",
)

from pages.dashboard import render
render()
