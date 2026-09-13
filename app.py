import streamlit as st

from ui_v2 import apply_global_styles, render_sidebar

st.set_page_config(
    page_title="Route 192 — MSc Decision-Support Prototype",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Global styling
apply_global_styles()

# Pages
pages = [
    st.Page(
        "views/1_EDA.py",
        title="Exploratory Data Analysis",
        icon=":material/explore:",
        default=True,
    ),
    st.Page(
        "views/2_Predictive_Modelling.py",
        title="Predictive Modelling",
        icon=":material/model_training:",
    ),
    st.Page(
        "views/3_Equity_Analysis.py",
        title="Equity Analysis",
        icon=":material/balance:",
    ),
    st.Page(
        "views/4_Operational_Analysis.py",
        title="Operational Analysis",
        icon=":material/directions_bus:",
    ),
]

pg = st.navigation(
    pages,
    position="hidden",
)

active_title = getattr(
    pg,
    "title",
    "Exploratory Data Analysis",
)

render_sidebar(active_title=active_title)

pg.run()