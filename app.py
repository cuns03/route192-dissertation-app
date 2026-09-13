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
        "pages/1_EDA.py",
        title="Exploratory Data Analysis",
        icon=":material/explore:",
        default=True,
    ),
    st.Page(
        "pages/2_Predictive_Modelling.py",
        title="Predictive Modelling",
        icon=":material/model_training:",
    ),
    st.Page(
        "pages/3_Equity_Analysis.py",
        title="Equity Analysis",
        icon=":material/balance:",
    ),
    st.Page(
        "pages/4_Operational_Analysis.py",
        title="Operational Analysis",
        icon=":material/directions_bus:",
    ),
]

# IMPORTANT:
# Hide Streamlit's native navigation
pg = st.navigation(
    pages,
    position="hidden"
)

# Determine current page
active_title = getattr(
    pg,
    "title",
    "Exploratory Data Analysis",
)

# Render custom Route 192 sidebar
render_sidebar(active_title=active_title)

# Run current page
pg.run()