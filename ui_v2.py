from __future__ import annotations

import streamlit as st

BEE_YELLOW = "#FFD800"
BEE_TINT = "#FFF9D6"
APP_BG = "#F7F8FC"
SIDEBAR_BG = "#F1F3F8"
BORDER = "#D9DFEA"
TEXT = "#171B26"
MUTED = "#6F7685"


def apply_global_styles():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Public+Sans:wght@500;600;700;800&display=swap');

        html, body, .stApp {{
            font-family: "Public Sans", "Inter", sans-serif !important;
        }}

        .stApp {{
            background: {APP_BG} !important;
        }}

        /* Hide every known form of Streamlit's built-in nav */
        [data-testid="stSidebarNav"],
        nav[data-testid="stSidebarNav"],
        div[data-testid="stSidebarNav"],
        section[data-testid="stSidebar"] nav,
        [data-testid="stSidebar"] ul {{
            display: none !important;
        }}

        /* Sidebar */
        section[data-testid="stSidebar"] {{
            background: {SIDEBAR_BG} !important;
            border-right: 1px solid {BORDER} !important;
            width: 290px !important;
            min-width: 290px !important;
            max-width: 290px !important;
        }}

        section[data-testid="stSidebar"] > div {{
            width: 290px !important;
        }}

        [data-testid="stSidebarContent"] {{
            padding-top: 0.4rem !important;
        }}

        /* Main content */
        [data-testid="stMainBlockContainer"] {{
            max-width: 1180px !important;
            padding-top: 2rem !important;
            padding-bottom: 3rem !important;
        }}

        h1, h2, h3 {{
            font-family: "Public Sans", "Inter", sans-serif !important;
            color: #232734 !important;
            letter-spacing: -0.025em !important;
        }}

        h1 {{
            font-weight: 800 !important;
        }}

        /* Brand */
        .r192-brand {{
            background: rgba(255,255,255,.82);
            border: 1px solid {BORDER};
            border-radius: 12px;
            padding: 14px 14px 12px 14px;
            margin: 4px 0 18px 0;
        }}

        .r192-brand-top {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .r192-bus {{
            width: 36px;
            height: 36px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: {BEE_YELLOW};
            border-radius: 9px;
            font-size: 20px;
        }}

        .r192-route {{
            font-size: 15px;
            line-height: 1;
            font-weight: 800;
            color: {TEXT};
        }}

        .r192-brand-small {{
            margin-top: 4px;
            font-size: 9px;
            line-height: 1;
            color: {MUTED};
            font-weight: 700;
            letter-spacing: .10em;
            text-transform: uppercase;
        }}

        .r192-project {{
            margin-top: 10px;
            font-size: 12px;
            line-height: 1.4;
            font-weight: 650;
            color: #303541;
        }}

        .r192-prototype {{
            margin-top: 2px;
            font-size: 10px;
            color: {MUTED};
        }}

        .r192-section-label {{
            margin: 0 0 7px 5px;
            color: {MUTED};
            font-size: 9px;
            font-weight: 800;
            letter-spacing: .13em;
            text-transform: uppercase;
        }}

        /* page_link cards */
        section[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"],
        section[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] {{
            min-height: 52px !important;
            padding: 9px 10px !important;
            margin: 4px 0 !important;
            border-radius: 9px !important;
            border: 1px solid transparent !important;
            color: #3E4552 !important;
            background: transparent !important;
            text-decoration: none !important;
        }}

        section[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"]:hover,
        section[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"]:hover {{
            background: #FFFFFF !important;
            border-color: {BORDER} !important;
        }}

        section[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"][aria-current="page"] {{
            background: {BEE_TINT} !important;
            border-left: 4px solid {BEE_YELLOW} !important;
            border-top: 1px solid #F0DE77 !important;
            border-right: 1px solid #F0DE77 !important;
            border-bottom: 1px solid #F0DE77 !important;
            color: {TEXT} !important;
            font-weight: 750 !important;
        }}

        section[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] p {{
            font-size: 12px !important;
            font-weight: 700 !important;
            margin: 0 !important;
        }}

        .r192-desc {{
            margin: -5px 8px 6px 44px;
            color: {MUTED};
            font-size: 9.5px;
            line-height: 1.25;
        }}

        .r192-flow {{
            margin-top: 17px;
            padding: 11px 12px;
            background: rgba(255,255,255,.75);
            border: 1px solid {BORDER};
            border-radius: 9px;
        }}

        .r192-flow-head {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
            color: {MUTED};
            font-size: 8.5px;
            font-weight: 800;
            letter-spacing: .10em;
            text-transform: uppercase;
        }}

        .r192-flow-row {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 4px;
            font-size: 9px;
            color: #7A8190;
        }}

        .r192-flow-active {{
            background: {BEE_YELLOW};
            color: #191919;
            padding: 2px 5px;
            border-radius: 4px;
            font-weight: 800;
        }}

        .r192-footer {{
            margin-top: 18px;
            border-top: 1px solid {BORDER};
            padding-top: 11px;
            color: {MUTED};
            font-size: 9.5px;
            line-height: 1.55;
        }}

        .r192-footer strong {{
            color: #343A46;
            font-size: 10px;
        }}

        .r192-status {{
            margin-top: 9px;
            display: flex;
            align-items: center;
            gap: 6px;
            color: #525966;
            font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
            font-size: 8.5px;
        }}

        .r192-dot {{
            width: 6px;
            height: 6px;
            border-radius: 100%;
            background: #16A34A;
        }}

        /* Shared content style */
        [data-testid="stMetric"] {{
            background: #FFFFFF !important;
            border: 1px solid {BORDER} !important;
            border-top: 4px solid {BEE_YELLOW} !important;
            border-radius: 11px !important;
            padding: .85rem 1rem !important;
        }}

        [data-testid="stMetricLabel"] {{
            color: #6B7280 !important;
            font-size: 11px !important;
            letter-spacing: .035em !important;
            text-transform: uppercase !important;
        }}

        [data-testid="stMetricValue"] {{
            color: #141B28 !important;
            font-family: "Public Sans", "Inter", sans-serif !important;
            font-weight: 800 !important;
        }}

        [data-testid="stExpander"] {{
            background: #FFFFFF !important;
            border: 1px solid {BORDER} !important;
            border-radius: 9px !important;
        }}

        [data-baseweb="select"] > div {{
            background: #EFF1F6 !important;
            border-radius: 8px !important;
        }}

        .stButton button {{
            border-radius: 8px !important;
            font-family: "Public Sans", "Inter", sans-serif !important;
            font-weight: 650 !important;
        }}

        button[data-baseweb="tab"][aria-selected="true"] {{
            border-top: 3px solid {BEE_YELLOW} !important;
            color: #161616 !important;
        }}

        footer {{
            visibility: hidden !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _flow(active_stage: int):
    labels = ["EDA", "RQ1", "RQ2", "RQ3"]
    items = []
    for i, label in enumerate(labels, start=1):
        cls = "r192-flow-active" if i == active_stage else ""
        items.append(f'<span class="{cls}">{label}</span>')
        if i < 4:
            items.append("<span>→</span>")
    return "".join(items)


def render_sidebar(active_title: str):
    stage_lookup = {
        "Exploratory Data Analysis": 1,
        "Predictive Modelling": 2,
        "Equity Analysis": 3,
        "Operational Analysis": 4,
    }
    stage = stage_lookup.get(active_title, 1)

    with st.sidebar:
        st.markdown(
            """
            <div class="r192-brand">
                <div class="r192-brand-top">
                    <div class="r192-bus">🚌</div>
                    <div>
                        <div class="r192-route">ROUTE 192</div>
                        <div class="r192-brand-small">Greater Manchester</div>
                    </div>
                </div>
                <div class="r192-project">Health-Integrated Demand Analysis</div>
                <div class="r192-prototype">MSc Decision-Support Prototype</div>
            </div>
            <div class="r192-section-label">Analytical workflow</div>
            """,
            unsafe_allow_html=True,
        )

        st.page_link(
            "pages/1_EDA.py",
            label="01  Exploratory Data Analysis (EDA)",
            icon=":material/explore:",
        )

        st.page_link(
            "pages/2_Predictive_Modelling.py",
            label="02  Predictive Modelling (RQ1)",
            icon=":material/model_training:",
        )

        st.page_link(
            "pages/3_Equity_Analysis.py",
            label="03  Equity Analysis (RQ2)",
            icon=":material/balance:",
        )

        st.page_link(
            "pages/4_Operational_Analysis.py",
            label="04  Operational Analysis (RQ3)",
            icon=":material/directions_bus:",
        )

        st.markdown(
            f"""
            <div class="r192-flow">
                <div class="r192-flow-head">
                    <span>Research sequence</span>
                    <span>Stage {stage}/4</span>
                </div>
                <div class="r192-flow-row">
                    {_flow(stage)}
                </div>
            </div>

            <div class="r192-footer">
                <strong>Route 192 Corridor Study</strong><br>
                Greater Manchester · Bee Network<br>
                MSc Dissertation Research
                <div class="r192-status">
                    <span class="r192-dot"></span>
                    <span>Analytical dataset loaded</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
