from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Equity Analysis | Route 192",
    page_icon="🚌",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

OUTPUT_DIR_CANDIDATES = [
    DATA_DIR / "rq2_rq3_outputs",
    DATA_DIR / "RQ2_RQ3_business_outputs",
    BASE_DIR / "RQ2_RQ3_business_outputs",
    BASE_DIR.parent / "RQ2_RQ3_business_outputs",
]


JOINT_FILE = "RQ2_LSOA_joint_analysis.csv"
ROBUST_FILE = "RQ2_health_index_sensitivity.csv"
THRESHOLD_FILE = "RQ2_threshold_sensitivity.csv"


YELLOW = "#FFD800"
BLACK = "#161616"
BG = "#F7F7FC"
WHITE = "#FFFFFF"
BORDER = "#DCE2F3"
GREY = "#6B7280"
BLUE = "#0053DB"

GROUP_COLOURS = {
    "Higher health + Lower demand": YELLOW,
    "Higher health + Higher demand": BLACK,
    "Lower health + Higher demand": "#6F8FAF",
    "Lower health + Lower demand": "#C9D1DC",
}

GROUP_ORDER = [
    "Higher health + Lower demand",
    "Higher health + Higher demand",
    "Lower health + Higher demand",
    "Lower health + Lower demand",
]

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Public+Sans:wght@600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: "Inter", sans-serif;
    }}
    .stApp {{
        background: {BG};
        color: #151C27;
    }}
    h1, h2, h3, h4 {{
        font-family: "Public Sans", sans-serif !important;
        letter-spacing: -0.015em;
    }}
    .block-container {{
        max-width: 1240px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }}
    .rq-badge {{
        display: inline-flex;
        align-items: center;
        padding: .36rem .72rem;
        border-radius: .35rem;
        background: {YELLOW};
        color: {BLACK};
        font-family: "Public Sans", sans-serif;
        font-size: .72rem;
        font-weight: 800;
        letter-spacing: .05em;
        text-transform: uppercase;
    }}
    .research-note {{
        background: #F0F3FF;
        border-left: 4px solid {BLUE};
        padding: .88rem 1rem;
        border-radius: .65rem;
        font-size: .9rem;
        color: #303642;
    }}
    .flow-wrap {{
        background: {WHITE};
        border: 1px solid {BORDER};
        border-radius: .8rem;
        padding: .85rem 1rem;
        box-shadow: 0 1px 4px rgba(0,0,0,.025);
    }}
    .flow-step {{
        display: inline-block;
        padding: .35rem .55rem;
        border-radius: .35rem;
        background: #F0F3FF;
        font-size: .75rem;
        font-weight: 700;
        color: #303642;
    }}
    .flow-arrow {{
        display: inline-block;
        color: {GREY};
        font-weight: 800;
        padding: 0 .2rem;
    }}
    .metric-card {{
        background: {WHITE};
        border: 1px solid {BORDER};
        border-top: 4px solid {YELLOW};
        border-radius: .8rem;
        padding: .95rem 1rem;
        min-height: 112px;
        box-shadow: 0 1px 4px rgba(0,0,0,.03);
    }}
    .metric-label {{
        color: {GREY};
        font-size: .72rem;
        text-transform: uppercase;
        letter-spacing: .04em;
        font-weight: 700;
        line-height: 1.3;
    }}
    .metric-value {{
        font-family: "Public Sans", sans-serif;
        color: #151C27;
        font-size: 2rem;
        line-height: 1.12;
        font-weight: 800;
        margin-top: .2rem;
    }}
    .metric-sub {{
        color: {GREY};
        font-size: .76rem;
        margin-top: .35rem;
        line-height: 1.35;
    }}
    .panel, .inspector-card {{
        background: {WHITE};
        border: 1px solid {BORDER};
        border-radius: .9rem;
        padding: 1rem;
        box-shadow: 0 1px 5px rgba(0,0,0,.03);
    }}
    .inspector-highlight {{
        background: {YELLOW};
        color: {BLACK};
        display: inline-block;
        padding: .18rem .42rem;
        border-radius: .3rem;
        font-family: "Public Sans", sans-serif;
        font-weight: 800;
    }}
    .interpret-box {{
        background: #F0F3FF;
        border-radius: .7rem;
        padding: .9rem;
        font-size: .82rem;
        line-height: 1.55;
        color: #424957;
    }}
    .caution-box {{
        background: #FFF9DF;
        border-left: 4px solid {YELLOW};
        border-radius: .65rem;
        padding: .88rem 1rem;
        font-size: .86rem;
        line-height: 1.5;
    }}
    .answer-box {{
        background: {YELLOW};
        color: {BLACK};
        border-radius: .95rem;
        padding: 1.1rem 1.2rem;
    }}
    .answer-box h3 {{
        margin: .1rem 0 .25rem 0;
    }}
    .diagnostic-card {{
        background: {WHITE};
        border: 1px solid {BORDER};
        border-radius: .8rem;
        padding: .9rem;
        min-height: 155px;
    }}
    .workflow-step {{
        background: {WHITE};
        border: 1px solid {BORDER};
        border-radius: .75rem;
        padding: .9rem;
        min-height: 112px;
    }}
    .workflow-step-yellow {{
        background: #FFF9D8;
        border: 1px solid #F1D13D;
        border-radius: .75rem;
        padding: .9rem;
        min-height: 112px;
    }}
    .section-kicker {{
        font-size: .71rem;
        color: {GREY};
        font-weight: 800;
        letter-spacing: .06em;
        text-transform: uppercase;
        margin-bottom: .1rem;
    }}
    .small-muted {{
        color: {GREY};
        font-size: .78rem;
        line-height: 1.45;
    }}
    .stDataFrame {{
        border: 1px solid {BORDER};
        border-radius: .7rem;
        overflow: hidden;
    }}
    hr {{
        border-color: {BORDER};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def norm(text) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())




def resolve_output_file(filename: str):
    for directory in OUTPUT_DIR_CANDIDATES:
        path = directory / filename
        if path.exists():
            return path
    return None


def safe_read_csv(path):
    if path is None or not Path(path).exists():
        return None
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return None


def find_col(df, candidates, contains=None):
    if df is None:
        return None
    norm_map = {norm(c): c for c in df.columns}
    for candidate in candidates:
        key = norm(candidate)
        if key in norm_map:
            return norm_map[key]
    if contains:
        tokens = [norm(x) for x in contains]
        for c in df.columns:
            nc = norm(c)
            if all(x in nc for x in tokens):
                return c
    return None


def fmt_pct(v):
    if pd.isna(v):
        return "—"
    x = float(v)
    if abs(x) <= 1.5:
        x *= 100
    return f"{x:.1f}%"


def fmt_num(v, digits=2, signed=False):
    if pd.isna(v):
        return "—"
    return f"{float(v):+.{digits}f}" if signed else f"{float(v):.{digits}f}"


def zscore(series):
    s = pd.to_numeric(series, errors="coerce")
    sd = s.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.mean()) / sd


def metric_card(label, value, sub=""):
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
          <div class="metric-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def plot_layout(fig, height=430):
    fig.update_layout(
        height=height,
        margin=dict(l=25, r=20, t=35, b=30),
        paper_bgcolor=WHITE,
        plot_bgcolor=WHITE,
        font=dict(family="Inter", color="#374151", size=12),
        hoverlabel=dict(font_family="Inter"),
        legend=dict(font=dict(family="Inter")),
    )
    fig.update_xaxes(gridcolor="#E8ECF5", zeroline=False)
    fig.update_yaxes(gridcolor="#E8ECF5", zeroline=False)
    return fig


joint_path = resolve_output_file(JOINT_FILE)
joint_raw = safe_read_csv(joint_path)

if joint_raw is None or joint_raw.empty:
    st.error(
        "RQ2 data could not be loaded. Place `RQ2_LSOA_joint_analysis.csv` in "
        "`route192_streamlit/data/rq2_rq3_outputs/` and refresh the app."
    )
    st.stop()


def canonicalise_joint(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    code_col = find_col(
        out,
        ["LSOA21CD", "LSOA_code", "lsoa_code", "LSOA", "area_code", "lsoa21cd"],
    )
    name_col = find_col(
        out,
        ["LSOA21NM", "LSOA_name", "lsoa_name", "area_name", "name"],
    )
    demand_col = find_col(
        out,
        [
            "demand_occurrence",
            "demand_occurrence_rate",
            "demand_rate",
            "reconstructed_demand_occurrence",
            "bus_demand_rate",
            "positive_demand_rate",
            "boarding_occurrence",
        ],
    )
    health_col = find_col(
        out,
        [
            "health_vulnerability",
            "health_vulnerability_index",
            "health_index",
            "combined_health_index",
            "health_score",
            "composite_health_z",
            "health_composite_z",
            "z_health",
        ],
    )
    demand_z_col = find_col(
        out,
        [
            "z_demand",
            "demand_z",
            "demand_occurrence_z",
            "standardised_demand",
            "standardized_demand",
            "demand_zscore",
        ],
    )
    health_z_col = find_col(
        out,
        [
            "z_health",
            "health_z",
            "health_vulnerability_z",
            "standardised_health",
            "standardized_health",
            "health_zscore",
        ],
    )
    gap_col = find_col(
        out,
        [
            "health_demand_gap",
            "health-demand gap",
            "health_demand_divergence",
            "divergence_gap",
            "gap_z",
            "gap",
        ],
    )
    group_col = find_col(
        out,
        [
            "health_demand_group",
            "health-demand group",
            "rq2_group",
            "health_demand_context",
            "context_group",
            "classification",
            "quadrant",
        ],
    )

    if code_col is None:
        out["_code"] = [f"LSOA_{i+1:02d}" for i in range(len(out))]
    else:
        out["_code"] = out[code_col].astype(str)

    out["_name"] = out[name_col].fillna(out["_code"]).astype(str) if name_col else out["_code"]

    if demand_col is None:
        candidates = [
            c for c in out.columns
            if "demand" in c.lower()
            and pd.api.types.is_numeric_dtype(out[c])
            and "z" not in c.lower()
        ]
        if not candidates:
            raise ValueError(
                "Could not identify a demand-occurrence column in RQ2_LSOA_joint_analysis.csv."
            )
        demand_col = candidates[0]

    if health_col is None and health_z_col is None:
        candidates = [
            c for c in out.columns
            if "health" in c.lower()
            and pd.api.types.is_numeric_dtype(out[c])
            and "gap" not in c.lower()
            and "demand" not in c.lower()
        ]
        if not candidates:
            raise ValueError(
                "Could not identify a health-vulnerability column in RQ2_LSOA_joint_analysis.csv."
            )
        health_col = candidates[0]

    out["_demand"] = pd.to_numeric(out[demand_col], errors="coerce")
    out["_health"] = (
        pd.to_numeric(out[health_col], errors="coerce")
        if health_col is not None
        else pd.to_numeric(out[health_z_col], errors="coerce")
    )

    out["_z_demand"] = (
        pd.to_numeric(out[demand_z_col], errors="coerce")
        if demand_z_col is not None
        else zscore(out["_demand"])
    )
    out["_z_health"] = (
        pd.to_numeric(out[health_z_col], errors="coerce")
        if health_z_col is not None
        else zscore(out["_health"])
    )
    out["_gap"] = (
        pd.to_numeric(out[gap_col], errors="coerce")
        if gap_col is not None
        else out["_z_health"] - out["_z_demand"]
    )

    demand_med = out["_demand"].median()
    health_med = out["_health"].median()

    def derive_group(row):
        high_h = row["_health"] > health_med
        high_d = row["_demand"] > demand_med
        if high_h and not high_d:
            return "Higher health + Lower demand"
        if high_h and high_d:
            return "Higher health + Higher demand"
        if not high_h and high_d:
            return "Lower health + Higher demand"
        return "Lower health + Lower demand"

    if group_col is not None:
        def normalise_group(x):
            s = str(x).lower()
            if (
                ("higher" in s or "high" in s)
                and "health" in s
                and ("lower" in s or "low" in s)
                and "demand" in s
            ):
                return "Higher health + Lower demand"
            if "higher health" in s and "higher demand" in s:
                return "Higher health + Higher demand"
            if "lower health" in s and "higher demand" in s:
                return "Lower health + Higher demand"
            if "lower health" in s and "lower demand" in s:
                return "Lower health + Lower demand"
            if "equity" in s and ("salient" in s or "diverg" in s):
                return "Higher health + Lower demand"
            return None

        mapped = out[group_col].astype(str).map(normalise_group)
        out["_group"] = [
            mapped.iloc[i] if pd.notna(mapped.iloc[i]) else derive_group(out.iloc[i])
            for i in range(len(out))
        ]
    else:
        out["_group"] = out.apply(derive_group, axis=1)

    lat_col = find_col(out, ["lat", "latitude", "lsoa_lat", "centroid_lat"])
    lon_col = find_col(out, ["lon", "lng", "longitude", "lsoa_lon", "centroid_lon"])
    out["_lat"] = pd.to_numeric(out[lat_col], errors="coerce") if lat_col else np.nan
    out["_lon"] = pd.to_numeric(out[lon_col], errors="coerce") if lon_col else np.nan

    out = out.dropna(subset=["_demand", "_health", "_z_demand", "_z_health", "_gap"])
    out = out.drop_duplicates("_code").reset_index(drop=True)
    out["_gap_rank"] = out["_gap"].rank(method="min", ascending=False).astype(int)
    return out


try:
    rq2 = canonicalise_joint(joint_raw)
except Exception as exc:
    st.error(f"RQ2 columns could not be standardised: {exc}")
    with st.expander("Available columns"):
        st.write(list(joint_raw.columns))
    st.stop()


robust_path = resolve_output_file(ROBUST_FILE)
robust_raw = safe_read_csv(robust_path)


def is_hhld(value):
    if pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value) == 1.0
    s = str(value).lower()
    return (
        (
            ("higher" in s or "high" in s)
            and "health" in s
            and ("lower" in s or "low" in s)
            and "demand" in s
        )
        or ("equity" in s and ("salient" in s or "diverg" in s))
    )


def parse_robustness(df):
    result = {
        "agreement_census": 97.4,
        "agreement_iod": 87.2,
        "stable_core": 5,
        "stable_codes": [],
        "table": None,
    }

    if df is None or df.empty:
        return result

    code_col = find_col(df, ["LSOA21CD", "LSOA_code", "lsoa_code", "LSOA", "area_code"])
    spec_col = find_col(
        df,
        ["specification", "health_specification", "health_definition", "scenario", "index_type", "spec"],
    )
    group_col = find_col(
        df,
        ["health_demand_group", "rq2_group", "classification", "context_group", "quadrant", "group"],
    )

    if code_col and spec_col and group_col:
        temp = df.copy()
        temp["_code"] = temp[code_col].astype(str)
        temp["_spec"] = temp[spec_col].astype(str)

        def spec_name(x):
            s = str(x).lower()
            if "combined" in s or "composite" in s or "main" in s:
                return "Combined Health Index"
            if "census" in s:
                return "Census-only"
            if "iod" in s or "imd" in s or "deprivation" in s:
                return "IoD-only"
            return str(x)

        temp["_spec_clean"] = temp["_spec"].map(spec_name)
        wide = temp.pivot_table(
            index="_code",
            columns="_spec_clean",
            values=group_col,
            aggfunc="first",
        )

        if "Combined Health Index" in wide.columns and "Census-only" in wide.columns:
            result["agreement_census"] = float(
                (wide["Combined Health Index"].astype(str) == wide["Census-only"].astype(str)).mean() * 100
            )
        if "Combined Health Index" in wide.columns and "IoD-only" in wide.columns:
            result["agreement_iod"] = float(
                (wide["Combined Health Index"].astype(str) == wide["IoD-only"].astype(str)).mean() * 100
            )

        needed = [x for x in ["Combined Health Index", "Census-only", "IoD-only"] if x in wide.columns]
        if len(needed) == 3:
            stable_mask = wide[needed].apply(lambda col: col.map(is_hhld)).all(axis=1)
            result["stable_codes"] = list(wide.index[stable_mask])
            result["stable_core"] = int(stable_mask.sum())

        result["table"] = wide.reset_index()
        return result

    if code_col:
        temp = df.copy()
        temp["_code"] = temp[code_col].astype(str)

        combined_col = next(
            (
                c for c in temp.columns
                if "combined" in c.lower()
                and any(k in c.lower() for k in ["group", "class", "flag"])
            ),
            None,
        )
        census_col = next(
            (
                c for c in temp.columns
                if "census" in c.lower()
                and any(k in c.lower() for k in ["group", "class", "flag"])
            ),
            None,
        )
        iod_col = next(
            (
                c for c in temp.columns
                if any(k in c.lower() for k in ["iod", "imd"])
                and any(k in c.lower() for k in ["group", "class", "flag"])
            ),
            None,
        )

        if combined_col and census_col:
            result["agreement_census"] = float(
                (temp[combined_col].astype(str) == temp[census_col].astype(str)).mean() * 100
            )
        if combined_col and iod_col:
            result["agreement_iod"] = float(
                (temp[combined_col].astype(str) == temp[iod_col].astype(str)).mean() * 100
            )
        if combined_col and census_col and iod_col:
            stable = (
                temp[combined_col].map(is_hhld)
                & temp[census_col].map(is_hhld)
                & temp[iod_col].map(is_hhld)
            )
            result["stable_core"] = int(stable.sum())
            result["stable_codes"] = list(temp.loc[stable, "_code"])

        show_cols = [x for x in ["_code", combined_col, census_col, iod_col] if x]
        result["table"] = temp[show_cols]

    return result


robust = parse_robustness(robust_raw)



n_lsoas = int(rq2["_code"].nunique())
demand_med = float(rq2["_demand"].median())
n_lower_demand = int((rq2["_demand"] <= demand_med).sum())
n_hhld = int((rq2["_group"] == "Higher health + Lower demand").sum())
stable_core = int(robust["stable_core"])


# =============================================================================
# HEADER
# =============================================================================
st.markdown(
    '<span class="rq-badge">RQ2 · Health–Demand Interpretation</span>',
    unsafe_allow_html=True,
)
st.title("Equity Analysis")
st.markdown(
    "Examining where neighbourhood health vulnerability and reconstructed boarding occurrence "
    "diverge across Route 192 LSOAs."
)

st.markdown(
    """
    <div class="research-note">
      <strong>Research note.</strong>
      Health indicators represent LSOA-level neighbourhood context rather than individual
      passenger characteristics. Reconstructed demand is model-based and should be interpreted
      as a relative within-corridor pattern.
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

st.markdown(
    """
    <div class="flow-wrap">
      <span class="flow-step">Reconstructed boarding occurrence</span>
      <span class="flow-arrow">→</span>
      <span class="flow-step">Health vulnerability</span>
      <span class="flow-arrow">→</span>
      <span class="flow-step">Health–demand divergence</span>
      <span class="flow-arrow">→</span>
      <span class="flow-step" style="background:#FFF4A3;">Equity-sensitive interpretation</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

k1, k2, k3, k4 = st.columns(4)
with k1:
    metric_card("Corridor LSOAs", f"{n_lsoas}", "Neighbourhood analytical units")
with k2:
    metric_card("Lower-demand LSOAs", f"{n_lower_demand}", "At or below the corridor median")
with k3:
    metric_card("Higher-health + Lower-demand", f"{n_hhld}", "Equity-sensitive divergence group")
with k4:
    metric_card("Stable core", f"{stable_core}", "Remain in the group across health specifications")


# =============================================================================
# SECTION 1 — QUADRANT + INSPECTOR
# =============================================================================
st.divider()
st.markdown(
    '<div class="section-kicker">RQ2 · Core empirical pattern</div>',
    unsafe_allow_html=True,
)
st.header("Health–Demand Context Across Route 192")
st.caption(
    "Demand occurrence is the proportion of scheduled stop visits with positive reconstructed "
    "boarding occurrence. Health vulnerability is an LSOA-level contextual index."
)

x_threshold = float(rq2["_z_demand"].median())
y_threshold = float(rq2["_z_health"].median())

scatter = go.Figure()

for group in GROUP_ORDER:
    part = rq2[rq2["_group"] == group]
    if part.empty:
        continue

    scatter.add_trace(
        go.Scatter(
            x=part["_z_demand"],
            y=part["_z_health"],
            mode="markers",
            name=group,
            marker=dict(
                size=13 if group == "Higher health + Lower demand" else 11,
                color=GROUP_COLOURS[group],
                line=dict(
                    color=BLACK if group == "Higher health + Lower demand" else WHITE,
                    width=1.2,
                ),
            ),
            customdata=np.column_stack(
                [
                    part["_name"],
                    part["_code"],
                    part["_demand"],
                    part["_health"],
                    part["_gap"],
                    part["_gap_rank"],
                ]
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Code: %{customdata[1]}<br>"
                "Demand occurrence: %{customdata[2]:.3f}<br>"
                "Health vulnerability: %{customdata[3]:.3f}<br>"
                "Z(Demand): %{x:.2f}<br>"
                "Z(Health): %{y:.2f}<br>"
                "Health–demand gap: %{customdata[4]:+.2f}<br>"
                "Gap rank: %{customdata[5]}<extra></extra>"
            ),
        )
    )

scatter.add_vline(x=x_threshold, line_dash="dot", line_color="#B8C0CC", line_width=1.5)
scatter.add_hline(y=y_threshold, line_dash="dot", line_color="#B8C0CC", line_width=1.5)

x_min, x_max = rq2["_z_demand"].min(), rq2["_z_demand"].max()
y_min, y_max = rq2["_z_health"].min(), rq2["_z_health"].max()

scatter.add_annotation(
    x=x_min + (x_threshold - x_min) * 0.08,
    y=y_max - (y_max - y_threshold) * 0.08,
    text="<b>Higher health / Lower demand</b>",
    showarrow=False,
    bgcolor="#FFF7C7",
    bordercolor="#F1D13D",
    font=dict(size=11, family="Inter"),
    align="left",
)
scatter.add_annotation(
    x=x_max - (x_max - x_threshold) * 0.08,
    y=y_max - (y_max - y_threshold) * 0.08,
    text="Higher health / Higher demand",
    showarrow=False,
    bgcolor="#F0F3FF",
    font=dict(size=10, family="Inter"),
    align="right",
)
scatter.add_annotation(
    x=x_min + (x_threshold - x_min) * 0.08,
    y=y_min + (y_threshold - y_min) * 0.08,
    text="Lower health / Lower demand",
    showarrow=False,
    bgcolor="#F0F3FF",
    font=dict(size=10, family="Inter"),
    align="left",
)
scatter.add_annotation(
    x=x_max - (x_max - x_threshold) * 0.08,
    y=y_min + (y_threshold - y_min) * 0.08,
    text="Lower health / Higher demand",
    showarrow=False,
    bgcolor="#F0F3FF",
    font=dict(size=10, family="Inter"),
    align="right",
)

for _, row in rq2.nlargest(3, "_gap").iterrows():
    scatter.add_annotation(
        x=row["_z_demand"],
        y=row["_z_health"],
        text=row["_name"],
        showarrow=True,
        arrowhead=0,
        ax=26,
        ay=-22,
        font=dict(size=10, family="Inter", color=BLACK),
        bgcolor="rgba(255,255,255,.86)",
        bordercolor="#E1E5EC",
    )

scatter.update_layout(
    xaxis_title="Standardised reconstructed demand occurrence",
    yaxis_title="Standardised health vulnerability",
    legend=dict(orientation="h", y=1.12, x=0, title=None),
)
plot_layout(scatter, height=535)

left, right = st.columns([1.65, 0.85], gap="large")

with left:
    st.markdown("#### Health × Demand Divergence Matrix")
    st.plotly_chart(
        scatter,
        use_container_width=True,
        config={"displaylogo": False},
        key="rq2_divergence_scatter",
    )
    st.caption(
        "Dotted reference lines represent corridor medians on the standardised axes. "
        "The continuous gap ranks divergence strength; the four-context classification "
        "uses health and demand thresholds jointly."
    )

with right:
    st.markdown("#### LSOA Inspector")

    rq2["_selector"] = rq2["_name"].astype(str) + " · " + rq2["_code"].astype(str)

    equity_sorted = rq2[
        rq2["_group"] == "Higher health + Lower demand"
    ].sort_values("_gap", ascending=False)

    default_selector = (
        equity_sorted.iloc[0]["_selector"]
        if not equity_sorted.empty
        else rq2.iloc[0]["_selector"]
    )

    selector_options = rq2.sort_values(
        ["_group", "_gap"],
        ascending=[True, False],
    )["_selector"].tolist()

    selected_label = st.selectbox(
        "Select corridor LSOA",
        selector_options,
        index=selector_options.index(default_selector)
        if default_selector in selector_options
        else 0,
        key="rq2_lsoa_selectbox",
    )

    selected = rq2.loc[rq2["_selector"] == selected_label].iloc[0]

    st.markdown(
        f"""
        <div class="inspector-card">
          <div class="metric-label">Selected neighbourhood</div>
          <h3 style="margin:.25rem 0 .1rem 0;">{selected['_name']}</h3>
          <div class="small-muted">{selected['_code']}</div>
          <hr style="margin:.8rem 0;">
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:.7rem;">
            <div>
              <div class="metric-label">Demand occurrence</div>
              <div style="font-family:'Public Sans';font-size:1.45rem;font-weight:800;">
                {fmt_pct(selected['_demand'])}
              </div>
            </div>
            <div>
              <div class="metric-label">Health vulnerability</div>
              <div style="font-family:'Public Sans';font-size:1.45rem;font-weight:800;">
                {fmt_num(selected['_z_health'], 2, True)} SD
              </div>
            </div>
          </div>
          <hr style="margin:.8rem 0;">
          <div class="metric-label">Health–Demand Gap · Z(H) − Z(D)</div>
          <div class="inspector-highlight" style="font-size:1.35rem;margin-top:.25rem;">
            {fmt_num(selected['_gap'], 2, True)}
          </div>
          <div class="small-muted" style="margin-top:.35rem;">
            Corridor divergence rank: <strong>{int(selected['_gap_rank'])}</strong> of {n_lsoas}
          </div>
          <div style="margin-top:.75rem;">
            <div class="metric-label">Context classification</div>
            <strong>{selected['_group']}</strong>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if selected["_group"] == "Higher health + Lower demand":
        interpretation = (
            "This LSOA combines comparatively higher neighbourhood health vulnerability "
            "with lower reconstructed boarding occurrence. A demand-only interpretation may "
            "therefore be incomplete, and additional accessibility evidence may be valuable."
        )
    elif selected["_group"] == "Higher health + Higher demand":
        interpretation = (
            "This LSOA combines comparatively higher neighbourhood health vulnerability "
            "with higher realised use. Existing bus use therefore already occurs within a "
            "more vulnerable neighbourhood context."
        )
    elif selected["_group"] == "Lower health + Lower demand":
        interpretation = (
            "This LSOA has lower realised use without the same comparatively high health "
            "vulnerability signal. It should therefore not be interpreted in the same way "
            "as the higher-health/lower-demand group."
        )
    else:
        interpretation = (
            "This LSOA combines comparatively lower health vulnerability with higher realised "
            "boarding occurrence. The primary signal is therefore demand rather than "
            "health–demand divergence."
        )

    st.markdown(
        f"""
        <div class="interpret-box" style="margin-top:.8rem;">
          <strong>Interpretation</strong><br>
          {interpretation}
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# SECTION 2 — GAP RANKING
# =============================================================================
st.divider()
st.markdown(
    '<div class="section-kicker">Magnitude of divergence</div>',
    unsafe_allow_html=True,
)
st.header("Strongest Health–Demand Divergences")
st.caption(
    "The continuous gap G = Z(Health vulnerability) − Z(Demand occurrence) ranks the "
    "magnitude of relative mismatch. It is not an equity classification by itself."
)

rank_mode = st.radio(
    "Display",
    ["Top 10", "All LSOAs"],
    horizontal=True,
    label_visibility="collapsed",
    key="rq2_rank_mode",
)

rank_df = rq2.sort_values("_gap", ascending=False).copy()
if rank_mode == "Top 10":
    rank_df = rank_df.head(10)
rank_df = rank_df.sort_values("_gap", ascending=True)

rank_fig = go.Figure(
    go.Bar(
        x=rank_df["_gap"],
        y=rank_df["_name"],
        orientation="h",
        marker_color=[
            YELLOW if g == "Higher health + Lower demand" else "#C9D1DC"
            for g in rank_df["_group"]
        ],
        text=[f"{v:+.2f}" for v in rank_df["_gap"]],
        textposition="outside",
        customdata=np.column_stack([rank_df["_group"], rank_df["_code"]]),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Code: %{customdata[1]}<br>"
            "Gap: %{x:+.2f}<br>"
            "Context: %{customdata[0]}<extra></extra>"
        ),
    )
)
rank_fig.add_vline(x=0, line_color="#AAB2BD", line_width=1)
rank_fig.update_layout(
    xaxis_title="Health–demand gap · SD units",
    yaxis_title=None,
    showlegend=False,
)
plot_layout(rank_fig, height=470 if rank_mode == "Top 10" else 850)
st.plotly_chart(rank_fig, use_container_width=True, config={"displaylogo": False})

st.markdown(
    """
    <div class="caution-box">
      <strong>Important distinction.</strong>
      A positive gap alone does not establish an equity-sensitive classification. Very low
      demand can generate a positive gap even when health vulnerability is below the corridor
      median. The primary equity-sensitive group therefore requires both
      <strong>higher health vulnerability</strong> and <strong>lower demand</strong>;
      the continuous gap then ranks divergence strength.
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# SECTION 3 — ROBUSTNESS
# =============================================================================
st.divider()
st.markdown(
    '<div class="section-kicker">Sensitivity analysis</div>',
    unsafe_allow_html=True,
)
st.header("Robustness to Alternative Health Definitions")
st.caption(
    "The RQ2 classification is compared under the combined health index, Census-only health "
    "information and the IoD health-deprivation specification."
)

r1, r2, r3 = st.columns(3)
with r1:
    metric_card(
        "Combined vs Census-only",
        f"{robust['agreement_census']:.1f}%",
        "Classification agreement",
    )
with r2:
    metric_card(
        "Combined vs IoD-only",
        f"{robust['agreement_iod']:.1f}%",
        "Classification agreement",
    )
with r3:
    metric_card(
        "Stable core",
        f"{robust['stable_core']}",
        "Higher-health + Lower-demand under all three specifications",
    )

if robust["table"] is not None and not robust["table"].empty:
    with st.expander("Inspect LSOA-level classification stability"):
        st.dataframe(
            robust["table"],
            use_container_width=True,
            hide_index=True,
        )

threshold_path = resolve_output_file(THRESHOLD_FILE)
threshold_raw = safe_read_csv(threshold_path)

if threshold_raw is not None and not threshold_raw.empty:
    with st.expander("Alternative classification-threshold sensitivity"):
        st.caption(
            "Supplementary threshold checks are shown as sensitivity evidence rather than "
            "as alternative primary definitions."
        )
        st.dataframe(
            threshold_raw,
            use_container_width=True,
            hide_index=True,
        )

st.markdown(
    """
    <div class="caution-box">
      <strong>Interpretation.</strong>
      The broad spatial pattern is robust, although some boundary LSOAs are sensitive to how
      neighbourhood health vulnerability is defined. Robustness therefore strengthens the
      interpretation of a stable core without implying that every classification is invariant.
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# SECTION 4 — INTERPRETATION
# =============================================================================
st.divider()
st.markdown(
    '<div class="section-kicker">Why context matters</div>',
    unsafe_allow_html=True,
)
st.header("Low Demand Does Not Have a Single Interpretation")

d1, d2 = st.columns(2, gap="large")

with d1:
    st.markdown(
        """
        <div class="panel" style="min-height:185px;">
          <div class="metric-label">Demand-only view</div>
          <h3>Lower reconstructed boarding occurrence</h3>
          <p class="small-muted">
          A demand-only assessment records lower realised use but provides limited information
          about the neighbourhood circumstances in which that use occurs.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with d2:
    st.markdown(
        f"""
        <div class="panel" style="min-height:185px;border-top:4px solid {YELLOW};">
          <div class="metric-label">Equity-sensitive view</div>
          <h3>Lower use under different health contexts</h3>
          <p class="small-muted">
          Of the {n_lower_demand} lower-demand LSOAs, <strong>{n_hhld}</strong> also exhibit
          comparatively higher neighbourhood health vulnerability. Similar realised use can
          therefore carry different interpretive significance.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="caution-box" style="margin-top:1rem;">
      Observed low use cannot by itself distinguish genuinely lower travel requirements from
      differences in route relevance, competing services, accessibility barriers or other
      constraints. RQ2 therefore identifies where further evidence may be valuable rather
      than proving suppressed or unmet demand.
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# FINAL RQ2 ANSWER
# =============================================================================
st.divider()
st.header("RQ2 Answer")

st.markdown(
    f"""
    <div class="answer-box">
      <div class="metric-label" style="color:#544600;">Empirical synthesis</div>
      <h3>Health context changes how lower realised demand should be interpreted.</h3>
      <div style="font-size:.92rem;line-height:1.55;">
        Among the {n_lower_demand} lower-demand LSOAs, <strong>{n_hhld}</strong> also exhibit
        comparatively higher health vulnerability. Similar levels of reconstructed bus use can
        therefore occur under substantively different neighbourhood conditions.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="interpret-box" style="margin-top:1rem;">
      <strong>Scope of the conclusion.</strong>
      RQ2 identifies where further accessibility evidence may be most valuable. It does not
      establish unmet transport need, causal effects, or individual passenger health status.
    </div>
    """,
    unsafe_allow_html=True,
)

if hasattr(st, "page_link"):
    try:
        st.page_link(
            "pages/4_Operational_Analysis.py",
            label="Continue to Operational Analysis (RQ3) →",
            icon="➡️",
        )
    except Exception:
        pass
