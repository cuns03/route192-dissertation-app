
from pathlib import Path
from urllib.parse import quote
import json
import re
import math
import time
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

st.set_page_config(page_title="EDA | Route 192", page_icon="🚌", layout="wide")

YELLOW="#FFD800"; BLACK="#161616"; TEXT="#151C27"; MUTED="#5F5E5E"
BG="#F9F9FF"; CARD="#FFFFFF"; LOW="#F0F3FF"; BORDER="#DCE2F3"; BLUE="#0053DB"

# Self-contained Bee-style bus icon for the map.
# It is embedded as an SVG data URL, so the app does not depend on an external image.
BUS_ICON_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" width="96" height="96" viewBox="0 0 96 96">
  <rect x="17" y="13" width="62" height="65" rx="13" fill="#FFD800" stroke="#161616" stroke-width="5"/>
  <rect x="25" y="23" width="46" height="19" rx="4" fill="#161616"/>
  <rect x="29" y="27" width="18" height="11" rx="2" fill="#F9F9FF"/>
  <rect x="51" y="27" width="16" height="11" rx="2" fill="#F9F9FF"/>
  <rect x="25" y="48" width="46" height="13" rx="4" fill="#161616"/>
  <circle cx="31" cy="76" r="8" fill="#161616"/>
  <circle cx="65" cy="76" r="8" fill="#161616"/>
  <circle cx="31" cy="76" r="3" fill="#DCE2F3"/>
  <circle cx="65" cy="76" r="3" fill="#DCE2F3"/>
</svg>
"""
BUS_ICON_URL = "data:image/svg+xml;charset=utf-8," + quote(BUS_ICON_SVG)


st.markdown(f"""
<style>
.stApp {{background:{BG};}}
.block-container {{max-width:1380px;padding-top:1.3rem;padding-bottom:4rem;}}
h1,h2,h3,h4 {{font-family:"Public Sans",-apple-system,BlinkMacSystemFont,sans-serif;color:{TEXT};}}
p,div,label,span {{font-family:"Inter",-apple-system,BlinkMacSystemFont,sans-serif;}}
.route-badge {{display:inline-flex;align-items:center;gap:7px;padding:7px 12px;background:{YELLOW};color:{BLACK};font-weight:800;border-radius:8px;font-size:15px;margin-bottom:8px;}}
.kicker {{text-transform:uppercase;letter-spacing:.08em;font-size:11px;font-weight:800;color:{MUTED};}}
.subtitle {{color:{MUTED};font-size:16px;line-height:1.55;max-width:950px;}}
.research-note {{background:{LOW};border:1px solid {BORDER};border-left:4px solid {BLUE};border-radius:10px;padding:11px 14px;color:{MUTED};font-size:13px;margin:10px 0 14px 0;}}
.insight {{background:#FFF7C2;border-left:5px solid {YELLOW};border-radius:10px;padding:13px 15px;color:{TEXT};margin-top:10px;}}
.soft-card {{background:{LOW};border:1px solid {BORDER};border-radius:12px;padding:14px 16px;height:100%;}}
.level-chip {{display:inline-block;padding:4px 8px;border-radius:6px;background:{YELLOW};font-size:11px;font-weight:800;color:{BLACK};}}
[data-testid="stMetric"] {{background:{CARD};border:1px solid {BORDER};border-top:4px solid {YELLOW};padding:15px 17px;border-radius:13px;box-shadow:0 1px 4px rgba(0,0,0,.035);}}
[data-testid="stMetricLabel"] {{text-transform:uppercase;letter-spacing:.045em;font-size:11px;font-weight:800;color:{MUTED};}}
[data-testid="stMetricValue"] {{font-family:"Public Sans",sans-serif;font-weight:800;color:{TEXT};}}
[data-testid="stExpander"] {{background:{CARD};border:1px solid {BORDER};border-radius:12px;}}
.stTabs [data-baseweb="tab-list"] {{gap:8px;border-bottom:1px solid {BORDER};}}
.stTabs [data-baseweb="tab"] {{background:transparent;border-radius:8px 8px 0 0;padding:10px 16px;font-weight:700;color:{MUTED};}}
.stTabs [aria-selected="true"] {{background:{CARD};color:{TEXT};border-top:3px solid {YELLOW};}}
[data-testid="stDataFrame"] {{border:1px solid {BORDER};border-radius:10px;overflow:hidden;}}
.passenger-grid {{
    display:grid;
    grid-template-columns:repeat(3,minmax(0,1fr));
    gap:10px;
    margin:12px 0 16px 0;
}}
.passenger-card {{
    background:#FFFFFF;
    border:1px solid #DCE2F3;
    border-top:4px solid #FFD800;
    border-radius:12px;
    padding:11px 8px 12px 8px;
    text-align:center;
    min-width:0;
}}
.passenger-card .metric-label {{
    display:block;
    font-size:10px;
    line-height:13px;
    letter-spacing:.035em;
    text-transform:uppercase;
    font-weight:800;
    color:#5F5E5E;
    white-space:normal;
}}
.passenger-card .metric-value {{
    display:block;
    margin-top:4px;
    font-family:"Public Sans",sans-serif;
    font-size:31px;
    line-height:35px;
    font-weight:800;
    color:#151C27;
}}
#MainMenu {{visibility:hidden;}} footer {{visibility:hidden;}}
</style>
""", unsafe_allow_html=True)

BASE_DIR=Path(__file__).resolve().parents[1]
DATA_PATH=BASE_DIR/"data"/"raw"/"route192_master.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, low_memory=False)

df=load_data()

def first_existing(cands, frame=None):
    frame=df if frame is None else frame
    return next((c for c in cands if c in frame.columns),None)

COL={
"trip":first_existing(["trip_id","Trip_ID"]),
"stop":first_existing(["stop_id","Stop_ID"]),
"stop_name":first_existing(["stop_name","Stop_Name"]),
"direction":first_existing(["Direction","direction"]),
"period":first_existing(["detailed_time_period","time_period","operating_period"]),
"headway":first_existing(["headway_minutes","headway_min","headway"]),
"target":first_existing(["bus_demand_binary","demand_binary"]),
"lsoa":first_existing(["LSOA21CD","lsoa_code","LSOA"]),
"lsoa_name":first_existing(["LSOA21NM","lsoa_name"]),
"lat":first_existing(["stop_lat","latitude","lat"]),
"lon":first_existing(["stop_lon","longitude","lon"]),
"sequence":first_existing(["stop_sequence","stop_seq","sequence"]),
"arrival_time":first_existing(["arrival_time","scheduled_arrival_time","time","arrival_datetime"]),
"boarding":first_existing(["boarding_offered_synthetic","boarding_synthetic","boardings","boarding_count","boarding"]),
"alighting":first_existing(["alighting_synthetic","alightings","alighting_count","alighting"]),
"load":first_existing(["loading_count","load_after_synthetic","onboard_load","passenger_load","load"]),
}

req=["trip","stop","direction","period","target","lsoa"]
missing=[k for k in req if COL[k] is None]
if missing:
    st.error("Missing required fields: "+", ".join(missing)); st.stop()

def num(s): return pd.to_numeric(s,errors="coerce")
def to_percent(s):
    x=num(s)
    return x*100 if (not x.dropna().empty and x.abs().max()<=1.5) else x
def clean_plot(fig,h=420):
    fig.update_layout(height=h,paper_bgcolor=CARD,plot_bgcolor=CARD,margin=dict(l=25,r=20,t=45,b=30),font=dict(family="Inter",color=TEXT),legend_title_text="")
    fig.update_xaxes(showgrid=False,linecolor=BORDER)
    fig.update_yaxes(gridcolor="#E9ECF4",zeroline=False)
    return fig

TIME_ORDER=["Night/Early morning off-peak","AM peak","AM interpeak","PM interpeak","PM peak","Evening off-peak"]

# Preferred display order only. If the dataset uses other direction labels,
# they are still retained after these values.
DIRECTION_ORDER=[
    "Hazel Grove to Manchester",
    "Manchester to Hazel Grove",
    "Hazel Grove → Manchester",
    "Manchester → Hazel Grove",
]
stop_df=df.drop_duplicates(COL["stop"]).copy()
lsoa_df=df.drop_duplicates(COL["lsoa"]).copy()


# ============================================================
# DISSERTATION-ALIGNED EDA HELPERS
# ============================================================

LSOA_GEO_CANDIDATES = [
    BASE_DIR / "data" / "geo" / "route192_lsoas.geojson",
    BASE_DIR / "data" / "geo" / "route192_lsoa_boundaries.geojson",
    BASE_DIR / "data" / "geo" / "lsoa_boundaries.geojson",
    BASE_DIR / "data" / "geo" / "Lower_Layer_Super_Output_Areas_2021.geojson",
]

def _valid_series(frame, col, percent=False):
    vals = num(frame[col]).dropna()
    return to_percent(vals) if percent else vals

def descriptive_row(frame, col, label, percent=False, include_sd=True):
    vals = _valid_series(frame, col, percent=percent)
    if vals.empty:
        return None

    row = {
        "Characteristic": label,
        "n": int(vals.count()),
        "Min": vals.min(),
        "Mean": vals.mean(),
        "Median": vals.median(),
        "Q1": vals.quantile(0.25),
        "Q3": vals.quantile(0.75),
        "Max": vals.max(),
    }
    if include_sd:
        row["SD"] = vals.std()
    return row

def descriptive_table(rows, digits=2, integer_cols=None):
    rows = [r for r in rows if r is not None]
    if not rows:
        return
    stat_df = pd.DataFrame(rows)
    integer_cols = set(integer_cols or [])

    fmt = {}
    for c in stat_df.columns:
        if c == "Characteristic":
            continue
        if c == "n":
            fmt[c] = "{:.0f}"
        elif c in integer_cols:
            fmt[c] = "{:,.0f}"
        else:
            fmt[c] = "{:,.%df}" % digits

    st.dataframe(
        stat_df.style.format(fmt),
        use_container_width=True,
        hide_index=True,
    )

def get_population_col(frame):
    return first_existing(
        [
            "gender_total_population",
            "population_total",
            "resident_population",
            "population_2021",
            "total_population",
        ],
        frame,
    )

def get_female_col(frame):
    return first_existing(
        [
            "female_rate",
            "female_rate_2021",
            "gender_female_rate",
            "female_share",
        ],
        frame,
    )

def representative_route():
    """Return one longest geocoded trip for a clean Route 192 overlay."""
    if map_lat_col is None or map_lon_col is None:
        return pd.DataFrame(columns=["lat", "lon"])

    valid = map_df.dropna(subset=[map_lat_col, map_lon_col]).copy()
    if valid.empty:
        return pd.DataFrame(columns=["lat", "lon"])

    counts = valid.groupby(COL["trip"])[COL["stop"]].nunique()
    if counts.empty:
        return pd.DataFrame(columns=["lat", "lon"])

    chosen_trip = counts.idxmax()
    route = valid[valid[COL["trip"]] == chosen_trip].copy()

    if COL["sequence"] is not None:
        route["_seq"] = num(route[COL["sequence"]])
        route = route.sort_values("_seq")

    route = route.drop_duplicates(COL["stop"])
    return route[[map_lat_col, map_lon_col]].rename(
        columns={map_lat_col: "lat", map_lon_col: "lon"}
    )

def lsoa_centroids():
    """Fallback LSOA points based on Route 192 stop coordinates."""
    if map_lat_col is None or map_lon_col is None:
        return pd.DataFrame(columns=[COL["lsoa"], "lat", "lon"])

    valid = map_df.dropna(
        subset=[COL["lsoa"], map_lat_col, map_lon_col]
    ).copy()

    if valid.empty:
        return pd.DataFrame(columns=[COL["lsoa"], "lat", "lon"])

    return (
        valid.groupby(COL["lsoa"], as_index=False)
        .agg(
            lat=(map_lat_col, "mean"),
            lon=(map_lon_col, "mean"),
        )
    )

ONS_LSOA_QUERY_URL = (
    "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/"
    "Lower_layer_Super_Output_Areas_December_2021_Boundaries_EW_BGC_V5/"
    "FeatureServer/0/query"
)

def _normalise_lsoa_geojson(geo, codes_tuple):
    """Keep only corridor LSOAs and create a stable code property for Plotly."""
    features = geo.get("features", []) if isinstance(geo, dict) else []
    if not features:
        return None

    corridor_codes = {str(x).strip() for x in codes_tuple}
    code_candidates = [
        "LSOA21CD",
        "LSOA21CD_1",
        "lsoa21cd",
        "lsoa_code",
        "LSOA11CD",
        "code",
    ]

    kept = []

    for feat in features:
        props = feat.get("properties", {}) or {}
        code_key = next((k for k in code_candidates if k in props), None)
        if code_key is None:
            continue

        code = str(props.get(code_key, "")).strip()
        if code not in corridor_codes:
            continue

        feat = dict(feat)
        props = dict(props)
        props["_lsoa_code"] = code
        feat["properties"] = props
        kept.append(feat)

    if not kept:
        return None

    return {
        "type": "FeatureCollection",
        "features": kept,
    }

@st.cache_data(show_spinner=False, ttl=86400)
def load_lsoa_geojson(codes_tuple):
    """
    Load Route 192 LSOA polygons.

    Priority:
    1. Local cached GeoJSON if available.
    2. ONS Open Geography ArcGIS API.
    3. Return None so the app can fall back to stop centroids.

    The API response is saved locally after the first successful request, so
    later app launches normally load instantly without repeating the request.
    """
    # --------------------------------------------------------
    # 1. Local cache first
    # --------------------------------------------------------
    geo_path = next((p for p in LSOA_GEO_CANDIDATES if p.exists()), None)

    if geo_path is not None:
        try:
            with open(geo_path, "r", encoding="utf-8") as f:
                local_geo = json.load(f)

            local_geo = _normalise_lsoa_geojson(local_geo, codes_tuple)

            if local_geo is not None:
                local_geo["_source"] = "Local ONS polygon cache"
                return local_geo
        except Exception:
            # If a local file is corrupt or incompatible, continue to API.
            pass

    # --------------------------------------------------------
    # 2. ONS Open Geography API
    # --------------------------------------------------------
    corridor_codes = sorted({str(x).strip() for x in codes_tuple if str(x).strip()})

    if not corridor_codes:
        return None

    quoted_codes = ",".join(f"'{code}'" for code in corridor_codes)

    params = {
        "where": f"LSOA21CD IN ({quoted_codes})",
        "outFields": "LSOA21CD,LSOA21NM",
        "returnGeometry": "true",
        "outSR": "4326",
        "f": "geojson",
    }

    try:
        response = requests.get(
            ONS_LSOA_QUERY_URL,
            params=params,
            headers={"User-Agent": "Route192-MSc-Streamlit/1.0"},
            timeout=30,
        )
        response.raise_for_status()

        api_geo = response.json()

        if isinstance(api_geo, dict) and api_geo.get("error"):
            return None

        api_geo = _normalise_lsoa_geojson(api_geo, codes_tuple)

        if api_geo is None:
            return None

        api_geo["_source"] = "ONS Open Geography API"

        # ----------------------------------------------------
        # 3. Save automatically for subsequent app launches
        # ----------------------------------------------------
        try:
            cache_path = BASE_DIR / "data" / "geo" / "route192_lsoas.geojson"
            cache_path.parent.mkdir(parents=True, exist_ok=True)

            # Remove internal metadata before writing GeoJSON.
            to_save = {
                "type": "FeatureCollection",
                "features": api_geo["features"],
            }

            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(to_save, f)
        except Exception:
            # Read-only deployments can still use the in-memory API response.
            pass

        return api_geo

    except Exception:
        return None

def _map_center():
    route = representative_route()
    if not route.empty:
        return {
            "lat": float(route["lat"].mean()),
            "lon": float(route["lon"].mean()),
        }
    return {"lat": 53.42, "lon": -2.16}


def _route_bbox(pad_lon=0.045, pad_lat=0.035):
    """Bounding box around Route 192, padded to show neighbouring LSOAs."""
    route = representative_route()

    if route.empty:
        return (-2.35, 53.30, -1.95, 53.58)

    return (
        float(route["lon"].min() - pad_lon),
        float(route["lat"].min() - pad_lat),
        float(route["lon"].max() + pad_lon),
        float(route["lat"].max() + pad_lat),
    )

@st.cache_data(show_spinner=False, ttl=86400)
def load_surrounding_lsoa_geojson(corridor_codes_tuple):
    """
    Retrieve LSOAs around Route 192 from the ONS Open Geography API.

    Route-192 LSOAs are excluded from the returned collection; these polygons
    are used only as a neutral contextual background.
    """
    xmin, ymin, xmax, ymax = _route_bbox()

    params = {
        "where": "1=1",
        "geometry": f"{xmin},{ymin},{xmax},{ymax}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "LSOA21CD,LSOA21NM",
        "returnGeometry": "true",
        "outSR": "4326",
        "f": "geojson",
        "resultRecordCount": 2000,
    }

    try:
        response = requests.get(
            ONS_LSOA_QUERY_URL,
            params=params,
            headers={"User-Agent": "Route192-MSc-Streamlit/1.0"},
            timeout=30,
        )
        response.raise_for_status()
        geo = response.json()

        if not isinstance(geo, dict) or geo.get("error"):
            return None

        corridor_codes = {
            str(code).strip()
            for code in corridor_codes_tuple
            if str(code).strip()
        }

        kept = []

        for feat in geo.get("features", []):
            props = feat.get("properties", {}) or {}
            code = str(props.get("LSOA21CD", "")).strip()

            if not code or code in corridor_codes:
                continue

            feat = dict(feat)
            props = dict(props)
            props["_lsoa_code"] = code
            feat["properties"] = props
            kept.append(feat)

        if not kept:
            return None

        return {
            "type": "FeatureCollection",
            "features": kept,
        }

    except Exception:
        return None

def make_lsoa_map(frame, value_col, label, percent=False):
    """Interactive LSOA choropleth; centroid map is used if polygon geometry is absent."""

    # Stable plotting names: avoid LSOA21CD_x / LSOA21CD_y after merge.
    plot_df = pd.DataFrame(
        {
            "LSOA code": frame[COL["lsoa"]].astype(str),
            "Value": (
                to_percent(frame[value_col])
                if percent
                else num(frame[value_col])
            ),
        }
    )

    if COL["lsoa_name"] is not None and COL["lsoa_name"] in frame.columns:
        plot_df["LSOA name"] = frame[COL["lsoa_name"]].astype(str)
    else:
        plot_df["LSOA name"] = plot_df["LSOA code"]

    plot_df = (
        plot_df
        .dropna(subset=["Value"])
        .drop_duplicates("LSOA code")
    )

    codes = tuple(plot_df["LSOA code"].tolist())
    geojson = load_lsoa_geojson(codes)
    center = _map_center()

    hover_fmt = ".1f" if percent else ",.0f"
    if "score" in value_col.lower():
        hover_fmt = ".2f"

    if geojson is not None:
        # ------------------------------------------------------------
        # Route 192 analytical polygons
        # ------------------------------------------------------------
        corridor_fig = px.choropleth_mapbox(
            plot_df,
            geojson=geojson,
            locations="LSOA code",
            featureidkey="properties._lsoa_code",
            color="Value",
            hover_name="LSOA name",
            hover_data={
                "LSOA code": True,
                "Value": f":{hover_fmt}",
            },
            color_continuous_scale="YlGnBu",
            opacity=0.82,
            center=center,
            zoom=9.35,
            mapbox_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
            labels={"Value": label},
        )
        corridor_fig.update_traces(
            marker_line_width=1.25,
            marker_line_color="#60666D",
        )

        # ------------------------------------------------------------
        # Neighbouring non-Route-192 LSOAs:
        # neutral light-grey contextual layer, similar to simulation.
        # ------------------------------------------------------------
        context_geojson = load_surrounding_lsoa_geojson(codes)

        fig = go.Figure()

        if context_geojson is not None:
            context_codes = [
                feature.get("properties", {}).get("_lsoa_code")
                for feature in context_geojson.get("features", [])
            ]
            context_codes = [c for c in context_codes if c]

            if context_codes:
                fig.add_trace(
                    go.Choroplethmapbox(
                        geojson=context_geojson,
                        locations=context_codes,
                        z=[0] * len(context_codes),
                        featureidkey="properties._lsoa_code",
                        colorscale=[
                            [0.0, "#EEF1F4"],
                            [1.0, "#EEF1F4"],
                        ],
                        showscale=False,
                        marker_opacity=0.10,
                        marker_line_width=0.65,
                        marker_line_color="#D9DEE3",
                        hoverinfo="skip",
                        name="Neighbouring LSOAs",
                        showlegend=False,
                    )
                )

        # Analytical corridor polygons sit above the neutral context layer.
        for trace in corridor_fig.data:
            fig.add_trace(trace)

        fig.update_layout(corridor_fig.layout)
        geometry_mode = "LSOA polygons"

    else:
        centroids = lsoa_centroids().rename(
            columns={COL["lsoa"]: "LSOA code"}
        )

        plot_df = (
            plot_df
            .merge(centroids, on="LSOA code", how="left")
            .dropna(subset=["lat", "lon"])
        )

        fig = px.scatter_mapbox(
            plot_df,
            lat="lat",
            lon="lon",
            color="Value",
            size=np.repeat(8, len(plot_df)),
            size_max=13,
            hover_name="LSOA name",
            hover_data={
                "LSOA code": True,
                "Value": f":{hover_fmt}",
                "lat": False,
                "lon": False,
            },
            color_continuous_scale="YlGnBu",
            center=center,
            zoom=9.35,
            mapbox_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
            labels={"Value": label},
        )
        geometry_mode = "LSOA stop-centroid fallback"

    # Overlay one representative Route 192 path.
    route = representative_route()
    if not route.empty:
        fig.add_trace(
            go.Scattermapbox(
                lat=route["lat"],
                lon=route["lon"],
                mode="lines",
                line=dict(width=4, color=YELLOW),
                hoverinfo="skip",
                name="Route 192",
                showlegend=False,
            )
        )

    fig.update_layout(
        height=560,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor=CARD,
        font=dict(family="Inter", color=TEXT),
        # Keep the user's zoom/pan position when another Streamlit widget reruns.
        uirevision=f"lsoa-map-{value_col}",
        coloraxis_colorbar=dict(
            title=label,
            thickness=12,
            len=0.65,
        ),
    )
    return fig, geometry_mode

def render_lsoa_inspector(frame, value_col, label, percent=False, related=None):
    """LSOA selector showing the chosen area's value and useful contextual fields."""
    available = frame.dropna(subset=[COL["lsoa"]]).copy()
    available["_code"] = available[COL["lsoa"]].astype(str)

    if COL["lsoa_name"] is not None and COL["lsoa_name"] in available.columns:
        available["_display"] = (
            available[COL["lsoa_name"]].astype(str)
            + " · "
            + available["_code"]
        )
    else:
        available["_display"] = available["_code"]

    lookup = dict(zip(available["_code"], available["_display"]))
    selected_code = st.selectbox(
        "Inspect LSOA",
        available["_code"].drop_duplicates().tolist(),
        format_func=lambda c: lookup.get(c, c),
        key=f"inspect_{re.sub(r'[^a-zA-Z0-9]+','_',value_col)}",
    )

    row = available[available["_code"] == selected_code].iloc[0]

    raw_value = num(pd.Series([row[value_col]])).iloc[0]
    shown_value = (
        to_percent(pd.Series([raw_value])).iloc[0]
        if percent
        else raw_value
    )

    corridor_vals = (
        to_percent(available[value_col])
        if percent
        else num(available[value_col])
    ).dropna()

    percentile = (
        float((corridor_vals <= shown_value).mean() * 100)
        if not corridor_vals.empty
        else np.nan
    )
    median = corridor_vals.median() if not corridor_vals.empty else np.nan

    if "score" in value_col.lower():
        value_text = f"{shown_value:.2f}"
        med_text = f"{median:.2f}"
    elif percent:
        value_text = f"{shown_value:.1f}%"
        med_text = f"{median:.1f}%"
    else:
        value_text = f"{shown_value:,.0f}"
        med_text = f"{median:,.0f}"

    st.metric(label, value_text)
    if not np.isnan(percentile):
        st.caption(
            f"Corridor percentile: {percentile:.0f}th · "
            f"Corridor median: {med_text}"
        )

    if related:
        st.markdown("##### Local profile")
        cols = st.columns(2)
        for i, item in enumerate(related):
            rel_label, rel_col, rel_percent = item
            if rel_col is None or rel_col not in row.index or pd.isna(row[rel_col]):
                continue

            value = num(pd.Series([row[rel_col]])).iloc[0]
            if rel_percent:
                value = to_percent(pd.Series([value])).iloc[0]
                display = f"{value:.1f}%"
            elif "score" in rel_col.lower():
                display = f"{value:.2f}"
            else:
                display = f"{value:,.0f}"

            cols[i % 2].metric(rel_label, display)

def render_lsoa_metric_section(
    frame,
    value_col,
    title,
    label,
    percent=False,
    include_sd=True,
    interpretation=None,
    related=None,
    key_prefix=None,
):
    st.markdown(f"### {title}")

    left, right = st.columns([2.15, 0.95], gap="large")
    with left:
        fig, geometry_mode = make_lsoa_map(
            frame,
            value_col=value_col,
            label=label,
            percent=percent,
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
            key=f"map_{key_prefix or value_col}",
            config={
                # Trackpad / mouse-wheel zoom.
                "scrollZoom": True,

                # Keep navigation buttons visible instead of only on hover.
                "displayModeBar": True,
                "displaylogo": False,

                # Convenient Mapbox-specific buttons:
                # + zoom in, - zoom out, home/reset view.
                "modeBarButtonsToAdd": [
                    "zoomInMapbox",
                    "zoomOutMapbox",
                    "resetViewMapbox",
                ],

                # Remove tools that are not useful for this dissertation map.
                "modeBarButtonsToRemove": [
                    "select2d",
                    "lasso2d",
                ],

                # Double-click returns to the default corridor view.
                "doubleClick": "reset",
            },
        )

        st.caption(
            "Coloured polygons = Route 192 LSOAs · light grey polygons = surrounding LSOAs. "
            "Use the mouse wheel/trackpad to zoom, drag to move, or use the + / − / reset controls."
        )
        if geometry_mode != "LSOA polygons":
            st.caption(
                "LSOA polygon boundaries could not be retrieved from the ONS Open "
                "Geography API, so the map is temporarily using stop-centroid markers."
            )

    with right:
        render_lsoa_inspector(
            frame,
            value_col=value_col,
            label=label,
            percent=percent,
            related=related,
        )

    st.markdown("##### Descriptive statistics")
    row = descriptive_row(
        frame,
        value_col,
        label,
        percent=percent,
        include_sd=include_sd,
    )
    descriptive_table(
        [row],
        digits=2 if ("score" in value_col.lower()) else 1,
        integer_cols={label} if not percent and "score" not in value_col.lower() else None,
    )

    if interpretation:
        st.markdown(
            f'<div class="insight"><b>EDA interpretation:</b> {interpretation}</div>',
            unsafe_allow_html=True,
        )


st.markdown('<div class="route-badge">🚌 Route 192</div>',unsafe_allow_html=True)
st.markdown('<div class="kicker">Bee Network corridor research · MSc dissertation extension</div>',unsafe_allow_html=True)
st.title("Exploratory Data Analysis")
st.markdown('<div class="subtitle">Explore service provision, reconstructed boarding demand and neighbourhood context along Greater Manchester’s Route 192.</div>',unsafe_allow_html=True)

st.markdown("### Dataset overview")
c1,c2,c3,c4=st.columns(4)
c1.metric("Scheduled Stop Visits",f"{len(df):,}")
c2.metric("Trips",f"{df[COL['trip']].nunique():,}")
c3.metric("Directional Stops",f"{df[COL['stop']].nunique():,}")
c4.metric("LSOAs",f"{df[COL['lsoa']].nunique():,}")

with st.expander("Dataset Structure & Multilevel Quality Architecture"):
    a,b,c=st.columns(3)
    with a:
        st.markdown('<span class="level-chip">LEVEL 1 · STOP VISIT</span>',unsafe_allow_html=True)
        st.markdown("#### Scheduled Stop Visits"); st.write("Direction, operating period, headway and reconstructed boarding occurrence.")
    with b:
        st.markdown('<span class="level-chip">LEVEL 2 · STOP</span>',unsafe_allow_html=True)
        st.markdown("#### Directional Stops"); st.write("Built-environment counts within 400 m.")
    with c:
        st.markdown('<span class="level-chip">LEVEL 3 · LSOA</span>',unsafe_allow_html=True)
        st.markdown("#### Neighbourhood Context"); st.write("Census, deprivation and health context.")

st.markdown("---")
st.markdown("## Explore Route 192 in Motion")
st.markdown(
    '<div class="research-note"><b>Research notice:</b> passenger activity is reconstructed '
    'from manual observations and scheduled service information. This is an analytical '
    'playback, not live bus tracking.</div>',
    unsafe_allow_html=True,
)

# ============================================================
# LOCAL GEO LOOKUP — PRODUCTION / DEMO MODE
# No live geocoding. Supervisor opens the app and the map loads immediately.
# ============================================================

GEO_PATH = BASE_DIR / "data" / "geo" / "route192_stop_coordinates.csv"

map_df = df.copy()
map_lat_col = COL["lat"]
map_lon_col = COL["lon"]

if map_lat_col is None or map_lon_col is None:
    if not GEO_PATH.exists():
        st.error(
            "Missing `data/geo/route192_stop_coordinates.csv`. "
            "Place the saved Route 192 coordinate lookup in that folder."
        )
        st.stop()

    geo = pd.read_csv(GEO_PATH, low_memory=False)

    geo_stop_col = next(
        (c for c in ["stop_id", "Stop_ID"] if c in geo.columns),
        None,
    )
    geo_lat_col = next(
        (c for c in ["stop_lat", "latitude", "lat"] if c in geo.columns),
        None,
    )
    geo_lon_col = next(
        (c for c in ["stop_lon", "longitude", "lon"] if c in geo.columns),
        None,
    )

    if geo_stop_col is None or geo_lat_col is None or geo_lon_col is None:
        st.error(
            "The local coordinate file must contain a stop identifier and latitude/longitude. "
            "Expected fields such as `stop_id`, `stop_lat`, `stop_lon`."
        )
        st.stop()

    # Normalise IDs on both sides before joining.
    map_df[COL["stop"]] = (
        map_df[COL["stop"]]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

    geo[geo_stop_col] = (
        geo[geo_stop_col]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

    geo_lookup = (
        geo[[geo_stop_col, geo_lat_col, geo_lon_col]]
        .rename(
            columns={
                geo_stop_col: "_geo_stop_id",
                geo_lat_col: "_stop_lat",
                geo_lon_col: "_stop_lon",
            }
        )
        .assign(
            _stop_lat=lambda x: pd.to_numeric(x["_stop_lat"], errors="coerce"),
            _stop_lon=lambda x: pd.to_numeric(x["_stop_lon"], errors="coerce"),
        )
        .dropna(subset=["_stop_lat", "_stop_lon"])
        .drop_duplicates(subset="_geo_stop_id")
    )

    map_df = map_df.merge(
        geo_lookup,
        left_on=COL["stop"],
        right_on="_geo_stop_id",
        how="left",
        validate="many_to_one",
    )

    map_lat_col = "_stop_lat"
    map_lon_col = "_stop_lon"

coordinate_coverage = (
    map_df[[map_lat_col, map_lon_col]]
    .notna()
    .all(axis=1)
    .mean()
)

if coordinate_coverage < 0.80:
    st.error(
        f"Only {coordinate_coverage:.1%} of analytical rows could be matched to coordinates. "
        "Check whether the stop IDs in `route192_stop_coordinates.csv` match the master dataset."
    )
elif coordinate_coverage < 0.98:
    st.warning(
        f"Coordinate coverage is {coordinate_coverage:.1%}. "
        "The map will use the matched stops and omit unmatched locations."
    )

# ============================================================
# MAP FILTERS
# ============================================================

map_controls = st.container()
with map_controls:
    m1, m2, m3 = st.columns([1.15, 1.15, 1.7])

    direction_values = (
        map_df[COL["direction"]]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    direction_values = (
        [d for d in DIRECTION_ORDER if d in direction_values]
        + [d for d in direction_values if d not in DIRECTION_ORDER]
    )

    with m1:
        map_direction = st.selectbox(
            "Direction",
            direction_values,
            key="map_direction",
        )

    period_values = (
        map_df.loc[
            map_df[COL["direction"]].astype(str).eq(str(map_direction)),
            COL["period"],
        ]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    period_values = (
        [p for p in TIME_ORDER if p in period_values]
        + [p for p in period_values if p not in TIME_ORDER]
    )

    with m2:
        map_period = st.selectbox(
            "Operating period",
            period_values,
            key="map_period",
        )

    trip_values = (
        map_df.loc[
            map_df[COL["direction"]].astype(str).eq(str(map_direction))
            & map_df[COL["period"]].astype(str).eq(str(map_period)),
            COL["trip"],
        ]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    with m3:
        map_trip = st.selectbox(
            "Trip",
            trip_values,
            key="map_trip",
        ) if trip_values else None

if map_trip is None:
    st.info("No trip is available for the selected direction and operating period.")

else:
    trip_df = map_df[
        map_df[COL["trip"]].astype(str).eq(str(map_trip))
    ].copy()

    # Order stops within the trip.
    if COL["sequence"] is not None:
        trip_df["_seq"] = num(trip_df[COL["sequence"]])
        trip_df = trip_df.sort_values("_seq")
    else:
        # Fallback preserves dataset order if no stop sequence is available.
        trip_df = trip_df.reset_index(drop=False).rename(columns={"index": "_seq"})

    trip_df = (
        trip_df
        .dropna(subset=[map_lat_col, map_lon_col])
        .drop_duplicates(subset=[COL["stop"], "_seq"])
        .reset_index(drop=True)
    )

    if len(trip_df) < 2:
        st.warning("The selected trip has fewer than two geocoded stops.")
    else:
        # ----------------------------------------------------
        # Reset animation automatically when trip selection changes
        # ----------------------------------------------------
        trip_signature = f"{map_direction}|{map_period}|{map_trip}"

        if st.session_state.get("_map_trip_signature") != trip_signature:
            st.session_state["_map_trip_signature"] = trip_signature
            st.session_state["_map_playing"] = False
            st.session_state["_pending_map_index"] = 0

        # Apply automatic step BEFORE slider widget is created.
        if "_pending_map_index" in st.session_state:
            pending = int(st.session_state.pop("_pending_map_index"))
            st.session_state["journey_slider"] = max(
                0, min(pending, len(trip_df) - 1)
            )

        if "journey_slider" not in st.session_state:
            st.session_state["journey_slider"] = 0

        # ----------------------------------------------------
        # Playback controls
        # One button toggles Play <-> Pause.
        # ----------------------------------------------------
        pc1, pc2, pc3 = st.columns([1.45, 0.85, 0.85])

        with pc1:
            # Match the vertical position of the Speed selectbox label
            # so Play/Pause, Speed and Reset sit on the same row.
            st.markdown(
                "<div style='height:28px'></div>",
                unsafe_allow_html=True,
            )

            is_playing = st.session_state.get("_map_playing", False)
            toggle_label = (
                "⏸ Pause Simulation"
                if is_playing
                else "▶ Play Simulation"
            )

            if st.button(
                toggle_label,
                use_container_width=True,
                key="toggle_map_playback",
                type="primary" if not is_playing else "secondary",
            ):
                st.session_state["_map_playing"] = not is_playing
                st.rerun()

        with pc2:
            speed = st.selectbox(
                "Speed",
                [0.5, 1.0, 2.0],
                index=1,
                format_func=lambda x: f"{x:g}×",
                key="map_speed",
            )

        with pc3:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            if st.button(
                "↺ Reset",
                use_container_width=True,
                key="reset_map_playback",
            ):
                st.session_state["_map_playing"] = False
                st.session_state["journey_slider"] = 0
                st.rerun()

        sim_idx = st.slider(
            "Journey playback",
            min_value=0,
            max_value=len(trip_df) - 1,
            step=1,
            key="journey_slider",
            help="Drag manually, or use Play/Pause to move through the selected trip.",
        )

        current = trip_df.iloc[sim_idx]
        next_row = trip_df.iloc[sim_idx + 1] if sim_idx < len(trip_df) - 1 else None

        # ----------------------------------------------------
        # Passenger variables
        # ----------------------------------------------------
        name_col = COL["stop_name"] if COL["stop_name"] else COL["stop"]

        def current_value(frame_row, key):
            col = COL[key]
            if col is None or pd.isna(frame_row[col]):
                return np.nan
            return float(num(pd.Series([frame_row[col]])).iloc[0])

        board = current_value(current, "boarding")
        alight = current_value(current, "alighting")
        load = current_value(current, "load")

        # Add clean numeric fields used by the spatial layers.
        trip_df["_boarding"] = (
            num(trip_df[COL["boarding"]]).fillna(0)
            if COL["boarding"] is not None
            else 0
        )
        trip_df["_alighting"] = (
            num(trip_df[COL["alighting"]]).fillna(0)
            if COL["alighting"] is not None
            else 0
        )
        trip_df["_load"] = (
            num(trip_df[COL["load"]]).fillna(0)
            if COL["load"] is not None
            else 0
        )

        # ----------------------------------------------------
        # Route geometry from ordered stop coordinates
        # ----------------------------------------------------
        route = trip_df[[map_lon_col, map_lat_col]].rename(
            columns={map_lon_col: "lon", map_lat_col: "lat"}
        )
        completed = route.iloc[: sim_idx + 1].copy()

        # Stop points, including passenger context for hover.
        stops = trip_df[
            [
                COL["stop"],
                name_col,
                map_lon_col,
                map_lat_col,
                "_boarding",
                "_alighting",
                "_load",
            ]
        ].copy()

        stops = stops.rename(
            columns={
                COL["stop"]: "stop_id",
                name_col: "name",
                map_lon_col: "lon",
                map_lat_col: "lat",
            }
        )

        # Keep stop markers visually stable. Passenger quantities are shown
        # in the information panel and tooltip rather than through a map-mode selector.
        stops["marker_radius"] = 28

        # ----------------------------------------------------
        # PyDeck layers
        # ----------------------------------------------------
        layers = []

        # Grey full route provides context.
        layers.append(
            pdk.Layer(
                "PathLayer",
                data=[{"path": route[["lon", "lat"]].values.tolist()}],
                get_path="path",
                get_width=6,
                width_min_pixels=4,
                get_color=[190, 195, 205, 180],
                pickable=False,
            )
        )

        # Completed route is highlighted in Bee yellow.
        layers.append(
            pdk.Layer(
                "PathLayer",
                data=[{"path": completed[["lon", "lat"]].values.tolist()}],
                get_path="path",
                get_width=8,
                width_min_pixels=5,
                get_color=[255, 216, 0, 240],
                pickable=False,
            )
        )

        # Neutral stop markers keep the map readable while the yellow route
        # and bus remain the visual focus.
        stop_fill = [255, 255, 255, 225]

        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=stops,
                get_position="[lon, lat]",
                get_radius="marker_radius",
                radius_min_pixels=3,
                radius_max_pixels=12,
                get_fill_color=stop_fill,
                get_line_color=[22, 22, 22, 220],
                line_width_min_pixels=1,
                stroked=True,
                pickable=True,
            )
        )

        # Current bus position: yellow halo + actual bus icon.
        bus_data = [
            {
                "lon": float(current[map_lon_col]),
                "lat": float(current[map_lat_col]),
                "name": "Route 192 bus",
                "icon": {
                    "url": BUS_ICON_URL,
                    "width": 96,
                    "height": 96,
                    "anchorY": 48,
                },
            }
        ]

        # Subtle halo makes the vehicle easy to find on the basemap.
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=bus_data,
                get_position="[lon, lat]",
                get_radius=35,
                radius_min_pixels=7,
                radius_max_pixels=10,
                get_fill_color=[255, 216, 0, 75],
                get_line_color=[22, 22, 22, 110],
                line_width_min_pixels=2,
                stroked=True,
                pickable=False,
            )
        )

        layers.append(
            pdk.Layer(
                "IconLayer",
                data=bus_data,
                get_position="[lon, lat]",
                get_icon="icon",
                get_size=24,
                size_units="pixels",
                size_min_pixels=18,
                size_max_pixels=26,
                pickable=False,
            )
        )

        # ----------------------------------------------------
        # Basemap and information panel
        # ----------------------------------------------------
        view = pdk.ViewState(
            latitude=float(route["lat"].mean()),
            longitude=float(route["lon"].mean()),
            zoom=10.7,
            pitch=0,
        )

        tooltip = {
            "html": (
                "<b>{name}</b><br/>"
                "Boarding: {_boarding}<br/>"
                "Alighting: {_alighting}<br/>"
                "Onboard: {_load}"
            ),
            "style": {
                "backgroundColor": "white",
                "color": "#151C27",
                "fontSize": "12px",
            },
        }

        map_col, info_col = st.columns([2.2, 1])

        with map_col:
            st.caption(
                f"Route 192 trip playback · stop {sim_idx + 1} of {len(trip_df)}"
            )

            st.pydeck_chart(
                pdk.Deck(
                    map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
                    initial_view_state=view,
                    layers=layers,
                    tooltip=tooltip,
                ),
                use_container_width=True,
                height=540,
            )

            st.progress((sim_idx + 1) / len(trip_df))
            st.caption(
                f"{trip_df.iloc[0][name_col]}  →  "
                f"**{current[name_col]}**  →  {trip_df.iloc[-1][name_col]}"
            )

        with info_col:
            st.markdown("### 🚌 Current stop")
            st.markdown(f"**{current[name_col]}**")
            st.caption(
                f"{current[COL['lsoa']]} · stop {sim_idx + 1} of {len(trip_df)}"
            )

            board_display = "—" if math.isnan(board) else f"+{board:.0f}"
            alight_display = "—" if math.isnan(alight) else f"−{alight:.0f}"
            load_display = "—" if math.isnan(load) else f"{load:.0f}"

            st.markdown(
                f"""
                <div class="passenger-grid">
                    <div class="passenger-card">
                        <span class="metric-label">Boarding</span>
                        <span class="metric-value">{board_display}</span>
                    </div>
                    <div class="passenger-card">
                        <span class="metric-label">Alighting</span>
                        <span class="metric-value">{alight_display}</span>
                    </div>
                    <div class="passenger-card">
                        <span class="metric-label">Onboard</span>
                        <span class="metric-value">{load_display}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if not math.isnan(load):
                st.markdown("**Onboard passenger load**")
                st.progress(min(max(load / 100, 0), 1))
                st.caption(
                    f"{load:.0f}/100 passengers · "
                    "73 passengers = seated-capacity reference"
                )

            if COL["arrival_time"] is not None and pd.notna(current[COL["arrival_time"]]):
                st.write(f"**Time:** {current[COL['arrival_time']]}")

            if next_row is not None:
                st.write(f"**Next stop:** {next_row[name_col]}")
            else:
                st.write("**Next stop:** Final terminus")

            if COL["headway"] is not None and pd.notna(current[COL["headway"]]):
                current_headway = num(pd.Series([current[COL["headway"]]])).iloc[0]
                if pd.notna(current_headway):
                    st.write(f"**Scheduled headway:** {current_headway:.1f} min")

        # ----------------------------------------------------
        # Trip summary
        # ----------------------------------------------------
        s1, s2, s3, s4, s5 = st.columns(5)

        s1.metric(
            "Total Boardings",
            "—"
            if COL["boarding"] is None
            else f"{num(trip_df[COL['boarding']]).sum():.0f}",
        )
        s2.metric(
            "Total Alightings",
            "—"
            if COL["alighting"] is None
            else f"{num(trip_df[COL['alighting']]).sum():.0f}",
        )
        s3.metric(
            "Peak Onboard Load",
            "—"
            if COL["load"] is None
            else f"{num(trip_df[COL['load']]).max():.0f}",
        )
        s4.metric(
            "Demand-Positive Visits",
            f"{num(trip_df[COL['target']]).fillna(0).gt(0).sum()}/{len(trip_df)}",
        )
        s5.metric("Trip Stops", f"{len(trip_df)}")

        # ----------------------------------------------------
        # Auto-play engine.
        # It schedules the next index, sleeps according to speed,
        # then reruns. No third-party Streamlit component is required.
        # ----------------------------------------------------
        if st.session_state.get("_map_playing", False):
            if sim_idx < len(trip_df) - 1:
                st.session_state["_pending_map_index"] = sim_idx + 1
                time.sleep(1.15 / float(speed))
                st.rerun()
            else:
                st.session_state["_map_playing"] = False
                st.success("Trip playback complete.")

st.markdown("---")
tab_service,tab_built,tab_context,tab_synthesis=st.tabs(["1. Service & Demand","2. Built Environment","3. Neighbourhood Context","4. EDA Synthesis & Model Formulation"])

with tab_service:
    st.markdown("## Service Exposure & Reconstructed Demand")
    f1,f2=st.columns(2)
    with f1:
        sd=st.selectbox("Corridor direction",["All directions"]+df[COL["direction"]].dropna().astype(str).drop_duplicates().tolist(),key="service_direction")
    with f2:
        sp=st.selectbox("Operating period",["All periods"]+df[COL["period"]].dropna().astype(str).drop_duplicates().tolist(),key="service_period")
    filtered=df.copy()
    if sd!="All directions": filtered=filtered[filtered[COL["direction"]].astype(str)==sd]
    if sp!="All periods": filtered=filtered[filtered[COL["period"]].astype(str)==sp]

    k1,k2,k3,k4=st.columns(4)
    k1.metric("Filtered Stop Visits",f"{len(filtered):,}")
    k2.metric("Active Trips",f"{filtered[COL['trip']].nunique():,}")
    k3.metric("Median Headway","—" if COL["headway"] is None else f"{num(filtered[COL['headway']]).median():.1f} min")
    k4.metric("Demand Occurrence",f"{num(filtered[COL['target']]).mean()*100:.1f}%")

    left,right=st.columns([1,2])
    with left:
        st.markdown("#### Target Variable Balance")
        counts=num(filtered[COL["target"]]).value_counts().reindex([0,1],fill_value=0); total=counts.sum()
        tv=pd.DataFrame({"Outcome":["No boarding","Positive boarding"],"Share":[counts.loc[0]/total*100 if total else 0,counts.loc[1]/total*100 if total else 0]})
        fig=px.bar(tv,x="Share",y="Outcome",orientation="h",text="Share",color="Outcome",color_discrete_sequence=[MUTED,YELLOW])
        fig.update_traces(texttemplate="%{text:.1f}%",textposition="inside"); fig.update_layout(showlegend=False,xaxis_title="Share (%)",yaxis_title="")
        st.plotly_chart(clean_plot(fig,300),use_container_width=True)
    with right:
        st.markdown("#### Directional Disparity: Service vs Demand")
        ds=df.groupby(COL["direction"],as_index=False).agg(stop_visits=(COL["target"],"size"),demand_occurrence=(COL["target"],"mean"))
        ds["demand_occurrence"]*=100
        fig=go.Figure()
        fig.add_bar(x=ds[COL["direction"]],y=ds["stop_visits"],name="Scheduled stop visits",marker_color=BLACK)
        fig.add_scatter(x=ds[COL["direction"]],y=ds["demand_occurrence"],name="Demand occurrence (%)",mode="lines+markers",line=dict(color=YELLOW,width=4),marker=dict(size=10),yaxis="y2")
        fig.update_layout(yaxis=dict(title="Scheduled stop visits"),yaxis2=dict(title="Demand occurrence (%)",overlaying="y",side="right"),legend=dict(orientation="h",y=1.12))
        st.plotly_chart(clean_plot(fig,350),use_container_width=True)

    with st.expander("Directional summary · Dissertation Table 4.3"):
        direction_rows = []
        for direction, grp in df.groupby(COL["direction"]):
            direction_rows.append(
                {
                    "Direction": direction,
                    "Trips": grp[COL["trip"]].nunique(),
                    "Unique stop IDs": grp[COL["stop"]].nunique(),
                    "Scheduled stop visits": len(grp),
                    "LSOAs served": grp[COL["lsoa"]].nunique(),
                    "Demand-occurrence rate": num(grp[COL["target"]]).mean() * 100,
                }
            )
        direction_table = pd.DataFrame(direction_rows)
        st.dataframe(
            direction_table.style.format(
                {
                    "Trips": "{:,.0f}",
                    "Unique stop IDs": "{:,.0f}",
                    "Scheduled stop visits": "{:,.0f}",
                    "LSOAs served": "{:,.0f}",
                    "Demand-occurrence rate": "{:.2f}%",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("#### Operating Period Dynamics")
    ps=df.groupby(COL["period"],as_index=False).agg(stop_visits=(COL["target"],"size"),demand_occurrence=(COL["target"],"mean"))
    ps["demand_occurrence"]*=100
    ps["_order"]=ps[COL["period"]].apply(lambda x:TIME_ORDER.index(x) if x in TIME_ORDER else 99); ps=ps.sort_values("_order")
    p1,p2=st.columns(2)
    with p1:
        fig=px.bar(ps,x=COL["period"],y="stop_visits"); fig.update_traces(marker_color=BLACK); fig.update_xaxes(tickangle=-25)
        st.plotly_chart(clean_plot(fig,400),use_container_width=True)
    with p2:
        fig=px.bar(ps,x=COL["period"],y="demand_occurrence"); fig.update_traces(marker_color=YELLOW); fig.update_xaxes(tickangle=-25)
        st.plotly_chart(clean_plot(fig,400),use_container_width=True)
    with st.expander("Operating-period summary · Dissertation Table 4.4"):
        period_table = (
            df.groupby(COL["period"], as_index=False)
            .agg(
                stop_visits=(COL["target"], "size"),
                unique_trips=(COL["trip"], "nunique"),
                demand_occurrence=(COL["target"], "mean"),
            )
        )
        period_table["demand_occurrence"] *= 100
        period_table["_order"] = period_table[COL["period"]].apply(
            lambda x: TIME_ORDER.index(x) if x in TIME_ORDER else 99
        )
        period_table = (
            period_table.sort_values("_order")
            .drop(columns="_order")
            .rename(
                columns={
                    COL["period"]: "Operating period",
                    "stop_visits": "Stop visits",
                    "unique_trips": "Unique trips",
                    "demand_occurrence": "Demand-occurrence rate",
                }
            )
        )
        st.dataframe(
            period_table.style.format(
                {
                    "Stop visits": "{:,.0f}",
                    "Unique trips": "{:,.0f}",
                    "Demand-occurrence rate": "{:.2f}%",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    if COL["headway"] is not None:
        with st.expander("Scheduled-headway descriptive statistics · Dissertation Table 4.5"):
            h = num(df[COL["headway"]]).dropna()
            headway_table = pd.DataFrame(
                [
                    {
                        "Measure": "Headway",
                        "Min": h.min(),
                        "Mean": h.mean(),
                        "Median": h.median(),
                        "Q1": h.quantile(0.25),
                        "Q3": h.quantile(0.75),
                        "Max": h.max(),
                    }
                ]
            )
            st.dataframe(
                headway_table.style.format(
                    {
                        "Min": "{:.2f}",
                        "Mean": "{:.2f}",
                        "Median": "{:.2f}",
                        "Q1": "{:.2f}",
                        "Q3": "{:.2f}",
                        "Max": "{:.2f}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

    st.markdown('<div class="insight"><b>Analytical insight:</b> service exposure and reconstructed demand occurrence do not necessarily move together.</div>',unsafe_allow_html=True)

    interaction=df.groupby([COL["period"],COL["direction"]],as_index=False).agg(demand_occurrence=(COL["target"],"mean"))
    interaction["demand_occurrence"]*=100
    interaction["_order"]=interaction[COL["period"]].apply(lambda x:TIME_ORDER.index(x) if x in TIME_ORDER else 99); interaction=interaction.sort_values("_order")
    fig=px.line(interaction,x=COL["period"],y="demand_occurrence",color=COL["direction"],markers=True,color_discrete_sequence=[YELLOW,BLACK])
    fig.update_xaxes(tickangle=-20)
    st.plotly_chart(clean_plot(fig,420),use_container_width=True)

with tab_built:
    st.markdown("## Stop-Level Built Environment")
    st.caption("Dissertation Figure 4.7 and Table 4.6")

    built = {
        "Employment destinations within 400 m": "employment_count_osm_400m",
        "Retail destinations within 400 m": "retail_count_osm_400m",
        "Education destinations within 400 m": "education_count_osm_400m",
        "Healthcare destinations within 400 m": "healthcare_count_osm_400m",
    }
    built = {k: v for k, v in built.items() if v in stop_df.columns}

    if not built:
        st.warning("No built-environment count columns were found.")
    else:
        st.markdown(
            "The dissertation evaluates these variables at the **107 directional-stop level**, "
            "rather than weighting them by the 71,200 stop visits."
        )

        # Dissertation-style descriptive statistics.
        stats = []
        for label, col in built.items():
            vals = num(stop_df[col]).dropna()
            if vals.empty:
                continue
            stats.append(
                {
                    "Feature": label,
                    "Min": vals.min(),
                    "Median": vals.median(),
                    "Mean": vals.mean(),
                    "Q75": vals.quantile(0.75),
                    "Max": vals.max(),
                    "Skewness": vals.skew(),
                }
            )

        stat_df = pd.DataFrame(stats)
        st.dataframe(
            stat_df.style.format(
                {
                    "Min": "{:.0f}",
                    "Median": "{:.1f}",
                    "Mean": "{:.2f}",
                    "Q75": "{:.1f}",
                    "Max": "{:.0f}",
                    "Skewness": "{:.2f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        short_labels = {
            "Employment destinations within 400 m": "Employment",
            "Retail destinations within 400 m": "Retail",
            "Education destinations within 400 m": "Education",
            "Healthcare destinations within 400 m": "Healthcare",
        }

        selected_label = st.segmented_control(
            "Explore stop-level activity",
            list(built.keys()),
            default=list(built.keys())[0],
            format_func=lambda x: short_labels.get(x, x),
            key="built_domain",
        )
        if selected_label is None:
            selected_label = list(built.keys())[0]

        col = built[selected_label]
        vals = num(stop_df[col])

        b1, b2 = st.columns([1.05, 1.45], gap="large")
        with b1:
            fig = px.histogram(
                stop_df,
                x=col,
                nbins=25,
                title=f"{short_labels[selected_label]} distribution across directional stops",
            )
            fig.update_traces(marker_color=BLACK)
            st.plotly_chart(clean_plot(fig, 370), use_container_width=True)

        with b2:
            name_col = COL["stop_name"] if COL["stop_name"] else COL["stop"]
            top = stop_df[[name_col, COL["lsoa"], col]].copy()
            top["Corridor percentile"] = top[col].rank(pct=True) * 100
            top = (
                top.sort_values(col, ascending=False)
                .head(10)
                .rename(
                    columns={
                        name_col: "Stop",
                        COL["lsoa"]: "LSOA",
                        col: "400 m count",
                    }
                )
            )
            st.markdown("##### Highest-activity stop catchments")
            st.dataframe(
                top.style.format(
                    {
                        "400 m count": "{:.0f}",
                        "Corridor percentile": "{:.1f}%",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown(
            '<div class="insight"><b>EDA interpretation:</b> the built environment is highly heterogeneous. '
            'Retail and employment counts are particularly right-skewed, indicating that high-activity '
            'destinations are concentrated around a relatively small number of stops rather than distributed '
            'uniformly along Route 192.</div>',
            unsafe_allow_html=True,
        )

with tab_context:
    st.markdown("## Neighbourhood Context")
    st.markdown(
        "This section follows the dissertation EDA sequence: population and demographics, "
        "socioeconomic and mobility context, then health and disability."
    )

    demo, socio, health = st.tabs(
        [
            "3A. Population & Demographics",
            "3B. Deprivation & Mobility",
            "3C. Health & Disability",
        ]
    )

    # ========================================================
    # 3A POPULATION & DEMOGRAPHICS
    # ========================================================
    with demo:
        population_col = get_population_col(lsoa_df)

        age = {
            "Age 0–14": first_existing(["age_0_14_rate"], lsoa_df),
            "Age 15–24": first_existing(["age_15_24_rate"], lsoa_df),
            "Age 25–44": first_existing(["age_25_44_rate"], lsoa_df),
            "Age 45–64": first_existing(["age_45_64_rate"], lsoa_df),
            "Age 65+": first_existing(["age_65_plus_rate_refined", "age_65_plus_rate"], lsoa_df),
        }
        age = {k: v for k, v in age.items() if v is not None}

        student_col = first_existing(["student_rate"], lsoa_df)
        employment_col = first_existing(["employment_rate"], lsoa_df)
        female_col = get_female_col(lsoa_df)

        # --- Population: Figure 4.8 / Table 4.7
        if population_col is not None:
            related = [
                ("Age 65+", age.get("Age 65+"), True),
                ("Student rate", student_col, True),
                ("Employment rate", employment_col, True),
            ]
            render_lsoa_metric_section(
                lsoa_df,
                population_col,
                "Resident population across Route 192 LSOAs",
                "Resident population",
                percent=False,
                include_sd=False,
                related=related,
                key_prefix="population",
                interpretation=(
                    "population size varies substantially across the 39 LSOAs, while higher- "
                    "and lower-population areas are interspersed along the corridor rather than "
                    "forming a simple north–south gradient."
                ),
            )
            st.caption("Dissertation Figure 4.8 and Table 4.7")
        else:
            st.warning(
                "A raw resident-population column was not found. Expected a field such as "
                "`gender_total_population` or `population_total`."
            )

        st.markdown("---")

        # --- Age composition: Figure 4.9 / Table 4.8
        st.markdown("### Age composition across Route 192 LSOAs")
        st.caption("Dissertation Figure 4.9 and Table 4.8")

        if age:
            age_frames = []
            age_rows = []

            for label, col in age.items():
                age_frames.append(
                    pd.DataFrame(
                        {
                            "Age group": label,
                            "Rate (%)": to_percent(lsoa_df[col]),
                            "LSOA": (
                                lsoa_df[COL["lsoa_name"]]
                                if COL["lsoa_name"]
                                else lsoa_df[COL["lsoa"]]
                            ),
                        }
                    )
                )
                age_rows.append(
                    descriptive_row(
                        lsoa_df,
                        col,
                        label,
                        percent=True,
                        include_sd=True,
                    )
                )

            age_long = pd.concat(age_frames, ignore_index=True)

            fig = px.box(
                age_long,
                x="Age group",
                y="Rate (%)",
                points="all",
                hover_name="LSOA",
                color="Age group",
                color_discrete_sequence=[
                    "#4C78A8",
                    "#F58518",
                    "#54A24B",
                    "#E45756",
                    "#B279A2",
                ],
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(clean_plot(fig, 430), use_container_width=True)
            descriptive_table(age_rows, digits=1)

            st.markdown(
                '<div class="insight"><b>EDA interpretation:</b> working-age residents dominate the '
                'corridor, but the wide ranges for ages 15–24 and 25–44 show substantial demographic '
                'heterogeneity between LSOAs rather than a single corridor-wide age profile.</div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # --- Student and employment: Figure 4.10 / Table 4.9
        st.markdown("### Student and employment rates")
        st.caption("Dissertation Figure 4.10 and Table 4.9")

        se_rows = []
        se_frames = []

        if student_col is not None:
            se_rows.append(
                descriptive_row(
                    lsoa_df,
                    student_col,
                    "Student rate",
                    percent=True,
                    include_sd=True,
                )
            )
            se_frames.append(
                pd.DataFrame(
                    {
                        "Characteristic": "Student rate",
                        "Rate (%)": to_percent(lsoa_df[student_col]),
                        "LSOA": (
                            lsoa_df[COL["lsoa_name"]]
                            if COL["lsoa_name"]
                            else lsoa_df[COL["lsoa"]]
                        ),
                    }
                )
            )

        if employment_col is not None:
            se_rows.append(
                descriptive_row(
                    lsoa_df,
                    employment_col,
                    "Employment rate",
                    percent=True,
                    include_sd=True,
                )
            )
            se_frames.append(
                pd.DataFrame(
                    {
                        "Characteristic": "Employment rate",
                        "Rate (%)": to_percent(lsoa_df[employment_col]),
                        "LSOA": (
                            lsoa_df[COL["lsoa_name"]]
                            if COL["lsoa_name"]
                            else lsoa_df[COL["lsoa"]]
                        ),
                    }
                )
            )

        if se_frames:
            se_long = pd.concat(se_frames, ignore_index=True)
            fig = px.box(
                se_long,
                x="Characteristic",
                y="Rate (%)",
                points="all",
                hover_name="LSOA",
                color="Characteristic",
                color_discrete_sequence=["#4C78A8", "#F58518"],
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(clean_plot(fig, 390), use_container_width=True)
            descriptive_table(se_rows, digits=1)

        # --- Gender composition: Figure 4.11
        if female_col is not None:
            st.markdown("---")
            st.markdown("### Gender composition")
            st.caption("Dissertation Figure 4.11")

            female = to_percent(lsoa_df[female_col]).mean()
            male = 100 - female

            g1, g2 = st.columns([1.1, 1])
            with g1:
                gender_df = pd.DataFrame(
                    {
                        "Sex": ["Male", "Female"],
                        "Share": [male, female],
                    }
                )
                fig = px.pie(
                    gender_df,
                    names="Sex",
                    values="Share",
                    hole=0.48,
                    color="Sex",
                    color_discrete_sequence=["#4C78A8", "#F58518"],
                )
                fig.update_traces(
                    textposition="inside",
                    texttemplate="%{value:.1f}%",
                )
                fig.update_layout(
                    height=330,
                    margin=dict(l=15, r=15, t=20, b=20),
                    showlegend=True,
                    paper_bgcolor=CARD,
                )
                st.plotly_chart(fig, use_container_width=True)
            with g2:
                st.markdown(
                    '<div class="soft-card"><h4>Corridor interpretation</h4>'
                    'Gender composition is broadly balanced at corridor level. '
                    'The dissertation therefore treats sex composition as contextual demographic '
                    'information rather than a dominant source of aggregate spatial differentiation.'
                    '</div>',
                    unsafe_allow_html=True,
                )

    # ========================================================
    # 3B DEPRIVATION & MOBILITY
    # ========================================================
    with socio:
        dep = {
            "Education, skills & training": first_existing(
                ["education_skills_training_score_2025"], lsoa_df
            ),
            "Crime": first_existing(["crime_score_2025"], lsoa_df),
            "Barriers to housing & services": first_existing(
                ["barriers_housing_services_score_2025"], lsoa_df
            ),
            "Living environment": first_existing(
                ["living_environment_score_2025"], lsoa_df
            ),
        }
        dep = {k: v for k, v in dep.items() if v is not None}

        st.markdown("### Socioeconomic deprivation domains")
        st.caption("Dissertation Table 4.10")

        if dep:
            dep_choice = st.selectbox(
                "Explore deprivation domain",
                list(dep.keys()),
                key="deprivation_domain_selector",
            )
            dep_col = dep[dep_choice]

            render_lsoa_metric_section(
                lsoa_df,
                dep_col,
                dep_choice,
                dep_choice,
                percent=False,
                include_sd=True,
                key_prefix="deprivation",
                interpretation=(
                    "the retained deprivation domains identify different neighbourhood patterns, "
                    "supporting the dissertation's treatment of disadvantage as multidimensional "
                    "rather than as a single composite condition."
                ),
            )

            st.markdown("##### All retained deprivation domains")
            dep_rows = [
                descriptive_row(
                    lsoa_df,
                    col,
                    label,
                    percent=False,
                    include_sd=True,
                )
                for label, col in dep.items()
            ]
            descriptive_table(dep_rows, digits=2)

        st.markdown("---")
        st.markdown("### Mobility resources & travel behaviour")
        st.caption("Dissertation Figures 4.12–4.13 and Tables 4.11–4.12")

        mobility = {
            "No-car household rate": first_existing(
                ["no_car_rate"], lsoa_df
            ),
            "Working-from-home rate": first_existing(
                ["work_from_home_rate_2021", "work_from_home_rate"], lsoa_df
            ),
        }
        mobility = {k: v for k, v in mobility.items() if v is not None}

        if mobility:
            mobility_choice = st.segmented_control(
                "Mobility indicator",
                list(mobility.keys()),
                default=list(mobility.keys())[0],
                key="mobility_indicator",
            )
            if mobility_choice is None:
                mobility_choice = list(mobility.keys())[0]

            mobility_col = mobility[mobility_choice]

            mobility_interpretation = {
                "No-car household rate": (
                    "access to private transport differs substantially across the corridor, "
                    "with higher no-car rates indicating neighbourhoods that may rely more heavily "
                    "on public or other non-car transport options."
                ),
                "Working-from-home rate": (
                    "home-working prevalence varies strongly between LSOAs, implying different "
                    "levels of regular commuting exposure and supporting its inclusion as a "
                    "contextual demand predictor."
                ),
            }[mobility_choice]

            render_lsoa_metric_section(
                lsoa_df,
                mobility_col,
                mobility_choice,
                mobility_choice,
                percent=True,
                include_sd=True,
                key_prefix="mobility",
                interpretation=mobility_interpretation,
            )

            st.markdown("##### Descriptive statistics for mobility indicators")
            mobility_rows = [
                descriptive_row(
                    lsoa_df,
                    col,
                    label,
                    percent=True,
                    include_sd=True,
                )
                for label, col in mobility.items()
            ]
            descriptive_table(mobility_rows, digits=1)

    # ========================================================
    # 3C HEALTH & DISABILITY
    # ========================================================
    with health:
        hm = {
            "Poor general health rate": first_existing(
                [
                    "bad_or_very_bad_health_rate_census2021",
                    "poor_general_health_rate",
                ],
                lsoa_df,
            ),
            "Activity-limiting disability rate": first_existing(
                [
                    "disabled_under_equality_act_rate_census2021",
                    "activity_limiting_disability_rate",
                ],
                lsoa_df,
            ),
            "Health deprivation score": first_existing(
                [
                    "health_deprivation_disability_score_2025",
                    "health_deprivation_score_2025",
                ],
                lsoa_df,
            ),
        }
        hm = {k: v for k, v in hm.items() if v is not None}

        st.markdown("### Health context and incremental information")
        st.caption("Dissertation Table 4.13")

        if hm:
            health_choice = st.segmented_control(
                "Health indicator",
                list(hm.keys()),
                default=list(hm.keys())[0],
                key="health_indicator",
            )
            if health_choice is None:
                health_choice = list(hm.keys())[0]

            health_col = hm[health_choice]
            is_percent = "score" not in health_col.lower()

            related_health = [
                (
                    label,
                    col,
                    "score" not in col.lower(),
                )
                for label, col in hm.items()
                if col != health_col
            ]

            render_lsoa_metric_section(
                lsoa_df,
                health_col,
                health_choice,
                health_choice,
                percent=is_percent,
                include_sd=True,
                related=related_health,
                key_prefix="health",
                interpretation=(
                    "health conditions vary materially across the 39 corridor LSOAs. "
                    "This spatial heterogeneity provides the variation required to test whether "
                    "health adds information beyond conventional demand predictors."
                ),
            )

            st.markdown("##### Health and disability descriptive statistics")
            health_rows = []
            for label, col in hm.items():
                health_rows.append(
                    descriptive_row(
                        lsoa_df,
                        col,
                        label,
                        percent=("score" not in col.lower()),
                        include_sd=True,
                    )
                )
            descriptive_table(health_rows, digits=2)

            st.markdown(
                '<div class="research-note"><b>Interpretation boundary:</b> these are LSOA-level '
                'neighbourhood indicators. They do not identify the health status of individual '
                'passengers and should not be interpreted causally.</div>',
                unsafe_allow_html=True,
            )

with tab_synthesis:
    st.markdown("## From Exploration to Model Specification")
    cand={
        "Population":first_existing(["gender_total_population","population_total","log_population_total"],lsoa_df),
        "Age 65+":first_existing(["age_65_plus_rate_refined"],lsoa_df),
        "Student":first_existing(["student_rate"],lsoa_df),
        "No-car":first_existing(["no_car_rate"],lsoa_df),
        "Income deprivation":first_existing(["income_deprivation_score_rate_2025"],lsoa_df),
        "Employment deprivation":first_existing(["employment_deprivation_score_rate_2025"],lsoa_df),
        "Crime":first_existing(["crime_score_2025"],lsoa_df),
        "Poor health":first_existing(["bad_or_very_bad_health_rate_census2021","poor_general_health_rate"],lsoa_df),
        "Disability":first_existing(["disabled_under_equality_act_rate_census2021","activity_limiting_disability_rate"],lsoa_df),
        "Health deprivation":first_existing(["health_deprivation_disability_score_2025","health_deprivation_score_2025"],lsoa_df),
    }
    cand={k:v for k,v in cand.items() if v}
    cd=lsoa_df[list(cand.values())].apply(pd.to_numeric,errors="coerce"); corr=cd.corr(method="spearman")
    reverse={v:k for k,v in cand.items()}; corr=corr.rename(index=reverse,columns=reverse)
    fig=px.imshow(corr,zmin=-1,zmax=1,color_continuous_scale="RdBu_r",aspect="auto",labels={"color":"Spearman ρ"})
    st.plotly_chart(clean_plot(fig,650),use_container_width=True)

    st.markdown("### Controlled Variable Inclusion Protocol")
    cols=st.columns(4)
    blocks=[
        ("1","Operational & Temporal","Direction, operating period and headway remain core controls."),
        ("2","Built Environment","Employment, retail, education and healthcare remain distinct stop-level features."),
        ("3","Socioeconomic & Mobility","Retain non-redundant deprivation and mobility measures."),
        ("4","Health Integration","Add poor health, disability and health deprivation only to the Health-Integrated model.")
    ]
    for col,(n,title,text) in zip(cols,blocks):
        with col:
            st.markdown(f'<div class="soft-card"><span class="level-chip">{n}</span><h4>{title}</h4>{text}</div>',unsafe_allow_html=True)
    st.markdown('<div class="insight"><b>RQ1 rationale:</b> health varies meaningfully across the corridor but overlaps with conventional socioeconomic context. Its value should therefore be tested incrementally against a strong baseline.</div>',unsafe_allow_html=True)
