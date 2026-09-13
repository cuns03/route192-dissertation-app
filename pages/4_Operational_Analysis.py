from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
import streamlit as st


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Operational Analysis | Route 192",
    page_icon="🚌",
    layout="wide",
)


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

OUTPUT_DIR_CANDIDATES = [
    DATA_DIR / "rq2_rq3_outputs",
    DATA_DIR / "RQ2_RQ3_outputs",
    DATA_DIR / "RQ2_RQ3_business_outputs",
    BASE_DIR / "rq2_rq3_outputs",
    BASE_DIR / "RQ2_RQ3_business_outputs",
    BASE_DIR.parent / "rq2_rq3_outputs",
    BASE_DIR.parent / "RQ2_RQ3_business_outputs",
]

GEO_DIR_CANDIDATES = [
    DATA_DIR / "geo",
    BASE_DIR / "data" / "geo",
]

FILE_CANDIDATES = {
    "direction_period": [
        "RQ3_direction_period_summary.csv",
        "rq3_direction_period_summary.csv",
    ],
    "segments": [
        "RQ3_segment_direction_period_hotspots.csv",
        "rq3_segment_direction_period_hotspots.csv",
        "RQ3_segment_hotspots.csv",
    ],
    "rank": [
        "RQ3_load_vs_PMD_rank_check.csv",
        "rq3_load_vs_PMD_rank_check.csv",
        "RQ3_load_vs_pmd_rank_check.csv",
    ],
    "weight": [
        "RQ3_passenger_weight_sensitivity.csv",
        "rq3_passenger_weight_sensitivity.csv",
    ],
    "percentile": [
        "RQ3_percentile_sensitivity.csv",
        "rq3_percentile_sensitivity.csv",
    ],
    "overlay": [
        "RQ2_RQ3_contextual_overlay.csv",
        "rq2_rq3_contextual_overlay.csv",
        "RQ3_health_demand_context.csv",
        "RQ2_RQ3_business_context.csv",
    ],
}

STOP_COORD_CANDIDATES = [
    DATA_DIR / "geo" / "route192_stop_coordinates.csv",
    DATA_DIR / "route192_stop_coordinates.csv",
]

MASTER_DATA_CANDIDATES = [
    DATA_DIR / "raw" / "route192_master.csv",
    DATA_DIR / "route192_master.csv",
]


# =============================================================================
# CONSTANTS
# =============================================================================

YELLOW = "#FFD800"
BLACK = "#161616"
BG = "#F7F7FC"
WHITE = "#FFFFFF"
BORDER = "#DCE2F3"
GREY = "#6B7280"
BLUE = "#0053DB"
RED = "#BA1A1A"
PALE = "#F0F3FF"

TOTAL_CAPACITY = 100.0
SEATED_REFERENCE = 73.0
BASELINE_MASS = 68.0

PERIOD_ORDER_HINTS = [
    "night",
    "early morning",
    "am peak",
    "am interpeak",
    "pm interpeak",
    "pm peak",
    "evening",
    "off-peak",
]

RQ2_GROUP_ORDER = [
    "Higher health + Lower demand",
    "Higher health + Higher demand",
    "Lower health + Higher demand",
    "Lower health + Lower demand",
]


# =============================================================================
# STYLE
# =============================================================================

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
        background: {PALE};
        border-left: 4px solid {BLUE};
        padding: .9rem 1rem;
        border-radius: .65rem;
        font-size: .9rem;
        color: #303642;
        line-height: 1.55;
    }}
    .method-card {{
        background: {WHITE};
        border: 1px solid {BORDER};
        border-radius: .85rem;
        padding: .9rem 1rem;
        box-shadow: 0 1px 4px rgba(0,0,0,.025);
    }}
    .flow-step {{
        display: inline-block;
        padding: .38rem .58rem;
        border-radius: .35rem;
        background: {PALE};
        font-size: .75rem;
        font-weight: 700;
        color: #303642;
        margin: .12rem;
    }}
    .flow-step-yellow {{
        display: inline-block;
        padding: .38rem .58rem;
        border-radius: .35rem;
        background: {YELLOW};
        font-size: .75rem;
        font-weight: 800;
        color: {BLACK};
        margin: .12rem;
    }}
    .flow-arrow {{
        color: {GREY};
        font-weight: 800;
        padding: 0 .15rem;
    }}
    .metric-card {{
        background: {WHITE};
        border: 1px solid {BORDER};
        border-top: 4px solid {YELLOW};
        border-radius: .8rem;
        padding: .95rem 1rem;
        min-height: 122px;
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
        margin-top: .22rem;
    }}
    .metric-sub {{
        color: {GREY};
        font-size: .76rem;
        margin-top: .38rem;
        line-height: 1.4;
    }}
    .panel {{
        background: {WHITE};
        border: 1px solid {BORDER};
        border-radius: .9rem;
        padding: 1rem;
        box-shadow: 0 1px 5px rgba(0,0,0,.03);
    }}
    .inspector-card {{
        background: {WHITE};
        border: 1px solid {BORDER};
        border-radius: .85rem;
        padding: .9rem;
        box-shadow: 0 1px 4px rgba(0,0,0,.03);
        margin-bottom: .7rem;
    }}
    .mini-label {{
        color: {GREY};
        font-size: .7rem;
        text-transform: uppercase;
        letter-spacing: .04em;
        font-weight: 700;
    }}
    .mini-value {{
        font-family: "Public Sans", sans-serif;
        color: #151C27;
        font-size: 1.45rem;
        font-weight: 800;
        margin-top: .1rem;
    }}
    .caution-box {{
        background: #FFF9DF;
        border-left: 4px solid {YELLOW};
        border-radius: .65rem;
        padding: .88rem 1rem;
        font-size: .86rem;
        line-height: 1.5;
        color: #3D4350;
    }}
    .answer-box {{
        background: {YELLOW};
        color: {BLACK};
        border-radius: .95rem;
        padding: 1.1rem 1.2rem;
        line-height: 1.55;
    }}
    .answer-box h3 {{
        margin: .1rem 0 .3rem 0;
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


# =============================================================================
# HELPERS
# =============================================================================

def norm(text) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def clean_id(value) -> str:
    if pd.isna(value):
        return ""
    s = str(value).strip()
    s = re.sub(r"^stop\s+", "", s, flags=re.I).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def parse_segment_endpoints(value):
    """Extract from/to stop IDs from common segment-label formats."""
    if value is None or pd.isna(value):
        return "", ""
    s = str(value).strip()
    if not s:
        return "", ""
    parts = re.split(r"\s*(?:->|→|–|—|>)\s*", s, maxsplit=1)
    if len(parts) == 2:
        return clean_id(parts[0]), clean_id(parts[1])
    return "", ""


def haversine_m(lat1, lon1, lat2, lon2):
    vals = [lat1, lon1, lat2, lon2]
    if any(pd.isna(v) for v in vals):
        return np.nan
    r = 6371000.0
    p1, p2 = math.radians(float(lat1)), math.radians(float(lat2))
    dp = math.radians(float(lat2) - float(lat1))
    dl = math.radians(float(lon2) - float(lon1))
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def resolve_output_file(key: str):
    filenames = FILE_CANDIDATES[key]
    for directory in OUTPUT_DIR_CANDIDATES:
        for filename in filenames:
            p = directory / filename
            if p.exists():
                return p
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
        k = norm(candidate)
        if k in norm_map:
            return norm_map[k]
    if contains:
        tokens = [norm(t) for t in contains]
        for c in df.columns:
            nc = norm(c)
            if all(t in nc for t in tokens):
                return c
    return None


def numeric(series):
    return pd.to_numeric(series, errors="coerce")


def coerce_percent(series):
    s = numeric(series)
    valid = s.dropna()
    if valid.empty:
        return s
    # If represented as fractions, convert to percentage units.
    if valid.abs().quantile(0.95) <= 1.5:
        s = s * 100.0
    return s


def fmt_pct(v, digits=1):
    if pd.isna(v):
        return "—"
    x = float(v)
    if abs(x) <= 1.5:
        x *= 100
    return f"{x:.{digits}f}%"


def fmt_num(v, digits=1):
    if pd.isna(v):
        return "—"
    return f"{float(v):,.{digits}f}"


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


def plot_layout(fig, height=430, legend=True):
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=40, b=30),
        paper_bgcolor=WHITE,
        plot_bgcolor=WHITE,
        font=dict(family="Inter", color="#374151", size=12),
        hoverlabel=dict(font_family="Inter"),
        showlegend=legend,
        legend=dict(font=dict(family="Inter")),
    )
    fig.update_xaxes(gridcolor="#E8ECF5", zeroline=False)
    fig.update_yaxes(gridcolor="#E8ECF5", zeroline=False)
    return fig


def direction_label(value):
    s = str(value).strip()
    ns = norm(s)
    if ns in {"0", "direction0", "outbound"}:
        return "Manchester to Hazel Grove"
    if ns in {"1", "direction1", "inbound"}:
        return "Hazel Grove to Manchester"
    if "manchester" in ns and "hazel" in ns:
        # Preserve explicit textual direction.
        man_i = ns.find("manchester")
        haz_i = ns.find("hazel")
        return "Manchester to Hazel Grove" if man_i < haz_i else "Hazel Grove to Manchester"
    return s


def period_sort_key(value):
    s = str(value).lower()
    if "night" in s or "early" in s:
        return 0
    if "am" in s and "peak" in s and "inter" not in s:
        return 1
    if "am" in s and "inter" in s:
        return 2
    if "pm" in s and "inter" in s:
        return 3
    if "pm" in s and "peak" in s and "inter" not in s:
        return 4
    if "evening" in s:
        return 5
    if "off" in s:
        return 6
    return 99


def canonical_group(value):
    s = norm(value)
    if not s:
        return str(value)
    high_health = ("higherhealth" in s or "highhealth" in s)
    low_health = ("lowerhealth" in s or "lowhealth" in s)
    high_demand = ("higherdemand" in s or "highdemand" in s)
    low_demand = ("lowerdemand" in s or "lowdemand" in s)

    if high_health and low_demand:
        return "Higher health + Lower demand"
    if high_health and high_demand:
        return "Higher health + Higher demand"
    if low_health and high_demand:
        return "Lower health + Higher demand"
    if low_health and low_demand:
        return "Lower health + Lower demand"
    return str(value)


def segment_label(row):
    if "segment" in row and pd.notna(row["segment"]) and str(row["segment"]).strip():
        return str(row["segment"])
    f = clean_id(row.get("from_stop"))
    t = clean_id(row.get("to_stop"))
    if f or t:
        return f"{f} → {t}"
    return "Segment"


def load_stop_coordinates():
    """
    Load Route 192 stop coordinates and, where available, human-readable
    stop names. If the coordinate file does not contain names, supplement
    them from route192_master.csv.
    """
    path = next((p for p in STOP_COORD_CANDIDATES if p.exists()), None)
    if path is None:
        return None

    df = safe_read_csv(path)
    if df is None or df.empty:
        return None

    id_col = find_col(
        df,
        ["stop_id", "stopid", "atco_code", "atcocode", "naptan_code", "stop_code"],
    )
    lat_col = find_col(df, ["lat", "latitude", "stop_lat"])
    lon_col = find_col(df, ["lon", "lng", "longitude", "stop_lon"])
    name_col = find_col(
        df,
        ["stop_name", "name", "common_name", "stop_name_clean", "stop_label"],
    )

    if id_col is None or lat_col is None or lon_col is None:
        return None

    out = pd.DataFrame(
        {
            "stop_id": df[id_col].map(clean_id),
            "lat": numeric(df[lat_col]),
            "lon": numeric(df[lon_col]),
        }
    )

    if name_col:
        out["stop_name"] = df[name_col].astype(str).str.strip()
    else:
        out["stop_name"] = ""

    # Supplement missing names from the master analytical dataset.
    master_path = next((p for p in MASTER_DATA_CANDIDATES if p.exists()), None)
    if master_path is not None:
        master = safe_read_csv(master_path)
        if master is not None and not master.empty:
            master_id = find_col(
                master,
                ["stop_id", "stopid", "atco_code", "atcocode", "naptan_code", "stop_code"],
            )
            master_name = find_col(
                master,
                [
                    "stop_name",
                    "name",
                    "common_name",
                    "stop_name_clean",
                    "stop_label",
                ],
            )
            if master_id and master_name:
                name_lookup = (
                    pd.DataFrame(
                        {
                            "stop_id": master[master_id].map(clean_id),
                            "_master_name": master[master_name].astype(str).str.strip(),
                        }
                    )
                    .replace({"_master_name": {"nan": "", "None": ""}})
                    .query("stop_id != ''")
                    .drop_duplicates("stop_id")
                )
                out = out.merge(name_lookup, on="stop_id", how="left")
                missing_name = (
                    out["stop_name"].fillna("").str.strip().isin(["", "nan", "None"])
                )
                out.loc[missing_name, "stop_name"] = out.loc[
                    missing_name, "_master_name"
                ].fillna("")
                out = out.drop(columns=["_master_name"])

    # Add any additional stop IDs/coordinates present in the master dataset.
    master_path = next((p for p in MASTER_DATA_CANDIDATES if p.exists()), None)
    if master_path is not None:
        master = safe_read_csv(master_path)
        if master is not None and not master.empty:
            mid = find_col(master, ["stop_id", "stopid", "atco_code", "atcocode", "naptan_code", "stop_code"])
            mlat = find_col(master, ["stop_lat", "lat", "latitude"])
            mlon = find_col(master, ["stop_lon", "lon", "lng", "longitude"])
            mname = find_col(master, ["stop_name", "name", "common_name", "stop_name_clean", "stop_label"])
            if mid and mlat and mlon:
                extra = pd.DataFrame({
                    "stop_id": master[mid].map(clean_id),
                    "lat": numeric(master[mlat]),
                    "lon": numeric(master[mlon]),
                    "stop_name": master[mname].astype(str).str.strip() if mname else "",
                })
                extra = extra.dropna(subset=["lat", "lon"])
                extra = extra[extra["stop_id"] != ""].drop_duplicates("stop_id")
                out = pd.concat([out, extra], ignore_index=True)

    out = out.dropna(subset=["lat", "lon"])
    out = out[out["stop_id"] != ""].drop_duplicates("stop_id", keep="first")
    return out


def build_stop_name_lookup(coords_df):
    if coords_df is None or coords_df.empty:
        return {}
    if "stop_name" not in coords_df.columns:
        return {}
    lookup = {}
    for _, r in coords_df.iterrows():
        sid = clean_id(r.get("stop_id"))
        name = str(r.get("stop_name", "")).strip()
        if sid and name and name.lower() not in {"nan", "none"}:
            lookup[sid] = name
    return lookup


def stop_display(stop_id, lookup):
    sid = clean_id(stop_id)
    name = lookup.get(sid, "")
    return name if name else (f"Stop {sid}" if sid else "Unknown stop")


def friendly_segment_display(row, lookup):
    """Reader-facing section label: names first, technical stop IDs second."""
    from_id = clean_id(row.get("from_stop"))
    to_id = clean_id(row.get("to_stop"))

    from_name = str(row.get("from_stop_name", "")).strip()
    to_name = str(row.get("to_stop_name", "")).strip()
    if not from_name or from_name.lower() in {"nan", "none"}:
        from_name = stop_display(from_id, lookup)
    if not to_name or to_name.lower() in {"nan", "none"}:
        to_name = stop_display(to_id, lookup)

    if from_id or to_id or from_name or to_name:
        return f"{from_name or from_id} → {to_name or to_id}"

    raw = str(row.get("segment", "")).strip()
    return raw if raw else "Directed segment"


def segment_code_display(row):
    from_id = clean_id(row.get("from_stop"))
    to_id = clean_id(row.get("to_stop"))
    if from_id or to_id:
        return f"{from_id} → {to_id}"
    return str(row.get("segment", "")).strip()


def load_trip_segments_from_master():
    """
    Build a stop-to-stop trip dataset directly from route192_master.csv when
    the analytical segment export is period-aggregated. The load at stop s is
    treated as the passenger load carried on the directed segment s -> s+1.
    """
    path = next((p for p in MASTER_DATA_CANDIDATES if p.exists()), None)
    df = safe_read_csv(path)
    if df is None or df.empty:
        return None

    trip_col = find_col(df, ["trip_id", "tripid", "journey_id", "vehicle_journey_id"])
    stop_col = find_col(df, ["stop_id", "stopid", "atco_code", "atcocode", "naptan_code", "stop_code"])
    seq_col = find_col(df, ["stop_sequence", "stop_seq", "sequence", "stop_order", "sequence_no"])
    dir_col = find_col(df, ["direction", "direction_id", "direction_label", "route_direction"])
    period_col = find_col(df, ["detailed_time_period", "operating_period", "time_period", "period"])
    name_col = find_col(df, ["stop_name", "name", "common_name", "stop_name_clean", "stop_label"])
    lat_col = find_col(df, ["stop_lat", "lat", "latitude"])
    lon_col = find_col(df, ["stop_lon", "lon", "lng", "longitude"])
    load_col = find_col(
        df,
        [
            "loading_count", "onboard_load", "onboard_count", "reconstructed_onboard_load",
            "load_after_stop", "passenger_load", "loading_synthetic", "load_synthetic",
        ],
        contains=["load"],
    )
    lsoa_col = find_col(df, ["LSOA21CD", "lsoa_code", "LSOA_code", "LSOA21CD_x", "origin_lsoa"])

    required = [trip_col, stop_col, seq_col, load_col]
    if any(c is None for c in required):
        return None

    x = pd.DataFrame({
        "trip_id": df[trip_col].astype(str).str.strip(),
        "from_stop": df[stop_col].map(clean_id),
        "stop_sequence": numeric(df[seq_col]),
        "direction": df[dir_col].map(direction_label) if dir_col else "Route 192",
        "period": df[period_col].astype(str) if period_col else "All periods",
        "from_stop_name": df[name_col].astype(str).str.strip() if name_col else "",
        "from_lat": numeric(df[lat_col]) if lat_col else np.nan,
        "from_lon": numeric(df[lon_col]) if lon_col else np.nan,
        "load": numeric(df[load_col]),
        "lsoa": df[lsoa_col].astype(str) if lsoa_col else "",
    })

    x = x.dropna(subset=["stop_sequence"]).sort_values(["trip_id", "stop_sequence"])
    g = x.groupby("trip_id", sort=False)
    x["to_stop"] = g["from_stop"].shift(-1)
    x["to_stop_name"] = g["from_stop_name"].shift(-1)
    x["to_lat"] = g["from_lat"].shift(-1)
    x["to_lon"] = g["from_lon"].shift(-1)
    x = x[x["to_stop"].notna() & x["to_stop"].astype(str).ne("")].copy()

    x["distance_m"] = x.apply(
        lambda r: haversine_m(r["from_lat"], r["from_lon"], r["to_lat"], r["to_lon"]), axis=1
    )
    x["util"] = x["load"] / TOTAL_CAPACITY * 100.0
    x["pmd_68"] = x["load"] * BASELINE_MASS / 1000.0 * (x["distance_m"] / 1000.0)
    x["segment"] = x["from_stop"] + " → " + x["to_stop"].map(clean_id)

    # A trip sequence is already topologically valid; this check only suppresses
    # occasional bad geocodes rather than changing the passenger data.
    valid_trip_dist = x["distance_m"].dropna()
    if not valid_trip_dist.empty:
        q1 = float(valid_trip_dist.quantile(0.25))
        q3 = float(valid_trip_dist.quantile(0.75))
        iqr = max(q3 - q1, 1.0)
        trip_limit = max(1200.0, min(2000.0, q3 + 4.0 * iqr))
        bad_geo = x["distance_m"] > trip_limit
        x.loc[bad_geo, ["from_lat", "from_lon", "to_lat", "to_lon"]] = np.nan

    return x.reset_index(drop=True)


def build_master_adjacency(trip_segments, coords_df):
    """
    Build the authoritative Route 192 stop-to-stop adjacency table from the
    ordered stop sequence in route192_master.csv.

    This is deliberately used as the geometry source for the period-level RQ3
    map. The RQ3 hotspot export supplies the operational metric, while the
    master stop sequence supplies the physical stop-to-stop topology.
    """
    if trip_segments is None or trip_segments.empty:
        return None

    x = trip_segments.copy()
    x["from_stop"] = x["from_stop"].map(clean_id)
    x["to_stop"] = x["to_stop"].map(clean_id)
    x["direction"] = x["direction"].map(direction_label)

    # Recover names / coordinates from the unified stop lookup where needed.
    x = fill_segment_geometry(x, coords_df)

    # Count how often each directed pair appears in the actual trip sequences.
    group_cols = ["direction", "from_stop", "to_stop"]
    agg = (
        x.groupby(group_cols, as_index=False)
        .agg(
            from_stop_name=("from_stop_name", "first"),
            to_stop_name=("to_stop_name", "first"),
            from_lat=("from_lat", "median"),
            from_lon=("from_lon", "median"),
            to_lat=("to_lat", "median"),
            to_lon=("to_lon", "median"),
            master_distance_m=("geo_distance_m", "median"),
            adjacency_count=("trip_id", "nunique"),
        )
    )

    # Robustly remove implausible geocoded jumps. Route 192 is an all-stops
    # urban service, so consecutive stops should be comparatively close.
    valid_dist = agg["master_distance_m"].dropna()
    if not valid_dist.empty:
        q1 = float(valid_dist.quantile(0.25))
        q3 = float(valid_dist.quantile(0.75))
        iqr = max(q3 - q1, 1.0)
        robust_limit = q3 + 4.0 * iqr
        # Keep the rule conservative: never below 1.2 km, never above 2.0 km.
        max_plausible_m = max(1200.0, min(2000.0, robust_limit))
    else:
        max_plausible_m = 1600.0

    agg["_master_geometry_valid"] = (
        agg["from_lat"].notna()
        & agg["from_lon"].notna()
        & agg["to_lat"].notna()
        & agg["to_lon"].notna()
        & agg["master_distance_m"].le(max_plausible_m)
    )

    return agg


def derive_expected_distance_m(df):
    """
    Recover segment distance for validation / display.

    Priority:
      1. explicit distance_m from the RQ3 export;
      2. distance implied by PMD, passenger load and the 68 kg baseline mass.

    PMD = load * 0.068 tonnes * distance_km
    """
    if df is None or df.empty:
        return pd.Series(dtype=float)

    explicit = numeric(df["distance_m"]) if "distance_m" in df.columns else pd.Series(
        np.nan, index=df.index, dtype=float
    )

    load = (
        numeric(df["p90_load"])
        if "p90_load" in df.columns
        else numeric(df["load"])
        if "load" in df.columns
        else pd.Series(np.nan, index=df.index, dtype=float)
    )
    pmd = (
        numeric(df["pmd_68"])
        if "pmd_68" in df.columns
        else pd.Series(np.nan, index=df.index, dtype=float)
    )

    implied = pd.Series(np.nan, index=df.index, dtype=float)
    valid = load.gt(0) & pmd.notna() & pmd.ge(0)
    # distance_m = PMD * 1,000,000 / (passengers * kg_per_passenger)
    implied.loc[valid] = (
        pmd.loc[valid] * 1_000_000.0 / (load.loc[valid] * BASELINE_MASS)
    )

    return explicit.where(explicit.notna() & explicit.gt(0), implied)


def apply_master_segment_geometry(df, master_adjacency, coords_df):
    """
    Build safe period-level segment geometry.

    An RQ3 section is mapped when either:
      1) the directed stop pair is observed consecutively in route192_master.csv; or
      2) both stop coordinates exist and their straight-line separation is
         consistent with the analytical segment distance (explicit or PMD-implied).

    This is intentionally more tolerant than the v7 master-only rule, while still
    rejecting the implausible long diagonal artefacts seen in the earlier map.
    """
    if df is None or df.empty:
        return df

    out = df.copy()
    out["from_stop"] = out["from_stop"].map(clean_id)
    out["to_stop"] = out["to_stop"].map(clean_id)
    out["direction"] = out["direction"].map(direction_label)

    # First recover coordinates/names from the reusable stop-coordinate lookup.
    out = fill_segment_geometry(out, coords_df)

    # Analytical distance is also useful when the export did not explicitly
    # include segment_distance_m.
    out["_expected_distance_m"] = derive_expected_distance_m(out)

    # ------------------------------------------------------------------
    # Exact master-adjacency validation where available.
    # ------------------------------------------------------------------
    if master_adjacency is not None and not master_adjacency.empty:
        ref = master_adjacency.rename(
            columns={
                "from_stop_name": "_route_from_name",
                "to_stop_name": "_route_to_name",
                "from_lat": "_route_from_lat",
                "from_lon": "_route_from_lon",
                "to_lat": "_route_to_lat",
                "to_lon": "_route_to_lon",
                "master_distance_m": "_route_distance_m",
                "adjacency_count": "_route_adjacency_count",
                "_master_geometry_valid": "_route_geometry_valid",
            }
        )

        keep = [
            "direction",
            "from_stop",
            "to_stop",
            "_route_from_name",
            "_route_to_name",
            "_route_from_lat",
            "_route_from_lon",
            "_route_to_lat",
            "_route_to_lon",
            "_route_distance_m",
            "_route_adjacency_count",
            "_route_geometry_valid",
        ]

        out = out.merge(
            ref[keep],
            on=["direction", "from_stop", "to_stop"],
            how="left",
            validate="m:1",
        )

        master_match = (
            out["_route_adjacency_count"].notna()
            & out["_route_geometry_valid"].fillna(False)
        )

        # For exact matches, master sequence geometry is authoritative.
        for target, source in [
            ("from_lat", "_route_from_lat"),
            ("from_lon", "_route_from_lon"),
            ("to_lat", "_route_to_lat"),
            ("to_lon", "_route_to_lon"),
        ]:
            out.loc[master_match, target] = out.loc[master_match, source]

        good_from_name = (
            master_match
            & out["_route_from_name"].fillna("").astype(str).str.strip().ne("")
        )
        good_to_name = (
            master_match
            & out["_route_to_name"].fillna("").astype(str).str.strip().ne("")
        )
        out.loc[good_from_name, "from_stop_name"] = out.loc[
            good_from_name, "_route_from_name"
        ]
        out.loc[good_to_name, "to_stop_name"] = out.loc[
            good_to_name, "_route_to_name"
        ]

        out.loc[
            master_match & out["_expected_distance_m"].isna(),
            "_expected_distance_m",
        ] = out.loc[
            master_match & out["_expected_distance_m"].isna(),
            "_route_distance_m",
        ]
    else:
        master_match = pd.Series(False, index=out.index)

    # ------------------------------------------------------------------
    # Coordinate-distance validation for unmatched rows.
    # ------------------------------------------------------------------
    out["geo_distance_m"] = out.apply(
        lambda r: haversine_m(
            r.get("from_lat"),
            r.get("from_lon"),
            r.get("to_lat"),
            r.get("to_lon"),
        ),
        axis=1,
    )

    have_coords = (
        out["from_lat"].notna()
        & out["from_lon"].notna()
        & out["to_lat"].notna()
        & out["to_lon"].notna()
        & out["geo_distance_m"].notna()
    )

    have_expected = out["_expected_distance_m"].notna() & out["_expected_distance_m"].gt(0)

    # Road / analytical distance should normally be >= straight-line distance.
    # Allow generous tolerance for stop-geocoding precision and route curvature,
    # but reject multi-kilometre diagonal artefacts.
    ratio = out["geo_distance_m"] / out["_expected_distance_m"]
    distance_consistent = (
        have_coords
        & have_expected
        & out["geo_distance_m"].le(2000.0)
        & ratio.between(0.10, 1.60, inclusive="both")
    )

    # If no analytical distance is available, accept only a short urban link.
    short_coordinate_link = (
        have_coords
        & ~have_expected
        & out["geo_distance_m"].le(1200.0)
    )

    fallback_valid = (~master_match) & (distance_consistent | short_coordinate_link)
    valid = master_match | fallback_valid

    out["_geometry_valid"] = valid

    out["geometry_reason"] = np.select(
        [
            master_match,
            fallback_valid & have_expected,
            fallback_valid & ~have_expected,
            ~have_coords,
            have_coords & have_expected & ~distance_consistent,
        ],
        [
            "Validated by ordered Route 192 stop sequence",
            "Validated by stop coordinates and analytical segment distance",
            "Validated by short stop-to-stop coordinate separation",
            "Missing endpoint coordinates",
            "Rejected: coordinate separation inconsistent with segment distance",
        ],
        default="Rejected: unvalidated geometry",
    )

    # Never draw rejected coordinates.
    rejected = ~valid
    out.loc[
        rejected,
        ["from_lat", "from_lon", "to_lat", "to_lon"],
    ] = np.nan

    # Populate segment distance for the inspector when it can be reconstructed
    # from the PMD definition.
    if "distance_m" not in out.columns:
        out["distance_m"] = np.nan
    out["distance_m"] = numeric(out["distance_m"])
    missing_dist = out["distance_m"].isna() | out["distance_m"].le(0)
    out.loc[missing_dist, "distance_m"] = out.loc[
        missing_dist, "_expected_distance_m"
    ]

    drop_cols = [c for c in out.columns if c.startswith("_route_")]
    out = out.drop(columns=drop_cols, errors="ignore")
    return out


def fill_segment_geometry(df, coords_df):
    """Fill missing endpoint coordinates/names from the unified stop lookup."""
    if df is None or df.empty:
        return df
    out = df.copy()

    for c in ["from_lat", "from_lon", "to_lat", "to_lon"]:
        if c not in out.columns:
            out[c] = np.nan
    for c in ["from_stop_name", "to_stop_name"]:
        if c not in out.columns:
            out[c] = ""

    if coords_df is not None and not coords_df.empty:
        ref = coords_df[["stop_id", "lat", "lon", "stop_name"]].copy()
        fr = ref.rename(columns={
            "stop_id": "from_stop", "lat": "_fr_lat", "lon": "_fr_lon", "stop_name": "_fr_name"
        })
        to = ref.rename(columns={
            "stop_id": "to_stop", "lat": "_to_lat", "lon": "_to_lon", "stop_name": "_to_name"
        })
        out = out.merge(fr, on="from_stop", how="left").merge(to, on="to_stop", how="left")
        out["from_lat"] = out["from_lat"].fillna(out["_fr_lat"])
        out["from_lon"] = out["from_lon"].fillna(out["_fr_lon"])
        out["to_lat"] = out["to_lat"].fillna(out["_to_lat"])
        out["to_lon"] = out["to_lon"].fillna(out["_to_lon"])
        missing = out["from_stop_name"].fillna("").str.strip().isin(["", "nan", "None"])
        out.loc[missing, "from_stop_name"] = out.loc[missing, "_fr_name"].fillna("")
        missing = out["to_stop_name"].fillna("").str.strip().isin(["", "nan", "None"])
        out.loc[missing, "to_stop_name"] = out.loc[missing, "_to_name"].fillna("")
        out = out.drop(columns=["_fr_lat", "_fr_lon", "_fr_name", "_to_lat", "_to_lon", "_to_name"])

    # Reject clearly implausible joins caused by incompatible stop-ID systems.
    out["geo_distance_m"] = out.apply(
        lambda r: haversine_m(r["from_lat"], r["from_lon"], r["to_lat"], r["to_lon"]), axis=1
    )
    out.loc[out["geo_distance_m"] > 3500, ["from_lat", "from_lon", "to_lat", "to_lon"]] = np.nan
    return out


def normalise_stop_name(value):
    s = str(value or "").strip().lower()
    s = re.sub(r"\([^)]*\)", " ", s)
    s = re.sub(r"\bstop\s+[a-z0-9]+\b", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def fill_coordinates_by_stop_name(df, coords_df):
    """
    Secondary coordinate recovery for cases where the analytical/master stop ID
    and the coordinate-file stop ID use different coding systems.

    Only exact matches after conservative name normalisation are used.
    """
    if df is None or df.empty or coords_df is None or coords_df.empty:
        return df

    out = df.copy()

    if "stop_name" not in coords_df.columns:
        return out

    ref = coords_df[["stop_name", "lat", "lon"]].copy()
    ref["_name_key"] = ref["stop_name"].map(normalise_stop_name)
    ref = ref[
        ref["_name_key"].ne("")
        & ref["lat"].notna()
        & ref["lon"].notna()
    ].copy()

    if ref.empty:
        return out

    # If the same normalised name appears more than once, use the median
    # coordinate only when the points are tightly clustered.
    name_rows = []
    for key, g in ref.groupby("_name_key"):
        if len(g) == 1:
            name_rows.append(
                {
                    "_name_key": key,
                    "_name_lat": float(g.iloc[0]["lat"]),
                    "_name_lon": float(g.iloc[0]["lon"]),
                }
            )
        else:
            lat_med = float(g["lat"].median())
            lon_med = float(g["lon"].median())
            spread = max(
                haversine_m(lat_med, lon_med, float(r["lat"]), float(r["lon"]))
                for _, r in g.iterrows()
            )
            # Conservative: only collapse duplicate stop names if their mapped
            # points are effectively the same physical stop area.
            if spread <= 250:
                name_rows.append(
                    {
                        "_name_key": key,
                        "_name_lat": lat_med,
                        "_name_lon": lon_med,
                    }
                )

    if not name_rows:
        return out

    name_ref = pd.DataFrame(name_rows)

    for prefix in ["from", "to"]:
        name_col = f"{prefix}_stop_name"
        lat_col = f"{prefix}_lat"
        lon_col = f"{prefix}_lon"

        if name_col not in out.columns:
            continue

        out[f"_{prefix}_name_key"] = out[name_col].map(normalise_stop_name)
        tmp = name_ref.rename(
            columns={
                "_name_key": f"_{prefix}_name_key",
                "_name_lat": f"_{prefix}_name_lat",
                "_name_lon": f"_{prefix}_name_lon",
            }
        )
        out = out.merge(tmp, on=f"_{prefix}_name_key", how="left")

        if lat_col not in out.columns:
            out[lat_col] = np.nan
        if lon_col not in out.columns:
            out[lon_col] = np.nan

        out[lat_col] = numeric(out[lat_col]).fillna(out[f"_{prefix}_name_lat"])
        out[lon_col] = numeric(out[lon_col]).fillna(out[f"_{prefix}_name_lon"])

        out = out.drop(
            columns=[
                f"_{prefix}_name_key",
                f"_{prefix}_name_lat",
                f"_{prefix}_name_lon",
            ],
            errors="ignore",
        )

    return out


def build_canonical_route_backbone(trip_segments, coords_df, direction, period=None):
    """
    Build a COMPLETE physical Route 192 backbone for one direction.

    IMPORTANT:
    Operating period is intentionally NOT used to filter the physical route.
    A period is a temporal analytical filter, not part of route topology.

    The previous implementation filtered stop-visit rows by period before
    choosing a representative trip. That removed many stops from the route and
    made Manchester -> Hazel Grove appear as disconnected fragments.

    The corrected logic is:
        direction -> choose the most complete full trip -> ordered stop sequence
        -> recover coordinates -> draw all consecutive stop-to-stop sections.

    RQ3 period-specific P90 values are overlaid later.
    """
    if trip_segments is None or trip_segments.empty:
        return None

    # Physical topology depends ONLY on direction.
    x = trip_segments.loc[
        trip_segments["direction"].map(direction_label) == direction_label(direction)
    ].copy()

    if x.empty:
        return None

    # Choose the fullest route traversal in the selected direction.
    # Coordinate coverage is only a tie-breaker.
    trip_stats = (
        x.groupby("trip_id", as_index=False)
        .agg(
            sections=("to_stop", "size"),
            coordinate_pairs=(
                "from_lat",
                lambda s: int(s.notna().sum()),
            ),
        )
        .sort_values(
            ["sections", "coordinate_pairs", "trip_id"],
            ascending=[False, False, True],
        )
    )

    canonical_trip = str(trip_stats.iloc[0]["trip_id"])

    # Crucially: pull the FULL trip from direction-filtered x, not a
    # period-filtered subset.
    bb = x.loc[x["trip_id"].astype(str) == canonical_trip].copy()
    bb = bb.sort_values("stop_sequence").reset_index(drop=True)

    # ID-based coordinate recovery first.
    bb = fill_segment_geometry(bb, coords_df)

    # Name-based fallback for incompatible stop-ID systems.
    bb = fill_coordinates_by_stop_name(bb, coords_df)

    bb["from_stop"] = bb["from_stop"].map(clean_id)
    bb["to_stop"] = bb["to_stop"].map(clean_id)

    bb["_segment_key"] = bb["from_stop"] + "__" + bb["to_stop"]
    bb["_segment_display"] = bb.apply(
        lambda r: friendly_segment_display(r, STOP_NAME_LOOKUP), axis=1
    )
    bb["_segment_code"] = bb.apply(segment_code_display, axis=1)

    # Recompute geometry AFTER both coordinate-recovery stages.
    bb["geo_distance_m"] = bb.apply(
        lambda r: haversine_m(
            r.get("from_lat"),
            r.get("from_lon"),
            r.get("to_lat"),
            r.get("to_lon"),
        ),
        axis=1,
    )

    # Because sections come from one ordered trip, they are topologically
    # consecutive by construction. Only suppress impossible coordinate jumps.
    bb["_geometry_valid"] = (
        bb["from_lat"].notna()
        & bb["from_lon"].notna()
        & bb["to_lat"].notna()
        & bb["to_lon"].notna()
        & bb["geo_distance_m"].notna()
        & bb["geo_distance_m"].le(2500.0)
    )

    bb["_canonical_trip"] = canonical_trip

    if "distance_m" not in bb.columns:
        bb["distance_m"] = np.nan
    bb["distance_m"] = numeric(bb["distance_m"])
    missing_dist = bb["distance_m"].isna() | bb["distance_m"].le(0)
    bb.loc[missing_dist, "distance_m"] = bb.loc[
        missing_dist, "geo_distance_m"
    ]

    return bb


def attach_period_metrics_to_backbone(backbone, period_metrics):
    """
    Overlay RQ3 period-level pressure values onto the continuous physical route.

    Matching priority:
      1. exact directed stop-ID pair;
      2. normalised from/to stop-name pair.

    Backbone sections with no matching RQ3 row are deliberately retained and
    shown in neutral grey. This avoids a visually broken route.
    """
    if backbone is None or backbone.empty:
        return backbone
    out = backbone.copy()

    metric_cols = [
        "p90_load", "p90_util", "pmd_68", "lsoa", "segment",
    ]
    for c in metric_cols:
        if c not in out.columns:
            out[c] = np.nan if c != "lsoa" and c != "segment" else ""

    # Clear trip-level values that must not masquerade as P90 period values.
    out["p90_load"] = np.nan
    out["p90_util"] = np.nan
    out["pmd_68"] = np.nan
    out["_metric_match"] = False
    out["_metric_match_method"] = "No period-level metric matched"

    if period_metrics is None or period_metrics.empty:
        return out

    m = period_metrics.copy()
    m["from_stop"] = m["from_stop"].map(clean_id)
    m["to_stop"] = m["to_stop"].map(clean_id)

    # Ensure names exist before fallback matching.
    m = fill_segment_geometry(m, coords)

    for c in ["p90_load", "p90_util", "pmd_68"]:
        if c not in m.columns:
            m[c] = np.nan
        m[c] = numeric(m[c])

    # Aggregate duplicate analytical rows for the same directed stop pair.
    agg_map = {
        "p90_load": "max",
        "p90_util": "max",
        "pmd_68": "max",
    }
    if "lsoa" in m.columns:
        agg_map["lsoa"] = "first"
    if "segment" in m.columns:
        agg_map["segment"] = "first"
    if "from_stop_name" in m.columns:
        agg_map["from_stop_name"] = "first"
    if "to_stop_name" in m.columns:
        agg_map["to_stop_name"] = "first"

    exact = (
        m.groupby(["from_stop", "to_stop"], as_index=False)
        .agg(agg_map)
    )

    exact = exact.rename(columns={
        "p90_load": "_m_p90_load",
        "p90_util": "_m_p90_util",
        "pmd_68": "_m_pmd_68",
        "lsoa": "_m_lsoa",
        "segment": "_m_segment",
        "from_stop_name": "_m_from_name",
        "to_stop_name": "_m_to_name",
    })

    keep = ["from_stop", "to_stop"] + [
        c for c in [
            "_m_p90_load", "_m_p90_util", "_m_pmd_68", "_m_lsoa",
            "_m_segment", "_m_from_name", "_m_to_name"
        ] if c in exact.columns
    ]

    out = out.merge(exact[keep], on=["from_stop", "to_stop"], how="left")

    exact_match = (
        out.get("_m_p90_load", pd.Series(np.nan, index=out.index)).notna()
        | out.get("_m_p90_util", pd.Series(np.nan, index=out.index)).notna()
        | out.get("_m_pmd_68", pd.Series(np.nan, index=out.index)).notna()
    )

    for dst, src_col in [
        ("p90_load", "_m_p90_load"),
        ("p90_util", "_m_p90_util"),
        ("pmd_68", "_m_pmd_68"),
    ]:
        if src_col in out.columns:
            out.loc[exact_match, dst] = out.loc[exact_match, src_col]

    if "_m_lsoa" in out.columns:
        out.loc[exact_match, "lsoa"] = out.loc[exact_match, "_m_lsoa"].fillna("")
    if "_m_segment" in out.columns:
        out.loc[exact_match, "segment"] = out.loc[exact_match, "_m_segment"].fillna("")

    out.loc[exact_match, "_metric_match"] = True
    out.loc[exact_match, "_metric_match_method"] = "Matched by directed stop IDs"

    # -------------------------------------------------------------
    # Fallback by normalised stop names for ID-system inconsistencies.
    # -------------------------------------------------------------
    unmatched = ~out["_metric_match"]

    if unmatched.any():
        if "from_stop_name" not in m.columns:
            m["from_stop_name"] = ""
        if "to_stop_name" not in m.columns:
            m["to_stop_name"] = ""

        m["_from_name_key"] = m["from_stop_name"].map(normalise_stop_name)
        m["_to_name_key"] = m["to_stop_name"].map(normalise_stop_name)
        name_m = m.loc[
            m["_from_name_key"].ne("") & m["_to_name_key"].ne("")
        ].copy()

        if not name_m.empty:
            name_agg = (
                name_m.groupby(["_from_name_key", "_to_name_key"], as_index=False)
                .agg(
                    _n_p90_load=("p90_load", "max"),
                    _n_p90_util=("p90_util", "max"),
                    _n_pmd_68=("pmd_68", "max"),
                    _n_lsoa=("lsoa", "first") if "lsoa" in name_m.columns else ("from_stop", "first"),
                )
            )

            out["_from_name_key"] = out["from_stop_name"].map(normalise_stop_name)
            out["_to_name_key"] = out["to_stop_name"].map(normalise_stop_name)
            out = out.merge(
                name_agg,
                on=["_from_name_key", "_to_name_key"],
                how="left",
            )

            name_match = (
                ~out["_metric_match"]
                & (
                    out["_n_p90_load"].notna()
                    | out["_n_p90_util"].notna()
                    | out["_n_pmd_68"].notna()
                )
            )

            out.loc[name_match, "p90_load"] = out.loc[name_match, "_n_p90_load"]
            out.loc[name_match, "p90_util"] = out.loc[name_match, "_n_p90_util"]
            out.loc[name_match, "pmd_68"] = out.loc[name_match, "_n_pmd_68"]
            if "_n_lsoa" in out.columns:
                out.loc[name_match, "lsoa"] = out.loc[name_match, "_n_lsoa"].fillna("")
            out.loc[name_match, "_metric_match"] = True
            out.loc[name_match, "_metric_match_method"] = "Matched by stop names"

    # Clean temporary merge columns.
    tmp = [
        c for c in out.columns
        if c.startswith("_m_") or c.startswith("_n_")
        or c in {"_from_name_key", "_to_name_key"}
    ]
    out = out.drop(columns=tmp, errors="ignore")

    return out


def apply_rq2_context_as_highlight(backbone_df, overlay_df, selected_context):
    """
    Preserve the whole route geometry. When an RQ2 context is selected, mark
    matching sections instead of deleting all other sections.
    """
    if backbone_df is None or backbone_df.empty:
        return backbone_df

    out = backbone_df.copy()
    out["_context_match"] = True

    if (
        selected_context == "All RQ2 contexts"
        or overlay_df is None
        or overlay_df.empty
        or "lsoa" not in out.columns
    ):
        return out

    context_codes = set(
        overlay_df.loc[
            overlay_df["group"] == selected_context, "lsoa_code"
        ].dropna().astype(str)
    )
    if not context_codes:
        return out

    out["_context_match"] = out["lsoa"].astype(str).isin(context_codes)
    return out


PRESSURE_COLORSCALE = [
    [0.00, "#D9DEE8"],
    [0.35, "#FDE68A"],
    [0.62, "#FFD400"],
    [0.82, "#F59E0B"],
    [1.00, "#991B1B"],
]


def pressure_colour(value, vmin, vmax):
    if pd.isna(value):
        return "#D9DEE8"
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        t = 0.5
    else:
        t = (float(value) - float(vmin)) / (float(vmax) - float(vmin))
    t = max(0.0, min(1.0, t))
    return sample_colorscale(PRESSURE_COLORSCALE, [t])[0]


def build_pressure_map(df, value_col, value_label, selected_key=None):
    """
    Draw the complete stop-to-stop route backbone.

    Sections with a valid selected metric are coloured by pressure.
    Sections without a matched metric are retained in neutral grey so the
    Route 192 corridor never appears artificially broken.
    """
    if df is None or df.empty:
        return None, 0, 0, 0

    x = df.dropna(subset=["from_lat", "from_lon", "to_lat", "to_lon"]).copy()
    total = len(df)
    mapped_n = len(x)
    if x.empty:
        return None, mapped_n, total, 0

    measured = x[value_col].notna()
    measured_n = int(measured.sum())

    if measured_n:
        vals = numeric(x.loc[measured, value_col]).dropna()
        vmin, vmax = float(vals.min()), float(vals.max())
    else:
        vmin, vmax = 0.0, 1.0

    fig = go.Figure()

    # Draw unmeasured route sections first in neutral grey.
    draw_order = x.copy()
    draw_order["_draw_priority"] = np.where(draw_order[value_col].notna(), 1, 0)
    if selected_key is not None:
        draw_order["_draw_priority"] += (
            draw_order["_segment_key"].astype(str) == str(selected_key)
        ).astype(int) * 10
    draw_order = draw_order.sort_values(
        ["_draw_priority", value_col],
        ascending=[True, True],
        na_position="first",
    )

    for _, row in draw_order.iterrows():
        v = row.get(value_col, np.nan)
        is_measured = pd.notna(v)
        context_match = bool(row.get("_context_match", True))

        if is_measured:
            colour = pressure_colour(v, vmin, vmax)
        else:
            colour = "#C7CEDA"

        # If an RQ2 context filter is active, retain non-matching route sections
        # but de-emphasise them rather than deleting them.
        if not context_match:
            colour = "#DDE2EA"

        from_name = (
            str(row.get("from_stop_name", "")).strip()
            or stop_display(row.get("from_stop"), STOP_NAME_LOOKUP)
        )
        to_name = (
            str(row.get("to_stop_name", "")).strip()
            or stop_display(row.get("to_stop"), STOP_NAME_LOOKUP)
        )
        code = segment_code_display(row)

        is_selected = (
            selected_key is not None
            and str(row.get("_segment_key", "")) == str(selected_key)
        )
        width = 9 if is_selected else (5.0 if is_measured and context_match else 3.2)

        load_value = row.get("p90_load", row.get("load", np.nan))
        util_value = row.get("p90_util", row.get("util", np.nan))
        pmd_value = row.get("pmd_68", np.nan)

        if is_measured:
            metric_text = (
                fmt_pct(v)
                if "Utilisation" in value_label
                else fmt_num(v, 2)
            )
        else:
            metric_text = "No matched period-level value"

        hover = (
            f"<b>{from_name} → {to_name}</b><br>"
            f"Stop IDs: {code}<br>"
            f"{value_label}: {metric_text}<br>"
            f"Onboard load: {fmt_num(load_value, 1)} passengers<br>"
            f"Capacity utilisation: {fmt_pct(util_value)}<br>"
            f"PMD at 68 kg: {fmt_num(pmd_value, 2)} tonne-km"
        )

        if not is_measured:
            hover += "<br><i>Route geometry shown for continuity; no matching RQ3 metric row.</i>"
        if not context_match:
            hover += "<br><i>Outside the selected RQ2 health–demand context.</i>"

        fig.add_trace(
            go.Scattermapbox(
                lat=[row["from_lat"], row["to_lat"]],
                lon=[row["from_lon"], row["to_lon"]],
                mode="lines",
                line=dict(color=colour, width=width),
                hovertext=[hover, hover],
                hoverinfo="text",
                showlegend=False,
            )
        )

    # Stop points from the complete route backbone.
    stops = pd.concat(
        [
            x[["from_stop", "from_stop_name", "from_lat", "from_lon"]].rename(
                columns={
                    "from_stop": "stop_id",
                    "from_stop_name": "stop_name",
                    "from_lat": "lat",
                    "from_lon": "lon",
                }
            ),
            x[["to_stop", "to_stop_name", "to_lat", "to_lon"]].rename(
                columns={
                    "to_stop": "stop_id",
                    "to_stop_name": "stop_name",
                    "to_lat": "lat",
                    "to_lon": "lon",
                }
            ),
        ],
        ignore_index=True,
    ).drop_duplicates("stop_id")

    stops["stop_name"] = stops.apply(
        lambda r: (
            str(r["stop_name"]).strip()
            if str(r["stop_name"]).strip() not in {"", "nan", "None"}
            else stop_display(r["stop_id"], STOP_NAME_LOOKUP)
        ),
        axis=1,
    )
    stops["hover"] = (
        "<b>" + stops["stop_name"] + "</b><br>Stop ID: " + stops["stop_id"].astype(str)
    )

    fig.add_trace(
        go.Scattermapbox(
            lat=stops["lat"],
            lon=stops["lon"],
            mode="markers",
            marker=dict(size=5.2, color=BLACK),
            hovertext=stops["hover"],
            hoverinfo="text",
            showlegend=False,
        )
    )

    centre_lat = float(stops["lat"].mean())
    centre_lon = float(stops["lon"].mean())

    # Colourbar is based only on sections that actually have the selected metric.
    if measured_n:
        fig.add_trace(
            go.Scattermapbox(
                lat=[centre_lat, centre_lat],
                lon=[centre_lon, centre_lon],
                mode="markers",
                marker=dict(
                    size=1,
                    opacity=0.01,
                    color=[vmin, vmax],
                    cmin=vmin,
                    cmax=vmax,
                    colorscale=PRESSURE_COLORSCALE,
                    showscale=True,
                    colorbar=dict(
                        title=value_label,
                        thickness=14,
                        len=0.66,
                        x=1.01,
                    ),
                ),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.update_layout(
        mapbox=dict(
            style="carto-positron",
            center=dict(lat=centre_lat, lon=centre_lon),
            zoom=10.7,
        ),
        height=570,
        margin=dict(l=0, r=55, t=8, b=0),
        paper_bgcolor=WHITE,
        font=dict(family="Inter"),
        uirevision="rq3_pressure_map",
    )

    return fig, mapped_n, total, measured_n


# =============================================================================
# DATA CANONICALISATION
# =============================================================================

def canonicalise_direction_period(df):
    if df is None or df.empty:
        return None
    out = df.copy()

    dir_col = find_col(
        out,
        ["direction", "direction_label", "route_direction", "direction_name"],
        contains=["direction"],
    )
    period_col = find_col(
        out,
        ["operating_period", "detailed_time_period", "time_period", "period"],
    )
    util_col = find_col(
        out,
        [
            "p90_capacity_utilisation",
            "p90_capacity_utilization",
            "capacity_utilisation_p90",
            "capacity_utilization_p90",
            "p90_utilisation",
            "p90_utilization",
            "p90_capacity_pct",
        ],
        contains=["p90", "util"],
    )
    load_col = find_col(
        out,
        ["p90_load", "load_p90", "p90_onboard_load", "p90_loading_count", "p90_passengers"],
        contains=["p90", "load"],
    )
    mean_load_col = find_col(
        out,
        ["mean_load", "avg_load", "mean_onboard_load", "average_load"],
        contains=["mean", "load"],
    )

    result = pd.DataFrame(index=out.index)
    result["direction"] = out[dir_col].map(direction_label) if dir_col else "Route 192"
    result["period"] = out[period_col].astype(str) if period_col else "All periods"
    result["p90_util"] = coerce_percent(out[util_col]) if util_col else np.nan
    result["p90_load"] = numeric(out[load_col]) if load_col else np.nan
    result["mean_load"] = numeric(out[mean_load_col]) if mean_load_col else np.nan

    if result["p90_util"].isna().all() and result["p90_load"].notna().any():
        result["p90_util"] = result["p90_load"] / TOTAL_CAPACITY * 100

    return result.dropna(how="all", subset=["p90_util", "p90_load", "mean_load"])


def canonicalise_segments(df):
    if df is None or df.empty:
        return None
    out = df.copy()

    dir_col = find_col(out, ["direction", "direction_label", "route_direction"])
    period_col = find_col(out, ["operating_period", "detailed_time_period", "time_period", "period"])
    from_col = find_col(
        out,
        ["from_stop_id", "origin_stop_id", "segment_origin_stop_id", "stop_id", "from_stop"],
    )
    to_col = find_col(
        out,
        ["to_stop_id", "destination_stop_id", "next_stop_id", "segment_destination_stop_id", "to_stop"],
    )
    seg_col = find_col(
        out,
        ["segment", "segment_id", "segment_label", "directed_segment"],
    )
    util_col = find_col(
        out,
        [
            "p90_capacity_utilisation",
            "p90_capacity_utilization",
            "capacity_utilisation_p90",
            "capacity_utilization_p90",
            "p90_utilisation",
            "p90_utilization",
            "p90_capacity_pct",
        ],
        contains=["p90", "util"],
    )
    load_col = find_col(
        out,
        [
            "p90_load",
            "load_p90",
            "p90_onboard_load",
            "p90_loading_count",
            "p90_passengers",
            "p90_reconstructed_onboard_passengers",
        ],
        contains=["p90", "load"],
    )
    dist_m_col = find_col(
        out,
        ["segment_distance_m", "distance_m", "segment_length_m", "distance_metres", "distance_meters"],
    )
    dist_km_col = find_col(
        out,
        ["segment_distance_km", "distance_km", "segment_length_km"],
    )
    pmd_col = find_col(
        out,
        [
            "p90_pmd_tonne_km",
            "p90_pmd_tkm",
            "pmd_p90",
            "p90_passenger_mass_distance",
            "p90_pmd_68kg",
            "pmd_68kg",
            "passenger_mass_distance",
        ],
        contains=["pmd"],
    )
    lsoa_col = find_col(
        out,
        ["LSOA21CD", "lsoa_code", "LSOA_code", "origin_lsoa", "segment_lsoa"],
    )
    trip_col = find_col(out, ["trip_id", "tripid", "journey_id", "vehicle_journey_id"])
    from_lat_col = find_col(out, ["from_lat", "origin_lat", "from_stop_lat", "segment_origin_lat"])
    from_lon_col = find_col(out, ["from_lon", "from_lng", "origin_lon", "origin_lng", "from_stop_lon"])
    to_lat_col = find_col(out, ["to_lat", "destination_lat", "to_stop_lat", "segment_destination_lat"])
    to_lon_col = find_col(out, ["to_lon", "to_lng", "destination_lon", "destination_lng", "to_stop_lon"])

    result = pd.DataFrame(index=out.index)
    result["direction"] = out[dir_col].map(direction_label) if dir_col else "Route 192"
    result["period"] = out[period_col].astype(str) if period_col else "All periods"
    result["from_stop"] = out[from_col].map(clean_id) if from_col else ""
    result["to_stop"] = out[to_col].map(clean_id) if to_col else ""
    result["segment"] = out[seg_col].astype(str) if seg_col else ""
    result["p90_util"] = coerce_percent(out[util_col]) if util_col else np.nan
    result["p90_load"] = numeric(out[load_col]) if load_col else np.nan

    if dist_m_col:
        result["distance_m"] = numeric(out[dist_m_col])
    elif dist_km_col:
        result["distance_m"] = numeric(out[dist_km_col]) * 1000.0
    else:
        result["distance_m"] = np.nan

    result["pmd_68"] = numeric(out[pmd_col]) if pmd_col else np.nan
    result["lsoa"] = out[lsoa_col].astype(str) if lsoa_col else ""
    result["trip_id"] = out[trip_col].astype(str).str.strip() if trip_col else ""
    result["from_lat"] = numeric(out[from_lat_col]) if from_lat_col else np.nan
    result["from_lon"] = numeric(out[from_lon_col]) if from_lon_col else np.nan
    result["to_lat"] = numeric(out[to_lat_col]) if to_lat_col else np.nan
    result["to_lon"] = numeric(out[to_lon_col]) if to_lon_col else np.nan

    # Recover endpoints from the segment label where dedicated from/to columns are absent.
    parsed = result["segment"].map(parse_segment_endpoints)
    parsed_from = parsed.map(lambda x: x[0])
    parsed_to = parsed.map(lambda x: x[1])
    # When the segment label itself explicitly contains both endpoints, prefer
    # those IDs. This avoids inconsistent origin/next-stop fields creating
    # implausible long lines in one direction.
    explicit_pair = parsed_from.ne("") & parsed_to.ne("")
    result.loc[explicit_pair, "from_stop"] = parsed_from[explicit_pair]
    result.loc[explicit_pair, "to_stop"] = parsed_to[explicit_pair]
    result.loc[result["from_stop"].eq(""), "from_stop"] = parsed_from[result["from_stop"].eq("")]
    result.loc[result["to_stop"].eq(""), "to_stop"] = parsed_to[result["to_stop"].eq("")]

    if result["p90_util"].isna().all() and result["p90_load"].notna().any():
        result["p90_util"] = result["p90_load"] / TOTAL_CAPACITY * 100

    if result["pmd_68"].isna().all():
        have = result["p90_load"].notna() & result["distance_m"].notna()
        result.loc[have, "pmd_68"] = (
            result.loc[have, "p90_load"]
            * BASELINE_MASS
            / 1000.0
            * (result.loc[have, "distance_m"] / 1000.0)
        )

    blank_seg = result["segment"].astype(str).str.strip().isin(["", "nan", "None"])
    result.loc[blank_seg, "segment"] = result.loc[blank_seg].apply(segment_label, axis=1)

    return result


def canonicalise_rank(df):
    if df is None or df.empty:
        return None
    out = df.copy()

    from_col = find_col(out, ["from_stop_id", "origin_stop_id", "from_stop"])
    to_col = find_col(out, ["to_stop_id", "destination_stop_id", "to_stop", "next_stop_id"])
    seg_col = find_col(out, ["segment", "segment_id", "segment_label", "directed_segment"])

    load_col = find_col(
        out,
        ["p90_load", "p90_onboard_load", "load_p90", "p90_reconstructed_onboard_passengers"],
        contains=["p90", "load"],
    )
    pmd_col = find_col(
        out,
        ["p90_pmd_tonne_km", "p90_pmd_tkm", "pmd_p90", "pmd_68kg", "p90_pmd_68kg"],
        contains=["pmd"],
    )
    dist_m_col = find_col(out, ["segment_distance_m", "distance_m", "segment_length_m"])
    dist_km_col = find_col(out, ["segment_distance_km", "distance_km", "segment_length_km"])

    result = pd.DataFrame(index=out.index)
    result["from_stop"] = out[from_col].map(clean_id) if from_col else ""
    result["to_stop"] = out[to_col].map(clean_id) if to_col else ""
    result["segment"] = out[seg_col].astype(str) if seg_col else ""
    result["p90_load"] = numeric(out[load_col]) if load_col else np.nan
    result["pmd_68"] = numeric(out[pmd_col]) if pmd_col else np.nan

    if dist_m_col:
        result["distance_m"] = numeric(out[dist_m_col])
    elif dist_km_col:
        result["distance_m"] = numeric(out[dist_km_col]) * 1000.0
    else:
        result["distance_m"] = np.nan

    if result["pmd_68"].isna().all():
        have = result["p90_load"].notna() & result["distance_m"].notna()
        result.loc[have, "pmd_68"] = (
            result.loc[have, "p90_load"]
            * BASELINE_MASS
            / 1000.0
            * (result.loc[have, "distance_m"] / 1000.0)
        )

    blank_seg = result["segment"].astype(str).str.strip().isin(["", "nan", "None"])
    result.loc[blank_seg, "segment"] = result.loc[blank_seg].apply(segment_label, axis=1)

    # Reduce duplicate rows to one row per directed segment for ranking.
    agg = {
        "p90_load": "max",
        "pmd_68": "max",
        "distance_m": "max",
        "from_stop": "first",
        "to_stop": "first",
    }
    result = result.groupby("segment", as_index=False).agg(agg)
    return result


def canonicalise_overlay(df):
    if df is None or df.empty:
        return None
    out = df.copy()

    code_col = find_col(out, ["LSOA21CD", "lsoa_code", "LSOA_code", "lsoa"])
    name_col = find_col(out, ["LSOA21NM", "lsoa_name", "LSOA_name", "area_name"])
    group_col = find_col(
        out,
        ["RQ2_group", "rq2_context", "health_demand_group", "health_demand_context", "group"],
    )
    pressure_col = find_col(
        out,
        [
            "higher_operational_pressure",
            "high_operational_pressure",
            "operational_pressure",
            "pressure_group",
            "high_pressure",
        ],
    )
    util_col = find_col(
        out,
        [
            "max_p90_capacity_utilisation",
            "max_p90_capacity_utilization",
            "max_p90_utilisation",
            "max_p90_utilization",
            "p90_capacity_utilisation",
            "p90_capacity_utilization",
        ],
        contains=["p90", "util"],
    )
    pmd_col = find_col(
        out,
        ["max_p90_pmd", "p90_pmd_tonne_km", "pmd_68kg", "pmd"],
        contains=["pmd"],
    )
    demand_col = find_col(
        out,
        [
            "demand_occurrence",
            "boarding_occurrence_rate",
            "demand_rate",
            "reconstructed_demand_occurrence",
            "lsoa_demand_occurrence",
        ],
    )
    health_col = find_col(
        out,
        [
            "health_vulnerability_index",
            "health_index",
            "composite_health_index",
            "health_vulnerability",
            "z_health",
            "health_z",
        ],
    )
    gap_col = find_col(
        out,
        [
            "health_demand_gap",
            "health-demand_gap",
            "divergence_gap",
            "gap_z",
            "standardised_gap",
            "standardized_gap",
            "gap",
        ],
    )

    result = pd.DataFrame(index=out.index)
    result["lsoa_code"] = out[code_col].astype(str) if code_col else ""
    result["lsoa_name"] = out[name_col].astype(str) if name_col else result["lsoa_code"]
    result["group"] = out[group_col].map(canonical_group) if group_col else "Unknown"
    result["demand_occurrence"] = coerce_percent(out[demand_col]) if demand_col else np.nan
    result["health_index"] = numeric(out[health_col]) if health_col else np.nan
    result["health_demand_gap"] = numeric(out[gap_col]) if gap_col else np.nan
    result["max_p90_util"] = coerce_percent(out[util_col]) if util_col else np.nan
    result["pmd"] = numeric(out[pmd_col]) if pmd_col else np.nan

    if pressure_col:
        raw = out[pressure_col]
        if pd.api.types.is_bool_dtype(raw):
            result["high_pressure"] = raw.fillna(False)
        else:
            result["high_pressure"] = raw.astype(str).map(
                lambda x: norm(x) in {
                    "1",
                    "true",
                    "yes",
                    "high",
                    "higher",
                    "higherpressure",
                    "highpressure",
                    "higherrelativeoperationalpressure",
                }
            )
    else:
        # Derive the study-specific upper-quartile rule if the file contains LSOA utilisation.
        valid = result["max_p90_util"].dropna()
        if not valid.empty:
            threshold = float(valid.quantile(0.75))
            result["high_pressure"] = result["max_p90_util"] > threshold
        else:
            result["high_pressure"] = False

    # One row per LSOA if the export contains duplicates.
    if result["lsoa_code"].ne("").any():
        result = (
            result.sort_values("max_p90_util", na_position="first")
            .groupby("lsoa_code", as_index=False)
            .agg(
                lsoa_name=("lsoa_name", "first"),
                group=("group", "first"),
                demand_occurrence=("demand_occurrence", "first"),
                health_index=("health_index", "first"),
                health_demand_gap=("health_demand_gap", "first"),
                max_p90_util=("max_p90_util", "max"),
                pmd=("pmd", "max"),
                high_pressure=("high_pressure", "max"),
            )
        )
    return result


# =============================================================================
# LOAD DATA
# =============================================================================

dp_raw = safe_read_csv(resolve_output_file("direction_period"))
seg_raw = safe_read_csv(resolve_output_file("segments"))
rank_raw = safe_read_csv(resolve_output_file("rank"))
weight_raw = safe_read_csv(resolve_output_file("weight"))
percentile_raw = safe_read_csv(resolve_output_file("percentile"))
overlay_raw = safe_read_csv(resolve_output_file("overlay"))

dp = canonicalise_direction_period(dp_raw)
segments = canonicalise_segments(seg_raw)
rank_df = canonicalise_rank(rank_raw)
overlay = canonicalise_overlay(overlay_raw)
coords = load_stop_coordinates()
STOP_NAME_LOOKUP = build_stop_name_lookup(coords)
trip_segments = load_trip_segments_from_master()
if trip_segments is not None and not trip_segments.empty:
    trip_segments = fill_segment_geometry(trip_segments, coords)

# Authoritative physical topology for the period-level segment map.
MASTER_ADJACENCY = build_master_adjacency(trip_segments, coords)

# Add reader-facing stop names to segment/rank datasets once the lookup is available.
if segments is not None and not segments.empty:
    segments["segment_display"] = segments.apply(
        lambda r: friendly_segment_display(r, STOP_NAME_LOOKUP), axis=1
    )
    segments["segment_code"] = segments.apply(segment_code_display, axis=1)

if rank_df is not None and not rank_df.empty:
    rank_df["segment_display"] = rank_df.apply(
        lambda r: friendly_segment_display(r, STOP_NAME_LOOKUP), axis=1
    )
    rank_df["segment_code"] = rank_df.apply(segment_code_display, axis=1)

# If the dedicated rank export is unavailable OR cannot be interpreted,
# derive the load-vs-PMD ranking directly from the segment hotspot dataset.
# This makes the app robust even when RQ3_load_vs_PMD_rank_check.csv has
# different column names or is not copied into the Streamlit data folder.
_rank_needs_fallback = (
    rank_df is None
    or rank_df.empty
    or not {"p90_load", "pmd_68"}.issubset(rank_df.columns)
    or rank_df[["p90_load", "pmd_68"]].dropna().empty
)

if _rank_needs_fallback and segments is not None and not segments.empty:
    usable_segments = segments.dropna(subset=["p90_load", "pmd_68"]).copy()

    if not usable_segments.empty:
        rank_df = (
            usable_segments.groupby("segment", as_index=False)
            .agg(
                p90_load=("p90_load", "max"),
                pmd_68=("pmd_68", "max"),
                distance_m=("distance_m", "max"),
                from_stop=("from_stop", "first"),
                to_stop=("to_stop", "first"),
            )
        )

if rank_df is not None and not rank_df.empty:
    rank_df["segment_display"] = rank_df.apply(
        lambda r: friendly_segment_display(r, STOP_NAME_LOOKUP), axis=1
    )
    rank_df["segment_code"] = rank_df.apply(segment_code_display, axis=1)


# =============================================================================
# DERIVED STUDY KPIs
# =============================================================================

# Highest direction-period P90 utilisation
if dp is not None and dp["p90_util"].notna().any():
    dp_peak_idx = dp["p90_util"].idxmax()
    dp_peak = dp.loc[dp_peak_idx]
    direction_period_peak = float(dp_peak["p90_util"])
    direction_period_peak_label = f'{dp_peak["direction"]} · {dp_peak["period"]}'
else:
    direction_period_peak = 44.0
    direction_period_peak_label = "Manchester to Hazel Grove · AM peak"

# Highest segment P90 utilisation
if segments is not None and segments["p90_util"].notna().any():
    seg_peak_idx = segments["p90_util"].idxmax()
    seg_peak = segments.loc[seg_peak_idx]
    max_segment_util = float(seg_peak["p90_util"])
    max_segment_label = friendly_segment_display(seg_peak, STOP_NAME_LOOKUP)
else:
    max_segment_util = 76.1
    max_segment_label = "48651 → 48677"

# Within-corridor P75 operational pressure threshold and count
if overlay is not None and overlay["max_p90_util"].notna().any():
    p75_threshold = float(overlay["max_p90_util"].quantile(0.75))
    # Prefer explicit high-pressure classification if available.
    high_pressure_n = int(overlay["high_pressure"].sum())
    lsoa_n = int(len(overlay))
else:
    p75_threshold = 56.8
    high_pressure_n = 10
    lsoa_n = 39

# Dissertation values are the intended benchmark. If the underlying overlay has exactly
# 39 LSOAs, its empirical q75 should reproduce approximately 56.8%.
if lsoa_n == 39 and abs(p75_threshold - 56.8) < 3:
    p75_display = p75_threshold
else:
    p75_display = p75_threshold

# Maximum PMD
if rank_df is not None and rank_df["pmd_68"].notna().any():
    pmd_peak_idx = rank_df["pmd_68"].idxmax()
    pmd_peak = rank_df.loc[pmd_peak_idx]
    max_pmd = float(pmd_peak["pmd_68"])
    max_pmd_segment = friendly_segment_display(pmd_peak, STOP_NAME_LOOKUP)
    max_pmd_load = float(pmd_peak["p90_load"]) if pd.notna(pmd_peak["p90_load"]) else np.nan
    max_pmd_distance = float(pmd_peak["distance_m"]) if pd.notna(pmd_peak["distance_m"]) else np.nan
else:
    max_pmd = 2.57
    max_pmd_segment = "48680 → 134267"
    max_pmd_load = 72.4
    max_pmd_distance = 523.0


# =============================================================================
# RQ2 CARRY-OVER INSIGHTS
# =============================================================================

# Dissertation-level fallbacks are retained so the page still communicates the
# RQ2 -> RQ3 logic even if the local overlay export contains only operational fields.
rq2_lower_demand_n = 19
rq2_equity_n = 9
rq2_strongest_cases = ["Stockport 014F", "Stockport 019C", "Stockport 014E"]
rq2_joint_cases = ["Stockport 007C", "Stockport 007D"]
rq2_joint_pressure_text = "62.4% · 60.0%"

if overlay is not None and not overlay.empty:
    overlay["group"] = overlay["group"].map(canonical_group)

    lower_mask = overlay["group"].isin(
        ["Higher health + Lower demand", "Lower health + Lower demand"]
    )
    equity_mask = overlay["group"] == "Higher health + Lower demand"

    if lower_mask.any():
        rq2_lower_demand_n = int(lower_mask.sum())
    if equity_mask.any():
        rq2_equity_n = int(equity_mask.sum())

    # If the continuous RQ2 gap is available, use it to identify the strongest
    # divergence cases dynamically; otherwise retain the dissertation result.
    if equity_mask.any() and overlay["health_demand_gap"].notna().any():
        strongest = (
            overlay.loc[equity_mask]
            .sort_values("health_demand_gap", ascending=False)
            .head(3)["lsoa_name"]
            .dropna()
            .astype(str)
            .tolist()
        )
        if strongest:
            rq2_strongest_cases = strongest

    joint = overlay.loc[equity_mask & overlay["high_pressure"]].copy()
    if not joint.empty:
        names = joint["lsoa_name"].dropna().astype(str).tolist()
        if names:
            rq2_joint_cases = names
        vals = joint["max_p90_util"].dropna().tolist()
        if vals:
            rq2_joint_pressure_text = " · ".join(f"{v:.1f}%" for v in vals[:3])

rq2_equity_low_pressure_n = max(rq2_equity_n - len(rq2_joint_cases), 0)


# =============================================================================
# HEADER
# =============================================================================

st.markdown('<span class="rq-badge">RQ3 · Segment-Level Operational Implications</span>', unsafe_allow_html=True)
st.title("Operational Analysis")
st.write(
    "Examine reconstructed onboard passenger pressure, capacity utilisation and "
    "passenger-carrying burden across Route 192."
)

st.markdown(
    """
    <div class="research-note">
      <strong>Methodological note.</strong> Operational measures are derived from reconstructed
      passenger loads and should be interpreted as <strong>relative within-corridor patterns</strong>,
      not verified year-round crowding. Post-stop onboard load is assigned to the following
      directed inter-stop segment because that is the load physically carried through the route section.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="method-card" style="margin-top:.8rem;">
      <span class="flow-step">RQ2: Health vulnerability</span>
      <span class="flow-arrow">+</span>
      <span class="flow-step">RQ2: Reconstructed demand</span>
      <span class="flow-arrow">→</span>
      <span class="flow-step-yellow">Health–demand context</span>
      <span class="flow-arrow">→</span>
      <span class="flow-step">RQ3: Onboard load by directed segment</span>
      <span class="flow-arrow">+</span>
      <span class="flow-step-yellow">Operational differentiation</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("## From RQ2 to RQ3: does equity-sensitive low demand also imply operational pressure?")
st.write(
    "RQ2 identified where realised boarding demand looked low relative to neighbourhood health "
    "vulnerability. RQ3 does not restart the analysis from zero: it asks whether those same "
    "health–demand contexts are traversed by lightly loaded buses or by buses already carrying "
    "substantial upstream passenger load."
)

r1, r2, r3 = st.columns(3)
with r1:
    metric_card(
        "RQ2 lower-demand LSOAs",
        f"{rq2_lower_demand_n} / 39",
        "Lower realised boarding occurrence within the Route 192 corridor.",
    )
with r2:
    metric_card(
        "RQ2 equity-sensitive subset",
        f"{rq2_equity_n} / {rq2_lower_demand_n}",
        "Higher health vulnerability combined with lower realised demand.",
    )
with r3:
    metric_card(
        "Strongest RQ2 divergence",
        " · ".join(rq2_strongest_cases),
        "Largest positive health–demand gaps; these are equity-screening signals, not unmet-demand proof.",
    )

st.markdown(
    f"""
    <div class="research-note" style="margin-top:.8rem;">
      <strong>RQ3 bridge question:</strong> among the <strong>{rq2_equity_n}</strong>
      higher-health / lower-demand LSOAs identified in RQ2, only
      <strong>{len(rq2_joint_cases)}</strong> also fall into the higher operational-pressure group:
      <strong>{", ".join(rq2_joint_cases)}</strong> ({rq2_joint_pressure_text} maximum P90 utilisation).
      The remaining <strong>{rq2_equity_low_pressure_n}</strong> therefore represent a different
      operational context. This is the key reason RQ3 is needed: the same RQ2 equity signal can sit
      alongside very different onboard loading conditions.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("### Operational evidence used to differentiate the RQ2 contexts")
c1, c2, c3, c4 = st.columns(4)
with c1:
    metric_card(
        "Highest direction-period P90",
        fmt_pct(direction_period_peak),
        direction_period_peak_label,
    )
with c2:
    metric_card(
        "Maximum segment P90 utilisation",
        fmt_pct(max_segment_util),
        max_segment_label,
    )
with c3:
    metric_card(
        "Upper-quartile screening threshold",
        fmt_pct(p75_display),
        "Study-specific within-corridor threshold; not an external crowding standard.",
    )
with c4:
    metric_card(
        "Higher-pressure LSOAs",
        f"{high_pressure_n} / {lsoa_n}",
        "LSOAs above the within-corridor operational-pressure threshold.",
    )

st.divider()


# =============================================================================
# SECTION 1 — SEGMENT-LEVEL PASSENGER PRESSURE
# =============================================================================

st.markdown('<div class="section-kicker">Segment-level operational state</div>', unsafe_allow_html=True)
st.header("Segment-Level Passenger Pressure")
st.caption(
    "Each coloured line is the road section between two consecutive Route 192 stops. "
    "Darker / warmer segments carry greater passenger pressure for the selected view."
)

if segments is None or segments.empty:
    st.warning(
        "Segment-level output was not found. Add `RQ3_segment_direction_period_hotspots.csv` "
        "to `data/rq2_rq3_outputs/` to enable this section."
    )
else:
    # -------------------------------------------------------------------------
    # Primary filters: direction -> period -> trip -> metric
    # -------------------------------------------------------------------------
    f1, f2, f3, f4 = st.columns([1.05, 1.05, 1.25, 1.25])

    direction_options = sorted(segments["direction"].dropna().astype(str).unique().tolist())
    with f1:
        selected_direction = st.selectbox(
            "Direction", direction_options, index=0, key="rq3_direction"
        )

    period_options = (
        segments.loc[segments["direction"] == selected_direction, "period"]
        .dropna().astype(str).unique().tolist()
    )
    period_options = sorted(period_options, key=period_sort_key)
    with f2:
        selected_period = st.selectbox(
            "Operating period", period_options, index=0, key="rq3_period"
        )

    # Trip filter is constructed from the master stop-visit dataset when available.
    available_trip_ids = []
    if trip_segments is not None and not trip_segments.empty:
        available_trip_ids = (
            trip_segments.loc[
                (trip_segments["direction"] == selected_direction)
                & (trip_segments["period"].astype(str) == selected_period),
                "trip_id",
            ]
            .dropna().astype(str).unique().tolist()
        )
        available_trip_ids = sorted(available_trip_ids)

    with f3:
        selected_trip = st.selectbox(
            "Trip",
            ["Period summary (P90)"] + available_trip_ids,
            index=0,
            key="rq3_trip",
            help=(
                "Period summary reproduces the dissertation P90 analysis. Select a trip to inspect "
                "its reconstructed stop-to-stop load profile."
            ),
        )

    is_trip_view = selected_trip != "Period summary (P90)"
    if is_trip_view:
        metric_options = {
            "Capacity Utilisation (%)": "util",
            "Onboard Load": "load",
            "Passenger-Mass-Distance (tonne-km)": "pmd_68",
        }
    else:
        metric_options = {
            "P90 Capacity Utilisation (%)": "p90_util",
            "P90 Onboard Load": "p90_load",
            "Passenger-Mass-Distance (tonne-km)": "pmd_68",
        }

    with f4:
        selected_metric_label = st.selectbox(
            "Passenger-pressure metric",
            list(metric_options), index=0, key="rq3_metric"
        )
    selected_metric = metric_options[selected_metric_label]

    # RQ3 is presented as a purely operational view here.
    # The optional RQ2-context filter has been removed from the UI.
    selected_context = "All RQ2 contexts"

    # -------------------------------------------------------------------------
    # Build active dataset
    # -------------------------------------------------------------------------
    if is_trip_view:
        active = trip_segments[
            (trip_segments["direction"] == selected_direction)
            & (trip_segments["period"].astype(str) == selected_period)
            & (trip_segments["trip_id"].astype(str) == str(selected_trip))
        ].copy()
        active["p90_load"] = active["load"]
        active["p90_util"] = active["util"]
        view_note = (
            f"Trip {selected_trip}: reconstructed onboard load is displayed on each stop-to-stop section. "
            "This is an exploratory trip view; the dissertation RQ3 comparison is based on P90 period summaries."
        )
    else:
        period_metrics = segments[
            (segments["direction"] == selected_direction)
            & (segments["period"].astype(str) == selected_period)
        ].copy()

        backbone = build_canonical_route_backbone(
            trip_segments,
            coords,
            selected_direction,
            selected_period,
        )

        if backbone is None or backbone.empty:
            # Last-resort fallback keeps the old behaviour if the master route
            # cannot supply a canonical trip.
            active = apply_master_segment_geometry(
                period_metrics,
                MASTER_ADJACENCY,
                coords,
            )
        else:
            active = attach_period_metrics_to_backbone(
                backbone,
                period_metrics,
            )

        view_note = (
            "Period summary: the physical Route 192 backbone is reconstructed from the most complete "
            "full trip in the selected direction and is independent of the operating-period filter. "
            "The selected period only determines the P90 passenger-pressure values overlaid on that "
            "backbone. Sections without a matched RQ3 value remain visible in neutral grey."
        )

    # RQ2 filtering now acts as a visual highlight, not a geometry deletion.
    active = apply_rq2_context_as_highlight(
        active,
        overlay,
        selected_context,
    )

    if active.empty:
        st.info("No stop-to-stop observations are available for the selected filters.")
    else:
        active["_segment_key"] = (
            active["from_stop"].map(clean_id) + "__" + active["to_stop"].map(clean_id)
        )
        active["_segment_display"] = active.apply(
            lambda r: friendly_segment_display(r, STOP_NAME_LOOKUP), axis=1
        )
        active["_segment_code"] = active.apply(segment_code_display, axis=1)

        st.markdown(
            f"""
            <div class="research-note" style="margin-bottom:.8rem;">
              <strong>How to read this map:</strong> passenger pressure belongs to the section
              <strong>after leaving the first stop and before arriving at the next stop</strong>.
              For sections with a matched metric, the colour scale is continuous within the selected view:
              pale pressure colours = lower pressure, yellow/orange = higher pressure, and dark red = the highest pressure.
              Neutral grey means the physical Route 192 section is retained for continuity but no matching metric row was found.
              The physical route is always built from a complete ordered Route 192 trip in the selected
              direction. <strong>The operating-period filter changes the pressure overlay, not the route
              geometry.</strong> Coloured sections have a matched RQ3 pressure value; neutral-grey sections
              are valid Route 192 links without a matched value for the current analytical view. {view_note}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # For period summaries, `active` is now a canonical route backbone.
        # Keep every section with usable endpoint geometry, including neutral-grey
        # sections that have no matching pressure metric.
        if "_geometry_valid" in active.columns:
            map_active = active.loc[active["_geometry_valid"]].copy()
            excluded_geometry_n = int((~active["_geometry_valid"]).sum())
        else:
            map_active = active.dropna(
                subset=["from_lat", "from_lon", "to_lat", "to_lon"]
            ).copy()
            excluded_geometry_n = len(active) - len(map_active)

        geometry_available = not map_active.empty

        if not geometry_available:
            st.warning(
                "No safe geographic section could be validated for this filtered view. "
                "The operational results are still available below, but the app will not draw "
                "unverified stop-to-stop lines."
            )
            # Keep the inspector usable even when the map is unavailable.
            inspector_active = active.copy()
        else:
            inspector_active = map_active.copy()

        # Inspector selector appears before the map so the chosen section can be highlighted.
        inspector_active = inspector_active.copy()
        inspector_active["_has_selected_metric"] = inspector_active[selected_metric].notna()
        inspector_active = inspector_active.sort_values(
            ["_has_selected_metric", "stop_sequence" if "stop_sequence" in inspector_active.columns else "_segment_key"],
            ascending=[False, True],
        )
        section_keys = inspector_active["_segment_key"].tolist()
        display_lookup = dict(
            zip(inspector_active["_segment_key"], inspector_active["_segment_display"])
        )
        inspect_col, summary_col = st.columns([2.4, 1])
        with inspect_col:
            selected_section = st.selectbox(
                "Inspect route section",
                section_keys,
                index=0,
                key="rq3_segment_inspector",
                format_func=lambda k: display_lookup.get(k, str(k)),
                help="Choose the section between two consecutive Route 192 stops.",
            )
        with summary_col:
            usable_vals = inspector_active[selected_metric].dropna()
            if not usable_vals.empty:
                suffix = "%" if "Utilisation" in selected_metric_label else ""
                st.caption(
                    f"Matched metric range · {usable_vals.min():.1f}{suffix} to "
                    f"{usable_vals.max():.1f}{suffix}"
                )
            elif not is_trip_view:
                st.caption("No matched period-level metric values in this view")

        row = inspector_active.loc[inspector_active["_segment_key"] == selected_section].iloc[0]

        # ---------------------------------------------------------------------
        # Stop-to-stop continuous pressure map + selected-section inspector
        # ---------------------------------------------------------------------
        left, right = st.columns([2.25, 1])
        with left:
            if geometry_available:
                fig, mapped_n, total_n, measured_n = build_pressure_map(
                    map_active,
                    selected_metric,
                    selected_metric_label,
                    selected_key=selected_section,
                )
                if fig is not None:
                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        config={"displaylogo": False, "scrollZoom": True},
                    )
                    if is_trip_view:
                        st.caption(
                            f"Route playback: {mapped_n} of {total_n} trip sections mapped."
                            + (
                                f" {excluded_geometry_n} sections have missing/implausible coordinates."
                                if excluded_geometry_n else ""
                            )
                        )
                    else:
                        grey_n = max(mapped_n - measured_n, 0)
                        st.caption(
                            f"Continuous Route 192 backbone: {mapped_n} stop-to-stop sections shown · "
                            f"{measured_n} sections have matched {selected_metric_label} values"
                            + (
                                f" · {grey_n} neutral-grey sections have no matched period-level metric."
                                if grey_n else ""
                            )
                            + (
                                f" {excluded_geometry_n} sections could not be mapped because endpoint "
                                "coordinates were unavailable or implausible."
                                if excluded_geometry_n else ""
                            )
                        )
            else:
                # Geographic display is withheld, but the user can still inspect the
                # pressure ranking for the selected direction / period.
                fallback = active.dropna(subset=[selected_metric]).copy()
                fallback = fallback.sort_values(selected_metric, ascending=False).head(20)
                fig = go.Figure(
                    go.Bar(
                        x=fallback[selected_metric],
                        y=fallback["_segment_display"],
                        orientation="h",
                        marker_color="#AEB8C7",
                        hovertemplate=(
                            "<b>%{y}</b><br>"
                            + selected_metric_label
                            + ": %{x:.2f}<extra></extra>"
                        ),
                    )
                )
                fig.update_yaxes(autorange="reversed")
                fig.update_layout(
                    height=560,
                    margin=dict(l=10, r=20, t=20, b=40),
                    paper_bgcolor=WHITE,
                    plot_bgcolor=WHITE,
                    font=dict(family="Inter", color="#374151", size=12),
                    xaxis_title=selected_metric_label,
                    yaxis_title="",
                )
                st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
                st.caption(
                    "Geographic lines are withheld for this filtered view because the stop-pair "
                    "geometry could not be validated safely. The ranking still uses the RQ3 "
                    "operational results."
                )


        if not is_trip_view:
            with st.expander("Route-backbone diagnostics", expanded=False):
                if "_canonical_trip" in active.columns and active["_canonical_trip"].notna().any():
                    st.write(
                        "Canonical full-trip backbone:",
                        str(active["_canonical_trip"].dropna().iloc[0]),
                    )
                st.write("Backbone sections:", int(len(active)))
                if "_geometry_valid" in active.columns:
                    st.write(
                        "Sections with usable endpoint geometry:",
                        int(active["_geometry_valid"].fillna(False).sum()),
                    )
                    st.write(
                        "Sections missing/rejected geometry:",
                        int((~active["_geometry_valid"].fillna(False)).sum()),
                    )
                if "_metric_match" in active.columns:
                    st.write(
                        "Sections matched to the selected RQ3 period metric:",
                        int(active["_metric_match"].fillna(False).sum()),
                    )

        if False and (not is_trip_view) and "geometry_reason" in active.columns:
            with st.expander("Geometry validation details", expanded=False):
                diag = (
                    active["geometry_reason"]
                    .fillna("Unknown")
                    .value_counts()
                    .rename_axis("Validation result")
                    .reset_index(name="Sections")
                )
                st.dataframe(diag, use_container_width=True, hide_index=True)

        with right:
            from_id = clean_id(row.get("from_stop"))
            to_id = clean_id(row.get("to_stop"))
            from_name = str(row.get("from_stop_name", "")).strip()
            to_name = str(row.get("to_stop_name", "")).strip()
            if not from_name or from_name.lower() in {"nan", "none"}:
                from_name = stop_display(from_id, STOP_NAME_LOOKUP)
            if not to_name or to_name.lower() in {"nan", "none"}:
                to_name = stop_display(to_id, STOP_NAME_LOOKUP)

            load = row.get("load", row.get("p90_load", np.nan)) if is_trip_view else row.get("p90_load", np.nan)
            util = row.get("util", row.get("p90_util", np.nan)) if is_trip_view else row.get("p90_util", np.nan)
            dist = row.get("distance_m", np.nan)
            if pd.isna(dist) or (isinstance(dist, (int, float, np.number)) and dist <= 0):
                dist = row.get("_expected_distance_m", np.nan)
            if pd.isna(dist):
                dist = row.get("geo_distance_m", np.nan)
            pmd = row.get("pmd_68", np.nan)

            st.markdown("#### Selected route section")
            st.markdown(
                f"""
                <div class="inspector-card">
                  <div class="mini-label">From</div>
                  <div class="mini-value" style="font-size:1.08rem;">{from_name}</div>
                  <div class="small-muted">Stop ID: {from_id or "—"}</div>
                </div>
                <div style="text-align:center; color:#6B7280; font-size:1.25rem; margin:-.1rem 0 .25rem 0;">↓</div>
                <div class="inspector-card">
                  <div class="mini-label">To</div>
                  <div class="mini-value" style="font-size:1.08rem;">{to_name}</div>
                  <div class="small-muted">Stop ID: {to_id or "—"}</div>
                </div>
                <div class="small-muted" style="margin:.2rem 0 .75rem 0;">
                  {selected_direction} · {selected_period}
                  {f' · Trip {selected_trip}' if is_trip_view else ' · P90 period summary'}
                </div>
                """,
                unsafe_allow_html=True,
            )

            a, b = st.columns(2)
            with a:
                st.markdown(
                    f"""<div class="inspector-card"><div class="mini-label">{'Onboard load' if is_trip_view else 'P90 onboard load'}</div>
                    <div class="mini-value">{fmt_num(load, 1)}</div><div class="small-muted">reconstructed passengers</div></div>""",
                    unsafe_allow_html=True,
                )
            with b:
                st.markdown(
                    f"""<div class="inspector-card"><div class="mini-label">{'Utilisation' if is_trip_view else 'P90 utilisation'}</div>
                    <div class="mini-value">{fmt_pct(util)}</div><div class="small-muted">of 100-passenger capacity</div></div>""",
                    unsafe_allow_html=True,
                )

            a, b = st.columns(2)
            with a:
                st.markdown(
                    f"""<div class="inspector-card"><div class="mini-label">Segment distance</div>
                    <div class="mini-value">{fmt_num(dist, 0)} m</div><div class="small-muted">stop to next stop</div></div>""",
                    unsafe_allow_html=True,
                )
            with b:
                st.markdown(
                    f"""<div class="inspector-card"><div class="mini-label">{'PMD' if is_trip_view else 'P90 PMD'}</div>
                    <div class="mini-value">{fmt_num(pmd, 2)}</div><div class="small-muted">tonne-km at 68 kg</div></div>""",
                    unsafe_allow_html=True,
                )

            if pd.notna(util):
                st.progress(max(0.0, min(float(util) / 100.0, 1.0)))
                st.caption(
                    f"{fmt_pct(util)} total-capacity utilisation · {SEATED_REFERENCE:.0f} passengers = seated-capacity reference."
                )


            if (not is_trip_view) and not bool(row.get("_metric_match", pd.notna(row.get(selected_metric, np.nan)))):
                st.caption(
                    "This section is part of the Route 192 backbone but has no matching "
                    "period-level RQ3 metric row for the current filters; it is shown in neutral grey."
                )

            if overlay is not None and not overlay.empty and str(row.get("lsoa", "")).strip():
                rq2_match = overlay[overlay["lsoa_code"].astype(str) == str(row.get("lsoa", "")).strip()]
                if not rq2_match.empty:
                    rq2_row = rq2_match.iloc[0]
                    st.markdown(
                        f"""
                        <div class="inspector-card">
                          <div class="mini-label">RQ2 health–demand context</div>
                          <div style="font-family:'Public Sans';font-weight:800;margin-top:.15rem;">{rq2_row['group']}</div>
                          <div class="small-muted" style="margin-top:.3rem;">{rq2_row['lsoa_name']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        # ---------------------------------------------------------------------
        # Ranked table: geographic names first, IDs secondary.
        # ---------------------------------------------------------------------
        st.markdown("#### Highest-pressure route sections in the selected view")
        table = active.dropna(subset=[selected_metric]).sort_values(
            selected_metric, ascending=False
        ).head(12).copy()
        table["Route section"] = table["_segment_display"]
        table["Stop IDs"] = table["_segment_code"]
        if is_trip_view:
            table["Onboard load"] = table["load"].round(1)
            table["Utilisation (%)"] = table["util"].round(1)
        else:
            table["P90 load"] = table["p90_load"].round(1)
            table["P90 utilisation (%)"] = table["p90_util"].round(1)
        table["PMD (t-km)"] = table["pmd_68"].round(3)

        cols = ["Route section", "Stop IDs"]
        cols += ["Onboard load", "Utilisation (%)"] if is_trip_view else ["P90 load", "P90 utilisation (%)"]
        cols += ["PMD (t-km)"]
        st.dataframe(table[cols], use_container_width=True, hide_index=True)

        st.download_button(
            "Download selected operational slice",
            active.drop(columns=["_segment_key", "_segment_display", "_segment_code"], errors="ignore")
            .to_csv(index=False).encode("utf-8"),
            file_name="route192_rq3_operational_slice.csv",
            mime="text/csv",
            use_container_width=False,
        )

st.divider()


# =============================================================================
# SECTION 2 — CORRIDOR AVERAGE VS LOCAL PRESSURE
# =============================================================================

st.header("Corridor Averages vs Local Segment Pressure")
st.caption(
    "Direction-period summaries can conceal short sections of concentrated loading."
)

delta = max_segment_util - direction_period_peak
b1, b2 = st.columns(2)
with b1:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=["Direction-period P90", "Maximum segment P90"],
            y=[direction_period_peak, max_segment_util],
            marker_color=["#AEB8C6", RED],
            text=[f"{direction_period_peak:.1f}%", f"{max_segment_util:.1f}%"],
            textposition="outside",
            hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
        )
    )
    fig.add_hline(
        y=SEATED_REFERENCE,
        line_dash="dash",
        line_color=YELLOW,
        annotation_text="73-passenger seated reference",
        annotation_position="top left",
    )
    fig.update_yaxes(
        title="P90 capacity utilisation (%)",
        range=[0, max(85, max_segment_util * 1.18)],
    )
    plot_layout(fig, height=360, legend=False)
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

with b2:
    st.markdown(
        f"""
        <div class="panel" style="min-height:330px;">
          <div class="section-kicker">Concentration gap</div>
          <div class="metric-value" style="color:{RED};">+{delta:.1f} pp</div>
          <p style="color:{GREY}; line-height:1.55;">
            The highest direction-period P90 utilisation is approximately
            <strong>{direction_period_peak:.1f}%</strong>, whereas the most heavily loaded
            individual segment reaches <strong>{max_segment_util:.1f}%</strong>.
          </p>
          <div class="caution-box">
            This difference is the core operational reason for retaining segment-level analysis:
            corridor or direction-period averages can mask short, high-load sections.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()


# =============================================================================
# SECTION 3 — DIRECTION × PERIOD
# =============================================================================

st.header("When Does Operational Pressure Occur?")
st.caption("Direction × operating-period comparison of upper-end reconstructed passenger loading.")

if dp is None or dp.empty:
    st.warning(
        "Direction-period summary not found. Add `RQ3_direction_period_summary.csv` "
        "to enable this section."
    )
else:
    heat = dp.copy()
    heat["period_order"] = heat["period"].map(period_sort_key)
    heat = heat.sort_values(["period_order", "period"])

    pivot = heat.pivot_table(
        index="direction",
        columns="period",
        values="p90_util",
        aggfunc="max",
    )
    ordered_cols = sorted(pivot.columns.tolist(), key=period_sort_key)
    pivot = pivot[ordered_cols]

    text_matrix = np.empty(pivot.shape, dtype=object)
    for i, direction in enumerate(pivot.index):
        for j, period in enumerate(pivot.columns):
            val = pivot.loc[direction, period]
            load_match = heat[
                (heat["direction"] == direction) & (heat["period"] == period)
            ]["p90_load"].dropna()
            load_text = ""
            if not load_match.empty:
                load_text = f"<br>{load_match.iloc[0]:.1f} pax"
            text_matrix[i, j] = "—" if pd.isna(val) else f"{val:.1f}%{load_text}"

    fig = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale=[
                [0.0, "#F0F3FF"],
                [0.55, "#DCE2F3"],
                [0.78, "#FFD800"],
                [1.0, "#BA1A1A"],
            ],
            text=text_matrix,
            texttemplate="%{text}",
            textfont=dict(family="Inter", size=12),
            colorbar=dict(title="P90 utilisation (%)"),
            hovertemplate="Direction: %{y}<br>Period: %{x}<br>P90: %{z:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        height=330,
        margin=dict(l=15, r=15, t=20, b=20),
        paper_bgcolor=WHITE,
        font=dict(family="Inter", color="#374151"),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

    peak_row = heat.loc[heat["p90_util"].idxmax()]
    st.markdown(
        f"""
        <div class="caution-box">
          <strong>Highest direction-period P90:</strong>
          {peak_row["direction"]} · {peak_row["period"]} at
          <strong>{peak_row["p90_util"]:.1f}%</strong>.
          This remains substantially below the maximum individual-segment value of
          <strong>{max_segment_util:.1f}%</strong>.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()


# =============================================================================
# SECTION 4 — PMD
# =============================================================================

st.markdown('<div class="section-kicker">Passenger-carrying burden</div>', unsafe_allow_html=True)
st.header("Passenger-Carrying Burden Beyond Occupancy")
st.write(
    "Passenger-mass-distance (PMD) combines reconstructed onboard load with the distance "
    "over which that passenger mass is carried."
)

st.markdown(
    """
    <div class="research-note">
      <strong>PMD boundary.</strong> PMD is an energy-relevant passenger-carrying burden,
      not a direct estimate of fuel consumption, electricity use or emissions. Actual energy
      requirements also depend on drivetrain, speed, acceleration, gradient and traffic conditions.
    </div>
    """,
    unsafe_allow_html=True,
)

if rank_df is None or rank_df.empty or rank_df[["p90_load", "pmd_68"]].dropna().empty:
    st.warning(
        "Load-vs-PMD data could not be constructed. The app first looks for "
        "`RQ3_load_vs_PMD_rank_check.csv` and then automatically falls back to "
        "`RQ3_segment_direction_period_hotspots.csv`. Please check that at least one "
        "of those files contains P90 onboard load and PMD (or load + segment distance)."
    )
else:
    scatter_df = rank_df.dropna(subset=["p90_load", "pmd_68"]).copy()
    rho = scatter_df["p90_load"].corr(scatter_df["pmd_68"], method="spearman")

    fig = go.Figure(
        go.Scatter(
            x=scatter_df["p90_load"],
            y=scatter_df["pmd_68"],
            mode="markers",
            marker=dict(
                size=8,
                color="#5F8DB8",
                opacity=0.72,
                line=dict(width=.5, color=WHITE),
            ),
            text=scatter_df.get("segment_display", scatter_df["segment"]),
            customdata=np.column_stack(
                [
                    scatter_df["distance_m"].fillna(np.nan),
                ]
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "P90 load: %{x:.1f} passengers<br>"
                "PMD: %{y:.2f} tonne-km<br>"
                "Distance: %{customdata[0]:.0f} m"
                "<extra></extra>"
            ),
            name="Segments",
        )
    )

    # Max PMD highlight
    max_row = scatter_df.loc[scatter_df["pmd_68"].idxmax()]
    fig.add_trace(
        go.Scatter(
            x=[max_row["p90_load"]],
            y=[max_row["pmd_68"]],
            mode="markers+text",
            marker=dict(size=15, color=BLUE, line=dict(width=3, color=YELLOW)),
            text=[f"  {max_row.get('segment_display', max_row['segment'])}"],
            textposition="middle right",
            textfont=dict(family="Public Sans", size=11, color=BLUE),
            hovertemplate=(
                "<b>Highest PMD</b><br>"
                f"{max_row.get('segment_display', max_row['segment'])}<br>"
                "P90 load: %{x:.1f}<br>"
                "PMD: %{y:.2f} tonne-km<extra></extra>"
            ),
            name="Highest PMD",
        )
    )

    fig.update_xaxes(title="P90 reconstructed onboard passengers")
    fig.update_yaxes(title="P90 passenger-mass-distance (tonne-km)")
    plot_layout(fig, height=480)
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

    p1, p2, p3, p4 = st.columns(4)
    with p1:
        metric_card("Spearman association", f"ρ = {rho:.3f}", "P90 load vs PMD across directed segments.")
    with p2:
        metric_card("Highest PMD", f"{max_pmd:.2f}", "tonne-km at the 68 kg baseline assumption.")
    with p3:
        metric_card("Peak-PMD segment", max_pmd_segment, "Segment with the largest passenger-carrying burden.")
    with p4:
        dist_text = "—" if pd.isna(max_pmd_distance) else f"{max_pmd_distance:.0f} m"
        metric_card(
            "Peak-PMD context",
            dist_text,
            "Segment distance associated with the highest PMD result.",
        )

    st.markdown(
        """
        <div class="caution-box">
          Strong association does not make the two metrics interchangeable. Capacity utilisation
          describes passenger concentration, whereas PMD also incorporates segment distance and can
          therefore change the ranking of operational hotspots.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()


# =============================================================================
# SECTION 5 — DIFFERENT HOTSPOTS
# =============================================================================

st.header("Different Metrics Identify Different Hotspots")
st.caption("Compare the top Route 192 segments ranked by P90 onboard load and PMD.")

if rank_df is not None and not rank_df.empty:
    ranking = rank_df.dropna(subset=["p90_load", "pmd_68"]).copy()

    top_n = min(15, len(ranking))
    top_load = ranking.nlargest(top_n, "p90_load").copy()
    top_pmd = ranking.nlargest(top_n, "pmd_68").copy()

    load_set = set(top_load["segment"])
    pmd_set = set(top_pmd["segment"])
    shared = load_set & pmd_set

    st.markdown(
        f"""
        <div class="method-card">
          <strong>{len(shared)} / {top_n} shared hotspots</strong> appear in both top-{top_n}
          rankings. The remaining segments are metric-specific, demonstrating that passenger
          concentration and passenger-carrying burden are related but operationally distinct.
        </div>
        """,
        unsafe_allow_html=True,
    )

    lcol, rcol = st.columns(2)
    with lcol:
        st.markdown("#### Top segments by P90 onboard load")
        show = top_load[["segment", "segment_display", "segment_code", "p90_load", "pmd_68"]].copy()
        show["Shared hotspot"] = show["segment"].isin(shared).map({True: "Yes", False: ""})
        show = show.drop(columns=["segment"]).rename(
            columns={
                "segment_display": "Route section",
                "segment_code": "Stop IDs",
                "p90_load": "P90 load",
                "pmd_68": "PMD (t-km)",
            }
        )
        st.dataframe(
            show,
            use_container_width=True,
            hide_index=True,
            column_config={
                "P90 load": st.column_config.NumberColumn(format="%.1f"),
                "PMD (t-km)": st.column_config.NumberColumn(format="%.2f"),
            },
        )

    with rcol:
        st.markdown("#### Top segments by PMD")
        show = top_pmd[["segment", "segment_display", "segment_code", "pmd_68", "p90_load"]].copy()
        show["Shared hotspot"] = show["segment"].isin(shared).map({True: "Yes", False: ""})
        show = show.drop(columns=["segment"]).rename(
            columns={
                "segment_display": "Route section",
                "segment_code": "Stop IDs",
                "pmd_68": "PMD (t-km)",
                "p90_load": "P90 load",
            }
        )
        st.dataframe(
            show,
            use_container_width=True,
            hide_index=True,
            column_config={
                "P90 load": st.column_config.NumberColumn(format="%.1f"),
                "PMD (t-km)": st.column_config.NumberColumn(format="%.2f"),
            },
        )

st.divider()


# =============================================================================
# SECTION 6 — PASSENGER MASS SENSITIVITY
# =============================================================================

st.markdown('<div class="section-kicker">Interactive sensitivity analysis</div>', unsafe_allow_html=True)
st.header("Passenger Mass Sensitivity Analysis")
st.caption(
    "Change the assumed average passenger mass. PMD rescales proportionally because mass "
    "enters the equation as a common multiplicative factor."
)

mass = st.slider(
    "Assumed average passenger mass (kg)",
    min_value=50,
    max_value=100,
    value=68,
    step=1,
    key="rq3_mass_slider",
)

scaled_peak_pmd = max_pmd * (mass / BASELINE_MASS)
delta_pct = (mass / BASELINE_MASS - 1.0) * 100.0

m1, m2, m3, m4 = st.columns(4)
with m1:
    metric_card("Selected passenger mass", f"{mass} kg", "Interactive assumption.")
with m2:
    metric_card("Peak PMD under selection", f"{scaled_peak_pmd:.2f}", "tonne-km; proportional rescaling.")
with m3:
    metric_card(
        "Change vs 68 kg",
        f"{delta_pct:+.1f}%",
        "Absolute PMD magnitude changes with passenger mass.",
    )
with m4:
    metric_card(
        "Segment ranking",
        "Unchanged",
        "Load and distance remain fixed, so a common mass scalar preserves ranking.",
    )

scenario_df = pd.DataFrame(
    {
        "Average passenger mass (kg)": [68, 75, 85, mass],
        "Peak PMD (tonne-km)": [
            max_pmd,
            max_pmd * 75 / 68,
            max_pmd * 85 / 68,
            scaled_peak_pmd,
        ],
        "Change vs 68 kg (%)": [
            0.0,
            (75 / 68 - 1) * 100,
            (85 / 68 - 1) * 100,
            delta_pct,
        ],
        "Scenario": ["Main specification", "Sensitivity", "Sensitivity", "Interactive"],
    }
)
scenario_df = scenario_df.drop_duplicates(subset=["Average passenger mass (kg)", "Scenario"])
st.dataframe(
    scenario_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Peak PMD (tonne-km)": st.column_config.NumberColumn(format="%.2f"),
        "Change vs 68 kg (%)": st.column_config.NumberColumn(format="%+.1f%%"),
    },
)

st.markdown(
    """
    <div class="research-note">
      <strong>Invariance:</strong> the sensitivity analysis changes PMD magnitudes but not
      relative segment rankings, because passenger mass is a common multiplicative factor
      when reconstructed load and segment distance are held fixed.
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()


# =============================================================================
# FINAL RQ3 ANSWER
# =============================================================================

st.markdown('<div class="section-kicker">Research question answer</div>', unsafe_allow_html=True)
st.header("RQ3 Answer")

st.markdown(
    f"""
    <div class="answer-box">
      <h3>RQ3 differentiates, rather than replaces, the RQ2 equity signal.</h3>
      <div>
        RQ2 identified <strong>{rq2_equity_n}</strong> higher-health / lower-demand LSOAs, showing
        that similarly low realised boarding can occur under different vulnerability conditions.
        RQ3 then asks whether those same contexts also coincide with elevated through-load.
        Only <strong>{len(rq2_joint_cases)}</strong> of the {rq2_equity_n} equity-sensitive LSOAs
        ({", ".join(rq2_joint_cases)}) fall into the higher-pressure group. This matters because
        local boarding and onboard load are not equivalent: passengers may have boarded upstream.
        Across the corridor, the highest direction-period P90 utilisation is approximately
        <strong>{direction_period_peak:.1f}%</strong>, while individual segments reach
        <strong>{max_segment_util:.1f}%</strong>. PMD adds a second operational dimension by
        incorporating distance, with a maximum of approximately
        <strong>{max_pmd:.2f} tonne-km</strong>. RQ3 therefore shows that the same RQ2
        health–demand context can sit alongside materially different operational conditions.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "Caution: reconstructed summer-period loads are used for relative screening within Route 192; "
    "they should not be interpreted as independently validated year-round passenger counts."
)
