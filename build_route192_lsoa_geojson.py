
from pathlib import Path
import json
import requests
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MASTER_PATH = BASE_DIR / "data" / "raw" / "route192_master.csv"
OUT_DIR = BASE_DIR / "data" / "geo"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "route192_lsoas.geojson"

# ONS Open Geography Portal:
# Lower layer Super Output Areas (December 2021) Boundaries EW BGC
# Generalised 20 m, clipped boundary layer.
ONS_QUERY_URL = (
    "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/"
    "Lower_layer_Super_Output_Areas_December_2021_Boundaries_EW_BGC_V5/"
    "FeatureServer/0/query"
)

df = pd.read_csv(MASTER_PATH, low_memory=False)

def first_existing(candidates):
    return next((c for c in candidates if c in df.columns), None)

LSOA_COL = first_existing(
    ["LSOA21CD", "lsoa21cd", "lsoa_code", "LSOA_code", "LSOA"]
)

if LSOA_COL is None:
    raise ValueError(
        "Could not find an LSOA code column in route192_master.csv."
    )

codes = (
    df[LSOA_COL]
    .dropna()
    .astype(str)
    .str.strip()
    .drop_duplicates()
    .tolist()
)

print(f"Found {len(codes)} unique corridor LSOAs.")

# ArcGIS where-clause. 39 codes are small enough for one request.
quoted = ",".join(f"'{c}'" for c in codes)
where = f"LSOA21CD IN ({quoted})"

params = {
    "where": where,
    "outFields": "LSOA21CD,LSOA21NM",
    "returnGeometry": "true",
    "outSR": "4326",        # WGS84 for Plotly/Mapbox
    "f": "geojson",
}

headers = {
    "User-Agent": "Route192-MSc-Research/1.0"
}

response = requests.get(
    ONS_QUERY_URL,
    params=params,
    headers=headers,
    timeout=60,
)
response.raise_for_status()
geojson = response.json()

features = geojson.get("features", [])
print(f"Downloaded {len(features)} polygon features.")

if len(features) != len(codes):
    downloaded = {
        str(f.get("properties", {}).get("LSOA21CD", "")).strip()
        for f in features
    }
    missing = sorted(set(codes) - downloaded)
    print("Missing codes:", missing)

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(geojson, f)

print("\nSaved:")
print(OUT_PATH.resolve())
print("\nDone. Restart Streamlit and the LSOA maps will use polygons.")
