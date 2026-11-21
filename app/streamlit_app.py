"""Read precomputed observations and spatially held-out predictions; never retrain in the UI."""

import json
from pathlib import Path

import folium
import geopandas as gpd
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
LAYERS = {
    "Observed satellite LST": "observed_lst_celsius",
    "Model estimate (spatially held out)": "oof_predicted_lst_celsius",
}

st.set_page_config(page_title="CoolBlocks", layout="wide")
st.title("CoolBlocks: Long Beach surface temperature")
st.caption(
    "City-clipped Census tracts are neighborhood proxies, not named neighborhoods. "
    "Land surface temperature (LST) is not air temperature or personal heat exposure."
)

required = [
    PROCESSED / "units.geojson",
    RESULTS / "predictions.parquet",
    RESULTS / "run_metadata.json",
]
if not all(path.exists() for path in required):
    st.warning("Data outputs are missing. From the project root, run `make data features train`.")
    st.stop()


@st.cache_data
def load_outputs(stamps):
    # Modification times invalidate the cache when the pipeline outputs change.
    return (
        gpd.read_file(required[0]),
        pd.read_parquet(required[1]),
        json.loads(required[2].read_text()),
    )


units, predictions, metadata = load_outputs(tuple(p.stat().st_mtime_ns for p in required))
dates = sorted(predictions.date.unique())
date = st.sidebar.selectbox("Satellite observation date", dates, index=len(dates) - 1)
label = st.sidebar.radio("Map layer", list(LAYERS))
field = LAYERS[label]
rows = predictions.loc[predictions.date == date]
mapped = units.merge(rows, on="unit_id", validate="one_to_one")
if mapped.empty:
    st.info("No tracts passed the coverage checks for this date.")
    st.stop()

st.subheader(f"{label} · {date}")
st.write(
    f"{len(mapped)} tracts with sufficient clear land pixels. "
    "Unfilled areas have no qualifying observation."
)
if field == "oof_predicted_lst_celsius":
    st.info(
        f"Estimates use {metadata['selected_model']}. Each tract was predicted by a model "
        "trained without that tract or any of its dates. These estimates use reflectance "
        "from the selected date; they are not forecasts."
    )
else:
    st.caption(
        "Observed means are averages of the USGS satellite surface temperature retrieval "
        "over valid land pixels, not ground thermometer measurements."
    )

bounds = units.total_bounds
map_view = folium.Map(
    location=[(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2],
    zoom_start=12,
    min_zoom=12,
    max_bounds=True,
    min_lat=bounds[1],
    max_lat=bounds[3],
    min_lon=bounds[0],
    max_lon=bounds[2],
    max_bounds_viscosity=1.0,
    tiles="OpenStreetMap",
)
# Style the legend inside the embedded map, rather than the surrounding Streamlit page.
map_view.get_root().header.add_child(
    folium.Element("""
    <style>
        .legend.leaflet-control {
            background: white;
            color: #222;
            padding: 8px 12px;
            border-radius: 6px;
            box-shadow: 0 1px 6px rgba(0, 0, 0, 0.3);
        }
        .legend.leaflet-control svg { display: block; }
    </style>
    """)
)
# Use one scale for observations and held-out predictions across all dates.
low = min(predictions.observed_lst_celsius.min(), predictions.oof_predicted_lst_celsius.min())
high = max(predictions.observed_lst_celsius.max(), predictions.oof_predicted_lst_celsius.max())
choropleth = folium.Choropleth(
    geo_data=mapped.to_json(),
    data=mapped,
    columns=["unit_id", field],
    key_on="feature.properties.unit_id",
    bins=[low + (high - low) * i / 6 for i in range(7)],
    fill_color="YlOrRd",
    fill_opacity=0.75,
    line_weight=0.5,
    legend_name=f"{label} (°C)",
    highlight=True,
).add_to(map_view)
folium.GeoJsonTooltip(
    fields=["name", "observed_lst_celsius", "oof_predicted_lst_celsius", "valid_fraction"],
    aliases=["Tract", "Observed LST (°C)", "Held-out model estimate (°C)", "Valid pixel fraction"],
    localize=True,
).add_to(choropleth.geojson)
st_folium(map_view, height=540, use_container_width=True, returned_objects=[])

st.subheader("Compare two tracts")
options = mapped.sort_values("unit_id").unit_id.tolist()
names = mapped.set_index("unit_id")["name"].to_dict()
chosen = st.multiselect(
    "Tracts",
    options,
    default=options[:2],
    max_selections=2,
    format_func=lambda unit: f"{names[unit]} ({unit})",
)
columns = {
    "name": "Tract",
    "observed_lst_celsius": "Observed LST (°C)",
    "oof_predicted_lst_celsius": "Held-out estimate (°C)",
    "ndvi": "NDVI",
    "ndbi": "NDBI",
    "ndwi": "NDWI",
    "valid_fraction": "Valid pixel fraction",
    "valid_pixels": "Valid pixels",
}
comparison = mapped.set_index("unit_id").loc[chosen]
st.dataframe(comparison[list(columns)].rename(columns=columns).round(3), hide_index=True)
if len(chosen) == 2:
    difference = comparison.iloc[1].observed_lst_celsius - comparison.iloc[0].observed_lst_celsius
    st.write(f"Observed LST difference (second minus first): **{difference:+.2f}°C**.")
st.caption(
    "NDVI describes vegetation, NDBI built-up spectral response, and NDWI water response. "
    "These indices are associations; they do not measure canopy or impervious percentages."
)

with st.expander("Evaluation and limitations"):
    st.write(
        "Five-fold spatial validation holds out whole tracts across all three dates. "
        "The table pools held-out predictions; model selection used mean fold MAE."
    )
    st.dataframe(
        pd.DataFrame(metadata["pooled_oof_metrics"])
        .T.rename(columns={"mae": "MAE (°C)", "rmse": "RMSE (°C)", "r2": "R²"})
        .round(3)
    )
    st.write(
        "Three summer 2023 dates are a small sample. Neighboring tracts can occur in "
        "different folds; no spatial buffer or temporal holdout was evaluated. "
        "Model selection and evaluation reuse the same folds. Low coverage and shoreline "
        "areas can bias comparisons. Thermal measurements have roughly 100 m native "
        "resolution, resampled to a 30 m grid. No causal cooling, health, or equity claims "
        "are supported by this prototype."
    )
st.download_button(
    "Download this date's observations and estimates",
    rows.to_csv(index=False),
    file_name=f"coolblocks_{date}.csv",
    mime="text/csv",
)
