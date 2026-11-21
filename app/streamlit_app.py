"""Explore precomputed, spatially held-out LST estimates without retraining."""

import json
from pathlib import Path

import folium
import geopandas as gpd
import pandas as pd
import streamlit as st
from shapely.geometry import Point
from streamlit_folium import st_folium

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
VIEWS = {
    "Predicted temperature": "oof_predicted_lst_celsius",
    "Observed temperature": "observed_lst_celsius",
    "Prediction error": "prediction_error_celsius",
}

st.set_page_config(page_title="CoolBlocks · Long Beach", layout="wide")
st.title("CoolBlocks")
st.subheader("Neighborhood surface temperatures in Long Beach")
st.write(
    "Explore machine learning estimates of land surface temperature across Long Beach. "
    "Compare the model's predictions with satellite observations for the same date."
)
st.caption(
    "These are precomputed same-date estimates, not future temperature forecasts. "
    "Surface temperature describes the ground and rooftops—not the air temperature people feel."
)
st.info(
    "Choose a date and map view → click a colored tract to inspect it → "
    "compare two tracts below. Census tracts serve as neighborhood-scale areas."
)

required = [
    PROCESSED / "units.geojson",
    RESULTS / "predictions.parquet",
    RESULTS / "run_metadata.json",
]
if not all(path.exists() for path in required):
    st.warning(
        "The bundled demo files are missing. Restore them from Git, or regenerate them "
        "from the project root with `make data features train`."
    )
    st.stop()


@st.cache_data
def load_outputs(stamps):
    # File modification times invalidate the cache after a new pipeline run.
    return (
        gpd.read_file(required[0]),
        pd.read_parquet(required[1]),
        json.loads(required[2].read_text()),
    )


units, predictions, metadata = load_outputs(tuple(p.stat().st_mtime_ns for p in required))
# Always use spatially held-out estimates, never the final model's fitted predictions.
predictions["prediction_error_celsius"] = (
    predictions.oof_predicted_lst_celsius - predictions.observed_lst_celsius
)

dates = sorted(predictions.date.unique())
view_control, date_control = st.columns([3, 1])
with view_control:
    view = st.radio("Map view", list(VIEWS), horizontal=True, key="map_view")
with date_control:
    date = st.selectbox(
        "Satellite observation date",
        dates,
        index=len(dates) - 1,
        format_func=lambda value: pd.Timestamp(value).strftime("%b %d, %Y"),
        key="observation_date",
    )
field = VIEWS[view]
rows = predictions.loc[predictions.date == date].copy()
mapped = units.merge(rows, on="unit_id", validate="one_to_one").sort_values("unit_id")
if mapped.empty:
    st.info("No tracts passed the coverage checks for this date.")
    st.stop()

st.subheader(view)
if view == "Predicted temperature":
    st.caption(
        "Each tract's temperature was estimated by a model trained on other tracts. "
        "Darker orange and red indicate higher surface temperatures."
    )
elif view == "Observed temperature":
    st.caption(
        "Tract averages from the satellite's surface temperature retrieval. "
        "The colors use exactly the same temperature scale as the predictions."
    )
else:
    st.caption(
        "Error = predicted − observed. Blue means the model estimated too cool; "
        "red means it estimated too warm; near-white means closer agreement."
    )
st.caption(
    f"{len(mapped)} tracts shown · {pd.Timestamp(date).strftime('%B %d, %Y')} · "
    "Uncolored areas have no qualifying observation."
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
# Fixed across dates. The error scale is symmetric so zero stays at its center.
if view == "Prediction error":
    extent = max(float(predictions.prediction_error_celsius.abs().max()), 0.1)
    bins = [-extent + 2 * extent * i / 7 for i in range(8)]
    palette, legend = "RdBu_r", "Prediction error: predicted − observed (°C)"
else:
    low = min(predictions.observed_lst_celsius.min(), predictions.oof_predicted_lst_celsius.min())
    high = max(predictions.observed_lst_celsius.max(), predictions.oof_predicted_lst_celsius.max())
    high = max(high, low + 0.1)
    bins = [low + (high - low) * i / 7 for i in range(8)]
    palette, legend = "YlOrRd", "Land surface temperature (°C)"

# Round only the map's display properties; calculations and downloads retain full precision.
map_rows = mapped[["unit_id", "name", "geometry", *VIEWS.values()]].copy()
map_rows[list(VIEWS.values())] = map_rows[list(VIEWS.values())].round(1)
choropleth = folium.Choropleth(
    geo_data=map_rows.to_json(),
    data=mapped,
    columns=["unit_id", field],
    key_on="feature.properties.unit_id",
    bins=bins,
    fill_color=palette,
    fill_opacity=0.85,
    line_weight=0.6,
    line_color="#555555",
    legend_name=legend,
    highlight=True,
).add_to(map_view)
folium.GeoJsonTooltip(
    fields=["name", *VIEWS.values()],
    aliases=["Tract", "Predicted (°C)", "Observed (°C)", "Error: predicted − observed (°C)"],
    localize=True,
    sticky=False,
).add_to(choropleth.geojson)

options = mapped.unit_id.tolist()
names = mapped.set_index("unit_id").name.to_dict()
by_id = mapped.set_index("unit_id")
# Keep selections when changing views/dates, unless a tract has no data on that date.
for key in ("selected_tract", "comparison_a", "comparison_b"):
    if st.session_state.get(key) not in options:
        st.session_state[key] = None

map_column, detail_column = st.columns([3, 1.2])
with map_column:
    interaction = st_folium(
        map_view,
        key=f"tract-map-{date}-{view}",
        height=540,
        use_container_width=True,
        returned_objects=["last_object_clicked", "last_object_clicked_count"],
    )
    click = (interaction or {}).get("last_object_clicked")
    # Resolve click coordinates against polygons, without parsing tooltip text or custom JS.
    if click:
        click_key = (
            date,
            view,
            interaction.get("last_object_clicked_count"),
            click["lng"],
            click["lat"],
        )
        if click_key != st.session_state.get("last_map_click"):
            st.session_state["last_map_click"] = click_key
            hits = mapped.loc[mapped.geometry.covers(Point(click["lng"], click["lat"]))]
            if not hits.empty:
                st.session_state["selected_tract"] = hits.iloc[0].unit_id

with detail_column:
    st.subheader("Tract details")
    selected = st.selectbox(
        "Click the map or choose a tract",
        options,
        index=None,
        placeholder="Select a tract…",
        format_func=names.get,
        key="selected_tract",
    )
    if selected is None:
        st.write("Select a colored area on the map to see its temperatures and prediction error.")
    else:
        tract = by_id.loc[selected]
        st.metric("Predicted surface temperature", f"{tract.oof_predicted_lst_celsius:.1f} °C")
        st.metric("Observed surface temperature", f"{tract.observed_lst_celsius:.1f} °C")
        st.metric("Prediction error", f"{tract.prediction_error_celsius:+.1f} °C")
        st.caption("Predicted minus observed. Positive = too warm; negative = too cool.")
        if st.button("Compare this tract", use_container_width=True):
            st.session_state["comparison_a"] = selected

st.divider()
st.subheader("Compare two tracts")
st.write("Choose two areas to compare their surface temperatures on the selected date.")
first, second = st.columns(2)
with first:
    tract_a = st.selectbox(
        "Tract A",
        options,
        index=None,
        placeholder="Choose first tract…",
        format_func=names.get,
        key="comparison_a",
    )
# Avoid comparing a tract with itself, including after using the details button.
second_options = [unit for unit in options if unit != tract_a]
if st.session_state.get("comparison_b") not in second_options:
    st.session_state["comparison_b"] = None
with second:
    tract_b = st.selectbox(
        "Tract B",
        second_options,
        index=None,
        placeholder="Choose second tract…",
        format_func=names.get,
        key="comparison_b",
    )
if tract_a and tract_b:
    a, b = by_id.loc[tract_a], by_id.loc[tract_b]
    difference = a.oof_predicted_lst_celsius - b.oof_predicted_lst_celsius
    if abs(difference) < 0.05:
        st.write("The two predicted temperatures are approximately equal (difference below 0.1°C).")
    else:
        st.write(
            f"**{names[tract_a]}** is predicted to be **{abs(difference):.1f}°C "
            f"{'warmer' if difference > 0 else 'cooler'}** than **{names[tract_b]}**."
        )
    comparison = by_id.loc[[tract_a, tract_b], list(VIEWS.values())].T
    comparison.columns = [f"A · {names[tract_a]}", f"B · {names[tract_b]}"]
    comparison.index = [
        "Predicted surface temperature",
        "Observed surface temperature",
        "Prediction error",
    ]
    display = comparison.map(lambda value: f"{value:.1f} °C")
    display.loc["Prediction error"] = comparison.loc["Prediction error"].map(
        lambda value: f"{value:+.1f} °C"
    )
    st.table(display)
    st.caption("Error = predicted − observed. Differences describe these tracts on this date only.")
else:
    st.caption("Select both tracts to see their temperatures side by side.")

with st.expander("About the model, data quality, and limitations"):
    st.write(
        "The model estimates tract-mean land surface temperature from same-date satellite "
        "vegetation, built-up, and water-response indices. Every displayed prediction is "
        "spatially held out: the model that predicted a tract was trained without that tract "
        "or any of its dates. The final model's fitted, in-sample predictions are not displayed."
    )
    st.write(
        f"Selected model: **{metadata['selected_model'].replace('_', ' ').title()}**. "
        "Five-fold spatial validation holds out whole tracts. The table below pools held-out "
        "predictions across all three dates; selection used mean fold MAE. MAE is the average "
        "absolute prediction error. RMSE gives larger errors more weight; higher R² indicates "
        "more variation explained."
    )
    st.dataframe(
        pd.DataFrame(metadata["pooled_oof_metrics"])
        .T.rename(
            columns={
                "mae": "MAE (°C)",
                "rmse": "RMSE (°C)",
                "r2": "R²",
            }
        )
        .round(3)
    )
    st.write(
        "Observed values are averages of USGS satellite surface temperature retrievals, "
        "not ground thermometer measurements. Tracts are clipped to a generalized Long Beach "
        "boundary and are not named neighborhoods. An observation needs at least 30 valid "
        "pixels and 50% valid coverage. Missing areas are not evidence of cooler conditions."
    )
    if selected:
        tract = by_id.loc[selected]
        st.write(f"**{names[selected]}** · GEOID `{selected}` · {date}")
        st.dataframe(
            pd.DataFrame(
                {
                    "Measure": [
                        "NDVI (vegetation response)",
                        "NDBI (built-up response)",
                        "NDWI (water response)",
                        "Valid pixel fraction",
                        "Valid pixels",
                    ],
                    "Value": [
                        f"{tract.ndvi:.3f}",
                        f"{tract.ndbi:.3f}",
                        f"{tract.ndwi:.3f}",
                        f"{tract.valid_fraction:.1%}",
                        str(tract.valid_pixels),
                    ],
                }
            ),
            hide_index=True,
        )
    st.write(
        "Spectral indices are associations, not canopy or impervious percentages. "
        "Three summer 2023 dates are a small sample. Adjacent tracts can be in different "
        "folds; no spatial buffer or temporal holdout was evaluated. Model selection and "
        "evaluation reuse the same folds. Shoreline effects, retrieval error, and missing "
        "pixels can bias comparisons. Native thermal resolution is roughly 100 m, resampled "
        "to a 30 m grid. No causal cooling, health, equity, or future forecasting conclusions "
        "are supported, and no confidence intervals are available."
    )
    st.download_button(
        "Download this date's data (CSV)",
        rows[["unit_id", "date", *VIEWS.values()]]
        .rename(columns={"oof_predicted_lst_celsius": "spatially_held_out_predicted_lst_celsius"})
        .to_csv(index=False),
        file_name=f"coolblocks_{date}.csv",
        mime="text/csv",
    )
