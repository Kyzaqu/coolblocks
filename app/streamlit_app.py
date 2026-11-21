"""CoolBlocks prototype. Reads precomputed files, so it runs without retraining.

Expected input: data/processed/neighborhoods.geojson with columns
  unit_id, name, lst_mean, svi_overall, canopy_deficit, ...
"""
from pathlib import Path

import streamlit as st

from coolblocks.prioritization.score import priority_score

DATA = Path("data/processed/neighborhoods.geojson")

st.set_page_config(page_title="CoolBlocks", layout="wide")
st.title("CoolBlocks: neighborhood heat analysis (Long Beach)")
st.caption("Land surface temperature is not the same as the air temperature people feel on the street.")

if not DATA.exists():
    st.warning(f"{DATA} not found. Run `make data features train` first.")
    st.stop()

import folium
import geopandas as gpd
from streamlit_folium import st_folium

gdf = gpd.read_file(DATA)

st.sidebar.header("Prioritization weights")
w_heat = st.sidebar.slider("Heat (LST)", 0.0, 1.0, 0.5)
w_svi = st.sidebar.slider("Social vulnerability (SVI)", 0.0, 1.0, 0.3)
w_can = st.sidebar.slider("Canopy deficit", 0.0, 1.0, 0.2)

weights = {"lst_mean": w_heat, "svi_overall": w_svi, "canopy_deficit": w_can}
if sum(weights.values()) == 0:
    st.info("Set at least one weight above zero.")
    st.stop()

ranked = priority_score(gdf, weights)

m = folium.Map(location=[33.80, -118.17], zoom_start=12, tiles="cartodbpositron")
folium.Choropleth(
    geo_data=ranked,
    data=ranked,
    columns=["unit_id", "priority_score"],
    key_on="feature.properties.unit_id",
    fill_color="YlOrRd",
    legend_name="Priority score",
).add_to(m)

col1, col2 = st.columns([3, 2])
with col1:
    st_folium(m, height=550, use_container_width=True)
with col2:
    st.subheader("Top 10 priority neighborhoods")
    st.dataframe(ranked[["priority_rank", "name", "lst_mean", "svi_overall"]].head(10))
