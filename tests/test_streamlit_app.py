"""UI integration checks use existing real outputs, with simulated map click events.

Run `make data features train` first for the output-dependent checks. CI can still
check missing-data guidance without downloading satellite rasters.
"""

from pathlib import Path

import folium
import geopandas as gpd
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit_app.py"


@pytest.fixture
def live_app(monkeypatch):
    paths = [
        ROOT / "data/processed/units.geojson",
        ROOT / "results/predictions.parquet",
        ROOT / "results/run_metadata.json",
    ]
    if not all(path.exists() for path in paths):
        pytest.skip("Real-output UI checks require make data features train")
    captured, event = [], {}

    def render_map(map_view, **kwargs):
        assert kwargs["returned_objects"] == ["last_object_clicked", "last_object_clicked_count"]
        # Render real Folium HTML, while simulating only the browser's click response.
        map_view.get_root().render()
        captured.append(map_view)
        return dict(event)

    monkeypatch.setattr("streamlit_folium.st_folium", render_map)
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception
    return app, event, captured, pd.read_parquet(paths[1]), gpd.read_file(paths[0])


def assert_ok(app):
    assert not app.exception, [str(error) for error in app.exception]


def test_missing_outputs_explain_how_to_prepare_data(tmp_path):
    source = APP.read_text().replace(
        "ROOT = Path(__file__).resolve().parents[1]", f"ROOT = Path({str(tmp_path)!r})"
    )
    app = AppTest.from_string(source, default_timeout=30).run()
    assert_ok(app)
    assert "make data features train" in app.warning[0].value


def test_default_prediction_view_and_consistent_scales(live_app):
    app, _, maps, predictions, _ = live_app
    assert app.radio(key="map_view").value == "Predicted temperature"
    assert app.selectbox(key="selected_tract").value is None
    assert all(not section.proto.expanded for section in app.expander)
    temperature_scale = None
    for date in sorted(predictions.date.unique()):
        app.selectbox(key="observation_date").set_value(date).run()
        for view in ("Predicted temperature", "Observed temperature", "Prediction error"):
            app.radio(key="map_view").set_value(view).run()
            assert_ok(app)
            layer = next(
                child
                for child in maps[-1]._children.values()
                if isinstance(child, folium.Choropleth)
            )
            scale = layer.color_scale
            if view == "Prediction error":
                assert scale.vmin == pytest.approx(-scale.vmax)
                assert scale.colors[0][2] > scale.colors[0][0]  # negative: blue
                assert scale.colors[-1][0] > scale.colors[-1][2]  # positive: red
            else:
                if temperature_scale is None:
                    temperature_scale = scale.index
                assert scale.index == temperature_scale
            for feature in layer.geojson.data["features"]:
                properties = feature["properties"]
                assert "predicted_lst_celsius" not in properties  # fitted predictions stay hidden
                assert "prediction_error_celsius" in properties
            assert maps[-1].options["max_bounds_viscosity"] == 1.0
            assert "background: white" in maps[-1].get_root().header.render()


def test_click_inspection_comparison_and_repeated_clicks(live_app):
    app, event, _, predictions, units = live_app
    date = app.selectbox(key="observation_date").value
    rows = predictions.loc[predictions.date == date].set_index("unit_id")
    tract_a, tract_b = rows.index[:2]
    point = units.set_index("unit_id").loc[tract_a].geometry.representative_point()
    event.update(last_object_clicked={"lng": point.x, "lat": point.y}, last_object_clicked_count=1)
    app.run()
    assert_ok(app)
    assert app.selectbox(key="selected_tract").value == tract_a
    a = rows.loc[tract_a]
    expected_error = a.oof_predicted_lst_celsius - a.observed_lst_celsius
    assert [metric.value for metric in app.metric] == [
        f"{a.oof_predicted_lst_celsius:.1f} °C",
        f"{a.observed_lst_celsius:.1f} °C",
        f"{expected_error:+.1f} °C",
    ]
    # A stale browser event must not override the accessible dropdown selection.
    app.selectbox(key="selected_tract").set_value(tract_b).run()
    assert app.selectbox(key="selected_tract").value == tract_b
    # Clicking the exact same map point again should select it again.
    event["last_object_clicked_count"] = 2
    app.run()
    assert app.selectbox(key="selected_tract").value == tract_a
    app.button[0].click().run()
    assert app.selectbox(key="comparison_a").value == tract_a
    app.selectbox(key="comparison_b").set_value(tract_b).run()
    assert_ok(app)
    comparison = app.table[0].value
    assert comparison.loc["Predicted surface temperature"].iloc[0] == (
        f"{a.oof_predicted_lst_celsius:.1f} °C"
    )
    assert comparison.loc["Prediction error"].iloc[0] == f"{expected_error:+.1f} °C"
    # Selecting B as A clears B rather than comparing a tract with itself.
    app.selectbox(key="comparison_a").set_value(tract_b).run()
    assert_ok(app)
    assert app.selectbox(key="comparison_b").value is None


def test_date_change_clears_tracts_without_observations(live_app):
    app, event, _, predictions, _ = live_app
    dates = sorted(predictions.date.unique())
    latest = set(predictions.loc[predictions.date == dates[-1], "unit_id"])
    earlier = set(predictions.loc[predictions.date == dates[0], "unit_id"])
    missing = latest - earlier
    if not missing:
        pytest.skip("Dataset has no missing-date tract to exercise this case")
    tract = min(missing)
    app.selectbox(key="selected_tract").set_value(tract).run()
    app.button[0].click().run()
    event.clear()
    app.selectbox(key="observation_date").set_value(dates[0]).run()
    assert_ok(app)
    assert app.selectbox(key="selected_tract").value is None
    assert app.selectbox(key="comparison_a").value is None
    assert not app.metric
