from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def build_model(name: str, seed: int = 42):
    if name == "mean_baseline":
        return DummyRegressor(strategy="mean")
    if name == "linear":
        return make_pipeline(StandardScaler(), LinearRegression())
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=seed
        )
    raise ValueError(f"Unknown model: {name}")
