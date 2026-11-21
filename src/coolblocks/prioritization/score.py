"""Configurable neighborhood prioritization. Pure functions only, so the app can call them."""
import pandas as pd


def minmax(s: pd.Series) -> pd.Series:
    """Scale to 0-1. A constant column maps to 0."""
    rng = s.max() - s.min()
    return (s - s.min()) / rng if rng else s * 0.0


def priority_score(df: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    """Weighted priority score; higher means higher priority.

    weights maps column name -> non-negative weight (normalized internally). Columns are
    min-max scaled first. For 'low is bad' columns (e.g. canopy), pass a pre-computed
    deficit column such as canopy_deficit = 100 - tree_canopy_pct.
    """
    if not weights:
        raise ValueError("weights must not be empty")
    if any(w < 0 for w in weights.values()):
        raise ValueError("weights must be non-negative")
    total = sum(weights.values())
    if total == 0:
        raise ValueError("at least one weight must be positive")
    missing = [c for c in weights if c not in df.columns]
    if missing:
        raise KeyError(f"columns not in dataframe: {missing}")

    out = df.copy()
    out["priority_score"] = sum(minmax(out[c]) * (w / total) for c, w in weights.items())
    out["priority_rank"] = out["priority_score"].rank(ascending=False, method="min").astype(int)
    return out.sort_values("priority_rank")
