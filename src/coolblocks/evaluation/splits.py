"""Spatial and temporal splits. Never split randomly by pixel: neighbors are near-duplicates."""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def spatial_group_splits(df: pd.DataFrame, group_col: str, n_splits: int = 5):
    """Yield (train_idx, test_idx) so every spatial unit is entirely in train or test."""
    gkf = GroupKFold(n_splits=n_splits)
    yield from gkf.split(df, groups=df[group_col])


def temporal_split(df: pd.DataFrame, date_col: str, holdout_year: int):
    """Train on all other years, test on one held-out year."""
    year = pd.to_datetime(df[date_col]).dt.year
    return np.where(year != holdout_year)[0], np.where(year == holdout_year)[0]
