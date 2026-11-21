"""Simple leakage checks. Log the output of these in results/ and discuss them in the report."""
import numpy as np
import pandas as pd


def assert_no_group_overlap(df, train_idx, test_idx, group_col: str) -> None:
    train_groups = set(df.iloc[train_idx][group_col])
    test_groups = set(df.iloc[test_idx][group_col])
    overlap = train_groups & test_groups
    if overlap:
        raise AssertionError(f"{len(overlap)} {group_col} values appear in both train and test")


def forbidden_features(columns, forbidden) -> list[str]:
    """Return any model feature that is on the forbidden list (e.g. raw thermal bands)."""
    forbidden = {f.lower() for f in forbidden}
    return [c for c in columns if c.lower() in forbidden]


def high_target_correlation(df: pd.DataFrame, target: str, threshold: float = 0.98) -> pd.Series:
    """Features nearly perfectly correlated with the target are suspicious."""
    corr = df.select_dtypes(include=np.number).corr()[target].drop(target).abs()
    return corr[corr >= threshold]
