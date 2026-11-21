"""Census GEOID helpers.

A block group GEOID has 12 digits: state (2) + county (3) + tract (6) + block group (1).
The tract GEOID is the first 11 digits, so block groups in the same tract share a tract_id
(and therefore share CDC SVI values, which are published at tract level).
"""
import pandas as pd

BLOCK_GROUP_LEN = 12
TRACT_LEN = 11


def tract_id_from_unit_id(unit_id: pd.Series) -> pd.Series:
    """Return the 11-digit tract GEOID for each 12-digit block group GEOID."""
    unit_id = unit_id.astype(str)
    bad = unit_id[unit_id.str.len() != BLOCK_GROUP_LEN]
    if len(bad):
        raise ValueError(
            f"{len(bad)} unit_id values are not 12-digit block group GEOIDs, e.g. {bad.iloc[0]!r}. "
            "Keep GEOIDs as strings so leading zeros (e.g. '06037...') are not lost."
        )
    return unit_id.str[:TRACT_LEN]


def add_tract_id(df: pd.DataFrame, unit_col: str = "unit_id") -> pd.DataFrame:
    out = df.copy()
    out[unit_col] = out[unit_col].astype(str)
    out["tract_id"] = tract_id_from_unit_id(out[unit_col])
    return out
